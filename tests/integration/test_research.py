"""Research pipeline + CLIs: training verdicts, promotion rules, tick download."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from app.controller import Controller
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.deriv.websocket import DerivWebSocket
from app.ml.artifacts import read_metadata
from app.ml.dataset import load_ticks_csv
from app.ml.pipeline import (
    evaluate_demo_promotion,
    start_demo_validation,
    train_and_validate,
)
from app.ml.predict import ModelPredictor
from app.models.schemas import Direction, Mode, ModelStatus
from app.storage.database import Database
from app.storage.repositories import Repositories
from research import build_dataset, download_ticks, evaluate, train_model, walk_forward
from tests.mocks.deriv_mock import MockDeriv
from tests.mocks.stubs import make_stub_model
from tests.unit.synth import momentum_walk, random_walk


class Quick(TypedDict):
    n_shuffles: int
    min_trades: int
    horizon: int
    symbol: str


QUICK: Quick = {"n_shuffles": 3, "min_trades": 50, "horizon": 5, "symbol": "R_100"}


def write_ticks(path: Path, epochs: np.ndarray, prices: np.ndarray) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "quote"])
        for e, p in zip(epochs, prices, strict=True):
            w.writerow([int(e), f"{p:.6f}"])
    return path


def test_random_walk_is_rejected_and_the_bot_will_not_trade_it(tmp_path: Path) -> None:
    ep, p = random_walk(20000, seed=21)
    report = train_and_validate(ep, p, model_dir=tmp_path / "m", **QUICK)
    assert report.verdict == "NO EVIDENCE OF EDGE"
    assert report.status == ModelStatus.REJECTED.value
    meta = read_metadata(tmp_path / "m")
    assert meta.status == "REJECTED" and meta.verdict_reasons
    assert (tmp_path / "m" / "report.json").exists()
    with pytest.raises(ValueError, match="CALIBRATED"):
        start_demo_validation(tmp_path / "m")  # a rejected model cannot enter demo validation


def test_planted_signal_is_found_and_needs_manual_steps_to_trade(tmp_path: Path) -> None:
    ep, p = momentum_walk(20000, seed=22)
    report = train_and_validate(ep, p, model_dir=tmp_path / "m", **QUICK)
    assert report.verdict == "EDGE EVIDENCE"
    assert report.status in (ModelStatus.CALIBRATED.value, ModelStatus.WALK_FORWARD_VALID.value)
    assert report.edge["significant"] and report.trades >= 50
    meta = read_metadata(tmp_path / "m")
    assert meta.feature_version == "v1" and meta.label_horizon == 5 and meta.git_commit != ""
    if report.status == ModelStatus.CALIBRATED.value:
        assert start_demo_validation(tmp_path / "m").status == "DEMO_VALIDATING"
    pred = ModelPredictor.load(tmp_path / "m")
    assert pred.status is not ModelStatus.PROMOTABLE  # never automatic


def test_too_little_data_is_refused(tmp_path: Path) -> None:
    ep, p = random_walk(500)
    with pytest.raises(ValueError, match="not enough"):
        train_and_validate(ep, p, model_dir=tmp_path / "m", **QUICK)


def seed_demo_trades(db_path: Path, wins: int, losses: int, be: float = 0.5128) -> None:
    db = Database(str(db_path))
    repos = Repositories(db)
    from app.models.schemas import OrderState, Signal

    now = datetime(2026, 9, 30, tzinfo=UTC)
    with db.transaction():
        for i in range(wins + losses):
            sig = Signal(f"s{i}", "R_100", Direction.CALL, i, "ml", "1", "stub-1", 0.7, 5, 0.0, 0.0)
            repos.insert_signal(Mode.DEMO, sig, now)
            repos.insert_order(
                order_id=f"o{i}",
                signal=sig,
                mode=Mode.DEMO,
                stake=Decimal("1"),
                break_even=be,
                ts=now,
            )
            won = i < wins
            repos.insert_trade(
                contract_id=i + 1,
                order_id=f"o{i}",
                mode=Mode.DEMO,
                symbol="R_100",
                direction="CALL",
                stake=Decimal("1"),
                buy_price=Decimal("1"),
                payout=Decimal("1.95"),
                probability=0.7,
                break_even=be,
                ts=now,
            )
            repos.settle_trade(
                i + 1,
                OrderState.WON if won else OrderState.LOST,
                Decimal("0.95") if won else Decimal("-1"),
                None,
                None,
                now,
            )
    db.close()


def test_promotion_requires_many_demo_trades_and_statistical_significance(tmp_path: Path) -> None:
    make_stub_model(tmp_path / "m", status=ModelStatus.DEMO_VALIDATING)
    few = tmp_path / "few.db"
    seed_demo_trades(few, 90, 60)  # 60% but only 150 trades
    ok, test, why = evaluate_demo_promotion(tmp_path / "m", few, min_demo_trades=1000)
    assert not ok and "150" in why and test.n == 150

    weak = tmp_path / "weak.db"
    seed_demo_trades(weak, 620, 580)  # 51.7% vs 51.28% break-even: not significant
    ok, test, why = evaluate_demo_promotion(tmp_path / "m", weak, min_demo_trades=1000)
    assert not ok and "not significantly" in why

    strong = tmp_path / "strong.db"
    seed_demo_trades(strong, 720, 480)  # 60% over 1200 trades
    ok, test, _ = evaluate_demo_promotion(tmp_path / "m", strong, min_demo_trades=1000)
    assert ok and test.significant
    assert read_metadata(tmp_path / "m").status == "DEMO_VALIDATING"  # dry run changed nothing
    ok, _, _ = evaluate_demo_promotion(tmp_path / "m", strong, min_demo_trades=1000, apply=True)
    assert ok and read_metadata(tmp_path / "m").status == "PROMOTABLE"
    assert read_metadata(tmp_path / "m").demo_stats["significant"] is True


def test_promotion_only_from_demo_validating(tmp_path: Path) -> None:
    make_stub_model(tmp_path / "m", status=ModelStatus.CALIBRATED)
    with pytest.raises(ValueError, match="DEMO_VALIDATING"):
        evaluate_demo_promotion(tmp_path / "m", tmp_path / "x.db")


async def test_rejected_model_never_trades_in_the_controller(
    controller: Controller, tmp_path: Path
) -> None:
    make_stub_model(Path(controller.config.app.model_dir), status=ModelStatus.REJECTED, p_up=0.9)
    assert controller.reload_model()
    await controller.start()
    from tests.integration.test_executor import mk_signal

    await controller.process_signal_now(mk_signal(p=0.9))
    assert controller.repos.db.query("SELECT * FROM orders") == []
    rules = [r["rule"] for r in controller.repos.recent_risk_events(20)]
    assert "model_status" in rules


async def test_download_ticks_pages_and_writes_csv(mock: MockDeriv, tmp_path: Path) -> None:
    async def url() -> str:
        return mock.url

    ws = DerivWebSocket(url, RateLimiter(500, 100), Backoff(0.05, 2, 0.2, 0.0))
    await ws.start()
    try:
        rows = await download_ticks.download(ws, "R_100", total=250, chunk=100)
    finally:
        await ws.close()
    assert len(rows) == 250 and len(set(rows)) == 250
    out = tmp_path / "t.csv"
    download_ticks.write_csv(rows, out)
    epochs, prices = load_ticks_csv(out)
    assert len(epochs) > 0 and np.all(np.diff(epochs) >= 0) and len(prices) == len(epochs)


def test_cli_entry_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no config.yaml here => safe defaults
    ep, p = momentum_walk(12000, seed=5)
    ticks = write_ticks(tmp_path / "t.csv", ep, p)
    npz = tmp_path / "d.npz"
    assert build_dataset.main(["--ticks", str(ticks), "--horizon", "5", "--out", str(npz)]) == 0
    assert np.load(npz)["x"].shape[1] == 12
    assert walk_forward.main(["--ticks", str(ticks), "--horizon", "5", "--shuffles", "2"]) == 0
    assert "VERDICT" in capsys.readouterr().out
    model_dir = tmp_path / "model"
    args = [
        "--symbol",
        "R_100",
        "--ticks",
        str(ticks),
        "--horizon",
        "5",
        "--model-dir",
        str(model_dir),
        "--shuffles",
        "2",
        "--min-trades",
        "50",
    ]
    assert train_model.main(args) == 0
    assert "VERDICT" in capsys.readouterr().out
    assert (model_dir / "model.joblib").exists() and (model_dir / "metadata.json").exists()
    assert evaluate.main(["--model-dir", str(model_dir), "--ticks", str(ticks)]) == 0
    out = capsys.readouterr().out
    assert "win_rate=" in out
    status = read_metadata(model_dir).status
    if status == "CALIBRATED":
        assert evaluate.main(["--model-dir", str(model_dir), "--start-demo-validation"]) == 0
        assert read_metadata(model_dir).status == "DEMO_VALIDATING"
