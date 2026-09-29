"""End-to-end research pipeline: dataset -> walk-forward -> shuffled control -> final model.

A model is only ever marked WALK_FORWARD_VALID / CALIBRATED when the out-of-sample evidence is
statistically credible AND it beats the shuffled-label control. Otherwise it is REJECTED and the
bot will refuse to trade it. Promotion beyond CALIBRATED requires demo evidence (see
`evaluate_demo_promotion`) and an explicit manual command.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from app.ml.artifacts import ModelMetadata, read_metadata, save_artifact, set_status
from app.ml.dataset import Dataset, build_dataset
from app.ml.features import FEATURE_VERSION
from app.ml.statistics import EdgeTest, break_even_from_ratio, edge_test
from app.ml.train import fit_heads
from app.ml.validate import (
    EDGE,
    NO_EDGE,
    Verdict,
    WalkForwardResult,
    run_walk_forward,
    shuffled_label_control,
)
from app.models.schemas import ModelStatus
from app.products import Product, ProductSpec, assumed_terms

MAX_CALIBRATION_GAP = 0.05  # |predicted - actual| per calibration bin


@dataclass
class TrainReport:
    status: str
    verdict: str
    reasons: list[str]
    samples: int
    oos_samples: int
    trades: int
    wins: int
    edge: dict[str, Any]
    pooled_metrics: dict[str, Any] | None
    calibration: list[dict[str, float]]
    calibration_max_gap: float | None
    shuffled_aucs: list[float]
    real_auc: float | None
    shuffled_auc_p95: float | None
    payout_ratio: float
    edge_margin: float
    horizon: int
    model_kind: str
    folds: int
    note: str = (
        "payout_ratio is a research assumption; the live bot always uses the actual proposal payout"
    )


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).isoformat()


def calibration_gap(table: list[dict[str, float]]) -> float | None:
    if not table:
        return None
    return max(abs(r["predicted"] - r["actual"]) for r in table)


def train_and_validate(
    epochs: npt.NDArray[np.int64],
    prices: npt.NDArray[np.float64],
    *,
    symbol: str,
    horizon: int,
    model_dir: str | Path,
    kind: str = "logistic",
    payout_ratio: float = 1.95,
    edge_margin: float = 0.03,
    alpha: float = 0.05,
    min_trades: int = 300,
    n_shuffles: int = 20,
    n_comparisons: int = 1,
    initial_train_fraction: float = 0.5,
    test_fraction: float = 0.1,
    seed: int = 7,
    spec: ProductSpec | None = None,
) -> TrainReport:
    if spec is None:  # classic Rise/Fall
        spec = ProductSpec(
            product=Product.RISE_FALL, horizon_ticks=horizon, payout_ratio=payout_ratio
        )
    horizon = spec.horizon_ticks
    win, loss = assumed_terms(spec)
    terms = (win, loss)
    payout_ratio = (win + loss) / loss  # payout-equivalent ratio of the assumed terms
    ds: Dataset = build_dataset(epochs, prices, horizon, spec)
    n = len(ds)
    initial_train = int(n * initial_train_fraction)
    test_size = max(int(n * test_fraction), 200)
    gap = max(horizon, 1)
    if n < 2000 or initial_train + gap + test_size > n:
        raise ValueError(f"not enough ticks for walk-forward validation (usable samples: {n})")

    real: WalkForwardResult = run_walk_forward(
        ds,
        kind=kind,
        initial_train=initial_train,
        gap=gap,
        test_size=test_size,
        payout_ratio=payout_ratio,
        terms=terms,
        edge_margin=edge_margin,
        alpha=alpha,
        seed=seed,
    )
    verdict: Verdict = shuffled_label_control(
        ds,
        real,
        kind=kind,
        initial_train=initial_train,
        gap=gap,
        test_size=test_size,
        n_shuffles=n_shuffles,
        alpha=alpha,
        min_trades=min_trades,
        n_comparisons=n_comparisons,
        payout_ratio=payout_ratio,
        terms=terms,
        edge_margin=edge_margin,
    )
    cal_gap = calibration_gap(real.calibration)
    if verdict.label == EDGE and cal_gap is not None and cal_gap <= MAX_CALIBRATION_GAP:
        status = ModelStatus.CALIBRATED
    elif verdict.label == EDGE:
        status = ModelStatus.WALK_FORWARD_VALID
    else:
        status = ModelStatus.REJECTED

    try:
        final = fit_heads(ds, kind=kind, seed=seed)
    except ValueError as exc:
        raise ValueError(
            f"{exc}: the profit target is (almost) never or always reached with these trade "
            "terms on this data - adjust the terms (targets, barriers, horizon)"
        ) from exc
    version = f"{symbol}-{spec.product.value}-{kind}-h{horizon}-{datetime.now(UTC):%Y%m%d%H%M%S}"
    n_cal = int(n * 0.25)
    meta = ModelMetadata(
        model_version=version,
        model_kind=kind,
        status=status.value,
        feature_version=FEATURE_VERSION,
        label_horizon=horizon,
        symbols=[symbol],
        training_dates={"start": _iso(int(epochs[0])), "end": _iso(int(epochs[-1]))},
        training_samples=n - n_cal,
        validation_samples=n_cal,
        test_samples=int(len(real.y)),
        payout_assumption=payout_ratio,
        break_even=loss / (win + loss),
        product=spec.product.value,
        spec=spec.model_dump(mode="json"),
        edge_margin=edge_margin,
        observed_win_rate=real.edge.win_rate if real.trades else None,
        confidence_interval=[real.edge.ci_low, real.edge.ci_high],
        p_value=real.edge.p_value if real.trades else None,
        calibration_metrics={
            "table": real.calibration,
            "max_gap": cal_gap,
            "pooled": real.pooled_metrics.as_dict() if real.pooled_metrics else None,
        },
        tie_rate=ds.tie_rate,
        verdict=verdict.label,
        verdict_reasons=verdict.reasons,
    )
    save_artifact(model_dir, final, meta)
    report = TrainReport(
        status=status.value,
        verdict=verdict.label,
        reasons=verdict.reasons,
        samples=n,
        oos_samples=int(len(real.y)),
        trades=real.trades,
        wins=real.wins,
        edge=real.edge.as_dict(),
        pooled_metrics=real.pooled_metrics.as_dict() if real.pooled_metrics else None,
        calibration=real.calibration,
        calibration_max_gap=cal_gap,
        shuffled_aucs=verdict.shuffled_aucs,
        real_auc=verdict.real_auc,
        shuffled_auc_p95=verdict.shuffled_auc_p95,
        payout_ratio=payout_ratio,
        edge_margin=edge_margin,
        horizon=horizon,
        model_kind=kind,
        folds=len(real.folds),
    )
    Path(model_dir, "report.json").write_text(json.dumps(asdict(report), indent=2, default=str))
    return report


def start_demo_validation(model_dir: str | Path) -> ModelMetadata:
    """Manual step: CALIBRATED -> DEMO_VALIDATING (allows trading on the DEMO account only)."""
    meta = read_metadata(model_dir)
    if meta.model_status() is not ModelStatus.CALIBRATED:
        raise ValueError(f"only a CALIBRATED model can start demo validation (is {meta.status})")
    return set_status(model_dir, ModelStatus.DEMO_VALIDATING)


def evaluate_demo_promotion(
    model_dir: str | Path,
    db_path: str | Path,
    *,
    min_demo_trades: int = 1000,
    alpha: float = 0.05,
    apply: bool = False,
) -> tuple[bool, EdgeTest, str]:
    """Decide DEMO_VALIDATING -> PROMOTABLE from REAL demo trades of this exact model version.

    Requires enough settled demo trades AND a win rate significantly above the average
    break-even actually paid. Never automatic: `apply=True` is an explicit operator action."""
    meta = read_metadata(model_dir)
    if meta.model_status() not in (ModelStatus.DEMO_VALIDATING, ModelStatus.PROMOTABLE):
        raise ValueError(f"model must be DEMO_VALIDATING (is {meta.status})")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT t.profit, t.break_even, t.target_hit FROM trades t "
            "JOIN orders o ON o.order_id=t.order_id "
            "JOIN signals s ON s.signal_id=o.signal_id "
            "WHERE t.mode='demo' AND t.settled_at IS NOT NULL AND s.model_version=?",
            (meta.model_version,),
        ).fetchall()
    finally:
        conn.close()
    wins = sum(1 for p, _, hit in rows if (bool(hit) if hit is not None else Decimal(str(p)) > 0))
    bes = [float(b) for _, b, _ in rows if b is not None]
    be = sum(bes) / len(bes) if bes else break_even_from_ratio(meta.payout_assumption)
    test = edge_test(wins, len(rows), be, alpha)
    if len(rows) < min_demo_trades:
        return False, test, f"only {len(rows)} settled demo trades (< {min_demo_trades})"
    if not test.significant:
        return (
            False,
            test,
            (
                f"demo win rate {test.win_rate:.4f} not significantly above break-even {be:.4f} "
                f"(p={test.p_value:.4f})"
            ),
        )
    if apply:
        set_status(model_dir, ModelStatus.PROMOTABLE, demo_stats=test.as_dict())
    return True, test, "demo evidence supports promotion" + ("" if apply else " (dry run)")


__all__ = [
    "EDGE",
    "NO_EDGE",
    "TrainReport",
    "evaluate_demo_promotion",
    "start_demo_validation",
    "train_and_validate",
]
