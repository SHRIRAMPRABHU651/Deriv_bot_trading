"""Statistics, protocol parsing, backoff, rate limiter, logging redaction, storage, telegram."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from app.alerts.telegram import TelegramAlerter
from app.deriv import protocol
from app.deriv.auth import parse_accounts
from app.deriv.protocol import DerivError, ProtocolError, RateLimitError
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.logging_setup import JsonFormatter, redact, register_secrets
from app.metrics.latency import FeedHealth, percentile
from app.ml.statistics import (
    binomial_pvalue_greater,
    break_even_from_ratio,
    cohens_h,
    edge_test,
    expected_value,
    required_sample_size,
    rolling_edge_check,
    wilson_interval,
)
from app.models.schemas import Direction, Mode
from app.products import Product, ProductSpec, proposal_params
from app.storage.database import Database
from app.storage.repositories import Repositories
from tests.conftest import make_settings


# ---------------------------------------------------------------- statistics
def test_break_even_and_ev() -> None:
    assert break_even_from_ratio(1.95) == pytest.approx(0.5128205)
    assert break_even_from_ratio(2.0) == 0.5
    assert expected_value(0.6, 1.95) == pytest.approx(0.17)
    assert expected_value(1 / 1.95, 1.95) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        break_even_from_ratio(0)


def test_binomial_test_matches_known_value() -> None:
    assert binomial_pvalue_greater(60, 100, 0.5) == pytest.approx(0.02844, abs=1e-4)
    assert binomial_pvalue_greater(50, 100, 0.5) > 0.4
    assert binomial_pvalue_greater(0, 0, 0.5) == 1.0


def test_wilson_interval_and_effect_size() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert cohens_h(0.6, 0.5) == pytest.approx(0.2013, abs=1e-3)


def test_edge_test_significance_needs_sample_size() -> None:
    small = edge_test(8, 10, 0.5128)  # 80% but only 10 trades
    assert not small.significant
    big = edge_test(600, 1000, 0.5128)
    assert big.significant and big.ci_low > 0.5128
    assert edge_test(0, 0, 0.5).significant is False


def test_small_edges_need_thousands_of_trades() -> None:
    assert required_sample_size(0.5128, 0.5428) > 1000
    with pytest.raises(ValueError):
        required_sample_size(0.55, 0.5)


def test_rolling_check_halts_only_when_significantly_below_break_even() -> None:
    assert rolling_edge_check(30, 100, 0.5128) is True
    assert rolling_edge_check(50, 100, 0.5128) is False
    assert rolling_edge_check(0, 0, 0.5) is False


# ---------------------------------------------------------------- protocol
def test_builders_use_documented_shapes() -> None:
    assert protocol.ping() == {"ping": 1}
    assert protocol.ticks("R_100") == {"ticks": "R_100", "subscribe": 1}
    spec = ProductSpec(product=Product.RISE_FALL, horizon_ticks=5)
    p = protocol.proposal(
        symbol="R_100",
        amount=Decimal("1.5"),
        currency="USD",
        product_params=proposal_params(spec, Direction.CALL, Decimal("1.5")),
    )
    assert p["basis"] == "stake" and p["duration_unit"] == "t" and p["amount"] == 1.5
    assert p["contract_type"] == "CALL" and p["duration"] == 5 and p["symbol"] == "R_100"
    assert protocol.sell(9) == {"sell": 9, "price": 0}
    assert protocol.buy("abc", Decimal("1.5"))["price"] == 1.5
    assert protocol.proposal_open_contract(7, True) == {
        "proposal_open_contract": 1,
        "contract_id": 7,
        "subscribe": 1,
    }
    assert protocol.forget("sid") == {"forget": "sid"}


def test_parsers_and_error_mapping() -> None:
    t = protocol.parse_tick({"tick": {"symbol": "R_100", "epoch": 5, "quote": 1.5, "id": "x"}}, 9.0)
    assert t is not None and t.epoch == 5 and t.tick_id == "x"
    assert protocol.parse_tick({"tick": {"symbol": "R_100"}}, 1.0) is None
    assert protocol.parse_tick({}, 1.0) is None
    prop = protocol.parse_proposal({"proposal": {"id": "p", "ask_price": 10, "payout": 19.5}})
    assert prop.payout_ratio == Decimal("1.95")
    with pytest.raises(ProtocolError):
        protocol.parse_proposal({"proposal": {"id": "p"}})
    with pytest.raises(ProtocolError):
        protocol.parse_buy({})
    assert protocol.parse_contract({"proposal_open_contract": {}}) is None
    bal = protocol.parse_balance({"balance": {"balance": 12.5, "currency": "USD", "loginid": "X"}})
    assert bal.balance == Decimal("12.5")
    err = protocol.error_from_message(
        {"error": {"code": "RateLimit", "message": "slow"}, "req_id": 3}
    )
    assert isinstance(err, RateLimitError) and err.req_id == 3
    err2 = protocol.error_from_message({"error": {"code": "Other", "message": "m"}})
    assert isinstance(err2, DerivError) and not isinstance(err2, RateLimitError)
    assert protocol.parse_history({"history": {"times": [1, 2], "prices": [1.0, 2.0]}}) == [
        (1, 1.0),
        (2, 2.0),
    ]
    with pytest.raises(ProtocolError):
        protocol.parse_history({"history": {"times": [1], "prices": []}})


def test_parse_accounts_variants() -> None:
    a = parse_accounts({"data": [{"account_id": "DOT1", "account_type": "DEMO", "balance": 10}]})
    assert a[0].account_type == "demo" and a[0].balance == Decimal("10")
    b = parse_accounts({"data": {"accounts": [{"account_id": "X", "account_type": "real"}]}})
    assert b[0].account_type == "real"
    assert parse_accounts({"data": []}) == []


# ---------------------------------------------------------------- backoff / rate limiter
def test_backoff_sequence_caps_and_resets() -> None:
    b = Backoff(rng=lambda: 0.0)
    assert [b.next_delay() for _ in range(8)] == [1, 2, 4, 8, 16, 30, 30, 30]
    b.reset()
    assert b.next_delay() == 1
    j = Backoff(rng=lambda: 1.0)
    delays = [j.next_delay() for _ in range(8)]
    assert all(d <= 30 for d in delays) and delays[0] == pytest.approx(1.25)


async def test_rate_limiter_enforces_rate() -> None:
    rl = RateLimiter(rate_per_second=20, burst=2)
    t0 = time.monotonic()
    for _ in range(6):
        await rl.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.15  # 2 burst + 4 more at 20/s
    with pytest.raises(ValueError):
        RateLimiter(0, 1)


# ---------------------------------------------------------------- latency / feed
def test_percentile_and_feed_health() -> None:
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5], 50) == 5
    f = FeedHealth()
    f.watch(["A", "B"], 100.0)
    assert not f.any_stale(105.0, 10)
    f.on_tick("A", 120.0)
    assert f.is_stale("B", 120.0, 10) and f.any_stale(120.0, 10)
    assert f.is_stale("nope", 1.0, 10)


# ---------------------------------------------------------------- logging
def test_logs_redact_tokens_and_registered_secrets() -> None:
    register_secrets("SUPERSECRETVALUE123")
    text = redact(
        'Authorization: Bearer abc.DEF-123 url=wss://x/ws?otp=ONETIME token="SUPERSECRETVALUE123"'
    )
    assert "abc.DEF-123" not in text and "ONETIME" not in text and "SUPERSECRETVALUE123" not in text
    rec = logging.LogRecord("derivbot.x", logging.INFO, "f", 1, "msg SUPERSECRETVALUE123", (), None)
    rec.symbol = "R_100"
    rec.latency_ms = 12
    out = json.loads(JsonFormatter().format(rec))
    assert out["symbol"] == "R_100" and "SUPERSECRETVALUE123" not in json.dumps(out)
    assert {"timestamp", "level", "component", "event"} <= set(out)


# ---------------------------------------------------------------- storage
def test_tables_indexes_unique_and_cleanup(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "x.db"))
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "bot_state",
        "risk_state",
        "account_snapshots",
        "signals",
        "orders",
        "trades",
        "risk_events",
        "latency_samples",
        "ticks",
        "model_runs",
        "reconciliation_events",
    } <= tables
    assert db.query("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
    repos = Repositories(db)
    now = datetime(2026, 9, 30, tzinfo=UTC)
    repos.add_tick("R_100", int((now - timedelta(days=10)).timestamp()), 1.0)
    repos.add_tick("R_100", int(now.timestamp()), 2.0)
    repos.add_latency(Mode.DEMO, "x", 1.0, now - timedelta(days=40))
    repos.add_latency(Mode.DEMO, "x", 2.0, now)
    repos.add_risk_event(
        Mode.DEMO,
        __import__("app.models.schemas", fromlist=["Severity"]).Severity.INFO,
        "r",
        "m",
        ts=now - timedelta(days=400),
    )
    assert repos.cleanup(now, ticks_days=7, latency_days=30) == (1, 1)
    assert len(db.query("SELECT * FROM ticks")) == 1
    assert len(db.query("SELECT * FROM risk_events")) == 1  # risk events are never auto-deleted
    with pytest.raises(Exception, match="CHECK"):
        db.execute(
            "INSERT INTO risk_events(ts,mode,severity,rule,message) VALUES('t','paper','I','r','m')"
        )
    db.close()


def test_transaction_rolls_back_atomically() -> None:
    db = Database(":memory:")
    repos = Repositories(db)
    with pytest.raises(RuntimeError), db.transaction():
        repos.set_bot_state("a", "1")
        raise RuntimeError("boom")
    assert repos.get_bot_state("a") is None
    with db.transaction(), db.transaction():
        repos.set_bot_state("b", "2")
    assert repos.get_bot_state("b") == "2"


# ---------------------------------------------------------------- telegram
async def test_telegram_is_noop_when_unconfigured_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = TelegramAlerter(make_settings())
    assert not a.configured
    a.notify("x", "y")
    assert a.sent == [] and await a.send("hi") is False

    from pydantic import SecretStr

    from app.config import Settings

    s = Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        telegram_chat_id="1",
    )
    calls: list[str] = []

    class FakeHttp:
        async def post(self, url: str, *, json: dict[str, str]) -> httpx.Response:
            calls.append(json["text"])
            return httpx.Response(200)

    b = TelegramAlerter(s, FakeHttp())
    b.notify("rule", "first")
    b.notify("rule", "second")  # de-duplicated
    await asyncio.sleep(0.05)
    assert len(calls) == 1 and "first" in calls[0]


def test_shipped_example_configs_are_valid_and_identical() -> None:
    from pathlib import Path

    from app.config import load_config

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "config.example.yaml")
    assert cfg.trading.product.product.value == "multiplier"
    assert cfg.trading.product.stop_loss_pct <= 1
    assert (root / "config.example.yaml").read_text() == (root / "config.yaml.example").read_text()
