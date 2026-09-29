"""Executor against the mock Deriv server: proposal/buy/settlement, errors, timeouts,
ambiguous buys, reconnects, rate limits, stale quotes, duplicates."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import AppConfig
from app.controller import Controller
from app.deriv.protocol import DerivError, RateLimitError
from app.models.schemas import Direction, Mode, OrderState, Signal
from tests.conftest import (
    build_controller,
    client_of,
    count_rows,
    make_settings,
    wait_until,
)
from tests.mocks.deriv_mock import MockDeriv
from tests.mocks.stubs import make_stub_model

_seq = 0


def mk_signal(
    symbol: str = "R_100", p: float = 0.7, age: float = 0.0, sid: str | None = None
) -> Signal:
    global _seq
    _seq += 1
    now = time.time() - age
    return Signal(
        signal_id=sid or f"sig-{_seq}-{now}",
        symbol=symbol,
        direction=Direction.CALL,
        tick_epoch=_seq,
        strategy="ml",
        strategy_version="1",
        model_version="stub-1",
        probability=p,
        horizon_ticks=5,
        tick_received_at=now,
        created_at=now,
    )


def buys(mock: MockDeriv) -> list[dict[str, object]]:
    return [r for r in mock.requests if "buy" in r]


def orders(ctl: Controller) -> list[dict[str, object]]:
    return [dict(r) for r in ctl.repos.db.query("SELECT * FROM orders ORDER BY id")]


def event_rules(ctl: Controller) -> list[str]:
    return [str(r["rule"]) for r in ctl.repos.recent_risk_events(200)]


def recon_kinds(ctl: Controller) -> list[str]:
    return [str(r["kind"]) for r in ctl.repos.db.query("SELECT kind FROM reconciliation_events")]


async def test_successful_buy_win_and_loss_update_state(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal())
    o = orders(controller)[0]
    assert o["state"] == OrderState.OPEN.value and o["ask_price"] == "10.0"
    assert Decimal(str(o["stake"])) == Decimal("10.00")
    cid = int(str(o["contract_id"]))
    assert controller.open_trade_count() == 1
    await mock.settle(cid, win=False)
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.LOST.value)
    assert controller.risk.state.consecutive_losses(Mode.DEMO) == 1
    assert controller.risk.state.daily_pnl(Mode.DEMO) == Decimal("-10.0")
    assert controller.open_trade_count() == 0
    # balance was refreshed from the broker after settlement
    assert controller.account.balance == mock.balance and controller.account.balance_ts is not None


async def test_proposal_error_fails_order_and_frees_the_trade_slot(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    mock.errors["proposal"] = {"code": "ContractCreationFailure", "message": "nope"}
    await controller.process_signal_now(mk_signal())
    assert orders(controller)[0]["state"] == OrderState.FAILED.value
    assert buys(mock) == []
    assert controller.risk.state.daily_trades(Mode.DEMO) == 0
    assert "proposal_failed" in event_rules(controller)


async def test_proposal_timeout_rejects(
    config: AppConfig, controller: Controller, mock: MockDeriv
) -> None:
    config.deriv.proposal_timeout_s = 0.3
    await controller.start()
    mock.proposal_delay = 2.0
    await controller.process_signal_now(mk_signal())
    assert orders(controller)[0]["state"] == OrderState.FAILED.value
    assert buys(mock) == []


async def test_buy_error_is_not_retried(controller: Controller, mock: MockDeriv) -> None:
    await controller.start()
    mock.errors["buy"] = {"code": "ContractBuyValidationError", "message": "price moved"}
    await controller.process_signal_now(mk_signal())
    assert orders(controller)[0]["state"] == OrderState.FAILED.value
    assert len(buys(mock)) == 1
    assert mock.contracts == {}


async def test_rate_limited_buy_is_never_retried(controller: Controller, mock: MockDeriv) -> None:
    await controller.start()
    mock.errors["buy"] = {"code": "RateLimit", "message": "slow down"}
    await controller.process_signal_now(mk_signal())
    assert len(buys(mock)) == 1 and mock.contracts == {}
    assert orders(controller)[0]["state"] == OrderState.FAILED.value


async def test_rate_limited_safe_requests_are_retried_with_backoff(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    before = len([r for r in mock.requests if "balance" in r])
    mock.rate_limit_next = 2
    bal = await client_of(controller).get_balance()
    assert bal.balance == mock.balance
    assert len([r for r in mock.requests if "balance" in r]) - before == 3
    mock.rate_limit_next = 10  # persistent limiting eventually surfaces the error
    with pytest.raises(RateLimitError):
        await client_of(controller).get_balance()
    assert isinstance(RateLimitError("RateLimit", "x"), DerivError)


async def test_stale_quote_is_never_chased(controller: Controller, mock: MockDeriv) -> None:
    await controller.start()
    mock.ask_price_override = Decimal("10.05")
    await controller.process_signal_now(mk_signal())
    assert buys(mock) == []
    assert "stale_quote" in event_rules(controller)
    assert orders(controller)[0]["state"] == OrderState.FAILED.value


async def test_thin_edge_is_rejected_at_the_proposal_gate(
    config: AppConfig, mock: MockDeriv
) -> None:
    make_stub_model(Path(config.app.model_dir), p_up=0.53)
    ctl = build_controller(make_settings(), config, mock)
    await ctl.start()
    await ctl.process_signal_now(mk_signal(p=0.53))  # 0.53 < 1/1.95 + 0.03
    assert buys(mock) == [] and "edge_gate" in event_rules(ctl)
    await ctl.shutdown()


async def test_duplicate_signal_places_one_order(controller: Controller, mock: MockDeriv) -> None:
    await controller.start()
    sig = mk_signal(sid="dup-1")
    await controller.process_signal_now(sig)
    await controller.process_signal_now(sig)
    assert len(buys(mock)) == 1 and len(orders(controller)) == 1
    assert "duplicate_signal" in event_rules(controller)


async def test_stale_signal_is_dropped_not_traded(controller: Controller, mock: MockDeriv) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal(age=60.0))
    assert orders(controller) == [] and "signal_stale" in event_rules(controller)


async def test_stake_follows_the_freshly_refreshed_balance(
    config: AppConfig, controller: Controller, mock: MockDeriv
) -> None:
    config.risk.balance_max_age_s = 0.2
    await controller.start()
    await asyncio.sleep(0.25)  # the cached balance is now too old
    mock.balance = Decimal("950")
    await controller.process_signal_now(mk_signal())
    assert Decimal(str(orders(controller)[0]["stake"])) == Decimal("9.50")


async def test_ambiguous_buy_timeout_reconciles_without_double_buying(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    assert controller.reconciler is not None
    controller.reconciler.grace_s = 0.05
    mock.swallow_buy = True  # the broker executes the buy but never answers
    await controller.process_signal_now(mk_signal())
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.OPEN.value)
    assert len(buys(mock)) == 1 and len(mock.contracts) == 1  # never retried
    assert orders(controller)[0]["contract_id"] == next(iter(mock.contracts))
    assert "ambiguous_buy" in event_rules(controller) and "buy_adopted" in recon_kinds(controller)
    assert count_rows(controller.db, "trades") == 1
    await mock.settle(next(iter(mock.contracts)), win=True)
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.WON.value)


async def test_ambiguous_buy_socket_drop_reconciles_after_reconnect(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    assert controller.reconciler is not None
    controller.reconciler.grace_s = 0.05
    mock.drop_on_buy = True
    await controller.process_signal_now(mk_signal())
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.OPEN.value, timeout_s=8)
    assert len(mock.contracts) == 1 and len(buys(mock)) == 1
    assert controller.client is not None and controller.client.ws.reconnect_count >= 1


async def test_buy_that_never_happened_is_marked_failed_not_retried(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    assert controller.reconciler is not None
    controller.reconciler.grace_s = 0.02
    mock.fail_buy_without_contract = True
    await controller.process_signal_now(mk_signal())
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.FAILED.value, timeout_s=8)
    assert mock.contracts == {} and len(buys(mock)) == 1
    assert "ambiguous_buy_no_contract" in recon_kinds(controller)


async def test_settlement_timeout_moves_to_reconciliation_and_resolves(
    config: AppConfig, controller: Controller, mock: MockDeriv
) -> None:
    config.deriv.settlement_timeout_s = 0.3
    await controller.start()
    await controller.process_signal_now(mk_signal())
    cid = int(str(orders(controller)[0]["contract_id"]))
    await wait_until(lambda: "settlement_timeout" in recon_kinds(controller))
    # the broker settles it but the update is lost (never pushed to the watcher)
    await mock.settle(cid, win=True, notify=False)
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.WON.value, timeout_s=5)
    assert "settled_by_reconciliation" in recon_kinds(controller)
    assert controller.open_trade_count() == 0


async def test_disconnect_marks_pending_then_reconnect_reconciles_and_resubscribes(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal())
    cid = int(str(orders(controller)[0]["contract_id"]))
    await mock.settle(cid, win=True, notify=False)  # settles while we are blind
    await mock.drop_all()
    await wait_until(lambda: "disconnect" in recon_kinds(controller))
    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.WON.value, timeout_s=8)
    assert controller.client is not None
    assert controller.client.ws.reconnect_count == 1 and controller.client.ws.disconnect_count >= 1
    # the tick subscription was restored on the new socket
    before = controller.counters["ticks"]
    await wait_until(lambda: len(mock.tick_subs) == 1)
    await mock.push_tick("R_100", 101.0)
    await wait_until(lambda: controller.counters["ticks"] > before)


async def test_stale_feed_halts_new_trades_then_recovers(
    config: AppConfig, controller: Controller, mock: MockDeriv
) -> None:
    config.risk.feed_stale_after_s = 0.5
    await controller.start()
    await asyncio.sleep(0.7)
    await controller.process_signal_now(mk_signal())
    assert buys(mock) == [] and "stale_feed" in event_rules(controller)
    await wait_until(lambda: any(r == "stale_feed" for r in event_rules(controller)))
    await mock.push_tick("R_100", 100.5)
    await wait_until(lambda: not controller.status()["feed"]["stale"])
    await controller.process_signal_now(mk_signal())
    assert len(buys(mock)) == 1


async def test_kill_switch_blocks_and_survives_restart(
    controller: Controller, config: AppConfig, mock: MockDeriv
) -> None:
    await controller.start()
    controller.kill("test")
    await controller.process_signal_now(mk_signal())
    assert buys(mock) == [] and "kill_switch" in event_rules(controller)
    await controller.stop()
    again = build_controller(make_settings(), config, mock, db=controller.db)
    from app.controller import StartError

    with pytest.raises(StartError, match="kill"):
        await again.start()


async def test_tick_queue_drops_oldest_and_keeps_latest(config: AppConfig, mock: MockDeriv) -> None:
    config.trading.tick_queue_size = 3
    ctl = build_controller(make_settings(), config, mock)
    from app.models.schemas import Tick

    for i in range(10):
        ctl.on_tick(Tick("R_100", i, 100.0 + i, time.time()))
    assert ctl.counters["ticks_dropped"] == 7
    kept = [ctl._tick_q.get_nowait().epoch for _ in range(3)]
    assert kept == [7, 8, 9]  # newest information wins
    await ctl.shutdown()


async def test_signal_queue_drops_oldest_signal(config: AppConfig, mock: MockDeriv) -> None:
    config.trading.signal_queue_size = 2
    ctl = build_controller(make_settings(), config, mock)
    sigs = [mk_signal(sid=f"q{i}") for i in range(5)]
    for s in sigs:
        ctl._enqueue_signal(s)
    assert [ctl._signal_q.get_nowait().signal_id for _ in range(2)] == ["q3", "q4"]
    assert ctl.counters["signals_dropped"] == 3
    await ctl.shutdown()
