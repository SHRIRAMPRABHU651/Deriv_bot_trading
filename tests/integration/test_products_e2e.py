"""All four Deriv 'Options' trade types (Multipliers, Accumulators, Turbos, Vanillas) end to end
against the mock broker: proposal fields, buy, bot-managed exits, settlement, spec safety."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from app.controller import Controller
from app.models.schemas import Direction, Mode, OrderState, Signal, Tick
from app.products import Product, ProductSpec
from tests.conftest import build_controller, make_config, make_settings, wait_until
from tests.mocks.deriv_mock import MockDeriv
from tests.mocks.stubs import make_stub_model

MULT = ProductSpec(
    product=Product.MULTIPLIER,
    horizon_ticks=5,
    multiplier=20,
    take_profit_pct=0.5,
    stop_loss_pct=0.5,
    tick_seconds=0.1,
    hold_grace_s=0.3,
)
ACCU = ProductSpec(
    product=Product.ACCUMULATOR,
    horizon_ticks=5,
    growth_rate=0.01,
    barrier_pct=0.0006,
    tick_seconds=0.1,
    hold_grace_s=0.3,
)
TURBO = ProductSpec(product=Product.TURBO, horizon_ticks=5, barrier_offset=0.5, take_profit_pct=0.5)
VANILLA = ProductSpec(product=Product.VANILLA, horizon_ticks=5, needed_move=1.0, target_pct=0.5)

_n = 0


def sig(spec: ProductSpec, p: float = 0.9, direction: Direction | None = None) -> Signal:
    global _n
    _n += 1
    now = time.time()
    if direction is None:
        direction = Direction.NEUTRAL if spec.product is Product.ACCUMULATOR else Direction.CALL
    return Signal(
        signal_id=f"prod-{_n}",
        symbol="R_100",
        direction=direction,
        tick_epoch=_n,
        strategy="ml",
        strategy_version="1",
        model_version="stub-1",
        probability=p,
        horizon_ticks=spec.horizon_ticks,
        tick_received_at=now,
        created_at=now,
        product=spec.product.value,
        entry_price=100.0,
    )


async def make(tmp_path: Path, mock: MockDeriv, spec: ProductSpec, p: float = 0.9) -> Controller:
    cfg = make_config(tmp_path, spec=spec)
    make_stub_model(Path(cfg.app.model_dir), p_up=p, p_down=p, spec=spec)
    ctl = build_controller(make_settings(), cfg, mock)
    assert ctl.reload_model(), ctl.model_error
    await ctl.start()
    return ctl


def proposals(mock: MockDeriv) -> list[dict[str, object]]:
    return [r for r in mock.requests if "proposal" in r]


def order(ctl: Controller) -> dict[str, object]:
    row = ctl.repos.db.query_one("SELECT * FROM orders ORDER BY id LIMIT 1")
    assert row is not None
    return dict(row)


def trade(ctl: Controller) -> dict[str, object]:
    row = ctl.repos.db.query_one("SELECT * FROM trades ORDER BY id LIMIT 1")
    assert row is not None
    return dict(row)


def rules(ctl: Controller) -> list[str]:
    return [str(r["rule"]) for r in ctl.repos.recent_risk_events(100)]


# ---------------------------------------------------------------- multipliers
async def test_multiplier_take_profit_uses_native_limit_orders(
    tmp_path: Path, mock: MockDeriv
) -> None:
    mock.commission = Decimal("0.2")
    ctl = await make(tmp_path, mock, MULT)
    await ctl.process_signal_now(sig(MULT))
    req = proposals(mock)[-1]
    assert req["contract_type"] == "MULTUP" and req["multiplier"] == 20
    assert req["limit_order"] == {"take_profit": 5.0, "stop_loss": 5.0}
    assert "duration" not in req and req["basis"] == "stake"
    o = order(ctl)
    assert o["state"] == OrderState.OPEN.value and o["product"] == "multiplier"
    assert o["contract_type"] == "MULTUP" and o["ask_price"] == "10.2"
    assert o["break_even"] == pytest.approx(0.52)
    cid = int(str(o["contract_id"]))
    assert mock.contracts[cid]["limit_order"] == {"take_profit": 5.0, "stop_loss": 5.0}
    await mock.close_with_profit(cid, 5.0)  # the broker's take-profit fires
    await wait_until(lambda: order(ctl)["state"] == OrderState.WON.value)
    assert trade(ctl)["target_hit"] == 1
    assert ctl.risk.state.daily_pnl(Mode.DEMO) == Decimal("5.0")
    await ctl.shutdown()


async def test_multiplier_stop_loss_is_a_bounded_loss(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, MULT)
    await ctl.process_signal_now(sig(MULT, direction=Direction.PUT))
    assert proposals(mock)[-1]["contract_type"] == "MULTDOWN"
    cid = int(str(order(ctl)["contract_id"]))
    await mock.close_with_profit(cid, -5.0)
    await wait_until(lambda: order(ctl)["state"] == OrderState.LOST.value)
    assert trade(ctl)["target_hit"] == 0
    assert ctl.risk.state.daily_pnl(Mode.DEMO) == Decimal("-5.0")  # never more than the stop
    assert ctl.risk.state.consecutive_losses(Mode.DEMO) == 1
    await ctl.shutdown()


async def test_multiplier_hold_cap_sells_at_market_and_is_not_counted_as_a_target_hit(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await make(tmp_path, mock, MULT)
    await ctl.process_signal_now(sig(MULT))
    cid = int(str(order(ctl)["contract_id"]))
    mock.set_profit(cid, 1.0)  # small open profit, neither TP nor SL
    await wait_until(lambda: cid in mock.sells, timeout_s=5)  # hold cap = 5*0.1 + 0.3 s
    await wait_until(lambda: order(ctl)["state"] == OrderState.WON.value)
    t = trade(ctl)
    assert t["target_hit"] == 0  # profit > 0 but far below the take-profit the model predicts
    assert ctl.tracker is not None and ctl.tracker.exits["hold_cap"] == 1
    assert ctl.open_trade_count() == 0
    await ctl.shutdown()


async def test_multiplier_commission_above_allowance_is_refused(
    tmp_path: Path, mock: MockDeriv
) -> None:
    mock.commission = Decimal("0.9")  # 9% of the stake > 5% allowance
    ctl = await make(tmp_path, mock, MULT)
    await ctl.process_signal_now(sig(MULT))
    assert [r for r in mock.requests if "buy" in r] == []
    assert "stale_quote" in rules(ctl)
    await ctl.shutdown()


# ---------------------------------------------------------------- accumulators
async def test_accumulator_request_target_and_hold(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, ACCU, p=0.995)
    await ctl.process_signal_now(sig(ACCU, p=0.995))
    req = proposals(mock)[-1]
    assert req["contract_type"] == "ACCU" and req["growth_rate"] == 0.01
    assert req["limit_order"] == {"take_profit": 0.51} and "duration" not in req
    o = order(ctl)
    assert o["state"] == OrderState.OPEN.value and o["direction"] == "NEUTRAL"
    assert o["break_even"] == pytest.approx(10 / 10.51)
    cid = int(str(o["contract_id"]))
    await mock.close_with_profit(cid, 0.51)  # native take-profit after N surviving ticks
    await wait_until(lambda: order(ctl)["state"] == OrderState.WON.value)
    assert trade(ctl)["target_hit"] == 1
    await ctl.shutdown()


async def test_accumulator_knockout_loses_the_stake_only(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, ACCU, p=0.995)
    await ctl.process_signal_now(sig(ACCU, p=0.995))
    cid = int(str(order(ctl)["contract_id"]))
    await mock.close_with_profit(cid, -10.0)
    await wait_until(lambda: order(ctl)["state"] == OrderState.LOST.value)
    assert ctl.risk.state.daily_pnl(Mode.DEMO) == Decimal("-10.0")  # exactly the stake
    await ctl.shutdown()


async def test_accumulator_with_a_narrower_live_barrier_is_refused(
    tmp_path: Path, mock: MockDeriv
) -> None:
    mock.accu_barrier_pct = 0.0002  # trained on 0.0006
    ctl = await make(tmp_path, mock, ACCU, p=0.995)
    await ctl.process_signal_now(sig(ACCU, p=0.995))
    assert [r for r in mock.requests if "buy" in r] == []
    assert "spec_mismatch" in rules(ctl)
    await ctl.shutdown()


async def test_accumulator_needs_probability_above_its_high_break_even(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await make(tmp_path, mock, ACCU, p=0.97)  # 0.97 < 0.9515 + 0.03
    await ctl.process_signal_now(sig(ACCU, p=0.97))
    assert [r for r in mock.requests if "buy" in r] == [] and "edge_gate" in rules(ctl)
    await ctl.shutdown()


# ---------------------------------------------------------------- turbos
async def test_turbo_bot_managed_take_profit(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, TURBO, p=0.85)
    await ctl.process_signal_now(sig(TURBO, p=0.85))
    req = proposals(mock)[-1]
    assert req["contract_type"] == "TURBOSLONG" and req["barrier"] == "-0.5"
    assert req["duration"] == 5 and "limit_order" not in req  # turbos have no native TP
    cid = int(str(order(ctl)["contract_id"]))
    mock.set_profit(cid, 4.9)
    await mock.push_contract(cid)
    await asyncio.sleep(0.2)  # let the update be processed: no exit below the target
    assert mock.sells == []  # 4.9 < 50% of the 10 paid
    mock.set_profit(cid, 5.0)
    await mock.push_contract(cid)
    await wait_until(lambda: cid in mock.sells)
    await wait_until(lambda: order(ctl)["state"] == OrderState.WON.value)
    assert trade(ctl)["target_hit"] == 1
    assert ctl.tracker is not None and ctl.tracker.exits["take_profit"] == 1
    await ctl.shutdown()


async def test_turbo_knockout_loses_the_stake(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, TURBO, p=0.85)
    await ctl.process_signal_now(sig(TURBO, p=0.85, direction=Direction.PUT))
    assert proposals(mock)[-1]["contract_type"] == "TURBOSSHORT"
    assert proposals(mock)[-1]["barrier"] == "+0.5"
    cid = int(str(order(ctl)["contract_id"]))
    await mock.close_with_profit(cid, -10.0)
    await wait_until(lambda: order(ctl)["state"] == OrderState.LOST.value)
    assert ctl.risk.state.daily_pnl(Mode.DEMO) == Decimal("-10.0")
    await ctl.shutdown()


async def test_turbo_whose_target_needs_a_bigger_move_than_trained_is_refused(
    tmp_path: Path, mock: MockDeriv
) -> None:
    mock.turbo_contracts = 10.0  # target would need a 1.0 move; the model saw 0.25
    ctl = await make(tmp_path, mock, TURBO, p=0.95)
    await ctl.process_signal_now(sig(TURBO, p=0.95))
    assert [r for r in mock.requests if "buy" in r] == [] and "spec_mismatch" in rules(ctl)
    await ctl.shutdown()


# ---------------------------------------------------------------- vanillas
async def test_vanilla_is_held_to_expiry(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, VANILLA, p=0.85)
    await ctl.process_signal_now(sig(VANILLA, p=0.85))
    req = proposals(mock)[-1]
    assert req["contract_type"] == "VANILLALONGCALL" and req["barrier"] == "+0.0"
    assert req["duration"] == 5 and req["duration_unit"] == "t"
    cid = int(str(order(ctl)["contract_id"]))
    assert ctl.tracker is not None and ctl.tracker._plans.get(cid) is None  # no bot exits
    await mock.close_with_profit(cid, 6.0)  # intrinsic payout at expiry
    await wait_until(lambda: order(ctl)["state"] == OrderState.WON.value)
    assert trade(ctl)["target_hit"] == 1 and mock.sells == []
    await ctl.shutdown()


async def test_vanilla_out_of_reach_target_is_refused(tmp_path: Path, mock: MockDeriv) -> None:
    mock.vanilla_contracts = 3.0  # target needs a 5.0 move, model trained on 1.0
    ctl = await make(tmp_path, mock, VANILLA, p=0.95)
    await ctl.process_signal_now(sig(VANILLA, p=0.95))
    assert [r for r in mock.requests if "buy" in r] == [] and "spec_mismatch" in rules(ctl)
    await ctl.shutdown()


# ---------------------------------------------------------------- pipeline + safety
async def test_real_ticks_drive_an_accumulator_trade(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await make(tmp_path, mock, ACCU, p=0.995)
    signal = None
    for i in range(80):
        t = Tick("R_100", 1_700_000_000 + i, 100.0 + 0.001 * (i % 5), time.time(), f"t{i}")
        signal = await ctl.process_tick(t) or signal
    assert signal is not None
    assert signal.direction is Direction.NEUTRAL and signal.product == "accumulator"
    assert signal.entry_price is not None
    await wait_until(lambda: ctl.executor is not None and ctl.executor.stats["buys"] >= 1)
    assert proposals(mock)[-1]["contract_type"] == "ACCU"
    await ctl.shutdown()


async def test_model_trained_for_other_terms_is_never_used(tmp_path: Path, mock: MockDeriv) -> None:
    cfg = make_config(tmp_path, spec=TURBO)
    make_stub_model(Path(cfg.app.model_dir), spec=MULT)  # a multiplier model, turbo configured
    ctl = build_controller(make_settings(), cfg, mock)
    assert ctl.predictor is None and ctl.model_error is not None
    assert "trained for" in ctl.model_error
    make_stub_model(Path(cfg.app.model_dir), spec=TURBO)
    assert ctl.reload_model()
    changed = TURBO.model_copy(update={"barrier_offset": 0.9})
    cfg.trading.product = changed  # the operator edits the config afterwards
    assert not ctl.reload_model()
    await ctl.shutdown()


async def test_restart_resumes_bot_managed_exit_for_open_turbo(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await make(tmp_path, mock, TURBO, p=0.85)
    await ctl.process_signal_now(sig(TURBO, p=0.85))
    cid = int(str(order(ctl)["contract_id"]))
    await ctl.stop()
    b = build_controller(make_settings(), ctl.config, mock, db=ctl.db)
    await b.start()
    mock.set_profit(cid, 5.0)
    await mock.push_contract(cid)
    await wait_until(lambda: cid in mock.sells)  # the resumed watcher still owns the exit
    await wait_until(lambda: order(b)["state"] == OrderState.WON.value)
    await b.shutdown()
