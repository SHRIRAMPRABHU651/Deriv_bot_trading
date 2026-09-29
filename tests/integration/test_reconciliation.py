"""Startup reconciliation: never double-trade because of a restart."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from app.config import AppConfig
from app.controller import Controller, StartError
from app.models.schemas import Mode, OrderState
from tests.conftest import build_controller, make_settings, wait_until
from tests.integration.test_executor import buys, mk_signal, orders, recon_kinds
from tests.mocks.deriv_mock import MockDeriv


def rules(ctl: Controller) -> list[str]:
    return [str(r["rule"]) for r in ctl.repos.recent_risk_events(200)]


async def test_restart_resumes_open_contract_and_blocks_double_trading(
    controller: Controller, config: AppConfig, mock: MockDeriv
) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal())
    cid = int(str(orders(controller)[0]["contract_id"]))
    await controller.stop()  # process "restarts": the contract is still open at the broker
    assert orders(controller)[0]["state"] in (
        OrderState.OPEN.value,
        OrderState.RECONCILIATION_PENDING.value,
    )

    b = build_controller(make_settings(), config, mock, db=controller.db)
    await b.start()
    assert b.open_trade_count() == 1  # remembered from SQLite + broker
    assert orders(b)[0]["state"] == OrderState.OPEN.value
    await b.process_signal_now(mk_signal())  # same symbol: max one open per symbol
    assert len(buys(mock)) == 1, "restart must not lead to a second purchase"
    assert "max_open_per_symbol" in rules(b)
    await mock.settle(cid, win=True)
    await wait_until(lambda: orders(b)[0]["state"] == OrderState.WON.value)
    await b.stop()


async def test_db_says_open_but_broker_says_closed(
    controller: Controller, config: AppConfig, mock: MockDeriv
) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal())
    cid = int(str(orders(controller)[0]["contract_id"]))
    await controller.stop()
    await mock.settle(cid, win=True, notify=False)  # closed while the bot was down

    b = build_controller(make_settings(), config, mock, db=controller.db)
    await b.start()
    await wait_until(lambda: orders(b)[0]["state"] == OrderState.WON.value)
    assert "db_open_api_closed" in recon_kinds(b)
    assert "settled_by_reconciliation" in recon_kinds(b)
    assert b.risk.state.daily_pnl(Mode.DEMO) == Decimal("9.50")
    await b.stop()


async def test_broker_position_unknown_to_database_is_adopted(
    controller: Controller, mock: MockDeriv
) -> None:
    cid = mock.add_external_contract("R_100", "5")
    await controller.start()
    assert "unknown_position" in recon_kinds(controller)
    assert "unknown_position" in rules(controller)
    assert controller.open_trade_count() == 1
    row = controller.repos.get_order_by_contract(cid)
    assert row is not None and row["order_id"] == f"external-{cid}"
    await mock.settle(cid, win=False)

    def settled() -> bool:
        r = controller.repos.get_order_by_contract(cid)
        return r is not None and r["state"] == OrderState.LOST.value

    await wait_until(settled)
    assert controller.open_trade_count() == 0


async def test_order_never_sent_before_crash_is_failed_safely(
    controller: Controller, mock: MockDeriv
) -> None:
    sig = mk_signal()
    controller.repos.insert_signal(Mode.DEMO, sig, controller.clock.now())
    controller.repos.insert_order(
        order_id="crashed",
        signal=sig,
        mode=Mode.DEMO,
        stake=Decimal("10"),
        break_even=None,
        ts=controller.clock.now(),
    )
    await controller.start()
    row = controller.repos.get_order("crashed")
    assert row is not None and row["state"] == OrderState.FAILED.value
    assert controller.open_trade_count() == 0


async def test_sent_but_unconfirmed_order_is_matched_to_the_broker_contract(
    controller: Controller, mock: MockDeriv
) -> None:
    sig = mk_signal()
    controller.repos.insert_signal(Mode.DEMO, sig, controller.clock.now())
    controller.repos.insert_order(
        order_id="ambig",
        signal=sig,
        mode=Mode.DEMO,
        stake=Decimal("10"),
        break_even=None,
        ts=controller.clock.now(),
    )
    controller.repos.update_order("ambig", state=OrderState.PROPOSED, order_sent_at=time.time())
    cid = mock.add_external_contract("R_100", "10")
    await controller.start()
    row = controller.repos.get_order("ambig")
    assert row is not None and row["contract_id"] == cid
    assert row["state"] == OrderState.OPEN.value
    assert "unknown_position" not in recon_kinds(controller)  # matched, not double-counted
    assert "buy_adopted" in recon_kinds(controller)


async def test_startup_aborts_trading_if_reconciliation_fails(
    controller: Controller, mock: MockDeriv
) -> None:
    mock.errors["portfolio"] = {"code": "InternalServerError", "message": "boom"}
    with pytest.raises(StartError, match="reconciliation"):
        await controller.start()
    assert not controller.running and not controller.trading_enabled
