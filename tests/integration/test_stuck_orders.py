"""A position the broker cannot describe must never block trading forever or fail silently."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import AppConfig
from app.controller import Controller
from app.execution import settlement
from app.models.schemas import OrderState
from tests.conftest import wait_until
from tests.integration.test_executor import mk_signal, orders
from tests.mocks.deriv_mock import MockDeriv


async def test_contract_unknown_to_the_broker_releases_the_slot_after_a_few_tries(
    config: AppConfig,
    controller: Controller,
    mock: MockDeriv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.deriv.settlement_timeout_s = 0.2
    monkeypatch.setattr(settlement, "RESOLVE_RETRY_S", 0.05)
    await controller.start()
    await controller.process_signal_now(mk_signal())
    cid = int(str(orders(controller)[0]["contract_id"]))
    assert controller.open_trade_count() == 1
    mock.contracts.pop(cid)  # the broker no longer knows it and sends nothing

    await wait_until(lambda: orders(controller)[0]["state"] == OrderState.FAILED.value, timeout_s=8)
    assert controller.open_trade_count() == 0  # the slot is free again
    rules = [r["rule"] for r in controller.repos.recent_risk_events(20)]
    assert "outcome_unknown" in rules


def test_contract_parser_tolerates_odd_field_formats() -> None:
    from app.deriv import protocol

    upd = protocol.parse_contract(
        {
            "proposal_open_contract": {
                "id": "123",  # new name for contract_id
                "is_sold": "true",
                "status": "Closed",
                "profit": "0.15",
                "date_expiry": "2026-10-01T03:00:00Z",  # not an int: must not raise
                "purchase_time": None,
                "sell_price": "5.15",
                "exit_tick": "628.12",
            }
        }
    )
    assert upd is not None and upd.contract_id == 123 and upd.is_sold
    assert upd.status == "sold" and str(upd.profit) == "0.15" and upd.date_expiry is None
    assert upd.exit_spot == 628.12
    assert protocol.parse_contract({"proposal_open_contract": {}}) is None


async def test_inspect_orders_lists_and_releases_a_stuck_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from decimal import Decimal

    from app.models.schemas import Mode
    from app.storage.database import Database
    from app.storage.repositories import Repositories
    from scripts import inspect_orders

    monkeypatch.chdir(tmp_path)  # no config.yaml / .env here => safe defaults, no credentials
    for name in ("DERIV_APP_ID", "DERIV_DEMO_TOKEN", "DERIV_DEMO_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    repos = Repositories(Database("data/derivbot.db"))
    sig = mk_signal()
    repos.insert_signal(Mode.DEMO, sig, datetime.now(UTC))
    repos.insert_order(
        order_id="abc12345",
        signal=sig,
        mode=Mode.DEMO,
        stake=Decimal("1"),
        break_even=None,
        ts=datetime.now(UTC),
    )
    assert await inspect_orders.main(release=False) == 0
    assert "abc12345"[:8] in capsys.readouterr().out
    assert len(repos.active_orders(Mode.DEMO)) == 1
    assert await inspect_orders.main(release=True) == 0
    assert repos.active_orders(Mode.DEMO) == []
