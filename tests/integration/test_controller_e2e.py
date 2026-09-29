"""MANDATORY: a real tick travels the whole pipeline
tick -> controller -> features -> ML prediction -> signal -> risk -> proposal -> buy -> settlement.

This would have caught the original `AttributeError: strategy.update` bug: the controller
calls the real Strategy.on_tick interface end to end.
"""

from __future__ import annotations

import time
from decimal import Decimal

from app.controller import Controller
from app.models.schemas import Mode, OrderState, Tick
from tests.conftest import wait_until
from tests.mocks.deriv_mock import MockDeriv


def _tick(i: int, quote: float | None = None) -> Tick:
    return Tick(
        symbol="R_100",
        epoch=1_700_000_000 + i,
        quote=quote if quote is not None else 100.0 + 0.01 * (i % 7),
        received_at=time.time(),
        tick_id=f"t{i}",
    )


async def test_tick_travels_through_entire_pipeline(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    assert controller.running and controller.trading_enabled
    mock.auto_settle = (0.1, True)

    signal = None
    for i in range(80):  # 64 ticks are needed to fill the bounded feature window
        signal = await controller.process_tick(_tick(i)) or signal
    assert signal is not None, "ML strategy produced no signal from real ticks"
    assert signal.probability is not None and signal.probability >= 0.7 - 1e-9
    assert signal.model_version == "stub-1"

    await wait_until(
        lambda: controller.tracker is not None and controller.tracker.settled_count >= 1
    )

    orders = controller.repos.db.query("SELECT * FROM orders")
    assert len(orders) >= 1
    order = orders[0]
    assert order["state"] == OrderState.WON.value
    assert order["contract_id"] is not None
    assert Decimal(order["stake"]) == Decimal("10.00")  # 1% of the 1000 mock balance
    trade = controller.repos.db.query_one(
        "SELECT * FROM trades WHERE contract_id=?", (order["contract_id"],)
    )
    assert trade is not None and trade["state"] == OrderState.WON.value
    assert Decimal(trade["profit"]) == Decimal("9.50")  # 10 * 1.95 - 10
    assert controller.risk.state.daily_pnl(Mode.DEMO) == Decimal("9.50")
    assert controller.risk.state.consecutive_losses(Mode.DEMO) == 0
    buys = [r for r in mock.requests if "buy" in r]
    assert len(buys) == 1
    # latency was recorded for every stage
    kinds = {
        r["kind"] for r in controller.repos.db.query("SELECT DISTINCT kind FROM latency_samples")
    }
    assert {"tick_to_signal", "signal_to_order", "order_to_confirmation"} <= kinds


async def test_ticks_from_websocket_use_queue_and_do_not_block(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    mock.auto_settle = (0.1, False)
    mock.proposal_delay = 0.3  # a slow proposal must not stall the tick receiver
    t0 = time.monotonic()
    for i in range(90):
        await mock.push_tick("R_100", 100.0 + 0.01 * (i % 5), epoch=1_700_000_000 + i)
    await wait_until(lambda: controller.counters["ticks"] >= 90, timeout_s=5)
    assert controller.counters["ticks"] >= 90
    # all ticks were ingested quickly even though proposal calls take 300 ms each
    assert time.monotonic() - t0 < 3.0
    await wait_until(
        lambda: controller.executor is not None and controller.executor.stats["buys"] >= 1
    )
    await wait_until(
        lambda: controller.tracker is not None and controller.tracker.settled_count >= 1
    )
    row = controller.repos.db.query_one("SELECT * FROM trades ORDER BY id LIMIT 1")
    assert row is not None and row["state"] == OrderState.LOST.value
    assert controller.risk.state.consecutive_losses(Mode.DEMO) >= 1


async def test_no_model_means_no_trading(controller: Controller, mock: MockDeriv) -> None:
    import shutil
    from pathlib import Path

    shutil.rmtree(Path(controller.config.app.model_dir))
    assert controller.reload_model() is False
    await controller.start()
    assert controller.running and not controller.trading_enabled
    for i in range(90):
        assert await controller.process_tick(_tick(i)) is None
    assert [r for r in mock.requests if "buy" in r] == []
    assert controller.status()["model"]["status"] == "NO_MODEL"
