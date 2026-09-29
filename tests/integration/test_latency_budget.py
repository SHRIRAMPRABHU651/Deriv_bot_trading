"""Internal pipeline latency (excludes real network round trips): must be far below one second."""

from __future__ import annotations

import statistics
import time

from app.controller import Controller
from app.models.schemas import Mode, Tick
from tests.conftest import wait_until
from tests.mocks.deriv_mock import MockDeriv


async def test_tick_to_confirmed_order_is_well_under_one_second_locally(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    mock.auto_settle = (0.05, True)
    for i in range(80):
        await controller.process_tick(
            Tick("R_100", 1_700_000_000 + i, 100.0 + 0.01 * (i % 7), time.time(), f"t{i}")
        )
    await wait_until(
        lambda: controller.executor is not None and controller.executor.stats["buys"] >= 1
    )
    stats = controller.latency.daily_stats(Mode.DEMO, controller.clock.now())
    total_ms = sum(
        stats[k]["max"] for k in ("tick_to_signal", "signal_to_order", "order_to_confirmation")
    )
    assert total_ms < 1000.0, stats  # tick -> signal -> order sent -> confirmation, mock broker
    assert statistics.median([stats["tick_to_signal"]["median"]]) < 100.0
