"""DEMO-ONLY pipeline probe: NOT a trading strategy and has no edge.

Emits a signal every `interval_ticks` ticks per symbol, alternating direction, so the real Deriv
proposal -> buy -> settlement -> record path can be exercised on a DEMO account without a model.
Every risk rule (stake, loss limits, exposure, cooldown, kill switch) still applies; only the
model/edge gates are skipped, and only when the risk manager sees DEMO mode + probe enabled.
Expect a slow loss: Deriv's fees are built into the payoff and nothing here beats them.
"""

from __future__ import annotations

import time

from app.models.schemas import Direction, Signal, Tick
from app.strategy.base import make_signal_id

PROBE_NAME = "demo_probe"


class ProbeStrategy:
    name = PROBE_NAME

    def __init__(self, interval_ticks: int, horizon: int, product: str, version: str = "1") -> None:
        self.version = version
        self._interval = max(1, interval_ticks)
        self._horizon = horizon
        self._product = product
        self._seen: dict[str, int] = {}
        self._flip: dict[str, bool] = {}

    def reset(self, symbol: str) -> None:
        self._seen.pop(symbol, None)

    async def on_tick(self, symbol: str, tick: Tick) -> Signal | None:
        if tick.symbol != symbol:
            return None
        n = self._seen.get(symbol, 0) + 1
        self._seen[symbol] = n
        if n % self._interval != 0:
            return None
        up = not self._flip.get(symbol, False)
        self._flip[symbol] = up
        direction = Direction.CALL if up else Direction.PUT
        return Signal(
            signal_id=make_signal_id(symbol, tick, self.version, direction, None),
            symbol=symbol,
            direction=direction,
            tick_epoch=tick.epoch,
            strategy=self.name,
            strategy_version=self.version,
            model_version=None,
            probability=None,
            horizon_ticks=self._horizon,
            tick_received_at=tick.received_at,
            created_at=time.time(),
            product=self._product,
            entry_price=tick.quote,
        )
