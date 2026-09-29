"""EMA-crossover reference strategy.

It carries NO calibrated probability, so the RiskManager's edge gate always refuses to trade it.
It exists as a deterministic baseline for tests/comparison, not as a trading edge.
"""

from __future__ import annotations

import time

from app.models.schemas import Direction, Signal, Tick
from app.strategy.base import make_signal_id


class _SymbolState:
    def __init__(self) -> None:
        self.fast: float | None = None
        self.slow: float | None = None
        self.prev_gap: float | None = None
        self.count = 0


class EmaStrategy:
    name = "ema"

    def __init__(
        self, fast: int = 5, slow: int = 20, horizon: int = 10, version: str = "1"
    ) -> None:
        if fast >= slow:
            raise ValueError("fast EMA span must be smaller than slow")
        self.version = version
        self._af = 2.0 / (fast + 1)
        self._as = 2.0 / (slow + 1)
        self._slow_n = slow
        self._horizon = horizon
        self._state: dict[str, _SymbolState] = {}  # independent state per symbol

    def reset(self, symbol: str) -> None:
        self._state.pop(symbol, None)

    async def on_tick(self, symbol: str, tick: Tick) -> Signal | None:
        if tick.symbol != symbol or not (tick.quote > 0):
            return None
        st = self._state.setdefault(symbol, _SymbolState())
        q = tick.quote
        st.fast = q if st.fast is None else self._af * q + (1 - self._af) * st.fast
        st.slow = q if st.slow is None else self._as * q + (1 - self._as) * st.slow
        st.count += 1
        if st.count < self._slow_n:
            return None
        gap = st.fast - st.slow
        prev, st.prev_gap = st.prev_gap, gap
        crossed_up = prev is not None and prev <= 0 < gap
        crossed_down = prev is not None and prev >= 0 > gap
        if not (crossed_up or crossed_down):
            return None
        direction = Direction.CALL if crossed_up else Direction.PUT
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
        )
