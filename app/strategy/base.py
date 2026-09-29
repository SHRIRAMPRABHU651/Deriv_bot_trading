"""Strategy interface and deterministic signal identity."""

from __future__ import annotations

import hashlib
from typing import Protocol

from app.models.schemas import Direction, Signal, Tick


class Strategy(Protocol):
    name: str
    version: str

    async def on_tick(self, symbol: str, tick: Tick) -> Signal | None: ...

    def reset(self, symbol: str) -> None: ...


def make_signal_id(
    symbol: str,
    tick: Tick,
    strategy_version: str,
    direction: Direction,
    model_version: str | None,
) -> str:
    """Deterministic identity: symbol + market tick identity + strategy/model version + direction.
    Deliberately independent of the local receive timestamp."""
    raw = "|".join(
        [
            symbol,
            str(tick.epoch),
            tick.tick_id or "",
            strategy_version,
            direction.value,
            model_version or "-",
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
