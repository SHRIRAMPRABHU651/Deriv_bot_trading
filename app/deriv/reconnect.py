"""Exponential backoff with jitter: 1s, 2s, 4s, 8s, 16s, 30s (max)."""

from __future__ import annotations

import random
from collections.abc import Callable


class Backoff:
    def __init__(
        self,
        base: float = 1.0,
        factor: float = 2.0,
        maximum: float = 30.0,
        jitter: float = 0.25,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.base = base
        self.factor = factor
        self.maximum = maximum
        self.jitter = jitter
        self._rng = rng
        self.attempt = 0

    def next_delay(self) -> float:
        """Nominal delay grows exponentially and is capped; jitter only ever adds up to +25%
        below the cap, and the result never exceeds `maximum`."""
        nominal = min(self.maximum, self.base * (self.factor**self.attempt))
        self.attempt += 1
        jittered = nominal * (1.0 + self.jitter * self._rng())
        return min(self.maximum, jittered)

    def reset(self) -> None:
        self.attempt = 0
