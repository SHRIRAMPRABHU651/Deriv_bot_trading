"""Async token-bucket rate limiter (Deriv does not publish fixed limits; see API_NOTES.md)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        rate_per_second: float,
        burst: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_second <= 0 or burst < 1:
            raise ValueError("rate_per_second must be > 0 and burst >= 1")
        self._rate = rate_per_second
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._monotonic = monotonic
        self._last = monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self._rate)
