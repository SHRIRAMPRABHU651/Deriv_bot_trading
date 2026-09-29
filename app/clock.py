"""Injectable clock so time-dependent risk logic (midnight rollover etc.) is testable."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    def time(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def time(self) -> float:
        return time.time()


class FakeClock:
    """Manually advanced clock used in tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock requires a timezone-aware datetime")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def time(self) -> float:
        return self._now.timestamp()

    def set(self, value: datetime) -> None:
        self._now = value

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._now = self._now + timedelta(seconds=seconds)
