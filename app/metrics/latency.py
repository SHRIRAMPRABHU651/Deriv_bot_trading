"""Latency tracking (tick->signal, signal->order, order->confirmation) and feed health."""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta

from app.models.schemas import Mode
from app.storage.repositories import Repositories

KINDS = ("tick_to_signal", "signal_to_order", "order_to_confirmation", "confirmation_to_settlement")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return ordered[int(rank) - 1]


class LatencyTracker:
    def __init__(self, repos: Repositories, window: int = 50) -> None:
        self._repos = repos
        self._recent: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, mode: Mode, kind: str, ms: float, ts: datetime) -> None:
        if ms < 0:
            ms = 0.0
        self._recent[kind].append(ms)
        self._repos.add_latency(mode, kind, ms, ts)

    def recent_p95(self, kind: str, min_samples: int = 5) -> float | None:
        vals = list(self._recent[kind])
        return percentile(vals, 95) if len(vals) >= min_samples else None

    def daily_stats(self, mode: Mode, now: datetime) -> dict[str, dict[str, float]]:
        since = now - timedelta(days=1)
        out: dict[str, dict[str, float]] = {}
        for kind in KINDS:
            vals = self._repos.latency_samples(mode, kind, since)
            if vals:
                out[kind] = {
                    "median": statistics.median(vals),
                    "p95": percentile(vals, 95),
                    "max": max(vals),
                    "n": float(len(vals)),
                }
        return out


class FeedHealth:
    """Tracks the last tick per symbol; the feed is stale if ANY subscribed symbol is stale."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._started_at: float | None = None

    def watch(self, symbols: list[str], now: float) -> None:
        self._started_at = now
        for s in symbols:
            self._last.setdefault(s, now)

    def on_tick(self, symbol: str, ts: float) -> None:
        self._last[symbol] = ts

    def age(self, symbol: str, now: float) -> float | None:
        last = self._last.get(symbol)
        return None if last is None else now - last

    def is_stale(self, symbol: str, now: float, threshold: float) -> bool:
        age = self.age(symbol, now)
        return age is None or age > threshold

    def any_stale(self, now: float, threshold: float) -> bool:
        return any(now - ts > threshold for ts in self._last.values()) if self._last else True

    def reset(self) -> None:
        self._last.clear()
