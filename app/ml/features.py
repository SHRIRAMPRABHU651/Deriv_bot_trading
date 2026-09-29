"""Deterministic features. Batch and streaming paths share ONE implementation.

Every feature at time t is a function of the last WINDOW prices ending at t (inclusive) only, so
no future information can leak in. Windows are bounded, hence memory is bounded too.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_VERSION = "v1"
WINDOW = 64
FEATURE_NAMES: tuple[str, ...] = (
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "volatility",
    "ema_fast",
    "ema_slow",
    "ema_gap",
    "trend_slope",
    "rsi",
    "streak",
    "direction_freq",
)
EMA_FAST = 5
EMA_SLOW = 20
VOL_WINDOW = 20
RSI_PERIOD = 14

FloatArray = npt.NDArray[np.float64]


def _ema(windows: FloatArray, span: int) -> FloatArray:
    alpha = 2.0 / (span + 1.0)
    e = windows[:, 0].copy()
    for j in range(1, windows.shape[1]):
        e = alpha * windows[:, j] + (1.0 - alpha) * e
    return e


def compute_features(windows: FloatArray) -> FloatArray:
    """windows: (n, WINDOW) of prices, last column = current tick. Returns (n, n_features)."""
    if windows.ndim != 2 or windows.shape[1] != WINDOW:
        raise ValueError(f"expected (n, {WINDOW}) windows")
    with np.errstate(divide="ignore", invalid="ignore"):
        last = windows[:, -1]
        steps = windows[:, 1:] / windows[:, :-1] - 1.0
        rets = [windows[:, -1] / windows[:, -1 - k] - 1.0 for k in (1, 3, 5, 10)]
        vol = steps[:, -VOL_WINDOW:].std(axis=1)
        ema_f = _ema(windows, EMA_FAST)
        ema_s = _ema(windows, EMA_SLOW)
        y = windows[:, -VOL_WINDOW:]
        x = np.arange(VOL_WINDOW, dtype=np.float64) - (VOL_WINDOW - 1) / 2.0
        slope = ((y - y.mean(axis=1, keepdims=True)) * x).sum(axis=1) / (x**2).sum() / last
        delta = np.diff(windows[:, -(RSI_PERIOD + 1) :], axis=1)
        gain = np.clip(delta, 0, None).mean(axis=1)
        loss = np.clip(-delta, 0, None).mean(axis=1)
        rsi = np.where(
            loss == 0, np.where(gain == 0, 50.0, 100.0), 100.0 - 100.0 / (1.0 + gain / loss)
        )
        sign = np.sign(steps)
        cur = sign[:, -1]
        streak = np.zeros(len(windows))
        alive = cur != 0
        for j in range(steps.shape[1] - 1, -1, -1):
            match = alive & (sign[:, j] == cur)
            streak += match
            alive = match
        streak = streak * cur
        dir_freq = (steps[:, -VOL_WINDOW:] > 0).mean(axis=1)
    out = np.column_stack(
        [
            *rets,
            vol,
            ema_f / last - 1.0,
            ema_s / last - 1.0,
            (ema_f - ema_s) / last,
            slope,
            rsi,
            streak,
            dir_freq,
        ]
    )
    return out.astype(np.float64)


def batch_features(prices: FloatArray, chunk: int = 20000) -> FloatArray:
    """Features for every index t >= WINDOW-1; row i corresponds to tick index i + WINDOW - 1."""
    if len(prices) < WINDOW:
        return np.empty((0, len(FEATURE_NAMES)))
    view = sliding_window_view(prices.astype(np.float64), WINDOW)
    parts = [
        compute_features(np.ascontiguousarray(view[i : i + chunk]))
        for i in range(0, len(view), chunk)
    ]
    return np.vstack(parts)


class SymbolFeatureState:
    """Per-symbol bounded rolling state used at runtime (streaming path)."""

    def __init__(self) -> None:
        self._buf: deque[float] = deque(maxlen=WINDOW)
        self.ticks_seen = 0

    def update(self, price: float) -> FloatArray | None:
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"invalid price {price!r}")
        self._buf.append(price)
        self.ticks_seen += 1
        if len(self._buf) < WINDOW:
            return None
        feats = compute_features(np.asarray(self._buf, dtype=np.float64).reshape(1, WINDOW))
        row: FloatArray = feats[0]
        return row

    def __len__(self) -> int:
        return len(self._buf)
