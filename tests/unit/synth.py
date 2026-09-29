"""Synthetic price series for ML tests (seeded, deterministic)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]


def random_walk(n: int, seed: int = 1, sigma: float = 0.1) -> tuple[npt.NDArray[np.int64], F]:
    rng = np.random.default_rng(seed)
    prices = 1000.0 + np.cumsum(rng.normal(0.0, sigma, n))
    return np.arange(n, dtype=np.int64) + 1_700_000_000, prices.astype(np.float64)


def momentum_walk(n: int, seed: int = 2, phi: float = 0.7) -> tuple[npt.NDArray[np.int64], F]:
    """Autocorrelated steps => direction is genuinely predictable from recent returns."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.05, n)
    steps = np.zeros(n)
    for i in range(1, n):
        steps[i] = phi * steps[i - 1] + eps[i]
    prices = 1000.0 + np.cumsum(steps)
    return np.arange(n, dtype=np.int64) + 1_700_000_000, prices.astype(np.float64)


def vol_regime_walk(
    n: int, seed: int = 3, low: float = 0.03, high: float = 0.3, block: int = 300
) -> tuple[npt.NDArray[np.int64], F]:
    """Volatility clusters: persistent calm and stormy regimes (predictable from recent vol)."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.integers(0, 2, n // block + 1), block)[:n]
    sigma = np.where(regime == 1, high, low)
    prices = 1000.0 + np.cumsum(rng.normal(0.0, 1.0, n) * sigma)
    return np.arange(n, dtype=np.int64) + 1_700_000_000, prices.astype(np.float64)
