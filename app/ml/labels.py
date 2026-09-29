"""Labels: does the price N ticks ahead exceed the entry price?

Expiry mapping: a Rise/Fall contract with duration N ticks is settled by comparing the exit tick
(the N-th tick after the entry tick) with the entry tick. We therefore label tick t with
    up   = price[t+N] >  price[t]      (CALL wins)
    down = price[t+N] <  price[t]      (PUT wins)
    tie  = price[t+N] == price[t]      (both lose)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def make_labels(
    prices: npt.NDArray[np.float64], horizon: int
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """Return (up, down, tie) for t in [0, len-horizon). Length is len(prices) - horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(prices) <= horizon:
        empty = np.zeros(0, dtype=bool)
        return empty, empty.copy(), empty.copy()
    entry = prices[:-horizon]
    exit_ = prices[horizon:]
    return exit_ > entry, exit_ < entry, exit_ == entry
