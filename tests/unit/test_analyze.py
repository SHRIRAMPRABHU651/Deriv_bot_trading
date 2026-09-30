"""research.analyze: the randomness tests must separate a random walk from a predictable series."""

from __future__ import annotations

import math

from research.analyze import randomness_tests, trades_needed
from tests.unit.synth import momentum_walk, random_walk


def test_random_walk_shows_no_structure() -> None:
    _, prices = random_walk(40_000, seed=11)
    assert randomness_tests(prices + 1000.0).significant == []


def test_momentum_series_is_flagged() -> None:
    _, prices = momentum_walk(40_000, seed=12)
    assert len(randomness_tests(prices).significant) >= 5


def test_trades_needed_grows_as_the_edge_shrinks() -> None:
    big, small = trades_needed(0.60, 0.50), trades_needed(0.53, 0.50)
    assert 0 < big < small
    assert math.isinf(trades_needed(0.5, 0.5))
