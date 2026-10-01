"""Accumulator study: a high win rate can still mean negative expected value."""

from __future__ import annotations

import numpy as np

from research.accumulator_study import ev_of, study, wilson


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = wilson(95, 100)
    assert lo < 0.95 < hi and lo >= 0.0 and hi <= 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_many_small_wins_can_still_lose_money() -> None:
    # 98.5 % of trades win 1 %, 1.5 % lose the whole stake
    assert ev_of(0.985, 0.01) < 0
    assert ev_of(0.995, 0.01) > 0  # break-even is 1 / 1.01 = 99.01 %


def test_study_finds_negative_ev_on_a_random_walk_and_positive_when_price_is_calm() -> None:
    rng = np.random.default_rng(3)
    barrier = 0.0006126
    risky = 628.0 * np.exp(np.cumsum(rng.normal(0, 0.00025, 60_000)))  # ~2.4 sigma barrier
    calm = 628.0 * np.exp(np.cumsum(rng.normal(0, 0.00005, 60_000)))  # barrier is 12 sigma
    r1 = study(risky, 0.01, barrier, 3, 2.0)
    assert 0.97 < r1[0].survival < 0.995  # wins almost every trade ...
    assert all(r.ev < 0 and r.verdict == "LOSES on average" for r in r1)  # ... and still loses
    assert all(r.ev > 0 for r in study(calm, 0.01, barrier, 3, 2.0))
