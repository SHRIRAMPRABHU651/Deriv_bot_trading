"""research.digit_study / martingale_sim: fair digits lose to the margin; doubling ends in ruin."""

from __future__ import annotations

import numpy as np

from research.digit_study import barrier_table, digit_tests, last_digits, payout_profit
from research.martingale_sim import simulate


def test_last_digit_is_rebuilt_from_the_decimals_not_the_trimmed_text() -> None:
    prices = np.array([6123.40, 6123.45, 6123.07, 6120.00])
    assert list(last_digits(prices, 2)) == [0, 5, 7, 0]


def test_uniform_digits_are_not_biased_and_every_bet_loses_the_margin() -> None:
    rng = np.random.default_rng(5)
    digits = rng.integers(0, 10, 200_000).astype(np.int64)
    t = digit_tests(digits)
    assert t.uniform_p > 0.01 and t.markov_p > 0.01
    rows = barrier_table(digits, margin=0.028)
    assert len(rows) == 18
    assert all(-0.05 < r.ev < 0.0 for r in rows)  # about -2.8 % each
    assert all(r.ev_high < 0.01 for r in rows)


def test_a_biased_digit_is_detected() -> None:
    rng = np.random.default_rng(6)
    digits = rng.choice(10, 100_000, p=[0.08] * 5 + [0.12] * 5).astype(np.int64)
    assert digit_tests(digits).uniform_p < 1e-6


def test_payout_profit_matches_the_fair_odds_less_margin() -> None:
    assert abs(payout_profit(0.4, 0.0) - 1.5) < 1e-12  # Over 5: 2.5x return
    assert payout_profit(0.4, 0.028) < 1.5


def test_doubling_after_losses_is_ruinous_on_a_small_account() -> None:
    small = simulate(capital=20, target=10, p_win=0.4, payout_profit=1.43, sessions=4000, seed=2)
    assert small.ruined > 0.3 and small.mean_final_profit < 0  # coin-flip to double, else bust
    big = simulate(capital=1000, target=10, p_win=0.4, payout_profit=1.43, sessions=4000, seed=2)
    assert (
        big.reached_target > 0.9 and big.mean_final_profit < 0
    )  # wins often, still loses on average
