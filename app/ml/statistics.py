"""Statistics for edge validation: break-even, EV, exact binomial test, Wilson CI, effect size."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import stats


def break_even_from_ratio(payout_ratio: float) -> float:
    """p* = 1 / R, R = payout / stake."""
    if payout_ratio <= 0:
        raise ValueError("payout ratio must be positive")
    return 1.0 / payout_ratio


def expected_value(p: float, payout_ratio: float) -> float:
    """EV per unit stake = p * R - 1."""
    return p * payout_ratio - 1.0


def wilson_interval(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = float(stats.norm.ppf(1 - alpha / 2))
    phat = wins / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_pvalue_greater(wins: int, n: int, p0: float) -> float:
    """Exact one-sided binomial test. H0: p <= p0  vs  H1: p > p0."""
    if n <= 0:
        return 1.0
    return float(stats.binomtest(wins, n, p0, alternative="greater").pvalue)


def cohens_h(p1: float, p0: float) -> float:
    """Effect size between two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))


def required_sample_size(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate trades needed to detect true win rate p1 > p0 (one-sided z-test)."""
    if p1 <= p0:
        raise ValueError("p1 must exceed p0")
    za = float(stats.norm.ppf(1 - alpha))
    zb = float(stats.norm.ppf(power))
    num = za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)


@dataclass(frozen=True)
class EdgeTest:
    n: int
    wins: int
    win_rate: float
    break_even: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float
    alpha: float
    significant: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def edge_test(wins: int, n: int, break_even: float, alpha: float = 0.05) -> EdgeTest:
    """Is the observed win rate statistically above break-even?"""
    if n <= 0:
        return EdgeTest(0, 0, 0.0, break_even, 0.0, 1.0, 1.0, 0.0, alpha, False)
    rate = wins / n
    lo, hi = wilson_interval(wins, n, alpha)
    pval = binomial_pvalue_greater(wins, n, break_even)
    return EdgeTest(
        n=n,
        wins=wins,
        win_rate=rate,
        break_even=break_even,
        ci_low=lo,
        ci_high=hi,
        p_value=pval,
        effect_size=cohens_h(rate, break_even) if 0 <= rate <= 1 else 0.0,
        alpha=alpha,
        significant=pval < alpha,
    )


def rolling_edge_check(wins: int, n: int, break_even: float, alpha: float = 0.05) -> bool:
    """True (=> HALT) when the win rate is significantly BELOW break-even (H1: p < p0)."""
    if n <= 0:
        return False
    return float(stats.binomtest(wins, n, break_even, alternative="less").pvalue) < alpha
