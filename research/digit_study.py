"""Is there any exploitable pattern in the LAST DIGIT of a Deriv index? (Digits Over / Under)

python -m research.download_ticks --symbol 1HZ10V --count 86400 --out data/1HZ10V.csv
python -m research.digit_study --ticks data/1HZ10V.csv --decimals 2

Tests (a) that the 10 digits are equally likely, (b) that the next digit does not depend on the
current one, and prints, for every Over / Under prediction, the observed win frequency, the win
frequency needed to break even and the expected profit per trade with a 95 % range.

`--margin` is the house margin assumed in the payout (payout = fair payout x (1 - margin)).
Read the REAL payout from `python -m scripts.check_proposal` (digit contracts) and pass it as
`--margin` to match; 0.028 is an assumption.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from app.ml.dataset import load_ticks_csv
from research.accumulator_study import wilson

IntArray = npt.NDArray[np.int64]


def last_digits(prices: npt.NDArray[np.float64], decimals: int) -> IntArray:
    """Last printed digit. CSVs drop trailing zeros, so the digit is rebuilt from the decimals."""
    return np.asarray(np.rint(prices * 10**decimals).astype(np.int64) % 10, dtype=np.int64)


@dataclass(frozen=True)
class DigitTests:
    n: int
    uniform_p: float  # chi-square: all 10 digits equally likely
    markov_p: float  # chi-square: next digit independent of the current digit
    frequencies: list[float]


def digit_tests(digits: IntArray) -> DigitTests:
    counts = np.bincount(digits, minlength=10)
    uniform = stats.chisquare(counts)
    table = np.zeros((10, 10), dtype=np.int64)
    np.add.at(table, (digits[:-1], digits[1:]), 1)
    markov_p = float(stats.chi2_contingency(table)[1])
    return DigitTests(
        len(digits), float(uniform.pvalue), markov_p, [float(c) / len(digits) for c in counts]
    )


@dataclass(frozen=True)
class BarrierRow:
    side: str  # "over" | "under"
    digit: int
    win_freq: float
    low: float
    high: float
    fair_p: float
    break_even: float
    ev: float
    ev_low: float
    ev_high: float


def payout_profit(fair_p: float, margin: float) -> float:
    """Profit per 1 staked on a win when the payout is the fair one less `margin`."""
    return (1.0 - margin) / fair_p - 1.0


def barrier_table(digits: IntArray, margin: float) -> list[BarrierRow]:
    n = len(digits)
    rows: list[BarrierRow] = []
    for side, rng_ in (("over", range(0, 9)), ("under", range(1, 10))):
        for d in rng_:
            wins = int((digits > d).sum() if side == "over" else (digits < d).sum())
            fair = (9 - d) / 10 if side == "over" else d / 10
            profit = payout_profit(fair, margin)
            lo, hi = wilson(wins, n)
            f = wins / n
            rows.append(
                BarrierRow(
                    side,
                    d,
                    f,
                    lo,
                    hi,
                    fair,
                    1.0 / (1.0 + profit),
                    f * profit - (1 - f),
                    lo * profit - (1 - lo),
                    hi * profit - (1 - hi),
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--decimals", type=int, default=2, help="decimals Deriv prints for this symbol")
    ap.add_argument("--margin", type=float, default=0.028)
    a = ap.parse_args(argv)
    _, prices = load_ticks_csv(a.ticks)
    digits = last_digits(prices, a.decimals)
    t = digit_tests(digits)
    print(
        f"{t.n} ticks. digit frequencies: "
        + " ".join(f"{i}:{f:.3f}" for i, f in enumerate(t.frequencies))
    )
    uni = "BIAS" if t.uniform_p < 0.01 else "consistent with uniform"
    dep = "DEPENDENCE" if t.markov_p < 0.01 else "consistent with independent"
    print(f"digits equally likely?   p={t.uniform_p:.4f}  ({uni})")
    print(f"next digit independent?  p={t.markov_p:.4f}  ({dep})")
    print(f"\nassumed house margin {a.margin:.1%}\n")
    print(f"{'bet':<10}{'win':>8}{'need':>8}{'EV/trade':>10}{'95% range':>20}  verdict")
    best: BarrierRow | None = None
    for r in barrier_table(digits, a.margin):
        verdict = "LOSES" if r.ev_high < 0 else ("EDGE?" if r.ev_low > 0 else "inconclusive")
        if best is None or r.ev > best.ev:
            best = r
        print(
            f"{r.side} {r.digit:<5}{r.win_freq:>8.1%}{r.break_even:>8.1%}{r.ev:>+10.2%}"
            f"  [{r.ev_low:+.2%}, {r.ev_high:+.2%}]  {verdict}"
        )
    if best is not None:
        print(
            f"\nbest bet in this sample: {best.side} {best.digit} at {best.ev:+.2%} per trade "
            "(picking the best of 18 bets in-sample overstates it; judge on fresh data)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
