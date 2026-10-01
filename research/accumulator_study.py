"""Accumulator short-hold study: does "open, hold a few ticks, close" make money?

python -m research.accumulator_study --ticks data/R_100.csv --growth 0.01 --barrier-percent 0.06126

An accumulator grows the stake by `growth` for every tick that the price stays within +-barrier
of the previous tick; one tick outside the barrier loses the WHOLE stake. Closing after k ticks
therefore wins (1+growth)^k - 1 with probability s_k and loses 1.0 (the stake) otherwise:

    expected profit per trade = s_k * ((1+g)^k - 1) - (1 - s_k)

It wins MOST trades yet can still lose money: many small wins, rare total losses. This tool
measures s_k on real ticks (non-overlapping trades, Wilson 95 % interval) and says which it is.
Spread/latency are NOT included, so real results are slightly worse than shown.
Read the barrier from `python -m scripts.check_proposal R_100 accumulator` ("within the +-x%").
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.ml.dataset import load_ticks_csv

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HoldResult:
    ticks: int
    seconds: float
    trades: int
    survival: float
    survival_low: float
    survival_high: float
    win_amount: float  # profit as a fraction of the stake when it survives
    break_even: float  # survival needed to break even
    ev: float  # expected profit per trade, fraction of stake
    ev_low: float
    ev_high: float

    @property
    def verdict(self) -> str:
        if self.ev_high < 0:
            return "LOSES on average"
        if self.ev_low > 0:
            return "positive edge"
        return "inconclusive"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def ev_of(survival: float, win_amount: float) -> float:
    return survival * win_amount - (1.0 - survival)


def study(
    prices: FloatArray, growth: float, barrier_frac: float, max_hold: int, tick_seconds: float
) -> list[HoldResult]:
    ret = np.abs(prices[1:] / prices[:-1] - 1.0)
    ok = ret < barrier_frac  # the tick stayed inside the band
    out: list[HoldResult] = []
    for k in range(1, max_hold + 1):
        n = len(ok) // k  # non-overlapping trades: independent samples, honest intervals
        survived = ok[: n * k].reshape(n, k).all(axis=1)
        wins, s = int(survived.sum()), float(survived.mean()) if n else 0.0
        lo, hi = wilson(wins, n)
        win = (1.0 + growth) ** k - 1.0
        out.append(
            HoldResult(
                k,
                k * tick_seconds,
                n,
                s,
                lo,
                hi,
                win,
                1.0 / (1.0 + win),
                ev_of(s, win),
                ev_of(lo, win),
                ev_of(hi, win),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--growth", type=float, default=0.01)
    ap.add_argument("--barrier-percent", type=float, required=True, help="e.g. 0.06126")
    ap.add_argument("--max-hold", type=int, default=10)
    a = ap.parse_args(argv)
    epochs, prices = load_ticks_csv(a.ticks)
    tick_s = float(np.median(np.diff(epochs)))
    print(
        f"{len(prices)} ticks, one every {tick_s:.1f} s; growth {a.growth:.1%}/tick, "
        f"barrier +-{a.barrier_percent}%\n"
    )
    print(
        f"{'hold':>10}{'trades':>8}{'wins':>8}{'need':>8}{'EV/trade':>10}{'95% range':>20}  verdict"
    )
    for r in study(prices, a.growth, a.barrier_percent / 100.0, a.max_hold, tick_s):
        print(
            f"{r.ticks:>3} ticks/{r.seconds:>3.0f}s{r.trades:>8}{r.survival:>8.1%}"
            f"{r.break_even:>8.1%}"
            f"{r.ev:>+10.2%}  [{r.ev_low:+.2%}, {r.ev_high:+.2%}]  {r.verdict}"
        )
    print(
        "\n'wins' is how often the trade ends in profit; 'need' is the win rate required "
        "to break even."
        "\nA high win rate below 'need' means many small wins and rare total losses: negative EV."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
