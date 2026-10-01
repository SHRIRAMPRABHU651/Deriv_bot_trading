"""Backtest rule-based strategies with realistic costs, adaptive exits and out-of-sample judgement.

python -m research.backtest --prices data/EURUSD_5m.csv --capital 20

Strategies (one position at a time, multiplier-style payoff):
  mean_reversion  fade a 2-sigma stretch from the rolling mean
  breakout        momentum breakout of the rolling N-bar high / low
  trend           fast / slow EMA cross

Exits are ADAPTIVE: take-profit and stop-loss are multiples of recent volatility (mean absolute
bar return over the last 14 bars x sqrt(horizon)), optionally with a trailing exit that locks in
profit after half of the target. Every trade pays Deriv-style commission and slippage.

Honest evaluation: parameters are chosen on the first 60 % of the data only; the verdict uses the
last 40 % (never seen), a one-sided t-test on the per-trade return with Bonferroni correction for
the number of strategies, and a Sharpe ratio on daily equity. "FAIL" is the expected result for
most markets; nothing here is a guarantee of profit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import lfilter

from app.ml.dataset import load_ticks_csv

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

STRATEGIES = ("mean_reversion", "breakout", "trend")


@dataclass(frozen=True)
class Costs:
    multiplier: int = 100
    fee_per_multiplier: float = 0.00044  # commission as a fraction of stake per unit of multiplier
    slippage_bps: float = 0.5  # per side, in basis points of price
    horizon: int = 12  # max bars a trade is held

    @property
    def round_trip(self) -> float:
        """Fraction of the stake lost to commission + two slippages."""
        return (
            self.fee_per_multiplier * self.multiplier
            + 2 * self.multiplier * self.slippage_bps / 1e4
        )


@dataclass(frozen=True)
class Params:
    strategy: str
    lookback: int
    k_tp: float
    k_sl: float
    trail: bool


@dataclass(frozen=True)
class Trade:
    entry: int
    exit: int
    side: int
    ret: float  # net return as a fraction of the stake
    sl_frac: float  # stop distance as a fraction of the stake (for risk sizing)


# ------------------------------------------------------------------ indicators / signals
def _rolling_mean_std(x: FloatArray, n: int) -> tuple[FloatArray, FloatArray]:
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    mean = np.full(len(x), np.nan)
    std = np.full(len(x), np.nan)
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    mean[n - 1 :] = s1 / n
    std[n - 1 :] = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    return mean, std


def ema(x: FloatArray, n: int) -> FloatArray:
    alpha = 2.0 / (n + 1.0)
    zi = np.array([x[0] * (1 - alpha)])
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)
    return np.asarray(out, dtype=np.float64)


def volatility(prices: FloatArray, n: int = 14) -> FloatArray:
    """Mean absolute bar return (fraction of price) over the last n bars; NaN until available."""
    ret = np.abs(np.diff(np.log(prices), prepend=np.log(prices[0])))
    mean, _ = _rolling_mean_std(ret, n)
    return mean


def signals(prices: FloatArray, strategy: str, lookback: int) -> IntArray:
    """+1 long / -1 short / 0 flat, using only information up to and including each bar."""
    n = len(prices)
    sig = np.zeros(n, dtype=np.int64)
    if strategy == "mean_reversion":
        mean, std = _rolling_mean_std(prices, lookback)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (prices - mean) / std
        sig[z < -2.0] = 1
        sig[z > 2.0] = -1
    elif strategy == "breakout":
        win = sliding_window_view(prices, lookback)  # window k covers bars k .. k+lookback-1
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        hi[lookback:] = win[:-1].max(axis=1)  # prior N bars only: excludes the current bar
        lo[lookback:] = win[:-1].min(axis=1)
        sig[prices > hi] = 1
        sig[prices < lo] = -1
    elif strategy == "trend":
        fast, slow = ema(prices, lookback), ema(prices, lookback * 3)
        diff = fast - slow
        up = (diff[1:] > 0) & (diff[:-1] <= 0)
        down = (diff[1:] < 0) & (diff[:-1] >= 0)
        sig[1:][up] = 1
        sig[1:][down] = -1
        sig[: lookback * 3] = 0  # EMAs not warmed up
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return sig


# ------------------------------------------------------------------ trade simulation
def simulate_trades(prices: FloatArray, p: Params, costs: Costs) -> list[Trade]:
    n, horizon, mult = len(prices), costs.horizon, costs.multiplier
    sig = signals(prices, p.strategy, p.lookback)
    vol = volatility(prices)
    trades: list[Trade] = []
    busy_until = -1
    for idx in np.flatnonzero(sig != 0):
        i = int(idx)
        if i <= busy_until or i + 2 + horizon > n or math.isnan(vol[i]) or vol[i] <= 0:
            continue
        side = int(sig[i])
        p0 = prices[i + 1]  # one-bar delay: the signal bar's close is already gone
        unit = vol[i] * math.sqrt(horizon)
        tp = p.k_tp * mult * unit
        sl = min(p.k_sl * mult * unit, 0.95)
        rel = side * mult * (prices[i + 2 : i + 2 + horizon] / p0 - 1.0)  # gross, stake fraction
        stop = rel <= -sl
        if p.trail:
            peak = np.maximum.accumulate(rel)
            take = ((peak >= 0.5 * tp) & (rel <= peak - 0.5 * tp)) | (rel >= 2.0 * tp)
        else:
            take = rel >= tp
        hits = np.flatnonzero(stop | take)
        k = int(hits[0]) if len(hits) else horizon - 1
        gross = float(rel[k])
        if not p.trail and take[k] and not stop[k]:
            gross = tp  # a fixed take-profit fills at its level, never better
        elif p.trail and rel[k] >= 2.0 * tp:
            gross = 2.0 * tp
        ret = max(gross - costs.round_trip, -1.0)  # a multiplier can never lose more than the stake
        exit_i = i + 2 + int(k)
        trades.append(Trade(i + 1, exit_i, side, ret, sl))
        busy_until = exit_i
    return trades


# ------------------------------------------------------------------ evaluation
@dataclass(frozen=True)
class Stats:
    trades: int
    win_rate: float
    mean_ret: float  # mean net return per trade, fraction of stake
    t_stat: float
    p_value: float  # one-sided: mean return > 0
    profit_factor: float
    sharpe: float
    max_drawdown: float
    final_equity: float
    forced_min_stake: int  # trades where the minimum stake exceeded the risk-sized stake
    effective_risk_pct: float  # mean % of equity actually risked per trade (stop distance)


def evaluate(
    trades: list[Trade],
    epochs: IntArray,
    capital: float,
    risk_pct: float,
    max_stake_pct: float,
    min_stake: float,
) -> Stats:
    rets = np.array([t.ret for t in trades], dtype=np.float64)
    if len(rets) < 2:
        return Stats(len(rets), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, capital, 0, 0.0)
    t_stat, p_two = stats.ttest_1samp(rets, 0.0)
    p_one = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    equity, peak, max_dd, forced = capital, capital, 0.0, 0
    eff: list[float] = []
    day_equity: dict[int, float] = {}
    for t in trades:
        if equity < min_stake:
            break  # account can no longer place a trade
        stake = min(risk_pct * equity / t.sl_frac, max_stake_pct * equity)
        if stake < min_stake:
            stake, forced = min_stake, forced + 1
        stake = min(stake, equity)
        eff.append(100.0 * stake * t.sl_frac / equity)
        equity += stake * t.ret
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        day_equity[int(epochs[t.exit]) // 86400] = equity
    days = sorted(day_equity)
    curve = []
    last = capital
    for d in range(days[0], days[-1] + 1) if days else []:
        last = day_equity.get(d, last)
        curve.append(last)
    daily = np.diff(np.array([capital, *curve])) / np.array([capital, *curve])[:-1]
    sharpe = (
        float(daily.mean() / daily.std() * math.sqrt(365))
        if len(daily) > 2 and daily.std() > 0
        else 0.0
    )
    return Stats(
        trades=len(rets),
        win_rate=float((rets > 0).mean()),
        mean_ret=float(rets.mean()),
        t_stat=float(t_stat),
        p_value=p_one,
        profit_factor=float(gains / losses) if losses > 0 else math.inf,
        sharpe=sharpe,
        max_drawdown=float(max_dd),
        final_equity=float(equity),
        forced_min_stake=forced,
        effective_risk_pct=float(np.mean(eff)) if eff else 0.0,
    )


def grid(strategy: str) -> list[Params]:
    return [
        Params(strategy, lb, k_tp, k_sl, trail)
        for lb, k_tp, k_sl, trail in product((20, 50), (0.5, 1.0, 1.5), (0.5, 1.0), (False, True))
    ]


@dataclass(frozen=True)
class StrategyReport:
    strategy: str
    params: Params
    in_sample: Stats
    out_of_sample: Stats
    approved: bool
    reason: str


def walk_forward(
    prices: FloatArray,
    epochs: IntArray,
    costs: Costs,
    *,
    capital: float,
    risk_pct: float,
    max_stake_pct: float,
    min_stake: float,
    alpha: float,
    min_oos_trades: int,
    train_frac: float = 0.6,
    strategies: tuple[str, ...] = STRATEGIES,
) -> list[StrategyReport]:
    cut = int(len(prices) * train_frac)
    out: list[StrategyReport] = []
    alpha_adj = alpha / len(strategies)
    for name in strategies:
        best: tuple[float, Params] | None = None
        for p in grid(name):
            tr = simulate_trades(prices[:cut], p, costs)
            if len(tr) < 30:
                continue
            score = float(np.mean([t.ret for t in tr]))
            if best is None or score > best[0]:
                best = (score, p)
        if best is None:
            empty = Stats(0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, capital, 0, 0.0)
            out.append(
                StrategyReport(name, grid(name)[0], empty, empty, False, "too few in-sample trades")
            )
            continue
        p = best[1]
        ins = evaluate(
            simulate_trades(prices[:cut], p, costs),
            epochs[:cut],
            capital,
            risk_pct,
            max_stake_pct,
            min_stake,
        )
        oos_trades = simulate_trades(prices[cut:], p, costs)
        oos = evaluate(oos_trades, epochs[cut:], capital, risk_pct, max_stake_pct, min_stake)
        reasons = []
        if oos.trades < min_oos_trades:
            reasons.append(f"only {oos.trades} out-of-sample trades (< {min_oos_trades})")
        if oos.mean_ret <= 0:
            reasons.append(
                f"mean net return {oos.mean_ret:+.2%} of stake per trade is not positive"
            )
        if oos.p_value >= alpha_adj:
            reasons.append(f"p={oos.p_value:.3f} not below corrected alpha {alpha_adj:.4f}")
        if oos.sharpe <= 0:
            reasons.append(f"Sharpe {oos.sharpe:.2f} is not positive")
        out.append(
            StrategyReport(
                name, p, ins, oos, not reasons, "; ".join(reasons) or "all checks passed"
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--prices", required=True, help="CSV epoch,quote (research.convert_csv / download)"
    )
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument(
        "--risk-pct", type=float, default=0.01, help="equity risked per trade at the stop"
    )
    ap.add_argument(
        "--max-stake-pct", type=float, default=0.25, help="stake cap as a fraction of equity"
    )
    ap.add_argument("--min-stake", type=float, default=1.0)
    ap.add_argument("--multiplier", type=int, default=100)
    ap.add_argument("--fee-per-multiplier", type=float, default=0.00044)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=12, help="max bars a trade is held")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-oos-trades", type=int, default=100)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)

    epochs, prices = load_ticks_csv(a.prices)
    costs = Costs(a.multiplier, a.fee_per_multiplier, a.slippage_bps, a.horizon)
    print(f"bars: {len(prices)}  span: {(epochs[-1] - epochs[0]) / 86400:.1f} days")
    print(f"round-trip cost: {costs.round_trip:.1%} of stake per trade (commission + slippage)\n")
    reports = walk_forward(
        prices,
        epochs,
        costs,
        capital=a.capital,
        risk_pct=a.risk_pct,
        max_stake_pct=a.max_stake_pct,
        min_stake=a.min_stake,
        alpha=a.alpha,
        min_oos_trades=a.min_oos_trades,
    )
    for r in reports:
        p, o = r.params, r.out_of_sample
        print(
            f"{r.strategy:<15} chosen in-sample: lookback={p.lookback} TP={p.k_tp}x SL={p.k_sl}x "
            f"{'trailing' if p.trail else 'fixed'}"
        )
        print(
            f"   in-sample  : {r.in_sample.trades:5d} trades  "
            f"mean {r.in_sample.mean_ret:+.2%}/trade"
        )
        print(
            f"   OUT-OF-SAMPLE: {o.trades:5d} trades  win {o.win_rate:.1%}  "
            f"mean {o.mean_ret:+.2%}/trade  "
            f"PF {o.profit_factor:.2f}  Sharpe {o.sharpe:+.2f}  maxDD {o.max_drawdown:.1%}  "
            f"{a.capital:.0f} -> {o.final_equity:.2f}"
        )
        print(
            f"   effective risk/trade {o.effective_risk_pct:.1f}% of equity "
            f"(min stake forced on {o.forced_min_stake} trades)"
        )
        print(f"   => {'APPROVED' if r.approved else 'REJECTED'}: {r.reason}\n")
        if o.sharpe < 0:
            print("   (negative Sharpe: fix or drop this strategy before going any further)\n")
    approved = [r.strategy for r in reports if r.approved]
    print(
        "APPROVED for DEMO validation: " + ", ".join(approved)
        if approved
        else "NO strategy passed out-of-sample after costs. "
        "Do not trade this market with these rules."
    )
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(
            json.dumps(
                [{**asdict(r), "approved": r.approved} for r in reports], indent=2, default=str
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
