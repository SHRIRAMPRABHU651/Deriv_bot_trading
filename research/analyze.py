"""Evidence report: is there ANY exploitable edge in these ticks after Deriv's fees?

python -m research.analyze --ticks data/R_100.csv [--fee-per-multiplier 0.00044]

1. Randomness tests on the tick returns (autocorrelation, variance ratio, runs test), Bonferroni
   adjusted for the number of tests.
2. Economics of each trade setting: the base rate at which the profit target is hit, the win rate
   needed to break even AFTER fees, and how many trades that edge would need to be provable.
3. A walk-forward machine-learning test per setting (train | gap | test), with the significance
   level divided by the number of settings tried, so trying many settings cannot manufacture a
   lucky "edge".

It never trades and never guarantees anything: "NO EVIDENCE OF EDGE" is a valid, common result.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from app.config import load_config
from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.validate import run_walk_forward
from app.products import Product, ProductSpec, assumed_terms, make_product_labels

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RandomnessResult:
    n: int
    tests: list[tuple[str, float, float]]  # (name, statistic, p-value)
    alpha_adjusted: float

    @property
    def significant(self) -> list[str]:
        return [name for name, _, p in self.tests if p < self.alpha_adjusted]


def randomness_tests(
    prices: FloatArray, alpha: float = 0.05, epochs: npt.NDArray[np.int64] | None = None
) -> RandomnessResult:
    ret = np.diff(np.log(prices))
    if epochs is not None and len(epochs) == len(prices):  # drop returns across market gaps
        dt = np.diff(epochs)
        ret = ret[dt <= 3 * np.median(dt)]
    n = len(ret)
    tests: list[tuple[str, float, float]] = []
    x = ret - ret.mean()
    var = float(np.dot(x, x) / n)
    for lag in range(1, 6):  # serial correlation of returns
        rho = float(np.dot(x[:-lag], x[lag:]) / (n * var))
        z = rho * math.sqrt(n)
        tests.append((f"autocorr lag {lag} (rho={rho:+.4f})", z, 2 * float(stats.norm.sf(abs(z)))))
    for q in (2, 5, 10, 20):  # Lo-MacKinlay variance ratio (homoskedastic)
        agg = np.convolve(ret, np.ones(q), mode="valid")
        vr = float(agg.var() / (q * ret.var()))
        se = math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
        z = (vr - 1.0) / se
        tests.append((f"variance ratio q={q} (VR={vr:.4f})", z, 2 * float(stats.norm.sf(abs(z)))))
    signs = np.sign(ret[ret != 0])  # Wald-Wolfowitz runs test on up/down moves
    n1, n2 = int((signs > 0).sum()), int((signs < 0).sum())
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    sd = math.sqrt((mu - 1) * (mu - 2) / (n1 + n2 - 1))
    z = (runs - mu) / sd
    tests.append(
        (f"runs test (runs={runs}, expected {mu:.0f})", z, 2 * float(stats.norm.sf(abs(z))))
    )
    return RandomnessResult(n, tests, alpha / len(tests))


def trades_needed(
    win_rate: float, break_even: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Independent trades needed to show win_rate > break_even (normal approximation)."""
    if win_rate <= break_even:
        return math.inf
    za, zb = float(stats.norm.isf(alpha)), float(stats.norm.isf(1 - power))
    s0, s1 = math.sqrt(break_even * (1 - break_even)), math.sqrt(win_rate * (1 - win_rate))
    return ((za * s0 + zb * s1) / (win_rate - break_even)) ** 2


def candidate_specs(base: ProductSpec, fee_per_multiplier: float) -> list[ProductSpec]:
    specs: list[ProductSpec] = []
    for mult, tp, sl in [
        (100, 0.1, 0.1),
        (100, 0.2, 0.1),
        (100, 0.1, 0.2),
        (40, 0.1, 0.1),
        (40, 0.2, 0.2),
    ]:
        specs.append(
            base.model_copy(
                update={
                    "product": Product.MULTIPLIER,
                    "multiplier": mult,
                    "take_profit_pct": tp,
                    "stop_loss_pct": sl,
                    "assumed_fee_pct": fee_per_multiplier * mult,
                }
            )
        )
    for horizon in (5, 10, 20):  # accumulators: how many surviving ticks before closing
        specs.append(
            base.model_copy(
                update={
                    "product": Product.ACCUMULATOR,
                    "growth_rate": 0.01,
                    "horizon_ticks": horizon,
                }
            )
        )
    return specs


def label(spec: ProductSpec) -> str:
    if spec.product is Product.MULTIPLIER:
        return (
            f"multiplier x{spec.multiplier} TP{spec.take_profit_pct:.0%}/SL{spec.stop_loss_pct:.0%}"
        )
    return f"accumulator {spec.growth_rate:.0%}/tick, {spec.horizon_ticks} ticks"


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticks", required=True)
    p.add_argument(
        "--fee-per-multiplier",
        type=float,
        default=0.00044,
        help="commission per unit of multiplier, as a fraction of stake (0.00044 = 4.4%% at x100)",
    )
    p.add_argument(
        "--skip-accumulators",
        action="store_true",
        help="accumulators exist only on synthetic indices: skip them for forex/crypto data",
    )
    p.add_argument("--horizon", type=int, default=cfg.trading.product.horizon_ticks)
    p.add_argument("--alpha", type=float, default=cfg.ml.alpha)
    args = p.parse_args(argv)

    epochs, prices = load_ticks_csv(args.ticks)
    print(f"ticks: {len(prices)}  span: {(epochs[-1] - epochs[0]) / 3600:.1f} h\n")

    rt = randomness_tests(prices, args.alpha, epochs)
    print(f"1) RANDOMNESS TESTS (Bonferroni alpha {rt.alpha_adjusted:.4f}, {len(rt.tests)} tests)")
    for name, z, pv in rt.tests:
        flag = "  <-- significant" if pv < rt.alpha_adjusted else ""
        print(f"   {name:<46} z={z:+7.2f}  p={pv:.4f}{flag}")
    print(
        "   =>",
        "structure found: " + ", ".join(rt.significant)
        if rt.significant
        else "consistent with a random walk: no linear predictability in these ticks",
    )

    base = cfg.trading.product.model_copy(update={"horizon_ticks": args.horizon})
    specs = candidate_specs(base, args.fee_per_multiplier)
    if args.skip_accumulators:
        specs = [x for x in specs if x.product is not Product.ACCUMULATOR]
    alpha_adj = args.alpha / len(specs)
    print(
        f"\n2+3) TRADE SETTINGS (walk-forward alpha {alpha_adj:.4f}, {len(specs)} settings tried)"
    )
    print(
        f"{'setting':<40}{'base':>7}{'need':>7}{'gap':>8}{'OOS n':>7}{'wins':>6}"
        f"{'win%':>7}{'p':>8}  verdict"
    )
    found = 0
    for spec in specs:
        ds = build_dataset(epochs, prices, spec.horizon_ticks, spec)
        win_b, win_u, _ = make_product_labels(prices, spec)
        base_rate = (
            float(np.concatenate([win_b, win_u]).mean())
            if spec.separate_bear_head
            else float(win_b.mean())
        )
        win_amt, loss_amt = assumed_terms(spec)
        need = loss_amt / (win_amt + loss_amt)
        n = len(ds)
        try:
            res = run_walk_forward(
                ds,
                kind="logistic",
                initial_train=int(n * 0.5),
                gap=spec.horizon_ticks,
                test_size=max(int(n * 0.1), 200),
                terms=(win_amt, loss_amt),
                edge_margin=cfg.ml.edge_margin,
                alpha=alpha_adj,
            )
        except ValueError as exc:
            print(f"{label(spec):<40} skipped: {exc}")
            continue
        e = res.edge
        ok = e.significant and e.n >= cfg.ml.min_validation_trades
        found += ok
        print(
            f"{label(spec):<40}{base_rate:7.1%}{need:7.1%}{(base_rate - need) * 100:+7.1f}p"
            f"{e.n:7d}{e.wins:6d}"
            f"{e.win_rate:7.1%}{e.p_value:8.3f}  {'CANDIDATE' if ok else 'no edge'}"
        )
    print()
    if found:
        print(
            f"{found} setting(s) show a statistically credible out-of-sample edge. "
            "Next: research.train_model for "
            "that setting, then DEMO validation over >= 1000 trades. This is still not a guarantee."
        )
    else:
        print(
            "NO EVIDENCE OF EDGE in any setting after fees and multiple-testing correction. "
            "The bot must not "
            "trade these settings for profit. ('gap' = base hit rate minus the fee-adjusted "
            "break-even; negative "
            "means the trade loses on average even before any model.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
