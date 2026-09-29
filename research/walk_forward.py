"""Walk-forward report (TRAIN | GAP | TEST, rolling) without writing any model artifact.

python -m research.walk_forward --ticks data/R_100.csv --horizon 10 --kind logistic
"""

from __future__ import annotations

import argparse

from app.config import load_config
from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.train import MODEL_KINDS
from app.ml.validate import run_walk_forward, shuffled_label_control


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticks", required=True)
    p.add_argument("--horizon", type=int, default=cfg.trading.duration_ticks)
    p.add_argument("--kind", choices=MODEL_KINDS, default="logistic")
    p.add_argument("--payout-ratio", type=float, default=1.95)
    p.add_argument("--edge-margin", type=float, default=cfg.ml.edge_margin)
    p.add_argument("--initial-train", type=float, default=0.5, help="fraction of samples")
    p.add_argument("--test-size", type=float, default=0.1, help="fraction of samples per fold")
    p.add_argument("--shuffles", type=int, default=10)
    args = p.parse_args(argv)
    epochs, prices = load_ticks_csv(args.ticks)
    ds = build_dataset(epochs, prices, args.horizon)
    n = len(ds)
    kw = {
        "initial_train": int(n * args.initial_train),
        "gap": args.horizon,
        "test_size": max(int(n * args.test_size), 200),
    }
    real = run_walk_forward(
        ds,
        kind=args.kind,
        payout_ratio=args.payout_ratio,
        edge_margin=args.edge_margin,
        alpha=cfg.ml.alpha,
        **kw,
    )
    print(f"samples={n} folds={len(real.folds)} horizon={args.horizon} gap={kw['gap']}")
    for i, f in enumerate(real.folds):
        m = f.metrics
        print(
            f"fold {i}: test={f.fold.test} n={m.n} acc={m.accuracy:.4f} "
            f"auc={m.roc_auc if m.roc_auc is None else round(m.roc_auc, 4)} "
            f"brier={m.brier:.4f} trades={f.n_trades} wins={f.wins}"
        )
    e = real.edge
    print(
        f"POOLED trades={e.n} wins={e.wins} win_rate={e.win_rate:.4f} "
        f"break_even={e.break_even:.4f} CI95=[{e.ci_low:.4f},{e.ci_high:.4f}] "
        f"p={e.p_value:.4f} h={e.effect_size:.4f}"
    )
    verdict = shuffled_label_control(
        ds,
        real,
        kind=args.kind,
        n_shuffles=args.shuffles,
        alpha=cfg.ml.alpha,
        min_trades=cfg.ml.min_validation_trades,
        payout_ratio=args.payout_ratio,
        edge_margin=args.edge_margin,
        **kw,
    )
    print(f"real AUC={verdict.real_auc} shuffled AUC p95={verdict.shuffled_auc_p95}")
    print(f"VERDICT: {verdict.label}")
    for r in verdict.reasons:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
