"""Evaluate / promote a model.

Out-of-sample check of a saved model on NEWER ticks:
    python -m research.evaluate --model-dir model --ticks data/new_ticks.csv

Manual lifecycle steps (never automatic):
    # CALIBRATED -> DEMO_VALIDATING
    python -m research.evaluate --model-dir model --start-demo-validation
    # dry-run promotion check
    python -m research.evaluate --model-dir model --demo-db data/derivbot.db
    # DEMO_VALIDATING -> PROMOTABLE (only with enough significant demo trades)
    python -m research.evaluate --model-dir model --demo-db data/derivbot.db --promote
"""

from __future__ import annotations

import argparse

from app.config import load_config
from app.ml.artifacts import read_metadata
from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.pipeline import evaluate_demo_promotion, start_demo_validation
from app.ml.predict import ModelPredictor
from app.ml.statistics import edge_test


def evaluate_ticks(model_dir: str, ticks: str, alpha: float, margin: float) -> None:
    pred = ModelPredictor.load(model_dir)
    meta = pred.meta
    epochs, prices = load_ticks_csv(ticks)
    ds = build_dataset(epochs, prices, meta.label_horizon, pred.spec).stride(meta.label_horizon)
    p_up, p_put = pred.model.direction_probs(ds.x)
    be = meta.break_even
    thr = be + margin
    call = (p_up >= thr) & (p_up >= p_put)
    put = (p_put >= thr) & (p_put > p_up)
    wins = int(ds.up[call].sum() + ds.down[put].sum())
    trades = int(call.sum() + put.sum())
    test = edge_test(wins, trades, be, alpha)
    print(
        f"model={meta.model_version} product={meta.product} status={meta.status} "
        f"samples={len(ds)} trades={trades}"
    )
    print(
        f"win_rate={test.win_rate:.4f} break_even={be:.4f} "
        f"CI95=[{test.ci_low:.4f},{test.ci_high:.4f}] p={test.p_value:.4f} "
        f"significant={test.significant}"
    )
    print(
        "No statistically credible edge => this model must not trade."
        if not test.significant
        else "Out-of-sample edge is statistically credible (still not a guarantee)."
    )


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-dir", default=cfg.app.model_dir)
    p.add_argument("--ticks")
    p.add_argument("--start-demo-validation", action="store_true")
    p.add_argument("--demo-db")
    p.add_argument("--promote", action="store_true", help="apply the promotion (else dry run)")
    p.add_argument("--min-demo-trades", type=int, default=cfg.ml.min_demo_trades)
    args = p.parse_args(argv)
    meta = read_metadata(args.model_dir)
    print(f"model {meta.model_version}: status={meta.status} verdict={meta.verdict or '-'}")
    if args.start_demo_validation:
        print("new status:", start_demo_validation(args.model_dir).status)
    if args.ticks:
        evaluate_ticks(args.model_dir, args.ticks, cfg.ml.alpha, cfg.ml.edge_margin)
    if args.demo_db:
        ok, test, why = evaluate_demo_promotion(
            args.model_dir,
            args.demo_db,
            min_demo_trades=args.min_demo_trades,
            alpha=cfg.ml.alpha,
            apply=args.promote,
        )
        print(
            f"demo trades={test.n} wins={test.wins} win_rate={test.win_rate:.4f} "
            f"break_even={test.break_even:.4f} CI95=[{test.ci_low:.4f},{test.ci_high:.4f}] "
            f"p={test.p_value:.4f}"
        )
        print(("PROMOTABLE: " if ok else "NOT promotable: ") + why)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
