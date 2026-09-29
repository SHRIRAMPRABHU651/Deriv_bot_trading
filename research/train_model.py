"""Train + validate a model and write model/model.joblib + model/metadata.json.

    python -m research.train_model --symbol R_100 --ticks data/R_100.csv --horizon 10

The result is REJECTED (=> the bot refuses to trade it) unless out-of-sample walk-forward evidence
is statistically credible AND beats the shuffled-label control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.config import load_config
from app.ml.dataset import load_ticks_csv
from app.ml.pipeline import train_and_validate
from app.ml.train import MODEL_KINDS


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbol", default="R_100")
    p.add_argument("--ticks", required=True, help="CSV with epoch,quote (research.download_ticks)")
    p.add_argument("--horizon", type=int, default=cfg.trading.duration_ticks)
    p.add_argument("--kind", choices=MODEL_KINDS, default="logistic")
    p.add_argument("--model-dir", default=cfg.app.model_dir)
    p.add_argument(
        "--payout-ratio",
        type=float,
        default=1.95,
        help="ASSUMED payout/stake for research only; live trading uses the real proposal",
    )
    p.add_argument("--edge-margin", type=float, default=cfg.ml.edge_margin)
    p.add_argument("--alpha", type=float, default=cfg.ml.alpha)
    p.add_argument("--min-trades", type=int, default=cfg.ml.min_validation_trades)
    p.add_argument("--shuffles", type=int, default=20)
    p.add_argument(
        "--comparisons",
        type=int,
        default=1,
        help="how many model/horizon variants you tried (Bonferroni-adjusts alpha)",
    )
    args = p.parse_args(argv)
    epochs, prices = load_ticks_csv(args.ticks)
    report = train_and_validate(
        epochs,
        prices,
        symbol=args.symbol,
        horizon=args.horizon,
        model_dir=args.model_dir,
        kind=args.kind,
        payout_ratio=args.payout_ratio,
        edge_margin=args.edge_margin,
        alpha=args.alpha,
        min_trades=args.min_trades,
        n_shuffles=args.shuffles,
        n_comparisons=args.comparisons,
    )
    print(json.dumps(asdict(report), indent=2, default=str))
    print(f"\nVERDICT: {report.verdict}  ->  model status {report.status}")
    if report.verdict != "EDGE EVIDENCE":
        print("The bot will NOT trade this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
