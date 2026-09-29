"""Train + validate a model and write model/model.joblib + model/metadata.json.

    python -m research.train_model --symbol R_100 --ticks data/R_100.csv --horizon 10

The result is REJECTED (=> the bot refuses to trade it) unless out-of-sample walk-forward evidence
is statistically credible AND beats the shuffled-label control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from app.config import load_config
from app.ml.dataset import load_ticks_csv
from app.ml.pipeline import TrainReport, train_and_validate
from app.ml.train import MODEL_KINDS
from app.products import ProductSpec
from research._spec import add_spec_args, spec_from_args


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbol", default="R_100")
    p.add_argument("--ticks", required=True, help="CSV with epoch,quote (research.download_ticks)")
    p.add_argument("--kind", choices=MODEL_KINDS, default="logistic")
    p.add_argument("--model-dir", default=cfg.app.model_dir)
    add_spec_args(p, cfg)
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
    spec = spec_from_args(args, cfg)
    epochs, prices = load_ticks_csv(args.ticks)
    try:
        report = _train(args, spec, epochs, prices)
    except ValueError as exc:
        print(f"cannot train: {exc}")
        return 2
    print(json.dumps(asdict(report), indent=2, default=str))
    print(f"\nVERDICT: {report.verdict}  ->  model status {report.status}")
    if report.verdict != "EDGE EVIDENCE":
        print("The bot will NOT trade this model.")
    return 0


def _train(args: argparse.Namespace, spec: ProductSpec, epochs: Any, prices: Any) -> TrainReport:
    return train_and_validate(
        epochs,
        prices,
        symbol=args.symbol,
        horizon=spec.horizon_ticks,
        model_dir=args.model_dir,
        kind=args.kind,
        spec=spec,
        edge_margin=args.edge_margin,
        alpha=args.alpha,
        min_trades=args.min_trades,
        n_shuffles=args.shuffles,
        n_comparisons=args.comparisons,
    )


if __name__ == "__main__":
    raise SystemExit(main())
