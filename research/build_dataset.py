"""Build a supervised dataset (features at t, label = price[t+N] > price[t]) from a ticks CSV.

python -m research.build_dataset --ticks data/R_100.csv --horizon 10 --out data/R_100_h10.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from app.config import load_config
from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.features import FEATURE_NAMES, FEATURE_VERSION
from research._spec import add_spec_args, spec_from_args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticks", required=True)
    p.add_argument("--out", required=True)
    add_spec_args(p, load_config())
    args = p.parse_args(argv)
    spec = spec_from_args(args, load_config())
    epochs, prices = load_ticks_csv(args.ticks)
    ds = build_dataset(epochs, prices, spec.horizon_ticks, spec)
    np.savez_compressed(
        args.out,
        x=ds.x,
        up=ds.up,
        down=ds.down,
        tie=ds.tie,
        epochs=ds.epochs,
        horizon=ds.horizon,
        feature_names=np.array(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
    )
    print(
        f"product={spec.product.value} {len(ds)} samples, win-rate bull={ds.up.mean():.4f} "
        f"bear={ds.down.mean():.4f} tie={ds.tie_rate:.4f} "
        f"feature_version={FEATURE_VERSION} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
