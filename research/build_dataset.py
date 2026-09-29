"""Build a supervised dataset (features at t, label = price[t+N] > price[t]) from a ticks CSV.

python -m research.build_dataset --ticks data/R_100.csv --horizon 10 --out data/R_100_h10.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.features import FEATURE_NAMES, FEATURE_VERSION


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ticks", required=True)
    p.add_argument("--horizon", type=int, default=10, help="N ticks ahead == contract duration")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    epochs, prices = load_ticks_csv(args.ticks)
    ds = build_dataset(epochs, prices, args.horizon)
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
        f"{len(ds)} samples, base rate up={ds.up.mean():.4f} tie={ds.tie_rate:.4f} "
        f"feature_version={FEATURE_VERSION} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
