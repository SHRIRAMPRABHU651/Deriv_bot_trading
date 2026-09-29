"""Dataset construction from tick history (features look back, labels look forward)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.ml.features import WINDOW, batch_features
from app.ml.labels import make_labels

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Dataset:
    x: FloatArray
    up: BoolArray
    down: BoolArray
    tie: BoolArray
    epochs: npt.NDArray[np.int64]
    horizon: int

    def __len__(self) -> int:
        return len(self.up)

    @property
    def tie_rate(self) -> float:
        return float(self.tie.mean()) if len(self) else 0.0

    def slice(self, start: int, stop: int) -> Dataset:
        return Dataset(
            self.x[start:stop],
            self.up[start:stop],
            self.down[start:stop],
            self.tie[start:stop],
            self.epochs[start:stop],
            self.horizon,
        )

    def stride(self, step: int, offset: int = 0) -> Dataset:
        """Non-overlapping subsample: with step >= horizon no two labels share a future tick."""
        idx = np.arange(offset, len(self), step)
        return Dataset(
            self.x[idx], self.up[idx], self.down[idx], self.tie[idx], self.epochs[idx], self.horizon
        )


def build_dataset(epochs: npt.NDArray[np.int64], prices: FloatArray, horizon: int) -> Dataset:
    """Row i <-> tick index i + WINDOW - 1; only rows with a full future horizon are kept."""
    feats = batch_features(prices)
    up, down, tie = make_labels(prices, horizon)
    n = min(len(feats), len(up) - (WINDOW - 1))
    if n <= 0:
        return Dataset(
            np.empty((0, feats.shape[1] if feats.size else 0)),
            np.zeros(0, bool),
            np.zeros(0, bool),
            np.zeros(0, bool),
            np.zeros(0, np.int64),
            horizon,
        )
    sl = slice(WINDOW - 1, WINDOW - 1 + n)
    return Dataset(feats[:n], up[sl], down[sl], tie[sl], epochs[sl], horizon)


def load_ticks_csv(path: str | Path) -> tuple[npt.NDArray[np.int64], FloatArray]:
    df = pd.read_csv(path)
    if not {"epoch", "quote"} <= set(df.columns):
        raise ValueError(f"{path}: expected columns epoch,quote")
    df = df.drop_duplicates(subset=["epoch", "quote"]).sort_values("epoch", kind="stable")
    return df["epoch"].to_numpy(dtype=np.int64), df["quote"].to_numpy(dtype=np.float64)
