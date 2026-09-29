"""Model training: conventional supervised ML only (LogisticRegression / HistGradientBoosting)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.calibrate import PlattCalibrator

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MODEL_KINDS = ("logistic", "hgb")


def build_pipeline(kind: str = "logistic", seed: int = 7) -> Pipeline:
    if kind == "logistic":
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0))]
        )
    if kind == "hgb":
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed
                    ),
                )
            ]
        )
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


class ProbaModel(Protocol):
    def predict_proba(self, x: FloatArray) -> FloatArray: ...


class Calibrator(Protocol):
    def transform(self, raw: FloatArray) -> FloatArray: ...


@dataclass
class TrainedModel:
    kind: str
    pipeline: ProbaModel
    calibrator: Calibrator
    tie_rate: float

    def predict_up(self, x: FloatArray) -> FloatArray:
        """Calibrated P(price_{t+N} > price_t)."""
        raw = np.asarray(self.pipeline.predict_proba(x)[:, 1], dtype=np.float64)
        return self.calibrator.transform(raw)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pipeline": self.pipeline,
            "calibrator": self.calibrator,
            "tie_rate": self.tie_rate,
        }


def fit_with_calibration(
    x: FloatArray,
    y: BoolArray,
    tie_rate: float,
    *,
    kind: str = "logistic",
    horizon: int,
    calib_fraction: float = 0.25,
    seed: int = 7,
) -> TrainedModel:
    """Time-ordered split:  [ model fit | gap (=horizon) | calibration ].

    The gap stops the calibration slice's labels from overlapping the fit slice's forward
    window; the caller must keep a further gap before any untouched test period.
    """
    n = len(y)
    n_cal = max(int(n * calib_fraction), 1)
    cut = n - n_cal
    fit_end = cut - horizon
    if fit_end < 50 or n_cal < 50:
        raise ValueError("not enough samples to fit and calibrate")
    pipe = build_pipeline(kind, seed)
    if len(np.unique(y[:fit_end])) < 2:
        raise ValueError("training slice contains a single class")
    pipe.fit(x[:fit_end], y[:fit_end].astype(int))
    raw_cal = np.asarray(pipe.predict_proba(x[cut:])[:, 1], dtype=np.float64)
    calibrator = PlattCalibrator().fit(raw_cal, y[cut:])
    return TrainedModel(kind, pipe, calibrator, tie_rate)
