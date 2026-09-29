"""Time-aware probability calibration (Platt scaling fit on a later, held-out calibration slice)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

FloatArray = npt.NDArray[np.float64]


def _logit(p: FloatArray) -> FloatArray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw probabilities to calibrated ones. Fit ONLY on data after the training slice."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")
        self.fitted = False

    def fit(self, raw: FloatArray, y: npt.NDArray[np.bool_]) -> PlattCalibrator:
        if len(np.unique(y)) < 2:
            raise ValueError("calibration slice needs both classes")
        self._lr.fit(_logit(raw), y.astype(int))
        self.fitted = True
        return self

    def transform(self, raw: FloatArray) -> FloatArray:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return np.asarray(self._lr.predict_proba(_logit(raw))[:, 1], dtype=np.float64)


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    roc_auc: float | None
    brier: float
    base_rate: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "base_rate": self.base_rate,
        }


def classification_metrics(y: npt.NDArray[np.bool_], p: FloatArray) -> Metrics:
    pred = p >= 0.5
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return Metrics(
        n=len(y),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        roc_auc=auc,
        brier=float(brier_score_loss(y, p)),
        base_rate=float(np.mean(y)),
    )


def calibration_table(
    y: npt.NDArray[np.bool_], p: FloatArray, bins: int = 10
) -> list[dict[str, float]]:
    """Calibration curve: mean predicted probability vs actual win rate per bin."""
    if len(np.unique(y)) < 2:
        return []
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return [
        {"predicted": float(m), "actual": float(f)} for f, m in zip(frac, mean_pred, strict=True)
    ]
