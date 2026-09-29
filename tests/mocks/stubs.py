"""Picklable stand-ins used to build model artifacts with predictable probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.ml.artifacts import ModelMetadata, save_artifact
from app.ml.features import FEATURE_VERSION
from app.ml.train import TrainedModel
from app.models.schemas import ModelStatus


class StubPipeline:
    def __init__(self, p_up: float) -> None:
        self.p_up = p_up

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = len(x)
        return np.tile(np.array([[1 - self.p_up, self.p_up]]), (n, 1))


class StubCalibrator:
    fitted = True

    def transform(self, raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return raw


def make_stub_model(
    model_dir: Path,
    *,
    p_up: float = 0.7,
    status: ModelStatus = ModelStatus.DEMO_VALIDATING,
    version: str = "stub-1",
    feature_version: str = FEATURE_VERSION,
    horizon: int = 5,
    tie_rate: float = 0.0,
) -> Path:
    model = TrainedModel(
        kind="stub",
        pipeline=StubPipeline(p_up),
        calibrator=StubCalibrator(),
        tie_rate=tie_rate,
    )
    meta = ModelMetadata(
        model_version=version,
        model_kind="stub",
        status=status.value,
        feature_version=feature_version,
        label_horizon=horizon,
        symbols=["R_100"],
        training_dates={},
        training_samples=0,
        validation_samples=0,
        test_samples=0,
        payout_assumption=1.95,
        break_even=1 / 1.95,
        edge_margin=0.03,
        observed_win_rate=None,
        confidence_interval=[0.0, 1.0],
        p_value=None,
        calibration_metrics={},
        tie_rate=tie_rate,
    )
    return save_artifact(model_dir, model, meta)
