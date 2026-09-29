"""Runtime predictor loaded from a validated artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ml.artifacts import ModelMetadata, load_artifact
from app.ml.features import FEATURE_VERSION
from app.ml.train import TrainedModel
from app.models.schemas import Direction, ModelStatus


class ModelPredictor:
    def __init__(self, model: TrainedModel, meta: ModelMetadata) -> None:
        self.model = model
        self.meta = meta

    @classmethod
    def load(cls, model_dir: str | Path, feature_version: str = FEATURE_VERSION) -> ModelPredictor:
        payload, meta = load_artifact(model_dir, feature_version)
        model = TrainedModel(
            kind=str(payload.get("kind", meta.model_kind)),
            pipeline=payload["pipeline"],
            calibrator=payload["calibrator"],
            tie_rate=float(payload.get("tie_rate", meta.tie_rate)),
        )
        return cls(model, meta)

    @property
    def version(self) -> str:
        return self.meta.model_version

    @property
    def status(self) -> ModelStatus:
        return self.meta.model_status()

    @property
    def horizon(self) -> int:
        return self.meta.label_horizon

    def probabilities(self, features: np.ndarray) -> dict[Direction, float]:
        """Calibrated win probabilities. P(CALL)=P(up); P(PUT)=1-P(up)-P(tie) (ties lose)."""
        p_up = float(self.model.predict_up(features.reshape(1, -1))[0])
        p_put = min(1.0, max(0.0, 1.0 - p_up - self.model.tie_rate))
        return {Direction.CALL: p_up, Direction.PUT: p_put}
