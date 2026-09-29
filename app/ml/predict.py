"""Runtime predictor loaded from a validated artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ml.artifacts import ModelMetadata, load_artifact
from app.ml.features import FEATURE_VERSION
from app.ml.train import TrainedModel
from app.models.schemas import Direction, ModelStatus
from app.products import Product, ProductSpec


class ModelPredictor:
    def __init__(self, model: TrainedModel, meta: ModelMetadata) -> None:
        self.model = model
        self.meta = meta
        if meta.spec:
            self.spec = ProductSpec.model_validate(meta.spec)
        else:  # legacy artifacts: Rise/Fall over `label_horizon` ticks
            self.spec = ProductSpec(product=Product.RISE_FALL, horizon_ticks=meta.label_horizon)

    @classmethod
    def load(cls, model_dir: str | Path, feature_version: str = FEATURE_VERSION) -> ModelPredictor:
        payload, meta = load_artifact(model_dir, feature_version)
        return cls(TrainedModel.from_payload(payload, meta.model_kind), meta)

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
        """Calibrated probabilities that the trade reaches its profit target.

        Rise/Fall: P(CALL)=P(up), P(PUT)=1-P(up)-P(tie). Multiplier/Turbo/Vanilla: separate heads.
        Accumulator: one non-directional survival probability keyed NEUTRAL."""
        p_bull, p_bear = self.model.direction_probs(features.reshape(1, -1))
        if self.spec.product is Product.ACCUMULATOR:
            return {Direction.NEUTRAL: float(p_bull[0])}
        return {Direction.CALL: float(p_bull[0]), Direction.PUT: float(p_bear[0])}
