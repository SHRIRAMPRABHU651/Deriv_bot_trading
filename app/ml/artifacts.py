"""Versioned model artifacts: model/model.joblib + model/metadata.json (+ integrity hash)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from app.ml.features import FEATURE_VERSION
from app.ml.train import TrainedModel
from app.models.schemas import ModelStatus


class ArtifactError(Exception):
    pass


class IncompatibleModelError(ArtifactError):
    pass


@dataclass
class ModelMetadata:
    model_version: str
    model_kind: str
    status: str
    feature_version: str
    label_horizon: int
    symbols: list[str]
    training_dates: dict[str, Any]
    training_samples: int
    validation_samples: int
    test_samples: int
    payout_assumption: float
    break_even: float
    edge_margin: float
    observed_win_rate: float | None
    confidence_interval: list[float]
    p_value: float | None
    calibration_metrics: dict[str, Any]
    tie_rate: float
    product: str = "rise_fall"
    spec: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    verdict_reasons: list[str] = field(default_factory=list)
    demo_stats: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    created_at: str = ""
    sha256: str = ""

    def model_status(self) -> ModelStatus:
        return ModelStatus(self.status)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_artifact(model_dir: str | Path, model: TrainedModel, meta: ModelMetadata) -> Path:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(model.to_payload(), model_path)
    meta.sha256 = _sha256(model_path)
    meta.git_commit = meta.git_commit or git_commit()
    meta.created_at = meta.created_at or datetime.now(UTC).isoformat()
    (directory / "metadata.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return directory


def read_metadata(model_dir: str | Path) -> ModelMetadata:
    path = Path(model_dir) / "metadata.json"
    if not path.exists():
        raise ArtifactError(f"no model metadata at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        return ModelMetadata(**raw)
    except TypeError as exc:
        raise IncompatibleModelError(f"metadata schema mismatch: {exc}") from exc


def set_status(model_dir: str | Path, status: ModelStatus, **updates: Any) -> ModelMetadata:
    meta = read_metadata(model_dir)
    meta.status = status.value
    for key, value in updates.items():
        setattr(meta, key, value)
    (Path(model_dir) / "metadata.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return meta


def load_artifact(
    model_dir: str | Path, expected_feature_version: str = FEATURE_VERSION
) -> tuple[dict[str, Any], ModelMetadata]:
    """Load and validate. Rejects: missing files, hash mismatch, incompatible feature version."""
    directory = Path(model_dir)
    meta = read_metadata(directory)
    model_path = directory / "model.joblib"
    if not model_path.exists():
        raise ArtifactError(f"missing {model_path}")
    if meta.sha256 and _sha256(model_path) != meta.sha256:
        raise ArtifactError("model.joblib does not match the recorded sha256 (tampered/corrupt)")
    if meta.feature_version != expected_feature_version:
        raise IncompatibleModelError(
            f"model feature_version {meta.feature_version!r} != "
            f"runtime {expected_feature_version!r}"
        )
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "pipeline" not in payload or "calibrator" not in payload:
        raise ArtifactError("model.joblib has an unexpected structure")
    return payload, meta
