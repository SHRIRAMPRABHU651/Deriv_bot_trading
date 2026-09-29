"""ML pipeline: features, labels, leakage, walk-forward, calibration, shuffled control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pytest

from app.ml.artifacts import (
    ArtifactError,
    IncompatibleModelError,
    load_artifact,
    read_metadata,
    save_artifact,
    set_status,
)
from app.ml.calibrate import PlattCalibrator, calibration_table, classification_metrics
from app.ml.dataset import build_dataset, load_ticks_csv
from app.ml.features import FEATURE_NAMES, WINDOW, SymbolFeatureState, batch_features
from app.ml.labels import make_labels
from app.ml.predict import ModelPredictor
from app.ml.train import fit_with_calibration
from app.ml.validate import (
    EDGE,
    NO_EDGE,
    run_walk_forward,
    shuffled_label_control,
    walk_forward_splits,
)
from app.models.schemas import Direction, ModelStatus
from tests.mocks.stubs import make_stub_model
from tests.unit.synth import momentum_walk, random_walk


# ------------------------------------------------------------------ features
def test_feature_shape_and_determinism() -> None:
    _, p = random_walk(400)
    a, b = batch_features(p), batch_features(p)
    assert a.shape == (400 - WINDOW + 1, len(FEATURE_NAMES))
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_streaming_features_equal_batch_features() -> None:
    _, p = random_walk(300)
    batch = batch_features(p)
    st = SymbolFeatureState()
    rows = [f for x in p if (f := st.update(float(x))) is not None]
    assert len(rows) == len(batch)
    assert np.allclose(np.vstack(rows), batch)


def test_features_never_use_future_prices() -> None:
    _, p = random_walk(300)
    full = batch_features(p)
    t = 200
    altered = p.copy()
    altered[t + 1 :] = altered[t + 1 :] * 3 + 17  # wreck the future
    changed = batch_features(altered)
    idx = t - (WINDOW - 1)
    assert np.allclose(full[: idx + 1], changed[: idx + 1])  # rows up to t are unchanged
    assert not np.allclose(full[idx + 1 :], changed[idx + 1 :])


def test_feature_window_is_bounded_and_rejects_bad_prices() -> None:
    st = SymbolFeatureState()
    for i in range(1000):
        st.update(100.0 + (i % 7) * 0.01)
    assert len(st) == WINDOW
    for bad in (float("nan"), 0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            st.update(bad)


def test_insufficient_history_returns_none() -> None:
    st = SymbolFeatureState()
    assert all(st.update(100.0 + i * 0.01) is None for i in range(WINDOW - 1))
    assert st.update(100.5) is not None


def test_rsi_and_streak_edge_cases() -> None:
    up = np.linspace(100, 110, WINDOW + 5)
    f = batch_features(up)[-1]
    names = list(FEATURE_NAMES)
    assert f[names.index("rsi")] == 100.0
    assert f[names.index("streak")] > 10
    flat = np.full(WINDOW + 5, 100.0)
    g = batch_features(flat)[-1]
    assert g[names.index("rsi")] == 50.0 and g[names.index("streak")] == 0


# ------------------------------------------------------------------ labels / dataset
def test_labels_look_forward_n_ticks() -> None:
    prices = np.array([1.0, 2.0, 2.0, 1.0, 5.0, 4.0])
    up, down, tie = make_labels(prices, 2)
    # entries [1,2,2,1] vs exits [2,1,5,4]
    assert list(up) == [True, False, True, True]
    assert list(down) == [False, True, False, False]
    assert list(tie) == [False, False, False, False]
    _, _, tie2 = make_labels(np.array([1.0, 1.0, 1.0]), 1)
    assert list(tie2) == [True, True]
    with pytest.raises(ValueError):
        make_labels(prices, 0)
    assert len(make_labels(prices, 10)[0]) == 0


def test_dataset_alignment_features_use_past_labels_use_future() -> None:
    ep, p = random_walk(500)
    ds = build_dataset(ep, p, horizon=7)
    assert len(ds) == 500 - (WINDOW - 1) - 7
    for i in (0, 10, 100, len(ds) - 1):
        t = i + WINDOW - 1
        assert ds.up[i] == (p[t + 7] > p[t])
        assert ds.epochs[i] == ep[t]
        assert np.allclose(ds.x[i], batch_features(p[: t + 1])[-1])


def test_non_overlapping_stride() -> None:
    ep, p = random_walk(500)
    ds = build_dataset(ep, p, horizon=5).stride(5)
    assert np.all(np.diff(ds.epochs) >= 5)


def test_load_ticks_csv_sorts_and_dedups(tmp_path: Path) -> None:
    f = tmp_path / "t.csv"
    f.write_text("epoch,quote\n3,1.3\n1,1.1\n2,1.2\n2,1.2\n")
    ep, q = load_ticks_csv(f)
    assert list(ep) == [1, 2, 3] and list(q) == [1.1, 1.2, 1.3]
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_ticks_csv(tmp_path / "bad.csv")


# ------------------------------------------------------------------ walk-forward
def test_walk_forward_splits_train_gap_test_roll_forward() -> None:
    folds = walk_forward_splits(1000, initial_train=400, gap=10, test_size=100, horizon=5)
    assert [f.train for f in folds] == [(0, 400), (0, 500), (0, 600), (0, 700), (0, 800)]
    for f in folds:
        assert f.gap == (f.train[1], f.train[1] + 10)
        assert f.test[0] == f.gap[1]  # test starts right after the gap
        assert f.test[1] - f.test[0] == 100
        assert f.train[1] <= f.test[0] - 10  # never adjacent
    assert folds[-1].test[1] <= 1000
    assert all(b.test[0] == a.test[1] for a, b in zip(folds, folds[1:], strict=False))


def test_walk_forward_gap_must_cover_the_label_horizon() -> None:
    with pytest.raises(ValueError, match="gap"):
        walk_forward_splits(1000, initial_train=400, gap=3, test_size=100, horizon=5)


# ------------------------------------------------------------------ calibration
def test_platt_calibrator_improves_brier_and_stays_in_bounds() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = rng.uniform(size=4000) < true_p
    overconfident = np.clip(0.5 + 3.0 * (true_p - 0.5), 0.01, 0.99)
    cal = PlattCalibrator().fit(overconfident[:2000], y[:2000])
    out = cal.transform(overconfident[2000:])
    assert (out >= 0).all() and (out <= 1).all()
    before = classification_metrics(y[2000:], overconfident[2000:]).brier
    after = classification_metrics(y[2000:], out).brier
    assert after < before
    with pytest.raises(ValueError):
        PlattCalibrator().fit(overconfident[:10], np.ones(10, dtype=bool))
    with pytest.raises(RuntimeError):
        PlattCalibrator().transform(overconfident[:3])
    table = calibration_table(y[2000:], out, bins=5)
    assert 3 <= len(table) <= 5 and all(0 <= r["actual"] <= 1 for r in table)


def test_calibration_slice_is_later_than_and_gapped_from_the_fit_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.train as train_mod

    ep, p = momentum_walk(4000)
    ds = build_dataset(ep, p, 5)
    seen: dict[str, int] = {}
    original = train_mod.build_pipeline

    class Spy:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def fit(self, x: Any, y: Any) -> Any:
            seen["n_fit"] = len(x)
            return self.inner.fit(x, y)

        def predict_proba(self, x: Any) -> Any:
            return self.inner.predict_proba(x)

    monkeypatch.setattr(
        train_mod, "build_pipeline", lambda kind="logistic", seed=7: Spy(original(kind, seed))
    )
    model = fit_with_calibration(ds.x, ds.up, ds.tie_rate, horizon=5, calib_fraction=0.25)
    n = len(ds)
    n_cal = int(n * 0.25)
    assert seen["n_fit"] == n - n_cal - 5  # fit slice ends `horizon` before calibration begins
    probs = model.predict_up(ds.x[-100:])
    assert ((probs >= 0) & (probs <= 1)).all()


# ------------------------------------------------------------------ pipeline + shuffled control
class WFArgs(TypedDict):
    initial_train: int
    gap: int
    test_size: int


WF: WFArgs = {"initial_train": 5000, "gap": 10, "test_size": 2000}


def test_pipeline_finds_planted_signal_and_control_agrees() -> None:
    ep, p = momentum_walk(14000)
    ds = build_dataset(ep, p, 5)
    real = run_walk_forward(ds, kind="logistic", **WF)
    assert real.pooled_metrics is not None and real.pooled_metrics.roc_auc is not None
    assert real.pooled_metrics.roc_auc > 0.65
    assert real.trades > 100 and real.edge.significant
    verdict = shuffled_label_control(ds, real, kind="logistic", n_shuffles=3, min_trades=100, **WF)
    assert verdict.label == EDGE and verdict.has_edge, verdict.reasons
    assert verdict.real_auc is not None and verdict.shuffled_auc_p95 is not None
    assert verdict.real_auc > verdict.shuffled_auc_p95


def test_random_walk_yields_no_evidence_of_edge() -> None:
    ep, p = random_walk(14000, seed=11)
    ds = build_dataset(ep, p, 5)
    real = run_walk_forward(ds, kind="logistic", **WF)
    verdict = shuffled_label_control(ds, real, kind="logistic", n_shuffles=3, min_trades=100, **WF)
    assert verdict.label == NO_EDGE == "NO EVIDENCE OF EDGE"
    assert verdict.reasons and not verdict.has_edge


def test_hgb_pipeline_runs() -> None:
    ep, p = momentum_walk(9000)
    ds = build_dataset(ep, p, 5)
    res = run_walk_forward(ds, kind="hgb", initial_train=4000, gap=10, test_size=2000)
    assert res.pooled_metrics is not None and res.pooled_metrics.roc_auc is not None
    with pytest.raises(ValueError):
        run_walk_forward(ds, kind="nope", **WF)


# ------------------------------------------------------------------ artifacts
def test_artifact_roundtrip_and_probabilities(tmp_path: Path) -> None:
    make_stub_model(tmp_path / "m", p_up=0.62, tie_rate=0.01)
    pred = ModelPredictor.load(tmp_path / "m")
    assert pred.version == "stub-1" and pred.status is ModelStatus.DEMO_VALIDATING
    probs = pred.probabilities(np.zeros(len(FEATURE_NAMES)))
    assert probs[Direction.CALL] == pytest.approx(0.62)
    assert probs[Direction.PUT] == pytest.approx(1 - 0.62 - 0.01)
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_artifact_rejects_incompatible_feature_version(tmp_path: Path) -> None:
    make_stub_model(tmp_path / "m", feature_version="v0")
    with pytest.raises(IncompatibleModelError):
        load_artifact(tmp_path / "m", "v1")


def test_artifact_missing_or_tampered_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path / "nothing")
    make_stub_model(tmp_path / "m")
    (tmp_path / "m" / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="sha256"):
        load_artifact(tmp_path / "m")
    (tmp_path / "m" / "model.joblib").unlink()
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path / "m")


def test_artifact_metadata_fields_and_status_update(tmp_path: Path) -> None:
    make_stub_model(tmp_path / "m")
    meta = read_metadata(tmp_path / "m")
    for field in (
        "model_version",
        "feature_version",
        "label_horizon",
        "symbols",
        "training_samples",
        "validation_samples",
        "test_samples",
        "payout_assumption",
        "break_even",
        "observed_win_rate",
        "confidence_interval",
        "p_value",
        "calibration_metrics",
        "git_commit",
        "training_dates",
        "created_at",
        "sha256",
    ):
        assert hasattr(meta, field)
    updated = set_status(tmp_path / "m", ModelStatus.PROMOTABLE)
    assert updated.status == "PROMOTABLE"
    assert ModelPredictor.load(tmp_path / "m").status is ModelStatus.PROMOTABLE


def test_real_trained_model_can_be_saved_and_loaded(tmp_path: Path) -> None:
    from app.ml.artifacts import ModelMetadata
    from app.ml.features import FEATURE_VERSION

    ep, p = momentum_walk(6000)
    ds = build_dataset(ep, p, 5)
    model = fit_with_calibration(ds.x, ds.up, ds.tie_rate, horizon=5)
    meta = ModelMetadata(
        model_version="real-1",
        model_kind="logistic",
        status=ModelStatus.BACKTESTED.value,
        feature_version=FEATURE_VERSION,
        label_horizon=5,
        symbols=["R_100"],
        training_dates={},
        training_samples=len(ds),
        validation_samples=0,
        test_samples=0,
        payout_assumption=1.95,
        break_even=1 / 1.95,
        edge_margin=0.03,
        observed_win_rate=None,
        confidence_interval=[0, 1],
        p_value=None,
        calibration_metrics={},
        tie_rate=ds.tie_rate,
    )
    save_artifact(tmp_path / "m", model, meta)
    pred = ModelPredictor.load(tmp_path / "m")
    probs = pred.probabilities(ds.x[-1])
    assert 0 <= probs[Direction.CALL] <= 1 and 0 <= probs[Direction.PUT] <= 1
