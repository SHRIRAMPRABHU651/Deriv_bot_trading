"""Walk-forward validation (TRAIN | GAP | TEST, rolling forward) with a shuffled-label control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from app.ml.calibrate import Metrics, calibration_table, classification_metrics
from app.ml.dataset import Dataset
from app.ml.statistics import EdgeTest, break_even_from_ratio, edge_test
from app.ml.train import MODEL_KINDS, TrainedModel, fit_heads

NO_EDGE = "NO EVIDENCE OF EDGE"
EDGE = "EDGE EVIDENCE"


@dataclass(frozen=True)
class Fold:
    train: tuple[int, int]
    gap: tuple[int, int]
    test: tuple[int, int]


def walk_forward_splits(
    n: int, *, initial_train: int, gap: int, test_size: int, horizon: int, step: int | None = None
) -> list[Fold]:
    """Expanding-window folds:  train[0:a]  gap[a:a+gap]  test[a+gap:a+gap+test_size].

    `gap` must be >= horizon, otherwise the last training labels (which look `horizon` ticks
    ahead) would overlap the first test features/labels.
    """
    if gap < horizon:
        raise ValueError(f"gap ({gap}) must be >= label horizon ({horizon}) to prevent leakage")
    if initial_train < 1 or test_size < 1:
        raise ValueError("initial_train and test_size must be positive")
    stride = step or test_size
    folds: list[Fold] = []
    a = initial_train
    while a + gap + test_size <= n:
        folds.append(Fold((0, a), (a, a + gap), (a + gap, a + gap + test_size)))
        a += stride
    return folds


@dataclass
class FoldResult:
    fold: Fold
    metrics: Metrics
    n_trades: int
    wins: int


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    pooled_metrics: Metrics | None
    calibration: list[dict[str, float]]
    trades: int
    wins: int
    edge: EdgeTest
    payout_ratio: float
    edge_margin: float
    tie_rate: float
    p_up: npt.NDArray[np.float64] = field(repr=False, default_factory=lambda: np.zeros(0))
    y: npt.NDArray[np.bool_] = field(repr=False, default_factory=lambda: np.zeros(0, dtype=bool))


def _select_and_score(
    p_call: npt.NDArray[np.float64],
    p_put: npt.NDArray[np.float64],
    ds: Dataset,
    threshold: float,
) -> tuple[int, int]:
    call = (p_call >= threshold) & (p_call >= p_put)
    put = (p_put >= threshold) & (p_put > p_call)
    trades = int(call.sum() + put.sum())
    wins = int(ds.up[call].sum() + ds.down[put].sum())
    return trades, wins


def run_walk_forward(
    ds: Dataset,
    *,
    kind: str,
    initial_train: int,
    gap: int,
    test_size: int,
    payout_ratio: float = 1.95,
    terms: tuple[float, float] | None = None,
    edge_margin: float = 0.03,
    alpha: float = 0.05,
    shuffle_labels: bool = False,
    seed: int = 7,
    step: int | None = None,
) -> WalkForwardResult:
    """Evaluate on non-overlapping test samples (stride = horizon). Trades are taken only when
    the calibrated probability clears break-even + margin, with break-even = L / (W + L) from
    `terms` = (win, loss) per unit stake, or 1 / payout_ratio for a plain Rise/Fall.
    NOTE: these terms are a research assumption; the live bot uses the ACTUAL proposal."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    rng = np.random.default_rng(seed)
    be = terms[1] / (terms[0] + terms[1]) if terms else break_even_from_ratio(payout_ratio)
    folds = walk_forward_splits(
        len(ds),
        initial_train=initial_train,
        gap=gap,
        test_size=test_size,
        horizon=ds.horizon,
        step=step,
    )
    results: list[FoldResult] = []
    all_p: list[npt.NDArray[np.float64]] = []
    all_y: list[npt.NDArray[np.bool_]] = []
    total_trades = total_wins = 0
    for fold in folds:
        train = ds.slice(*fold.train)
        if shuffle_labels:
            train = replace(train, up=rng.permutation(train.up), down=rng.permutation(train.down))
        try:
            model: TrainedModel = fit_heads(train, kind=kind, seed=seed)
        except ValueError:
            continue
        test = ds.slice(*fold.test).stride(ds.horizon)
        if len(test) < 10:
            continue
        p_up, p_down = model.direction_probs(test.x)
        trades, wins = _select_and_score(p_up, p_down, test, be + edge_margin)
        results.append(FoldResult(fold, classification_metrics(test.up, p_up), trades, wins))
        all_p.append(p_up)
        all_y.append(test.up)
        total_trades += trades
        total_wins += wins
    p_all = np.concatenate(all_p) if all_p else np.zeros(0)
    y_all = np.concatenate(all_y) if all_y else np.zeros(0, dtype=bool)
    pooled = classification_metrics(y_all, p_all) if len(y_all) else None
    return WalkForwardResult(
        folds=results,
        pooled_metrics=pooled,
        calibration=calibration_table(y_all, p_all) if len(y_all) else [],
        trades=total_trades,
        wins=total_wins,
        edge=edge_test(total_wins, total_trades, be, alpha),
        payout_ratio=payout_ratio,
        edge_margin=edge_margin,
        tie_rate=ds.tie_rate,
        p_up=p_all,
        y=y_all,
    )


@dataclass
class Verdict:
    label: str
    reasons: list[str]
    real_auc: float | None
    shuffled_auc_p95: float | None
    shuffled_aucs: list[float]

    @property
    def has_edge(self) -> bool:
        return self.label == EDGE


def shuffled_label_control(
    ds: Dataset,
    real: WalkForwardResult,
    *,
    kind: str,
    initial_train: int,
    gap: int,
    test_size: int,
    n_shuffles: int = 10,
    alpha: float = 0.05,
    min_trades: int = 300,
    n_comparisons: int = 1,
    payout_ratio: float = 1.95,
    terms: tuple[float, float] | None = None,
    edge_margin: float = 0.03,
    seed: int = 1000,
) -> Verdict:
    """Repeat the identical pipeline with shuffled training labels. If the real model is not
    clearly better than that control (and not statistically above break-even) -> NO EVIDENCE."""
    aucs: list[float] = []
    for i in range(n_shuffles):
        res = run_walk_forward(
            ds,
            kind=kind,
            initial_train=initial_train,
            gap=gap,
            test_size=test_size,
            payout_ratio=payout_ratio,
            terms=terms,
            edge_margin=edge_margin,
            alpha=alpha,
            shuffle_labels=True,
            seed=seed + i,
        )
        if res.pooled_metrics is not None and res.pooled_metrics.roc_auc is not None:
            aucs.append(res.pooled_metrics.roc_auc)
    real_auc = real.pooled_metrics.roc_auc if real.pooled_metrics else None
    p95 = float(np.percentile(aucs, 95)) if aucs else None
    reasons: list[str] = []
    if real.trades < min_trades:
        reasons.append(f"only {real.trades} out-of-sample trades (< {min_trades})")
    if real.edge.p_value >= alpha / max(n_comparisons, 1):
        reasons.append(
            f"win rate {real.edge.win_rate:.4f} not significantly above break-even "
            f"{real.edge.break_even:.4f} (p={real.edge.p_value:.4f})"
        )
    if real_auc is None or p95 is None or real_auc <= p95:
        reasons.append("real model AUC does not beat the shuffled-label control")
    return Verdict(NO_EDGE if reasons else EDGE, reasons, real_auc, p95, aucs)
