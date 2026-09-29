"""ML strategy: per-symbol bounded feature state -> calibrated probability -> signal."""

from __future__ import annotations

import logging
import time

from app.ml.features import SymbolFeatureState
from app.ml.predict import ModelPredictor
from app.models.schemas import Direction, Signal, Tick
from app.products import assumed_break_even
from app.strategy.base import make_signal_id

log = logging.getLogger("derivbot.strategy")


class MLStrategy:
    name = "ml"

    def __init__(
        self,
        version: str = "1",
        min_probability: float | None = None,
        horizon: int = 10,
        margin: float = 0.03,
    ) -> None:
        self.version = version
        # Coarse pre-filter only (avoids a proposal per tick); the risk gate uses the real terms.
        self._fixed_min_p = min_probability
        self._margin = margin
        self._min_p = 0.5 if min_probability is None else min_probability
        self._horizon = horizon
        self._features: dict[str, SymbolFeatureState] = {}  # independent state per symbol
        self._last_signal_tick: dict[str, int] = {}
        self._predictor: ModelPredictor | None = None
        self.last_probability: dict[str, dict[Direction, float]] = {}

    def reset(self, symbol: str) -> None:
        """Forget a symbol's rolling state (used after ticks were dropped => window has a gap)."""
        self._features.pop(symbol, None)
        self._last_signal_tick.pop(symbol, None)

    def set_predictor(self, predictor: ModelPredictor | None) -> None:
        self._predictor = predictor
        if predictor is not None:
            self._horizon = predictor.horizon
            if self._fixed_min_p is None:
                self._min_p = max(0.5, assumed_break_even(predictor.spec) + self._margin - 0.02)

    @property
    def predictor(self) -> ModelPredictor | None:
        return self._predictor

    async def on_tick(self, symbol: str, tick: Tick) -> Signal | None:
        if tick.symbol != symbol:
            return None
        state = self._features.setdefault(symbol, SymbolFeatureState())
        try:
            feats = state.update(tick.quote)
        except ValueError:
            log.warning("malformed_tick", extra={"event": "malformed_tick", "symbol": symbol})
            return None
        predictor = self._predictor
        if feats is None or predictor is None:
            return None
        # One signal per horizon per symbol: contracts must not overlap on the same symbol.
        last = self._last_signal_tick.get(symbol)
        if last is not None and state.ticks_seen - last < self._horizon:
            return None
        probs = predictor.probabilities(feats)
        self.last_probability[symbol] = dict(probs)
        direction = max(probs, key=lambda d: probs[d])
        p = probs[direction]
        if p < self._min_p:
            return None
        self._last_signal_tick[symbol] = state.ticks_seen
        return Signal(
            signal_id=make_signal_id(symbol, tick, self.version, direction, predictor.version),
            symbol=symbol,
            direction=direction,
            tick_epoch=tick.epoch,
            strategy=self.name,
            strategy_version=self.version,
            model_version=predictor.version,
            probability=p,
            horizon_ticks=self._horizon,
            tick_received_at=tick.received_at,
            created_at=time.time(),
            features_version=predictor.meta.feature_version,
            product=predictor.spec.product.value,
            entry_price=tick.quote,
        )


__all__ = ["Direction", "MLStrategy"]
