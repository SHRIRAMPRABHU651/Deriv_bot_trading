"""Strategies: EMA maths, ML strategy behaviour, per-symbol isolation, malformed input."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.ml.features import WINDOW
from app.ml.predict import ModelPredictor
from app.models.schemas import Direction, Signal, Tick
from app.strategy.base import make_signal_id
from app.strategy.ema import EmaStrategy
from app.strategy.ml_strategy import MLStrategy
from tests.mocks.stubs import make_stub_model


def tick(symbol: str, i: int, quote: float, received: float = 1.0) -> Tick:
    return Tick(symbol, 1_700_000_000 + i, quote, received, f"id{i}")


async def feed(
    strategy: EmaStrategy | MLStrategy, symbol: str, quotes: list[float]
) -> list[Signal]:
    out: list[Signal] = []
    for i, q in enumerate(quotes):
        sig = await strategy.on_tick(symbol, tick(symbol, i, q))
        if sig is not None:
            out.append(sig)
    return out


async def test_ema_needs_enough_history() -> None:
    s = EmaStrategy(fast=3, slow=10)
    assert await feed(s, "R_100", [100.0 + i for i in range(9)]) == []


async def test_ema_crossover_direction_and_probability_none() -> None:
    s = EmaStrategy(fast=3, slow=10, horizon=5)
    quotes = (
        [100.0] * 12 + [100 + 0.5 * i for i in range(1, 15)] + [107 - 0.8 * i for i in range(1, 25)]
    )
    sigs = await feed(s, "R_100", quotes)
    assert [x.direction for x in sigs][:2] == [Direction.CALL, Direction.PUT]
    assert all(x.probability is None for x in sigs)


async def test_ema_is_deterministic() -> None:
    quotes = [100 + math.sin(i / 3) * 2 for i in range(200)]
    a = await feed(EmaStrategy(3, 10), "R_100", quotes)
    b = await feed(EmaStrategy(3, 10), "R_100", quotes)
    assert [x.signal_id for x in a] == [x.signal_id for x in b]
    assert len(a) > 0


async def test_ema_multi_symbol_state_is_isolated() -> None:
    up = [100.0 + 0.3 * i for i in range(60)]
    down = [200.0 - 0.3 * i for i in range(60)]
    solo = EmaStrategy(3, 10)
    a_alone = await feed(solo, "R_100", up + down)
    mixed = EmaStrategy(3, 10)
    got: list[Signal] = []
    for i in range(120):
        q_a = (up + down)[i]
        q_b = (down + up)[i]
        for sym, q in (("R_100", q_a), ("R_50", q_b)):
            sig = await mixed.on_tick(sym, tick(sym, i, q))
            if sig is not None and sig.symbol == "R_100":
                got.append(sig)
    assert [x.signal_id for x in got] == [x.signal_id for x in a_alone]


async def test_malformed_ticks_are_ignored() -> None:
    s = EmaStrategy(3, 10)
    for bad in (0.0, -5.0, float("nan")):
        assert await s.on_tick("R_100", tick("R_100", 1, bad)) is None
    assert await s.on_tick("R_100", tick("R_50", 1, 100.0)) is None  # symbol mismatch


def test_signal_id_is_deterministic_and_ignores_receive_time() -> None:
    t1 = tick("R_100", 5, 100.0, received=1.0)
    t2 = tick("R_100", 5, 100.0, received=999.0)
    a = make_signal_id("R_100", t1, "1", Direction.CALL, "m1")
    assert a == make_signal_id("R_100", t2, "1", Direction.CALL, "m1")
    assert a != make_signal_id("R_100", t1, "1", Direction.PUT, "m1")
    assert a != make_signal_id("R_100", t1, "2", Direction.CALL, "m1")
    assert a != make_signal_id("R_100", t1, "1", Direction.CALL, "m2")
    assert a != make_signal_id("R_50", t1, "1", Direction.CALL, "m1")


@pytest.fixture
def predictor(tmp_path: Path) -> ModelPredictor:
    make_stub_model(tmp_path / "m", p_up=0.7, horizon=5)
    return ModelPredictor.load(tmp_path / "m")


async def test_ml_strategy_waits_for_window_then_signals_once_per_horizon(
    predictor: ModelPredictor,
) -> None:
    s = MLStrategy(min_probability=0.55, horizon=5)
    s.set_predictor(predictor)
    quotes = [100 + 0.01 * (i % 9) for i in range(WINDOW + 20)]
    sigs = await feed(s, "R_100", quotes)
    assert sigs, "expected signals"
    first = sigs[0]
    assert first.direction is Direction.CALL and first.probability == pytest.approx(0.7)
    assert first.tick_epoch == 1_700_000_000 + WINDOW - 1
    epochs = [x.tick_epoch for x in sigs]
    assert all(
        b - a >= 5 for a, b in zip(epochs, epochs[1:], strict=False)
    )  # no overlapping contracts


async def test_ml_strategy_no_predictor_no_signal() -> None:
    s = MLStrategy()
    assert await feed(s, "R_100", [100 + 0.01 * (i % 9) for i in range(200)]) == []


async def test_ml_strategy_symbol_isolation_and_bounded_memory(predictor: ModelPredictor) -> None:
    s = MLStrategy(min_probability=0.55, horizon=5)
    s.set_predictor(predictor)
    for i in range(WINDOW - 1):
        assert await s.on_tick("R_100", tick("R_100", i, 100 + 0.01 * (i % 5))) is None
    # R_50 has seen nothing: a single R_50 tick must not produce a signal from R_100's history
    assert await s.on_tick("R_50", tick("R_50", 0, 50.0)) is None
    sig = await s.on_tick("R_100", tick("R_100", WINDOW, 100.0))
    assert sig is not None and sig.symbol == "R_100"
    for i in range(5000):
        await s.on_tick("R_100", tick("R_100", 10 + i, 100 + 0.01 * (i % 5)))
    assert len(s._features["R_100"]) == WINDOW  # bounded, not growing


async def test_ml_strategy_malformed_tick_and_reset(predictor: ModelPredictor) -> None:
    s = MLStrategy(min_probability=0.55, horizon=5)
    s.set_predictor(predictor)
    assert await s.on_tick("R_100", tick("R_100", 1, float("nan"))) is None
    assert await s.on_tick("R_100", tick("R_100", 2, -1.0)) is None
    await feed(s, "R_100", [100 + 0.01 * (i % 5) for i in range(WINDOW + 3)])
    s.reset("R_100")
    assert "R_100" not in s._features
