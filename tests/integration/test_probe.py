"""DEMO probe: exercises buy -> settle without a model, and can never run outside DEMO."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.controller import Controller, ModeSwitchError, StartError
from app.models.schemas import Direction, Mode, Signal, Tick
from app.strategy.probe import ProbeStrategy
from tests.conftest import build_controller, make_config, make_settings, wait_until
from tests.mocks.deriv_mock import MockDeriv


def _tick(i: int) -> Tick:
    return Tick("R_100", 1_700_000_000 + i, 100.0 + 0.01 * (i % 7), time.time(), f"t{i}")


def probe_controller(tmp_path: Path, mock: MockDeriv, *, live: bool = False) -> Controller:
    cfg = make_config(tmp_path)  # NO model artifact on purpose
    cfg.probe.enabled = True
    cfg.probe.interval_ticks = 5
    if live:
        return build_controller(
            make_settings(allow_live=True, live_token="live-x", live_account="CR1"), cfg, mock
        )
    return build_controller(make_settings(), cfg, mock)


async def test_probe_trades_without_a_model_in_demo(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = probe_controller(tmp_path, mock)
    assert ctl.predictor is None
    await ctl.start()
    assert ctl.trading_enabled
    mock.auto_settle = (0.05, True)
    for i in range(12):
        await ctl.process_tick(_tick(i))
    await wait_until(lambda: ctl.tracker is not None and ctl.tracker.settled_count >= 1)
    row = ctl.repos.db.query("SELECT * FROM orders")[0]
    assert row["direction"] in {"CALL", "PUT"}
    assert ctl.status()["probe"] is True
    await ctl.shutdown()


async def test_probe_alternates_direction_and_paces_signals() -> None:
    s = ProbeStrategy(interval_ticks=3, horizon=5, product="rise_fall")
    got: list[Signal] = []
    for i in range(1, 10):
        sig = await s.on_tick("R_100", _tick(i))
        if sig:
            got.append(sig)
    assert [g.direction for g in got] == [Direction.CALL, Direction.PUT, Direction.CALL]
    assert all(g.probability is None and g.model_version is None for g in got)


async def test_probe_signals_are_rejected_when_probe_is_off(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = probe_controller(tmp_path, mock)
    await ctl.start()
    sig = await ProbeStrategy(1, 5, "rise_fall").on_tick("R_100", _tick(1))
    assert sig is not None
    ctl.config.probe.enabled = False  # disabled at runtime => risk must refuse
    assert ctl.risk.evaluate(sig).rule == "probe_not_allowed"
    await ctl.shutdown()


async def test_probe_can_never_go_live(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = probe_controller(tmp_path, mock, live=True)
    with pytest.raises(ModeSwitchError, match="probe"):
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=True, nonce="x")
    ctl.account.mode = Mode.LIVE  # even if mode were somehow LIVE, start must refuse
    with pytest.raises(StartError, match="probe"):
        await ctl.start()
    await ctl.shutdown()


async def test_no_probe_no_model_means_no_trading(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = build_controller(make_settings(), make_config(tmp_path), mock)
    await ctl.start()
    assert not ctl.trading_enabled and ctl.status()["probe"] is False
    await ctl.shutdown()


async def test_client_adapts_when_deriv_rejects_the_legacy_symbol_field(
    tmp_path: Path, mock: MockDeriv
) -> None:
    mock.reject_legacy_symbol = True  # what the real API answered
    ctl = probe_controller(tmp_path, mock)
    await ctl.start()
    mock.auto_settle = (0.05, True)
    for i in range(12):
        await ctl.process_tick(_tick(i))
    await wait_until(lambda: ctl.tracker is not None and ctl.tracker.settled_count >= 1)
    sent = [r for r in mock.requests if "proposal" in r]
    assert any("underlying_symbol" in r for r in sent)
    await ctl.shutdown()


async def test_trade_type_can_be_switched_and_probe_trades_accumulators(
    tmp_path: Path, mock: MockDeriv
) -> None:
    from app.controller import ControllerError

    ctl = probe_controller(tmp_path, mock)
    assert ctl.set_product("accumulator") == "accumulator"
    assert ctl.config.trading.product.product.value == "accumulator"
    with pytest.raises(ControllerError, match="unknown"):
        ctl.set_product("nonsense")
    await ctl.start()
    with pytest.raises(ControllerError, match="stop the bot"):
        ctl.set_product("multiplier")
    mock.auto_settle = (0.05, True)
    for i in range(12):
        await ctl.process_tick(_tick(i))
    await wait_until(lambda: ctl.tracker is not None and ctl.tracker.settled_count >= 1)
    assert ctl.repos.db.query("SELECT * FROM orders")[0]["product"] == "accumulator"
    await ctl.shutdown()
