"""DEMO -> LIVE gating. LIVE must stay impossible unless EVERY condition holds."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.config import AppConfig, Settings
from app.controller import Controller, ControllerError, ModeSwitchError
from app.models.schemas import Mode
from tests.conftest import build_controller, make_config, make_settings
from tests.mocks.deriv_mock import MockDeriv, make_http
from tests.mocks.stubs import make_stub_model


def live_settings() -> Settings:
    return make_settings(allow_live=True, live_token="live-token-zzz", live_account="LIVE1")


def critical_denials(ctl: Controller) -> int:
    return sum(
        1
        for r in ctl.repos.recent_risk_events(100)
        if r["rule"] == "live_switch_denied" and r["severity"] == "CRITICAL"
    )


async def ctl_for(
    tmp_path: Path,
    mock: MockDeriv,
    settings: Settings,
    *,
    live_type: str = "real",
    live_id: str = "LIVE1",
) -> Controller:
    cfg: AppConfig = make_config(tmp_path)
    make_stub_model(Path(cfg.app.model_dir), p_up=0.7)
    http, _ = make_http(mock, account_type=live_type, account_id=live_id)
    return build_controller(settings, cfg, mock, http=http)


async def test_live_impossible_when_allow_live_false(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await ctl_for(
        tmp_path, mock, make_settings(allow_live=False, live_token="t", live_account="LIVE1")
    )
    with pytest.raises(ModeSwitchError, match="ALLOW_LIVE"):
        await ctl.prepare_live_switch()
    with pytest.raises(ModeSwitchError, match="ALLOW_LIVE"):
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=True, nonce="x")
    assert ctl.mode is Mode.DEMO and critical_denials(ctl) == 2
    await ctl.shutdown()


@pytest.mark.parametrize(
    ("token", "account", "needle"), [("", "LIVE1", "TOKEN"), ("t", "", "ACCOUNT_ID")]
)
async def test_live_needs_credentials(
    tmp_path: Path, mock: MockDeriv, token: str, account: str, needle: str
) -> None:
    ctl = await ctl_for(
        tmp_path, mock, make_settings(allow_live=True, live_token=token, live_account=account)
    )
    with pytest.raises(ModeSwitchError, match=needle):
        await ctl.prepare_live_switch()
    assert ctl.mode is Mode.DEMO
    await ctl.shutdown()


async def test_live_refused_while_bot_running(tmp_path: Path, mock: MockDeriv) -> None:
    settings = live_settings()
    cfg = make_config(tmp_path)
    make_stub_model(Path(cfg.app.model_dir))
    ctl = build_controller(settings, cfg, mock)
    await ctl.start()
    with pytest.raises(ModeSwitchError, match="stopped"):
        await ctl.prepare_live_switch()
    assert ctl.mode is Mode.DEMO
    await ctl.shutdown()


async def test_live_refused_with_open_trade(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings())
    ctl.risk.update_balance(Decimal("1000"), "USD")
    ctl.account.verified_type = "demo"
    sig_id = "s1"
    from app.models.schemas import Direction, Signal

    sig = Signal(sig_id, "R_100", Direction.CALL, 1, "ml", "1", "stub-1", 0.7, 5, 0.0, 0.0)
    ctl.repos.insert_order(
        order_id="o1",
        signal=sig,
        mode=Mode.DEMO,
        stake=Decimal("1"),
        break_even=None,
        ts=ctl.clock.now(),
    )
    with pytest.raises(ModeSwitchError, match="open trades"):
        await ctl.prepare_live_switch()
    await ctl.shutdown()


async def test_live_refused_with_kill_switch(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings())
    ctl.kill("test")
    with pytest.raises(ModeSwitchError, match="kill"):
        await ctl.prepare_live_switch()
    await ctl.shutdown()


async def test_live_refused_when_api_says_account_is_not_real(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings(), live_type="demo")
    with pytest.raises(ModeSwitchError, match="verif"):
        await ctl.prepare_live_switch()
    assert ctl.mode is Mode.DEMO
    await ctl.shutdown()


async def test_wrong_confirmation_missing_nonce_and_missing_second_confirm(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings())
    for typed in ("live", "LIVE ", "", "Live"):
        info = await ctl.prepare_live_switch()
        with pytest.raises(ModeSwitchError, match="exactly LIVE"):
            await ctl.switch_mode(Mode.LIVE, typed=typed, second_confirm=True, nonce=info["nonce"])
    with pytest.raises(ModeSwitchError, match="expired|missing"):
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=True, nonce="forged")
    info = await ctl.prepare_live_switch()
    with pytest.raises(ModeSwitchError, match="second"):
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=False, nonce=info["nonce"])
    assert ctl.mode is Mode.DEMO
    assert critical_denials(ctl) >= 6
    await ctl.shutdown()


async def test_confirmation_summary_shows_everything_required(
    tmp_path: Path, mock: MockDeriv
) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings())
    info = await ctl.prepare_live_switch()
    for key in (
        "account_id",
        "account_type",
        "balance",
        "proposed_stake",
        "risk_limits",
        "drawdown",
        "daily_pnl",
        "weekly_pnl",
        "open_exposure",
        "model_status",
        "edge_gate",
        "must_type",
    ):
        assert key in info
    assert info["account_type"] == "real" and info["must_type"] == "LIVE"
    assert info["proposed_stake"] == "5.00"  # 0.5% of 1000
    await ctl.shutdown()


async def test_successful_switch_is_memory_only_and_nonce_single_use(
    tmp_path: Path, mock: MockDeriv
) -> None:
    settings = live_settings()
    ctl = await ctl_for(tmp_path, mock, settings)
    info = await ctl.prepare_live_switch()
    assert (
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=True, nonce=info["nonce"])
        is Mode.LIVE
    )
    assert (
        ctl.mode is Mode.LIVE and ctl.account.verified_type is None
    )  # must be re-verified on start
    with pytest.raises(ModeSwitchError):  # nonce cannot be replayed
        await ctl.switch_mode(Mode.LIVE, typed="LIVE", second_confirm=True, nonce=info["nonce"])
    # a fresh process (same database) always starts in DEMO
    again = build_controller(settings, ctl.config, mock, db=ctl.db)
    assert again.mode is Mode.DEMO
    await ctl.shutdown()


async def test_mode_switch_never_clears_kill_switch(tmp_path: Path, mock: MockDeriv) -> None:
    ctl = await ctl_for(tmp_path, mock, live_settings())
    ctl.kill("test")
    await ctl.switch_mode(Mode.DEMO)
    assert ctl.risk.kill_active()
    await ctl.shutdown()


async def test_kill_clear_requires_stopped_bot(controller: Controller) -> None:
    await controller.start()
    controller.kill("t")
    with pytest.raises(ControllerError, match="stop"):
        controller.clear_kill()
    await controller.stop()
    controller.clear_kill()
    assert not controller.risk.kill_active()
