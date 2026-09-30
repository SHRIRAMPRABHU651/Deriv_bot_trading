"""Start-up failures must surface as clear StartError (never crash the app or trade)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import AppConfig
from app.controller import Controller, StartError
from app.deriv.client import DerivClient
from app.models.schemas import Mode
from tests.conftest import build_controller, make_config, make_settings
from tests.mocks.deriv_mock import MockDeriv
from tests.mocks.stubs import make_stub_model


def controller_with_transport(
    tmp_path: Path, handler: httpx.MockTransport, *, token: str = "demo-token-xyz"
) -> Controller:
    settings = make_settings(demo_token=token)
    cfg: AppConfig = make_config(tmp_path)
    make_stub_model(Path(cfg.app.model_dir))
    http = httpx.AsyncClient(transport=handler)

    def factory(mode: Mode) -> DerivClient:
        return DerivClient(mode, settings, cfg, http)

    return Controller(settings, cfg, http=http, client_factory=factory)


async def test_rejected_token_gives_clear_error(tmp_path: Path) -> None:
    ctl = controller_with_transport(
        tmp_path, httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad"}))
    )
    with pytest.raises(StartError, match="401"):
        await ctl.start()
    assert not ctl.running and not ctl.trading_enabled
    await ctl.shutdown()


async def test_network_unreachable_gives_clear_error(tmp_path: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    ctl = controller_with_transport(tmp_path, httpx.MockTransport(boom))
    with pytest.raises(StartError):
        await ctl.start()
    assert not ctl.running
    await ctl.shutdown()


async def test_missing_credentials_tell_the_operator_what_to_configure(tmp_path: Path) -> None:
    ctl = controller_with_transport(
        tmp_path, httpx.MockTransport(lambda req: httpx.Response(200, json={})), token=""
    )
    with pytest.raises(StartError, match="DERIV_DEMO_TOKEN"):
        await ctl.start()
    await ctl.shutdown()


async def test_wrong_account_type_aborts_start(tmp_path: Path, mock: MockDeriv) -> None:
    from tests.mocks.deriv_mock import make_http

    http, _ = make_http(mock, account_type="real")  # demo mode but the API says REAL
    cfg = make_config(tmp_path)
    make_stub_model(Path(cfg.app.model_dir))
    ctl = build_controller(make_settings(), cfg, mock, http=http)
    with pytest.raises(StartError, match="real"):
        await ctl.start()
    assert not ctl.running
    await ctl.shutdown()


async def test_rejected_token_shows_deriv_message_without_leaking_the_token(tmp_path: Path) -> None:
    body = {"error": {"code": "InvalidToken", "message": "Invalid or expired token"}}
    ctl = controller_with_transport(
        tmp_path,
        httpx.MockTransport(lambda req: httpx.Response(401, json=body)),
        token="pat_" + "s3cr3t" * 5,
    )
    with pytest.raises(StartError) as exc:
        await ctl.start()
    text = str(exc.value)
    assert "Invalid or expired token" in text and "401" in text and "app id" in text
    assert "s3cr3t" not in text
    await ctl.shutdown()


def test_credential_lint_catches_common_copy_paste_mistakes() -> None:
    from scripts.check_auth import lint_value

    assert lint_value("T", "pat_abc") == []
    assert lint_value("T", "") == ["T is empty"]
    assert any("quotes" in p for p in lint_value("T", '"pat_abc"'))
    assert any("whitespace" in p for p in lint_value("T", "pat_abc "))
    assert any("space" in p for p in lint_value("T", "pat abc"))
