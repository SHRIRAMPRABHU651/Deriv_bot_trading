"""Dashboard security + actions: Host/Origin allowlists, bearer token, CSRF, JSON bodies."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest_asyncio
from fastapi import FastAPI

from app.controller import Controller
from app.main import create_app
from app.models.schemas import Mode
from tests.conftest import make_settings
from tests.mocks.deriv_mock import MockDeriv

HOST = "http://127.0.0.1:8000"
GOOD_ORIGIN = {"Origin": HOST}


@pytest_asyncio.fixture
async def app(controller: Controller) -> FastAPI:
    return create_app(
        controller, settings=make_settings(), config=controller.config, autostart=False
    )


@pytest_asyncio.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=HOST) as c:
        yield c


async def csrf(http: httpx.AsyncClient) -> str:
    return str((await http.get("/api/csrf")).json()["csrf"])


async def post(
    http: httpx.AsyncClient,
    path: str,
    body: object = None,
    *,
    token: str | None = "dash-token",
    token_csrf: str | None = "auto",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    h = dict(GOOD_ORIGIN)
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    if token_csrf == "auto":
        token_csrf = await csrf(http)
    if token_csrf:
        h["X-CSRF-Token"] = token_csrf
    h.update(headers or {})
    if body is None:
        return await http.post(path, headers=h)
    return await http.post(path, headers=h, json=body)


async def test_host_header_allowlist(http: httpx.AsyncClient) -> None:
    assert (await http.get("/api/status")).status_code == 200
    assert (await http.get("/api/status", headers={"Host": "evil.example"})).status_code == 400
    assert (
        await http.get("/api/status", headers={"Host": "127.0.0.1.evil.example"})
    ).status_code == 400
    assert (await http.get("/api/status", headers={"Host": "localhost:8000"})).status_code == 200


async def test_status_is_readable_and_leaks_no_secrets(http: httpx.AsyncClient) -> None:
    r = await http.get("/api/status")
    body = r.text
    assert "demo-token-xyz" not in body and "dash-token" not in body
    data = r.json()
    for key in (
        "mode",
        "running",
        "account",
        "daily_pnl",
        "weekly_pnl",
        "drawdown",
        "hwm",
        "consecutive_losses",
        "win_rate",
        "break_even_win_rate",
        "model",
        "risk_status",
        "feed",
        "latency",
        "edge_margin",
        "halts",
    ):
        assert key in data
    assert data["mode"] == "demo" and data["product"]["product"] == "rise_fall"
    assert "DERIVBOT" in (await http.get("/")).text


async def test_post_requires_origin_token_csrf_and_json(http: httpx.AsyncClient) -> None:
    body = {"confirm": True, "reason": "t"}
    # Origin
    r = await post(http, "/kill", body, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = await post(http, "/kill", body, headers={"Sec-Fetch-Site": "cross-site", "Origin": ""})
    assert r.status_code == 403
    # token
    assert (await post(http, "/kill", body, token=None)).status_code == 401
    assert (await post(http, "/kill", body, token="wrong")).status_code == 401
    # csrf
    assert (await post(http, "/kill", body, token_csrf=None)).status_code == 403
    assert (await post(http, "/kill", body, token_csrf="forged")).status_code == 403
    # bodyless / malformed / wrong confirm
    assert (await post(http, "/kill", None)).status_code == 415  # bodyless POST is refused
    r = await http.post(
        "/kill",
        content=b"",
        headers={
            **GOOD_ORIGIN,
            "Authorization": "Bearer dash-token",
            "X-CSRF-Token": await csrf(http),
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 422
    assert (await post(http, "/kill", {"confirm": False})).status_code == 422
    r = await http.post(
        "/kill",
        content=b"confirm=1",
        headers={
            **GOOD_ORIGIN,
            "Authorization": "Bearer dash-token",
            "X-CSRF-Token": await csrf(http),
            "Content-Type": "text/plain",
        },
    )
    assert r.status_code == 415
    # everything valid
    assert (await post(http, "/kill", body)).status_code == 200


async def test_all_protected_endpoints_reject_unauthenticated_requests(
    http: httpx.AsyncClient,
) -> None:
    for path in (
        "/start",
        "/stop",
        "/kill",
        "/kill/clear",
        "/mode",
        "/mode/prepare",
        "/model/reload",
        "/risk/drawdown/reset",
        "/risk/review/resume",
        "/orders/release",
        "/product",
    ):
        r = await http.post(path, headers=GOOD_ORIGIN, json={"confirm": True})
        assert r.status_code == 401, path
        r = await http.get(path)
        assert r.status_code == 405, path  # actions are POST-only


async def test_start_stop_kill_clear_cycle(
    http: httpx.AsyncClient, controller: Controller, mock: MockDeriv
) -> None:
    r = await post(http, "/start", {"confirm": True})
    assert r.status_code == 200 and controller.running
    assert (await http.get("/api/status")).json()["running"] is True
    r = await post(http, "/kill", {"confirm": True, "reason": "test"})
    assert r.status_code == 200 and controller.risk.kill_active()
    assert "KILL_SWITCH" in (await http.get("/api/status")).json()["halts"]
    # clearing needs the exact phrase AND a stopped bot
    assert (await post(http, "/kill/clear", {"confirm": True, "phrase": "nope"})).status_code == 422
    assert (
        await post(http, "/kill/clear", {"confirm": True, "phrase": "CLEAR KILL"})
    ).status_code == 409
    assert (
        await post(http, "/stop", {"confirm": True})
    ).status_code == 200 and not controller.running
    # cannot start while the kill switch is active
    assert (await post(http, "/start", {"confirm": True})).status_code == 409
    assert (
        await post(http, "/kill/clear", {"confirm": True, "phrase": "CLEAR KILL"})
    ).status_code == 200
    assert not controller.risk.kill_active()
    assert (await post(http, "/start", {"confirm": True})).status_code == 200


async def test_mode_endpoints_deny_live_without_prerequisites(
    http: httpx.AsyncClient, controller: Controller
) -> None:
    assert (await post(http, "/mode/prepare", {"confirm": True})).status_code == 403
    r = await post(
        http, "/mode", {"target": "live", "typed": "LIVE", "second_confirm": True, "nonce": "x"}
    )
    assert r.status_code == 403
    assert (await http.get("/api/status")).json()["mode"] == "demo"
    assert (await post(http, "/mode", {"target": "demo"})).status_code == 200
    assert (await post(http, "/mode", {"target": "bogus"})).status_code == 422


async def test_model_reload_and_drawdown_reset(
    http: httpx.AsyncClient, controller: Controller
) -> None:
    r = await post(http, "/model/reload", {"confirm": True})
    assert r.status_code == 200 and r.json()["model"]["version"] == "stub-1"
    ok = {"confirm": True, "phrase": "RESET DRAWDOWN"}
    assert (await post(http, "/risk/drawdown/reset", {"confirm": True})).status_code == 422
    controller.risk.state.set_drawdown_halt(Mode.DEMO, True)
    assert "DRAWDOWN_HALT" in (await http.get("/api/status")).json()["halts"]
    r = await post(http, "/risk/drawdown/reset", ok)  # balance comes from the account list
    assert r.status_code == 200
    assert "DRAWDOWN_HALT" not in (await http.get("/api/status")).json()["halts"]
    await controller.start()
    assert (await post(http, "/risk/drawdown/reset", ok)).status_code == 409  # not while running


async def test_lifespan_shuts_the_controller_down(app: FastAPI, controller: Controller) -> None:
    async with app.router.lifespan_context(app):
        await controller.start()
        assert controller.running
    assert not controller.running
    json.dumps({"ok": True})


async def test_review_resume_needs_phrase_note_and_an_active_loss_halt(
    http: httpx.AsyncClient, controller: Controller
) -> None:
    from decimal import Decimal

    ok = {"confirm": True, "phrase": "RESUME", "note": "reviewed the 3 losses"}
    assert (await post(http, "/risk/review/resume", {**ok, "phrase": "no"})).status_code == 422
    assert (await post(http, "/risk/review/resume", ok)).status_code == 409  # nothing to review
    for _ in range(3):
        controller.risk.on_settlement(Decimal("-0.1"))
    assert "DAILY_LOSS_COUNT" in (await http.get("/api/status")).json()["halts"]
    assert (await post(http, "/risk/review/resume", {**ok, "note": "  "})).status_code == 409
    r = await post(http, "/risk/review/resume", ok)
    assert r.status_code == 200 and r.json()["review"]["used_today"] == 1
    assert "DAILY_LOSS_COUNT" not in (await http.get("/api/status")).json()["halts"]
