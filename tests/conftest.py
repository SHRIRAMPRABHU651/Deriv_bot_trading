"""Shared fixtures: settings/config, mock Deriv server, controller wired to the mock."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from app.config import AppConfig, RiskProfile, Settings
from app.controller import Controller
from app.deriv.client import DerivClient
from app.deriv.reconnect import Backoff
from app.models.schemas import Mode
from app.products import Product, ProductSpec
from app.storage.database import Database
from tests.mocks.deriv_mock import MockDeriv, make_http
from tests.mocks.stubs import make_stub_model


def make_settings(
    *,
    allow_live: bool = False,
    live_token: str = "",
    live_account: str = "",
    demo_token: str = "demo-token-xyz",
    dashboard: str = "dash-token",
) -> Settings:
    return Settings(
        _env_file=None,
        deriv_app_id="app1",
        deriv_demo_token=SecretStr(demo_token),
        deriv_demo_account_id="DOT1",
        deriv_live_token=SecretStr(live_token),
        deriv_live_account_id=live_account,
        allow_live=allow_live,
        dashboard_token=SecretStr(dashboard),
    )


def make_config(
    tmp_path: Path, *, spec: ProductSpec | None = None, **profile_overrides: object
) -> AppConfig:
    cfg = AppConfig()
    cfg.app.db_path = str(tmp_path / "bot.db")
    cfg.app.model_dir = str(tmp_path / "model")
    cfg.app.log_dir = str(tmp_path / "logs")
    cfg.app.allowed_hosts = ["127.0.0.1", "localhost"]
    cfg.trading.symbols = ["R_100"]
    cfg.trading.product = spec or ProductSpec(product=Product.RISE_FALL, horizon_ticks=5)
    cfg.trading.max_signal_age_s = 5.0
    cfg.deriv.requests_per_second = 500.0
    cfg.deriv.burst = 100
    cfg.deriv.ping_interval_s = 30.0
    cfg.deriv.request_timeout_s = 3.0
    cfg.deriv.proposal_timeout_s = 1.0
    cfg.deriv.buy_timeout_s = 0.6
    cfg.deriv.settlement_timeout_s = 30.0
    cfg.risk.feed_stale_after_s = 30.0
    prof = cfg.risk.demo.model_dump()
    prof.update({"cooldown_seconds": 0.0, "max_open_trades": 3, "max_open_trades_per_symbol": 1})
    prof.update(profile_overrides)
    cfg.risk.demo = RiskProfile(**prof)
    return cfg


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest_asyncio.fixture
async def mock() -> AsyncIterator[MockDeriv]:
    server = MockDeriv()
    await server.start()
    yield server
    await server.stop()


def build_controller(
    settings: Settings,
    config: AppConfig,
    mock: MockDeriv,
    *,
    http: httpx.AsyncClient | None = None,
    account_type: str = "demo",
    db: Database | None = None,
) -> Controller:
    client_http = http or make_http(mock, account_type=account_type)[0]

    def factory(mode: Mode) -> DerivClient:
        return DerivClient(
            mode, settings, config, client_http, backoff=Backoff(0.05, 2.0, 0.2, 0.0)
        )

    return Controller(settings, config, http=client_http, client_factory=factory, db=db)


@pytest_asyncio.fixture
async def controller(
    settings: Settings, config: AppConfig, mock: MockDeriv
) -> AsyncIterator[Controller]:
    make_stub_model(Path(config.app.model_dir), p_up=0.7, horizon=5)
    ctl = build_controller(settings, config, mock)
    ctl.reload_model()
    yield ctl
    await ctl.shutdown()


def count_rows(db: Database, table: str) -> int:
    row = db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
    assert row is not None
    return int(row["n"])


def client_of(ctl: Controller) -> DerivClient:
    assert ctl.client is not None
    return ctl.client


async def wait_until(
    cond: Callable[[], bool | Awaitable[bool]], timeout_s: float = 5.0, step: float = 0.02
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        res = cond()
        if isinstance(res, Awaitable):
            res = await res
        if res:
            return
        await asyncio.sleep(step)
    raise AssertionError("condition not met within timeout")
