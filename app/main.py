"""FastAPI application + single-event-loop lifecycle (uvicorn owns SIGINT/SIGTERM)."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import build_router
from app.api.security import Guard, HostGuardMiddleware
from app.config import AppConfig, Settings, load_config
from app.controller import Controller, StartError
from app.logging_setup import register_secrets, setup_logging

log = logging.getLogger("derivbot.main")
STATIC = Path(__file__).parent / "dashboard" / "static"


def create_app(
    controller: Controller | None = None,
    *,
    settings: Settings | None = None,
    config: AppConfig | None = None,
    autostart: bool | None = None,
) -> FastAPI:
    settings = settings or Settings()
    config = config or load_config()
    token = settings.dashboard_token.get_secret_value()
    generated = not token
    if generated:
        token = secrets.token_urlsafe(24)
    register_secrets(
        settings.deriv_demo_token.get_secret_value(),
        settings.deriv_live_token.get_secret_value(),
        settings.telegram_bot_token.get_secret_value(),
        token,
    )
    guard = Guard(config.app.allowed_hosts, config.app.allowed_origins, token)
    ctl = controller or Controller(settings, config)
    auto = (
        autostart
        if autostart is not None
        else os.environ.get("AUTOSTART", "false").lower() == "true"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if generated:
            print(f"DERIVBOT dashboard token (this run only): {token}", flush=True)
        if auto:
            try:
                await ctl.start()
            except StartError as exc:
                log.error(
                    "autostart_failed", extra={"event": "autostart_failed", "error": str(exc)}
                )
        try:
            yield
        finally:
            await ctl.shutdown()  # ordered graceful shutdown inside uvicorn's own loop

    app = FastAPI(
        title="DERIVBOT", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.controller = ctl
    app.state.guard = guard
    app.include_router(build_router(ctl, guard))
    app.add_middleware(HostGuardMiddleware, guard=guard)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def run() -> None:
    config = load_config()
    setup_logging(config.app.log_dir)
    host = os.environ.get("DERIVBOT_HOST", config.app.host)
    uvicorn.run(create_app(config=config), host=host, port=config.app.port, log_level="info")


if __name__ == "__main__":
    run()
