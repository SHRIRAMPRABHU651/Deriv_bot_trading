"""Dashboard HTTP API. Reads need only a permitted Host; every POST needs
Origin + bearer token + CSRF token + a JSON body with an explicit confirmation."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.security import Guard
from app.controller import Controller, ControllerError, ModeSwitchError, StartError
from app.models.schemas import Mode


class Confirm(BaseModel):
    confirm: Literal[True]


class KillBody(BaseModel):
    confirm: Literal[True]
    reason: str = "manual"


class ClearKillBody(BaseModel):
    confirm: Literal[True]
    phrase: Literal["CLEAR KILL"]


class ReviewBody(BaseModel):
    confirm: Literal[True]
    phrase: Literal["RESUME"]
    note: str


class DrawdownBody(BaseModel):
    confirm: Literal[True]
    phrase: Literal["RESET DRAWDOWN"]


class ReleaseBody(BaseModel):
    confirm: Literal[True]
    phrase: Literal["RELEASE"]


class ProductBody(BaseModel):
    confirm: Literal[True]
    product: Literal["multiplier", "accumulator", "turbo", "vanilla", "rise_fall"]
    horizon_ticks: int | None = None  # how many ticks a trade is held (accumulators / multipliers)


class ModeBody(BaseModel):
    target: Literal["demo", "live"]
    typed: str = ""
    second_confirm: bool = False
    nonce: str = ""


def build_router(controller: Controller, guard: Guard) -> APIRouter:
    router = APIRouter()

    def action(request: Request) -> None:
        guard.check_action(request)

    act = Depends(action)

    @router.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/api/csrf")
    async def csrf() -> dict[str, str]:
        return {"csrf": guard.issue_csrf()}

    @router.get("/api/status")
    async def status() -> dict[str, Any]:
        return controller.status()

    @router.get("/api/lists")
    async def lists() -> dict[str, Any]:
        return controller.dashboard_lists()

    @router.post("/start", dependencies=[act])
    async def start(body: Confirm) -> dict[str, Any]:
        try:
            await controller.start()
        except StartError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "running": controller.running}

    @router.post("/stop", dependencies=[act])
    async def stop(body: Confirm) -> dict[str, Any]:
        await controller.stop()
        return {"ok": True, "running": controller.running}

    @router.post("/kill", dependencies=[act])
    async def kill(body: KillBody) -> dict[str, Any]:
        controller.kill(body.reason)
        return {"ok": True, "kill": True}

    @router.post("/kill/clear", dependencies=[act])
    async def kill_clear(body: ClearKillBody) -> dict[str, Any]:
        try:
            controller.clear_kill()
        except ControllerError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "kill": False}

    @router.post("/mode/prepare", dependencies=[act])
    async def mode_prepare(body: Confirm) -> dict[str, Any]:
        try:
            return await controller.prepare_live_switch()
        except ModeSwitchError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/mode", dependencies=[act])
    async def mode(body: ModeBody) -> dict[str, Any]:
        try:
            new_mode = await controller.switch_mode(
                Mode(body.target),
                typed=body.typed,
                second_confirm=body.second_confirm,
                nonce=body.nonce,
            )
        except ModeSwitchError as exc:
            raise HTTPException(403, str(exc)) from exc
        return {"ok": True, "mode": new_mode.value}

    @router.post("/model/reload", dependencies=[act])
    async def model_reload(body: Confirm) -> dict[str, Any]:
        loaded = controller.reload_model()
        return {"ok": loaded, "model": controller.status()["model"]}

    @router.post("/product", dependencies=[act])
    async def product(body: ProductBody) -> dict[str, Any]:
        try:
            chosen = controller.set_product(body.product, horizon_ticks=body.horizon_ticks)
        except ControllerError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "product": chosen}

    @router.post("/risk/review/resume", dependencies=[act])
    async def review_resume(body: ReviewBody) -> dict[str, Any]:
        try:
            controller.review_resume(body.note)
        except ControllerError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "review": controller.status()["review"]}

    @router.post("/orders/release", dependencies=[act])
    async def orders_release(body: ReleaseBody) -> dict[str, Any]:
        try:
            released = controller.release_stuck_orders()
        except ControllerError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "released": released}

    @router.post("/risk/drawdown/reset", dependencies=[act])
    async def drawdown_reset(body: DrawdownBody) -> dict[str, Any]:
        try:
            await controller.reset_drawdown_halt()
        except ControllerError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True}

    return router
