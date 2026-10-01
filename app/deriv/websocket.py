"""Resilient Deriv WebSocket: req_id correlation, subscriptions, ping/heartbeat, reconnect."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from app.deriv import protocol
from app.deriv.auth import AuthError
from app.deriv.protocol import ConnectionLost, DerivError, RateLimitError
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.logging_setup import redact

log = logging.getLogger("derivbot.ws")

Handler = Callable[[dict[str, Any]], None]


@dataclass
class Subscription:
    payload: dict[str, Any]
    handler: Handler
    req_id: int = 0
    sub_id: str | None = None
    active: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class DerivWebSocket:
    def __init__(
        self,
        url_provider: Callable[[], Awaitable[str]],
        limiter: RateLimiter,
        backoff: Backoff,
        *,
        ping_interval: float = 15.0,
        ping_timeout: float = 10.0,
        request_timeout: float = 10.0,
        healthy_after: float = 10.0,
    ) -> None:
        self._url_provider = url_provider
        self._limiter = limiter
        self._backoff = backoff
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._request_timeout = request_timeout
        self._healthy_after = healthy_after
        self._ws: ClientConnection | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._subs: dict[int, Subscription] = {}
        self._all_subs: list[Subscription] = []
        self._next_id = 1
        self._closing = False
        self._supervisor: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self.connected = asyncio.Event()
        self.on_connected: Callable[[bool], Awaitable[None]] | None = None
        self.on_disconnected: Callable[[], None] | None = None
        self.connect_count = 0
        self.reconnect_count = 0
        self.disconnect_count = 0
        self.last_message_at = 0.0
        self.last_error: str | None = None
        self.auth_retry_floor_s = 15.0

    # ---- lifecycle --------------------------------------------------------------------------
    async def start(self, timeout_s: float = 30.0) -> None:
        self._closing = False
        self._supervisor = asyncio.create_task(self._supervise(), name="deriv-ws-supervisor")
        try:
            await asyncio.wait_for(self.connected.wait(), timeout_s)
        except TimeoutError:
            await self.close()
            raise ConnectionLost("could not establish the Deriv WebSocket in time") from None

    async def close(self) -> None:
        self._closing = True
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(WebSocketException, OSError):
                await ws.close()
        if self._supervisor is not None:
            self._supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._supervisor
            self._supervisor = None
        self._fail_pending(ConnectionLost("client closed"))
        for task in list(self._tasks):
            task.cancel()
        self.connected.clear()

    # ---- supervisor / receive loop ----------------------------------------------------------
    async def _supervise(self) -> None:
        first = True
        while not self._closing:
            connected_at = 0.0
            auth_failed = False
            try:
                url = await self._url_provider()
                async with connect(url, ping_interval=None, open_timeout=15, max_size=2**22) as ws:
                    self._ws = ws
                    connected_at = time.monotonic()
                    self.connect_count += 1
                    if not first:
                        self.reconnect_count += 1
                    self.last_message_at = time.time()
                    self.last_error = None
                    self.connected.set()
                    hb = asyncio.create_task(self._heartbeat(ws), name="deriv-ws-heartbeat")
                    if self.on_connected is not None:
                        self._spawn(self.on_connected(not first))
                    first = False
                    try:
                        async for raw in ws:
                            self._handle(raw)
                    finally:
                        hb.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await hb
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail = redact(str(exc))[:240]
                self.last_error = (
                    f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
                )
                log.warning(
                    "ws_error",
                    extra={"event": "ws_error", "error": type(exc).__name__, "detail": detail},
                )
                auth_failed = isinstance(exc, AuthError)
            finally:
                self._ws = None
                if self.connected.is_set():
                    self.disconnect_count += 1
                self.connected.clear()
                self._fail_pending(ConnectionLost("connection lost"))
                if self.on_disconnected is not None:
                    self.on_disconnected()
            if self._closing:
                break
            if connected_at and time.monotonic() - connected_at >= self._healthy_after:
                self._backoff.reset()
            delay = self._backoff.next_delay()
            if auth_failed:  # Deriv refused a connection key: do not hammer it
                delay = max(delay, self.auth_retry_floor_s)
            log.info("ws_reconnect_wait", extra={"event": "ws_reconnect_wait", "delay_s": delay})
            await asyncio.sleep(delay)

    async def _heartbeat(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            try:
                await self.request(protocol.ping(), timeout_s=self._ping_timeout, _bypass_wait=True)
            except (TimeoutError, ConnectionLost, DerivError):
                log.warning("ws_ping_timeout", extra={"event": "ws_ping_timeout"})
                with contextlib.suppress(WebSocketException, OSError):
                    await ws.close(code=1011, reason="ping timeout")
                return

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("ws_task_failed", extra={"event": "ws_task_failed"})

    def _fail_pending(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        self._subs.clear()  # subscriptions are re-registered on reconnect via _all_subs

    def _handle(self, raw: str | bytes) -> None:
        self.last_message_at = time.time()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        req_id = msg.get("req_id")
        fut = self._pending.pop(req_id, None) if isinstance(req_id, int) else None
        if fut is not None and not fut.done():
            if "error" in msg:
                fut.set_exception(protocol.error_from_message(msg))
            else:
                fut.set_result(msg)
        sub = self._subs.get(req_id) if isinstance(req_id, int) else None
        if sub is not None and sub.active:
            if "subscription" in msg and isinstance(msg["subscription"], dict):
                sub.sub_id = str(msg["subscription"].get("id", "")) or sub.sub_id
            try:
                sub.handler(msg)
            except Exception:
                log.exception("subscription handler failed", extra={"event": "handler_error"})

    # ---- requests ---------------------------------------------------------------------------
    def _new_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def _send(self, payload: dict[str, Any], *, _bypass_wait: bool = False) -> int:
        if not _bypass_wait:
            await asyncio.wait_for(self.connected.wait(), self._request_timeout)
        ws = self._ws
        if ws is None:
            raise ConnectionLost("not connected")
        await self._limiter.acquire()
        req_id = payload.get("req_id") or self._new_id()
        body = {**payload, "req_id": req_id}
        try:
            await ws.send(json.dumps(body))
        except (WebSocketException, OSError) as exc:
            raise ConnectionLost(str(exc)) from exc
        return int(req_id)

    async def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
        safe_to_retry: bool = False,
        _bypass_wait: bool = False,
    ) -> dict[str, Any]:
        """Send one request and await its response. Only idempotent requests may set
        safe_to_retry (retried on rate-limit responses). NEVER set it for `buy`."""
        attempts = 4 if safe_to_retry else 1
        delay = 0.5
        for attempt in range(attempts):
            req_id = self._new_id()
            fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[req_id] = fut
            try:
                await self._send({**payload, "req_id": req_id}, _bypass_wait=_bypass_wait)
                return await asyncio.wait_for(fut, timeout_s or self._request_timeout)
            except RateLimitError:
                self._pending.pop(req_id, None)
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(delay * (1 + 0.25 * (attempt % 2)))
                delay *= 2
            except BaseException:
                self._pending.pop(req_id, None)
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def subscribe(self, payload: dict[str, Any], handler: Handler) -> Subscription:
        sub = Subscription(payload=payload, handler=handler)
        await self._start_subscription(sub)
        self._all_subs.append(sub)
        return sub

    async def _start_subscription(self, sub: Subscription) -> dict[str, Any]:
        req_id = self._new_id()
        sub.req_id = req_id
        self._subs[req_id] = sub
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._send({**sub.payload, "req_id": req_id})
            return await asyncio.wait_for(fut, self._request_timeout)
        except BaseException:
            self._pending.pop(req_id, None)
            self._subs.pop(req_id, None)
            raise

    async def resubscribe_all(self) -> None:
        for sub in list(self._all_subs):
            if not sub.active:
                continue
            try:
                await self._start_subscription(sub)
            except (DerivError, ConnectionLost, TimeoutError) as exc:
                log.warning(
                    "resubscribe_failed",
                    extra={"event": "resubscribe_failed", "error": type(exc).__name__},
                )

    def discard_subscription(self, sub: Subscription) -> None:
        """Forget a subscription locally (its connection is gone; nothing to send)."""
        sub.active = False
        if sub in self._all_subs:
            self._all_subs.remove(sub)
        self._subs.pop(sub.req_id, None)

    async def unsubscribe(self, sub: Subscription) -> None:
        sub.active = False
        if sub in self._all_subs:
            self._all_subs.remove(sub)
        self._subs.pop(sub.req_id, None)
        if sub.sub_id and self.connected.is_set():
            with contextlib.suppress(DerivError, ConnectionLost, TimeoutError):
                await self.request(protocol.forget(sub.sub_id), timeout_s=3.0)
