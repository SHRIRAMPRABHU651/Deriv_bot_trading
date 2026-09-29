"""Optional Telegram alerts. If not configured, every call is a silent no-op."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

import httpx

from app.config import Settings

log = logging.getLogger("derivbot.alerts")


class HttpPoster(Protocol):
    async def post(self, url: str, *, json: dict[str, str]) -> httpx.Response: ...


class TelegramAlerter:
    def __init__(self, settings: Settings, http: HttpPoster | None = None) -> None:
        self._token = settings.telegram_bot_token.get_secret_value()
        self._chat = settings.telegram_chat_id
        self._http = http
        self._last_sent: dict[str, float] = {}
        self._tasks: set[asyncio.Task[bool]] = set()
        self.sent: list[str] = []  # in-memory record (tests / dashboard)

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat)

    def notify(self, rule: str, message: str) -> None:
        """Fire-and-forget; de-duplicated per rule for 60 s. Never raises."""
        if not self.configured:
            return
        now = time.monotonic()
        if now - self._last_sent.get(rule, -1e9) < 60.0:
            return
        self._last_sent[rule] = now
        try:
            task = asyncio.get_running_loop().create_task(
                self.send(f"[DERIVBOT] {rule}: {message}")
            )
        except RuntimeError:
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send(self, text: str) -> bool:
        if not self.configured:
            return False
        self.sent.append(text)
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {"chat_id": self._chat, "text": text[:3500]}
        try:
            if self._http is not None:
                resp = await self._http.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(url, json=payload)
            return resp.status_code == 200
        except httpx.HTTPError:
            log.warning("telegram_failed", extra={"event": "telegram_failed"})
            return False
