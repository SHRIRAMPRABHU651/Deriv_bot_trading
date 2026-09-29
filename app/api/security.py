"""Dashboard hardening: Host allowlist, Origin allowlist, bearer token, CSRF token."""

from __future__ import annotations

import hmac
import secrets
import time

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class Guard:
    def __init__(
        self,
        allowed_hosts: list[str],
        allowed_origins: list[str],
        dashboard_token: str,
        csrf_ttl_s: float = 3600.0,
    ) -> None:
        if not dashboard_token:
            raise ValueError("dashboard token must not be empty")
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.allowed_origins = {o.rstrip("/").lower() for o in allowed_origins}
        self._token = dashboard_token
        self._csrf: dict[str, float] = {}
        self._ttl = csrf_ttl_s

    # ---- csrf -------------------------------------------------------------------------------
    def issue_csrf(self) -> str:
        now = time.monotonic()
        self._csrf = {t: exp for t, exp in self._csrf.items() if exp > now}
        if len(self._csrf) > 256:  # bounded memory
            self._csrf.pop(min(self._csrf, key=lambda k: self._csrf[k]))
        token = secrets.token_urlsafe(32)
        self._csrf[token] = now + self._ttl
        return token

    def csrf_valid(self, token: str | None) -> bool:
        if not token:
            return False
        exp = self._csrf.get(token)
        return exp is not None and exp > time.monotonic()

    def token_valid(self, presented: str | None) -> bool:
        return bool(presented) and hmac.compare_digest(presented or "", self._token)

    # ---- checks -----------------------------------------------------------------------------
    def host_allowed(self, host_header: str | None) -> bool:
        if not host_header:
            return False
        host = host_header.strip().lower()
        # IPv6 literal "[::1]:8000" vs "host:8000"
        host = host.split("]")[0] + "]" if host.startswith("[") else host.split(":")[0]
        return host in self.allowed_hosts

    def origin_ok(self, origin: str | None, sec_fetch_site: str | None) -> bool:
        if origin:
            return origin.rstrip("/").lower() in self.allowed_origins
        # No Origin: browsers always send Sec-Fetch-Site on cross-origin/fetch POSTs; a request
        # without either header is a non-browser client (still needs token + CSRF).
        return sec_fetch_site is None

    def check_action(self, request: Request) -> None:
        """Authenticate + CSRF + Origin for every state-changing request."""
        if not self.origin_ok(request.headers.get("origin"), request.headers.get("sec-fetch-site")):
            raise HTTPException(403, "origin not allowed")
        auth = request.headers.get("authorization", "")
        presented = auth[7:] if auth.lower().startswith("bearer ") else None
        if not self.token_valid(presented):
            raise HTTPException(401, "missing or invalid dashboard token")
        if not self.csrf_valid(request.headers.get("x-csrf-token")):
            raise HTTPException(403, "missing or invalid CSRF token")
        if "application/json" not in request.headers.get("content-type", ""):
            raise HTTPException(415, "JSON body required")


class HostGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, guard: Guard) -> None:
        super().__init__(app)
        self._guard = guard

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._guard.host_allowed(request.headers.get("host")):
            return JSONResponse({"detail": "host not allowed"}, status_code=400)
        return await call_next(request)
