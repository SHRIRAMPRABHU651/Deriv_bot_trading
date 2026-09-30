"""Authentication against the current Deriv API (PAT + Deriv-App-ID -> REST -> OTP -> WebSocket).

Flow (docs/API_NOTES.md):
  1. REST calls carry `Authorization: Bearer <PAT>` and `Deriv-App-ID: <app id>`.
  2. GET  {rest}/trading/v1/options/accounts                     -> account list (type demo/real)
  3. POST {rest}/trading/v1/options/accounts/{accountId}/otp      -> data.url (WebSocket URL with a
     single-use OTP valid ~120 s). A fresh OTP is requested for EVERY (re)connect.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.deriv.protocol import dec
from app.models.schemas import AccountInfo, Mode


class AuthError(Exception):
    pass


class LiveNotPermittedError(AuthError):
    pass


class DerivAuth:
    def __init__(self, settings: Settings, rest_base_url: str, http: httpx.AsyncClient) -> None:
        self._settings = settings
        self._base = rest_base_url.rstrip("/")
        self._http = http

    def _headers(self, mode: Mode) -> dict[str, str]:
        if mode is Mode.LIVE and not self._settings.live_permitted():
            raise LiveNotPermittedError(
                "LIVE credentials are unavailable: ALLOW_LIVE must be true and "
                "DERIV_LIVE_TOKEN / DERIV_LIVE_ACCOUNT_ID must be set"
            )
        token = self._settings.token_for(mode)
        if not token or not self._settings.deriv_app_id:
            raise AuthError(
                "Missing credentials. Put DERIV_APP_ID, DERIV_DEMO_TOKEN and "
                "DERIV_DEMO_ACCOUNT_ID in .env (never paste the token in chat)."
            )
        return {"Authorization": f"Bearer {token}", "Deriv-App-ID": self._settings.deriv_app_id}

    async def list_accounts(self, mode: Mode) -> list[AccountInfo]:
        resp = await self._http.get(
            f"{self._base}/trading/v1/options/accounts", headers=self._headers(mode)
        )
        if resp.status_code != 200:
            raise AuthError(describe_failure("account list", resp))
        return parse_accounts(resp.json())

    async def verify_account(self, mode: Mode) -> AccountInfo:
        """Confirm the configured account exists and its type matches `mode`."""
        wanted = self._settings.account_id_for(mode)
        expected = "demo" if mode is Mode.DEMO else "real"
        for acct in await self.list_accounts(mode):
            if acct.account_id == wanted:
                if acct.account_type != expected:
                    raise AuthError(
                        f"account {wanted} is '{acct.account_type}', expected '{expected}'"
                    )
                return acct
        raise AuthError(f"account {wanted} not found in the account list")

    async def get_ws_url(self, mode: Mode) -> str:
        account_id = self._settings.account_id_for(mode)
        resp = await self._http.post(
            f"{self._base}/trading/v1/options/accounts/{account_id}/otp",
            headers=self._headers(mode),
        )
        if resp.status_code not in (200, 201):
            raise AuthError(describe_failure("OTP request", resp))
        data = resp.json().get("data") or {}
        url = data.get("url")
        if not isinstance(url, str) or not url.startswith(("ws://", "wss://")):
            raise AuthError("OTP response did not contain a WebSocket url")
        return url


def describe_failure(what: str, resp: httpx.Response) -> str:
    """HTTP status + Deriv's message (truncated) + likely causes. Never echoes secrets."""
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")
            else:
                detail = str(err)
    except ValueError:
        detail = resp.text
    msg = f"{what} failed: HTTP {resp.status_code}" + (f" - {detail[:160]}" if detail else "")
    if resp.status_code in (401, 403):
        msg += (
            " | check: the token is copied in full (starts with pat_) with no spaces or quotes, "
            "has not expired or been revoked, was created for this account, and "
            "DERIV_APP_ID is the app id belonging to it"
        )
    return msg


def parse_accounts(payload: Any) -> list[AccountInfo]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = data.get("accounts", [data])
    out: list[AccountInfo] = []
    for item in data or []:
        if not isinstance(item, dict) or "account_id" not in item:
            continue
        bal = item.get("balance")
        out.append(
            AccountInfo(
                account_id=str(item["account_id"]),
                account_type=str(item.get("account_type", "")).lower(),
                balance=None if bal is None else dec(bal),
                currency=None if item.get("currency") is None else str(item["currency"]),
            )
        )
    return out
