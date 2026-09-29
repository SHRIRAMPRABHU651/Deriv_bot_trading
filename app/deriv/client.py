"""High-level Deriv client bound to a single account mode (demo or live)."""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx

from app.config import AppConfig, Settings
from app.deriv import protocol
from app.deriv.auth import DerivAuth
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.deriv.websocket import DerivWebSocket, Subscription
from app.models.schemas import (
    AccountInfo,
    BalanceInfo,
    BuyResult,
    ContractUpdate,
    Mode,
    Proposal,
    Tick,
)
from app.risk.permit import BuyPermit, PermitError


class DerivClient:
    def __init__(
        self,
        mode: Mode,
        settings: Settings,
        config: AppConfig,
        http: httpx.AsyncClient,
        *,
        ws: DerivWebSocket | None = None,
        backoff: Backoff | None = None,
    ) -> None:
        self.mode = mode
        self._cfg = config
        self.auth = DerivAuth(settings, config.deriv.rest_base_url, http)
        d = config.deriv
        self.ws = ws or DerivWebSocket(
            lambda: self.auth.get_ws_url(mode),
            RateLimiter(d.requests_per_second, d.burst),
            backoff or Backoff(),
            ping_interval=d.ping_interval_s,
            ping_timeout=d.ping_timeout_s,
            request_timeout=d.request_timeout_s,
        )
        self._tick_subs: dict[str, Subscription] = {}

    # ---- lifecycle / account ---------------------------------------------------------------
    async def connect(self) -> None:
        await self.ws.start()

    async def close(self) -> None:
        await self.ws.close()

    async def verify_account(self) -> AccountInfo:
        return await self.auth.verify_account(self.mode)

    async def get_balance(self) -> BalanceInfo:
        msg = await self.ws.request(protocol.balance(), safe_to_retry=True)
        return protocol.parse_balance(msg)

    # ---- market data -----------------------------------------------------------------------
    async def subscribe_ticks(self, symbol: str, on_tick: Callable[[Tick], None]) -> None:
        def handler(msg: dict[str, Any]) -> None:
            tick = protocol.parse_tick(msg, time.time())
            if tick is not None:
                on_tick(tick)

        self._tick_subs[symbol] = await self.ws.subscribe(protocol.ticks(symbol), handler)

    async def unsubscribe_ticks(self, symbol: str) -> None:
        sub = self._tick_subs.pop(symbol, None)
        if sub is not None:
            await self.ws.unsubscribe(sub)

    async def ticks_history(
        self, symbol: str, *, count: int, end: str | int = "latest", start: int | None = None
    ) -> list[tuple[int, float]]:
        msg = await self.ws.request(
            protocol.ticks_history(symbol, count=count, end=end, start=start),
            timeout_s=30.0,
            safe_to_retry=True,
        )
        return protocol.parse_history(msg)

    # ---- trading ---------------------------------------------------------------------------
    async def proposal(self, symbol: str, contract_type: str, stake: Decimal) -> Proposal:
        cfg = self._cfg.trading
        msg = await self.ws.request(
            protocol.proposal(
                symbol=symbol,
                contract_type=contract_type,
                amount=stake,
                currency=cfg.currency,
                duration=cfg.duration_ticks,
            ),
            timeout_s=self._cfg.deriv.proposal_timeout_s,
            safe_to_retry=True,
        )
        return protocol.parse_proposal(msg)

    async def buy(self, permit: BuyPermit, *, timeout_s: float | None = None) -> BuyResult:
        """Send a purchase. Requires a RiskManager-issued permit. Never auto-retried: on
        timeout/disconnect the outcome is unknown and the caller must reconcile."""
        if not isinstance(permit, BuyPermit):
            raise PermitError("buy requires a BuyPermit")
        if permit.mode is not self.mode:
            raise PermitError(f"permit mode {permit.mode.value} != client mode {self.mode.value}")
        msg = await self.ws.request(
            protocol.buy(permit.proposal_id, permit.max_price, {"order_id": permit.order_id}),
            timeout_s=timeout_s or self._cfg.deriv.buy_timeout_s,
            safe_to_retry=False,
        )
        return protocol.parse_buy(msg)

    async def contract_status(self, contract_id: int) -> ContractUpdate | None:
        msg = await self.ws.request(
            protocol.proposal_open_contract(contract_id, subscribe=False), safe_to_retry=True
        )
        return protocol.parse_contract(msg)

    async def subscribe_contract(
        self, contract_id: int, on_update: Callable[[ContractUpdate], None]
    ) -> Subscription:
        def handler(msg: dict[str, Any]) -> None:
            update = protocol.parse_contract(msg)
            if update is not None:
                on_update(update)

        return await self.ws.subscribe(
            protocol.proposal_open_contract(contract_id, subscribe=True), handler
        )

    async def unsubscribe(self, sub: Subscription) -> None:
        await self.ws.unsubscribe(sub)

    async def portfolio(self) -> list[dict[str, Any]]:
        msg = await self.ws.request(protocol.portfolio(), safe_to_retry=True)
        return protocol.parse_portfolio(msg)
