"""Contract tracking and settlement. Nothing stays OPEN forever: watchers time out into
RECONCILIATION_PENDING and are then resolved against the account state."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from decimal import Decimal

from app.clock import Clock
from app.config import AppConfig
from app.deriv.client import DerivClient
from app.deriv.protocol import ConnectionLost, DerivError
from app.deriv.websocket import Subscription
from app.metrics.latency import LatencyTracker
from app.models.schemas import ContractUpdate, Mode, OrderState
from app.risk.manager import ModelInfo, RiskManager
from app.storage.repositories import Repositories

log = logging.getLogger("derivbot.settlement")

SETTLABLE = (
    OrderState.BOUGHT.value,
    OrderState.OPEN.value,
    OrderState.RECONCILIATION_PENDING.value,
)


class SettlementTracker:
    def __init__(
        self,
        client: DerivClient,
        repos: Repositories,
        risk: RiskManager,
        config: AppConfig,
        clock: Clock,
        latency: LatencyTracker,
        *,
        refresh_balance: Callable[[], Awaitable[None]],
        model_getter: Callable[[], ModelInfo | None],
    ) -> None:
        self._client = client
        self._repos = repos
        self._risk = risk
        self._cfg = config
        self._clock = clock
        self._latency = latency
        self._refresh_balance = refresh_balance
        self._model = model_getter
        self._subs: dict[int, Subscription] = {}
        self._timers: dict[int, asyncio.Task[None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self.settled_count = 0

    @property
    def mode(self) -> Mode:
        return self._risk.mode

    # ---- watching ---------------------------------------------------------------------------
    async def watch(self, order_id: str, contract_id: int) -> None:
        if contract_id not in self._subs:

            def on_update(update: ContractUpdate) -> None:
                self._spawn(self.handle_update(order_id, update))

            try:
                self._subs[contract_id] = await self._client.subscribe_contract(
                    contract_id, on_update
                )
            except (DerivError, ConnectionLost, TimeoutError):
                self._repos.update_order(order_id, state=OrderState.RECONCILIATION_PENDING)
                self._repos.add_reconciliation(
                    self.mode, "watch_failed", "could not subscribe to contract", contract_id
                )
        self._arm_timer(order_id, contract_id)

    def _arm_timer(self, order_id: str, contract_id: int) -> None:
        """(Re)start the settlement deadline: an OPEN contract can never be forgotten."""
        old = self._timers.get(contract_id)
        if old is not None and old is not asyncio.current_task():
            old.cancel()
        self._timers[contract_id] = asyncio.create_task(self._timeout_guard(order_id, contract_id))

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _timeout_guard(self, order_id: str, contract_id: int) -> None:
        await asyncio.sleep(self._cfg.deriv.settlement_timeout_s)
        row = self._repos.get_order(order_id)
        if row is not None and row["state"] in SETTLABLE:
            self._repos.update_order(order_id, state=OrderState.RECONCILIATION_PENDING)
            self._repos.add_reconciliation(
                self.mode, "settlement_timeout", "no settlement before timeout", contract_id
            )
            await self.resolve(order_id, contract_id)

    # ---- state transitions ------------------------------------------------------------------
    def mark_all_pending(self) -> None:
        """Called on disconnect: open contracts can no longer be observed."""
        for row in self._repos.active_orders(self.mode):
            if row["contract_id"] is not None and row["state"] in (
                OrderState.OPEN.value,
                OrderState.BOUGHT.value,
            ):
                self._repos.update_order(row["order_id"], state=OrderState.RECONCILIATION_PENDING)
                self._repos.add_reconciliation(
                    self.mode, "disconnect", "watcher lost", int(row["contract_id"])
                )
        for sub in self._subs.values():
            self._client.ws.discard_subscription(sub)  # avoid stale re-subscription on reconnect
        self._subs.clear()

    async def handle_update(self, order_id: str, update: ContractUpdate) -> None:
        if update.is_sold or update.status in ("won", "lost", "sold", "cancelled"):
            await self.settle(order_id, update)

    async def resolve(self, order_id: str, contract_id: int) -> None:
        """Query the account for a contract's final state and settle / resume watching."""
        try:
            update = await self._client.contract_status(contract_id)
        except (DerivError, ConnectionLost, TimeoutError):
            return  # stays RECONCILIATION_PENDING; retried on reconnect / next sweep
        if update is None:
            self._repos.add_reconciliation(
                self.mode, "contract_unknown", "broker returned no contract", contract_id
            )
            return
        if update.is_sold or update.status in ("won", "lost", "sold", "cancelled"):
            await self.settle(order_id, update, reconciled=True)
        else:
            self._repos.update_order(order_id, state=OrderState.OPEN)
            await self.watch(order_id, contract_id)

    async def settle(
        self, order_id: str, update: ContractUpdate, *, reconciled: bool = False
    ) -> None:
        row = self._repos.get_order(order_id)
        if row is None or row["state"] not in SETTLABLE:
            return  # already settled: settlement is idempotent
        stake = Decimal(str(row["stake"]))
        if update.profit is not None:
            profit = update.profit
        elif update.sell_price is not None:
            profit = update.sell_price - (update.buy_price or stake)
        else:
            profit = Decimal(0) - stake if update.status == "lost" else Decimal(0)
        won = profit > 0
        state = OrderState.WON if won else OrderState.LOST
        contract_id = int(row["contract_id"])
        now = self._clock.now()
        with self._repos.db.transaction():
            self._repos.update_order(order_id, state=state)
            self._repos.settle_trade(
                contract_id, state, profit, update.entry_spot, update.exit_spot, now
            )
            if reconciled:
                self._repos.add_reconciliation(
                    self.mode, "settled_by_reconciliation", f"profit={profit}", contract_id
                )
            self._risk.on_settlement(profit)
        self.settled_count += 1
        if row["confirmed_at"] is not None:
            self._latency.record(
                self.mode,
                "confirmation_to_settlement",
                (self._clock.time() - float(row["confirmed_at"])) * 1000.0,
                now,
            )
        timer = self._timers.pop(contract_id, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        sub = self._subs.pop(contract_id, None)
        if sub is not None:
            with contextlib.suppress(Exception):
                await self._client.unsubscribe(sub)
        with contextlib.suppress(Exception):
            await self._refresh_balance()
        self._risk.check_model_performance(self._model())
        log.info(
            "settled",
            extra={"event": "settled", "contract_id": contract_id, "mode": self.mode.value},
        )

    async def close(self) -> None:
        for t in list(self._timers.values()):
            t.cancel()
        self._timers.clear()
        for task in list(self._tasks):
            task.cancel()
        for sub in list(self._subs.values()):
            with contextlib.suppress(Exception):
                await self._client.unsubscribe(sub)
        self._subs.clear()
