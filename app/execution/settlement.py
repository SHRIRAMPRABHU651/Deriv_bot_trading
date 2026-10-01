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
from app.deriv.protocol import CLOSED_STATUSES, ConnectionLost, DerivError
from app.deriv.websocket import Subscription
from app.metrics.latency import LatencyTracker
from app.models.schemas import ContractUpdate, Mode, OrderState, Severity
from app.products import ExitPlan, exit_plan
from app.risk.manager import ModelInfo, RiskManager
from app.storage.repositories import Repositories

log = logging.getLogger("derivbot.settlement")

RESOLVE_ATTEMPTS = 20  # settlement-timeout retries per contract
RESOLVE_RETRY_S = 15.0
UNKNOWN_LIMIT = 4  # consecutive "broker has no such contract" answers before releasing the slot

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
        self._holds: dict[int, asyncio.Task[None]] = {}
        self._plans: dict[int, ExitPlan | None] = {}
        self._selling: set[int] = set()
        self._unknown_counts: dict[int, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self.settled_count = 0
        self.exits = {"take_profit": 0, "hold_cap": 0}

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
        plan = self._plan_for(order_id)
        self._plans[contract_id] = plan
        hold = plan.max_hold_s if plan is not None and plan.max_hold_s else 0.0
        if hold and contract_id not in self._holds:
            self._holds[contract_id] = asyncio.create_task(
                self._hold_guard(order_id, contract_id, hold)
            )
        self._arm_timer(order_id, contract_id, self._cfg.deriv.settlement_timeout_s + hold)

    def _plan_for(self, order_id: str) -> ExitPlan | None:
        """Bot-managed exits derive from the order row + configured product (DB is authoritative,
        so a restart resumes the same plan)."""
        row = self._repos.get_order(order_id)
        spec = self._cfg.trading.product
        if row is None or row["product"] != spec.product.value:
            return None
        return exit_plan(spec, Decimal(str(row["ask_price"] or row["stake"])))

    def _arm_timer(self, order_id: str, contract_id: int, delay: float) -> None:
        """(Re)start the settlement deadline: an OPEN contract can never be forgotten."""
        old = self._timers.get(contract_id)
        if old is not None and old is not asyncio.current_task():
            old.cancel()
        self._timers[contract_id] = asyncio.create_task(
            self._timeout_guard(order_id, contract_id, delay)
        )

    async def _hold_guard(self, order_id: str, contract_id: int, hold_s: float) -> None:
        """Products that never expire (multipliers, accumulators) are closed after the hold cap."""
        await asyncio.sleep(hold_s)
        row = self._repos.get_order(order_id)
        if row is not None and row["state"] in SETTLABLE:
            await self._sell(order_id, contract_id, "hold_cap")

    async def _sell(self, order_id: str, contract_id: int, reason: str) -> None:
        if contract_id in self._selling:
            return
        self._selling.add(contract_id)
        try:
            await self._client.sell(contract_id)
        except (DerivError, ConnectionLost, TimeoutError) as exc:
            self._selling.discard(contract_id)  # allow another attempt
            self._repos.add_risk_event(
                self.mode,
                Severity.WARNING,
                "sell_failed",
                f"{reason}: {type(exc).__name__}",
                ts=self._clock.now(),
            )
            return
        self.exits[reason] = self.exits.get(reason, 0) + 1
        self._repos.add_reconciliation(self.mode, f"exit_{reason}", order_id, contract_id)

    async def _maybe_take_profit(self, order_id: str, update: ContractUpdate) -> None:
        plan = self._plans.get(update.contract_id)
        if plan is None or plan.take_profit_pct is None or update.profit is None:
            return
        target = Decimal(str(plan.take_profit_pct)) * plan.cost
        if update.valid_to_sell and update.profit >= target:
            await self._sell(order_id, update.contract_id, "take_profit")

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """A failed settlement must never be silent (the order would look open forever)."""
        self._tasks.discard(task)
        if task.cancelled() or task.exception() is None:
            return
        exc = task.exception()
        log.error("settlement_task_failed", exc_info=exc)
        self._repos.add_risk_event(
            self.mode,
            Severity.WARNING,
            "settlement_error",
            f"{type(exc).__name__}: {exc}"[:300],
            ts=self._clock.now(),
        )

    async def _timeout_guard(self, order_id: str, contract_id: int, delay: float) -> None:
        await asyncio.sleep(delay)
        row = self._repos.get_order(order_id)
        if row is None or row["state"] not in SETTLABLE:
            return
        self._repos.update_order(order_id, state=OrderState.RECONCILIATION_PENDING)
        self._repos.add_reconciliation(
            self.mode, "settlement_timeout", "no settlement before timeout", contract_id
        )
        for _ in range(RESOLVE_ATTEMPTS):
            try:
                await self.resolve(order_id, contract_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep trying: the order must not stay blocked silently
                log.exception("resolve_failed")
                self._repos.add_risk_event(
                    self.mode,
                    Severity.WARNING,
                    "settlement_error",
                    f"resolve failed: {type(exc).__name__}: {exc}"[:300],
                    ts=self._clock.now(),
                )
            row = self._repos.get_order(order_id)
            if row is None or row["state"] not in SETTLABLE:
                return  # settled (or given up on) by resolve()
            await asyncio.sleep(RESOLVE_RETRY_S)

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
        if update.is_sold or update.status in CLOSED_STATUSES:
            await self.settle(order_id, update)
        else:
            await self._maybe_take_profit(order_id, update)

    async def resolve(self, order_id: str, contract_id: int) -> None:
        """Query the account for a contract's final state and settle / resume watching."""
        try:
            update = await self._client.contract_status(contract_id)
        except (ConnectionLost, TimeoutError):
            return  # transient: stays RECONCILIATION_PENDING; retried on reconnect / next sweep
        except DerivError as exc:
            self._unknown(order_id, contract_id, f"{exc.code}: {exc.message}")
            return
        if update is None:
            self._unknown(order_id, contract_id, "broker returned no contract")
            return
        self._unknown_counts.pop(contract_id, None)
        if update.is_sold or update.status in CLOSED_STATUSES:
            await self.settle(order_id, update, reconciled=True)
        else:
            self._repos.update_order(order_id, state=OrderState.OPEN)
            await self.watch(order_id, contract_id)

    def _unknown(self, order_id: str, contract_id: int, why: str) -> None:
        """The broker cannot describe this contract. After a few attempts, stop letting it block
        trading: mark the order FAILED (outcome NOT recorded) and say so loudly. The balance is
        refreshed from the broker, so drawdown protection still sees any real loss."""
        n = self._unknown_counts.get(contract_id, 0) + 1
        self._unknown_counts[contract_id] = n
        self._repos.add_reconciliation(self.mode, "contract_unknown", why, contract_id)
        if n < UNKNOWN_LIMIT:
            return
        self._unknown_counts.pop(contract_id, None)
        self._repos.update_order(
            order_id, state=OrderState.FAILED, error=f"outcome unknown to broker: {why}"[:200]
        )
        self._repos.add_risk_event(
            self.mode,
            Severity.CRITICAL,
            "outcome_unknown",
            f"contract {contract_id}: {why}. Position released; result NOT recorded in P&L.",
            ts=self._clock.now(),
        )
        self._spawn(self._refresh_balance())

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
        win_amount = row["win_amount"]
        # 'target hit' is the event the model predicts; it drives the rolling monitor and promotion.
        target_hit = (
            profit >= Decimal(str(win_amount)) * Decimal("0.9") if win_amount else profit > 0
        )
        state = OrderState.WON if won else OrderState.LOST
        contract_id = int(row["contract_id"])
        now = self._clock.now()
        with self._repos.db.transaction():
            self._repos.update_order(order_id, state=state)
            self._repos.settle_trade(
                contract_id,
                state,
                profit,
                update.entry_spot,
                update.exit_spot,
                now,
                target_hit=target_hit,
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
        hold = self._holds.pop(contract_id, None)
        if hold is not None and hold is not asyncio.current_task():
            hold.cancel()
        self._plans.pop(contract_id, None)
        self._selling.discard(contract_id)
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
        for h in list(self._holds.values()):
            h.cancel()
        self._holds.clear()
        for task in list(self._tasks):
            task.cancel()
        for sub in list(self._subs.values()):
            with contextlib.suppress(Exception):
                await self._client.unsubscribe(sub)
        self._subs.clear()
