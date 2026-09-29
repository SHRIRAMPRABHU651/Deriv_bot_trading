"""Executor: signal -> risk -> proposal -> risk (edge gate) -> buy -> confirm -> track."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from app.clock import Clock
from app.config import AppConfig
from app.deriv.client import DerivClient
from app.deriv.protocol import ConnectionLost, DerivError, ProtocolError
from app.execution.reconciliation import Reconciler
from app.execution.settlement import SettlementTracker
from app.metrics.latency import LatencyTracker
from app.models.schemas import OrderState, Severity, Signal
from app.risk.manager import RiskManager
from app.storage.repositories import Repositories

log = logging.getLogger("derivbot.executor")


class Executor:
    def __init__(
        self,
        client: DerivClient,
        risk: RiskManager,
        repos: Repositories,
        config: AppConfig,
        clock: Clock,
        latency: LatencyTracker,
        tracker: SettlementTracker,
        reconciler: Reconciler,
        *,
        refresh_balance: Callable[[], Awaitable[None]],
    ) -> None:
        self._client = client
        self._risk = risk
        self._repos = repos
        self._cfg = config
        self._clock = clock
        self._latency = latency
        self._tracker = tracker
        self._reconciler = reconciler
        self._refresh_balance = refresh_balance
        self._background: set[asyncio.Task[object]] = set()
        self.stats = {
            "signals": 0,
            "accepted": 0,
            "proposals": 0,
            "buys": 0,
            "failed": 0,
            "stale_dropped": 0,
        }

    async def process(self, signal: Signal) -> None:
        self.stats["signals"] += 1
        mode = self._risk.mode
        now = self._clock.time()
        if now - signal.created_at > self._cfg.trading.max_signal_age_s:
            self.stats["stale_dropped"] += 1
            self._repos.add_risk_event(
                mode,
                Severity.WARNING,
                "signal_stale",
                f"signal older than {self._cfg.trading.max_signal_age_s}s dropped",
                symbol=signal.symbol,
                signal_id=signal.signal_id,
                ts=self._clock.now(),
            )
            return
        acct = self._risk.account
        if acct.balance_ts is None or now - acct.balance_ts > self._cfg.risk.balance_max_age_s / 2:
            # On failure risk.evaluate() rejects on stale/unknown balance.
            with contextlib.suppress(DerivError, ConnectionLost, TimeoutError):
                await self._refresh_balance()

        decision = self._risk.evaluate(signal)
        if not decision.approved or decision.order_id is None or decision.stake is None:
            return
        self.stats["accepted"] += 1
        order_id = decision.order_id
        self._latency.record(
            mode,
            "tick_to_signal",
            (signal.created_at - signal.tick_received_at) * 1000.0,
            self._clock.now(),
        )

        try:
            proposal = await asyncio.wait_for(
                self._client.proposal(signal.symbol, signal.direction.value, decision.stake),
                self._cfg.deriv.proposal_timeout_s + 1.0,
            )
            self.stats["proposals"] += 1
        except (TimeoutError, DerivError, ConnectionLost, ProtocolError) as exc:
            self._fail(
                order_id, signal, "proposal_failed", f"{type(exc).__name__}: {exc}", refund=True
            )
            return

        permit, _verdict = self._risk.authorize_buy(order_id, signal, proposal)
        if permit is None:
            return

        sent = self._clock.time()
        self._repos.update_order(order_id, order_sent_at=sent)
        self._latency.record(
            mode, "signal_to_order", (sent - signal.created_at) * 1000.0, self._clock.now()
        )
        try:
            result = await self._client.buy(permit)
        except DerivError as exc:
            self._fail(order_id, signal, "buy_rejected", f"{exc.code}: {exc.message}", refund=False)
            return
        except (TimeoutError, ConnectionLost):
            # The buy may have succeeded. NEVER retry blindly: reconcile against the account.
            self._repos.update_order(order_id, state=OrderState.RECONCILIATION_PENDING)
            self._repos.add_risk_event(
                mode,
                Severity.CRITICAL,
                "ambiguous_buy",
                "buy outcome unknown; reconciling instead of retrying",
                symbol=signal.symbol,
                signal_id=signal.signal_id,
                ts=self._clock.now(),
            )
            self._spawn(self._reconciler.resolve_ambiguous_buy(order_id))
            return
        except ProtocolError as exc:
            self._repos.update_order(
                order_id,
                state=OrderState.RECONCILIATION_PENDING,
                error=f"unparseable buy response: {exc}",
            )
            self._spawn(self._reconciler.resolve_ambiguous_buy(order_id))
            return

        confirmed = self._clock.time()
        order_row = self._repos.get_order(order_id)
        break_even = None if order_row is None else order_row["break_even"]
        self.stats["buys"] += 1
        self._latency.record(
            mode, "order_to_confirmation", (confirmed - sent) * 1000.0, self._clock.now()
        )
        with self._repos.db.transaction():
            self._repos.update_order(
                order_id,
                state=OrderState.OPEN,
                contract_id=result.contract_id,
                confirmed_at=confirmed,
                ask_price=result.buy_price,
                payout=result.payout,
            )
            self._repos.insert_trade(
                contract_id=result.contract_id,
                order_id=order_id,
                mode=mode,
                symbol=signal.symbol,
                direction=signal.direction.value,
                stake=decision.stake,
                buy_price=result.buy_price,
                payout=result.payout,
                probability=signal.probability,
                break_even=break_even,
                ts=self._clock.now(),
            )
            self._repos.set_signal_status(signal.signal_id, "EXECUTED")
        self._risk.on_purchase()
        if result.balance_after is not None:
            self._risk.update_balance(result.balance_after, self._risk.account.currency)
        await self._tracker.watch(order_id, result.contract_id)

    def _fail(self, order_id: str, signal: Signal, rule: str, msg: str, *, refund: bool) -> None:
        self.stats["failed"] += 1
        self._repos.update_order(order_id, state=OrderState.FAILED, error=f"{rule}: {msg}")
        self._repos.set_signal_status(signal.signal_id, "FAILED", f"{rule}: {msg}")
        self._repos.add_risk_event(
            self._risk.mode,
            Severity.WARNING,
            rule,
            msg,
            symbol=signal.symbol,
            signal_id=signal.signal_id,
            ts=self._clock.now(),
        )
        if refund:
            self._risk.state.refund_trade(self._risk.mode)

    def _spawn(self, coro: Awaitable[object]) -> None:
        task: asyncio.Task[object] = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def pending_tasks(self) -> list[asyncio.Task[object]]:
        return list(self._background)

    async def close(self) -> None:
        for t in list(self._background):
            t.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
