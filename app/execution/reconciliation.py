"""Startup / reconnect / ambiguous-buy reconciliation against the broker's account state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.clock import Clock
from app.deriv.client import DerivClient
from app.deriv.protocol import ConnectionLost, DerivError, dec
from app.execution.idempotency import match_portfolio_contract
from app.execution.settlement import SettlementTracker
from app.models.schemas import Mode, OrderState, Severity
from app.products import product_for_contract_type
from app.risk.manager import RiskManager
from app.storage.database import utc_iso
from app.storage.repositories import Repositories


@dataclass
class ReconReport:
    resumed: list[int] = field(default_factory=list)
    settled: list[int] = field(default_factory=list)
    adopted_unknown: list[int] = field(default_factory=list)
    failed_orders: list[str] = field(default_factory=list)
    ok: bool = True


class Reconciler:
    def __init__(
        self,
        client: DerivClient,
        repos: Repositories,
        risk: RiskManager,
        tracker: SettlementTracker,
        clock: Clock,
    ) -> None:
        self._client = client
        self._repos = repos
        self._risk = risk
        self._tracker = tracker
        self._clock = clock
        self.grace_s = 1.5  # settle time before looking for an ambiguous buy

    @property
    def mode(self) -> Mode:
        return self._risk.mode

    async def startup(self) -> ReconReport:
        """Compare SQLite with the broker's open positions BEFORE trading is enabled."""
        report = ReconReport()
        try:
            portfolio = await self._client.portfolio()
        except (DerivError, ConnectionLost, TimeoutError) as exc:
            report.ok = False
            self._repos.add_reconciliation(self.mode, "startup_failed", type(exc).__name__)
            return report
        api_by_id = {int(c["contract_id"]): c for c in portfolio if "contract_id" in c}
        active = self._repos.active_orders(self.mode)
        known = {int(o["contract_id"]) for o in active if o["contract_id"] is not None}

        for order in active:
            oid = str(order["order_id"])
            cid = order["contract_id"]
            if cid is None:
                await self._resolve_without_contract(order, portfolio, known, report)
            elif int(cid) in api_by_id:
                self._repos.update_order(oid, state=OrderState.OPEN)
                await self._tracker.watch(oid, int(cid))
                report.resumed.append(int(cid))
            else:
                # DB says open, broker says not open => it closed while we were away.
                self._repos.update_order(oid, state=OrderState.RECONCILIATION_PENDING)
                self._repos.add_reconciliation(
                    self.mode, "db_open_api_closed", "resolving final state", int(cid)
                )
                await self._tracker.resolve(oid, int(cid))
                report.settled.append(int(cid))

        for cid, c in api_by_id.items():
            if cid not in known and self._repos.get_order_by_contract(cid) is None:
                await self._adopt_unknown(cid, c)
                report.adopted_unknown.append(cid)
        self._repos.add_reconciliation(
            self.mode,
            "startup_done",
            f"resumed={len(report.resumed)} settled={len(report.settled)} "
            f"unknown={len(report.adopted_unknown)} failed={len(report.failed_orders)}",
        )
        return report

    async def _resolve_without_contract(
        self, order: Any, portfolio: list[dict[str, Any]], known: set[int], report: ReconReport
    ) -> None:
        oid = str(order["order_id"])
        sent = order["order_sent_at"]
        if sent is None:
            # The buy request was never sent: safe to fail.
            self._repos.update_order(oid, state=OrderState.FAILED, error="abandoned before send")
            report.failed_orders.append(oid)
            return
        match = match_portfolio_contract(order, portfolio, known, sent_at=float(sent))
        if match is None:
            self._repos.update_order(
                oid, state=OrderState.FAILED, error="ambiguous buy: no matching broker contract"
            )
            self._repos.add_reconciliation(self.mode, "ambiguous_buy_no_contract", oid)
            report.failed_orders.append(oid)
            return
        await self._adopt_order(order, match)
        report.resumed.append(int(match["contract_id"]))

    async def _adopt_order(self, order: Any, contract: dict[str, Any]) -> None:
        cid = int(contract["contract_id"])
        oid = str(order["order_id"])
        self._repos.update_order(oid, state=OrderState.OPEN, contract_id=cid)
        self._repos.insert_trade(
            contract_id=cid,
            order_id=oid,
            mode=self.mode,
            symbol=str(order["symbol"]),
            direction=str(order["direction"]),
            stake=Decimal(str(order["stake"])),
            buy_price=dec(contract.get("buy_price", order["stake"])),
            payout=None if contract.get("payout") is None else dec(contract["payout"]),
            probability=order["probability"],
            break_even=order["break_even"],
            ts=self._clock.now(),
            product=str(order["product"]),
        )
        self._repos.add_reconciliation(self.mode, "buy_adopted", f"order {oid}", cid)
        self._risk.on_purchase()
        await self._tracker.watch(oid, cid)

    async def _adopt_unknown(self, cid: int, contract: dict[str, Any]) -> None:
        """Adopt a broker position this database has never seen (never double-trade)."""
        buy_price = dec(contract.get("buy_price", "0"))
        oid = f"external-{cid}"
        ctype = str(contract.get("contract_type", "?"))
        found = product_for_contract_type(ctype)
        product = found.value if found is not None else "rise_fall"
        now = utc_iso(self._clock.now())
        self._repos.db.execute(
            "INSERT OR IGNORE INTO orders(order_id,signal_id,mode,symbol,direction,stake,state,"
            "contract_id,created_at,updated_at,product,contract_type) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                oid,
                f"external:{cid}",
                self.mode.value,
                str(contract.get("symbol", "?")),
                str(contract.get("contract_type", "?")),
                str(buy_price),
                OrderState.OPEN.value,
                cid,
                now,
                now,
                product,
                ctype,
            ),
        )
        self._repos.insert_trade(
            contract_id=cid,
            order_id=oid,
            mode=self.mode,
            symbol=str(contract.get("symbol", "?")),
            direction=str(contract.get("contract_type", "?")),
            stake=buy_price,
            buy_price=buy_price,
            payout=None if contract.get("payout") is None else dec(contract["payout"]),
            probability=None,
            break_even=None,
            ts=self._clock.now(),
            product=product,
        )
        self._repos.add_reconciliation(self.mode, "unknown_position", "adopted", cid)
        self._repos.add_risk_event(
            self.mode,
            Severity.WARNING,
            "unknown_position",
            f"broker reports contract {cid} that this database did not know",
            details=contract,
        )
        await self._tracker.watch(oid, cid)

    async def reconcile_pending(self) -> int:
        """Resolve RECONCILIATION_PENDING orders (after reconnect or a settlement timeout)."""
        done = 0
        for order in self._repos.orders_in_state(self.mode, OrderState.RECONCILIATION_PENDING):
            if order["contract_id"] is not None:
                await self._tracker.resolve(str(order["order_id"]), int(order["contract_id"]))
                done += 1
            else:
                await self.resolve_ambiguous_buy(str(order["order_id"]))
                done += 1
        return done

    async def resolve_ambiguous_buy(
        self, order_id: str, *, grace_s: float | None = None, retries: int = 2
    ) -> bool:
        """A buy timed out / the socket dropped mid-buy. Never retry the buy: look at the account.
        Returns True if a contract was found and adopted."""
        order = self._repos.get_order(order_id)
        if order is None or order["contract_id"] is not None:
            return order is not None
        sent = float(order["order_sent_at"] or 0)
        for attempt in range(retries + 1):
            await asyncio.sleep((grace_s if grace_s is not None else self.grace_s) * (attempt + 1))
            try:
                portfolio = await self._client.portfolio()
            except (DerivError, ConnectionLost, TimeoutError):
                continue
            known = {
                int(o["contract_id"])
                for o in self._repos.active_orders(self.mode)
                if o["contract_id"] is not None
            }
            match = match_portfolio_contract(order, portfolio, known, sent_at=sent)
            if match is not None:
                await self._adopt_order(order, match)
                return True
            if attempt == retries:
                self._repos.update_order(
                    order_id, state=OrderState.FAILED, error="ambiguous buy: no contract found"
                )
                self._repos.add_reconciliation(self.mode, "ambiguous_buy_no_contract", order_id)
                return False
        return (
            False  # portfolio unreachable: stays RECONCILIATION_PENDING (still counts as exposure)
        )
