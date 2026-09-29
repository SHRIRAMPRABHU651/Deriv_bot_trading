"""Test harness building a RiskManager on an in-memory database with a controllable clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.clock import FakeClock
from app.config import AppConfig, RiskProfile
from app.metrics.latency import FeedHealth, LatencyTracker
from app.models.schemas import (
    Direction,
    Mode,
    ModelStatus,
    OrderState,
    Proposal,
    Signal,
)
from app.products import Product, ProductSpec
from app.risk.manager import ModelInfo, RiskManager
from app.risk.state import AccountState
from app.storage.database import Database
from app.storage.repositories import Repositories

START = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)  # a Wednesday


@dataclass
class Harness:
    db: Database
    repos: Repositories
    clock: FakeClock
    config: AppConfig
    account: AccountState
    feed: FeedHealth
    latency: LatencyTracker
    risk: RiskManager
    flags: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def signal(
        self,
        symbol: str = "R_100",
        p: float | None = 0.70,
        direction: Direction = Direction.CALL,
        model_version: str | None = "m1",
        sid: str | None = None,
        product: str | None = None,
    ) -> Signal:
        self.seq += 1
        now = self.clock.time()
        self.feed.on_tick(symbol, now)
        return Signal(
            signal_id=sid or f"sig-{self.seq}",
            symbol=symbol,
            direction=direction,
            tick_epoch=self.seq,
            strategy="ml",
            strategy_version="1",
            model_version=model_version,
            probability=p,
            horizon_ticks=5,
            tick_received_at=now,
            created_at=now,
            product=product or self.config.trading.product.product.value,
            entry_price=100.0,
        )

    def new_risk(self) -> RiskManager:
        """A second RiskManager on the same database = process restart."""
        return RiskManager(
            self.repos,
            self.config,
            self.clock,
            account=AccountState(
                mode=self.account.mode,
                account_id="DOT1",
                balance=self.account.balance,
                balance_ts=self.clock.time(),
                verified_type=self.account.verified_type,
            ),
            feed=self.feed,
            latency=self.latency,
            model_getter=lambda: self.flags["model"],
            running_getter=lambda: bool(self.flags["running"]),
        )

    def set_balance(self, value: str) -> None:
        self.risk.update_balance(Decimal(value), "USD")

    def add_active_order(
        self, symbol: str, stake: str = "10", state: OrderState = OrderState.OPEN
    ) -> str:
        sig = self.signal(symbol)
        oid = f"ord-{self.seq}"
        self.repos.insert_order(
            order_id=oid,
            signal=sig,
            mode=self.risk.mode,
            stake=Decimal(stake),
            break_even=None,
            ts=self.clock.now(),
        )
        self.repos.update_order(oid, state=state)
        return oid


def proposal(
    stake: str = "10", payout: str = "19.5", pid: str = "p1", ask: str | None = None
) -> Proposal:
    return Proposal(
        proposal_id=pid,
        ask_price=Decimal(ask or stake),
        payout=Decimal(payout),
        min_stake=Decimal("0.35"),
        max_stake=Decimal("2000"),
    )


def make_harness(
    *,
    mode: Mode = Mode.DEMO,
    balance: str = "1000",
    profile: dict[str, Any] | None = None,
    risk_cfg: dict[str, Any] | None = None,
    model_status: ModelStatus = ModelStatus.DEMO_VALIDATING,
    start: datetime = START,
    spec: ProductSpec | None = None,
) -> Harness:
    config = AppConfig()
    config.trading.product = spec or ProductSpec(product=Product.RISE_FALL, horizon_ticks=5)
    prof_updates: dict[str, Any] = {
        "cooldown_seconds": 0.0,
        "max_open_trades": 10,
        "max_open_trades_per_symbol": 1,
        "max_total_exposure_percent": Decimal("0.5"),
        "max_group_exposure_percent": Decimal("0.5"),
        "max_trades_per_day": 100,
    }
    prof_updates.update(profile or {})
    base = config.risk.profile(mode).model_dump()
    base.update(prof_updates)
    new_prof = RiskProfile(**base)
    if mode is Mode.DEMO:
        config.risk.demo = new_prof
    else:
        config.risk.live = new_prof
    for k, v in (risk_cfg or {}).items():
        setattr(config.risk, k, v)
    db = Database(":memory:")
    repos = Repositories(db)
    clock = FakeClock(start)
    account = AccountState(
        mode=mode, account_id="DOT1", verified_type="demo" if mode is Mode.DEMO else "real"
    )
    feed = FeedHealth()
    latency = LatencyTracker(repos)
    flags: dict[str, Any] = {
        "running": True,
        "model": ModelInfo("m1", model_status, "v1"),
    }
    risk = RiskManager(
        repos,
        config,
        clock,
        account=account,
        feed=feed,
        latency=latency,
        model_getter=lambda: flags["model"],
        running_getter=lambda: bool(flags["running"]),
    )
    h = Harness(db, repos, clock, config, account, feed, latency, risk, flags)
    risk.update_balance(Decimal(balance), "USD")
    return h
