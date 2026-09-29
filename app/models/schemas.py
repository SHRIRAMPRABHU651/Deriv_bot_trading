"""Core domain types shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class Mode(StrEnum):
    DEMO = "demo"
    LIVE = "live"


class Direction(StrEnum):
    """Bullish (CALL) / bearish (PUT) view. NEUTRAL = non-directional (Accumulators)."""

    CALL = "CALL"
    PUT = "PUT"
    NEUTRAL = "NEUTRAL"


class ModelStatus(StrEnum):
    UNTRAINED = "UNTRAINED"
    BACKTESTED = "BACKTESTED"
    WALK_FORWARD_VALID = "WALK_FORWARD_VALID"
    CALIBRATED = "CALIBRATED"
    DEMO_VALIDATING = "DEMO_VALIDATING"
    PROMOTABLE = "PROMOTABLE"
    REJECTED = "REJECTED"


class OrderState(StrEnum):
    APPROVED = "APPROVED"  # passed risk, stake reserved, nothing sent to Deriv yet
    PROPOSED = "PROPOSED"
    BOUGHT = "BOUGHT"
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"
    EXPIRED = "EXPIRED"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    RECONCILED = "RECONCILED"
    FAILED = "FAILED"


# States that still carry worst-case exposure (the stake can still be lost).
ACTIVE_STATES: tuple[OrderState, ...] = (
    OrderState.APPROVED,
    OrderState.PROPOSED,
    OrderState.BOUGHT,
    OrderState.OPEN,
    OrderState.RECONCILIATION_PENDING,
)


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    epoch: int
    quote: float
    received_at: float  # wall-clock seconds when the tick arrived locally
    tick_id: str | None = None


@dataclass(frozen=True, slots=True)
class Signal:
    signal_id: str
    symbol: str
    direction: Direction
    tick_epoch: int
    strategy: str
    strategy_version: str
    model_version: str | None
    probability: float | None  # calibrated P(this direction wins); None => cannot pass edge gate
    horizon_ticks: int
    tick_received_at: float
    created_at: float
    features_version: str | None = None
    product: str = "rise_fall"
    entry_price: float | None = None


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    ask_price: Decimal
    payout: Decimal
    spot: float | None = None
    longcode: str = ""
    min_stake: Decimal | None = None
    max_stake: Decimal | None = None
    commission: float | None = None
    contracts: float | None = None  # implied number of contracts (turbos / vanillas)
    barrier_abs: float | None = None  # absolute barrier / strike (turbos / vanillas)
    barrier_pct_per_tick: float | None = None  # accumulator tick barrier as a FRACTION of spot
    max_ticks: int | None = None  # accumulator maximum duration

    @property
    def payout_ratio(self) -> Decimal:
        """R = payout / price paid. Break-even probability is 1 / R."""
        return self.payout / self.ask_price


@dataclass(frozen=True, slots=True)
class BuyResult:
    contract_id: int
    buy_price: Decimal
    payout: Decimal
    balance_after: Decimal | None
    purchase_time: int
    start_time: int | None = None
    shortcode: str = ""
    transaction_id: int | None = None


@dataclass(frozen=True, slots=True)
class ContractUpdate:
    contract_id: int
    is_sold: bool
    is_expired: bool
    status: str  # "open" | "won" | "lost" | "sold" | "cancelled"
    profit: Decimal | None
    buy_price: Decimal | None
    sell_price: Decimal | None
    payout: Decimal | None
    entry_spot: float | None = None
    exit_spot: float | None = None
    date_expiry: int | None = None
    symbol: str | None = None
    contract_type: str | None = None
    purchase_time: int | None = None
    bid_price: Decimal | None = None
    valid_to_sell: bool = False


@dataclass(frozen=True, slots=True)
class BalanceInfo:
    balance: Decimal
    currency: str
    loginid: str | None = None


@dataclass(frozen=True, slots=True)
class AccountInfo:
    account_id: str
    account_type: str  # "demo" | "real"
    balance: Decimal | None = None
    currency: str | None = None


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    rule: str = ""
    reason: str = ""
    stake: Decimal | None = None
    order_id: str | None = None
    details: dict[str, object] = field(default_factory=dict)
