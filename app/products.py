"""Trade-type ("product") layer: Rise/Fall, Multipliers, Accumulators, Turbos, Vanillas.

Every product is reduced to a **bounded two-outcome bet** so ONE edge gate covers all of them:

    win  event  -> the trade reaches its profit target       (net profit  W, in currency)
    loss event  -> anything else, valued CONSERVATIVELY as   (net loss    L, in currency)
    break-even probability = L / (W + L)         EV = p*W - (1-p)*L

W and L are computed from the ACTUAL proposal (ask price, fees, limit orders, contracts) at trade
time - never from constants. For Rise/Fall this collapses to the classic p* = 1/R.

Product mechanics (field names follow the Deriv proposal/buy schemas, see docs/API_NOTES.md):
  MULTIPLIER  MULTUP/MULTDOWN. Native `limit_order` take_profit + stop_loss close it server-side.
              win = take-profit reached before stop-loss within `horizon_ticks`. The bot closes any
              position still open after the hold cap (sell at market) and values that as a loss.
  ACCUMULATOR ACCU. Stake grows `growth_rate` per tick while each tick stays inside the barrier;
              native take_profit = the growth after `horizon_ticks` survived ticks.
              win = survive `horizon_ticks` ticks. Knock-out loses the stake.
  TURBO       TURBOSLONG/TURBOSSHORT with a knock-out barrier `barrier_offset` away. No native TP:
              the bot sells at market once profit >= take_profit_pct (bot-managed exit). Expires at
              the proposal duration. win = profit target before knock-out within the horizon.
  VANILLA     VANILLALONGCALL/VANILLALONGPUT, held to expiry (intrinsic payoff, no pricing model
              needed). win = terminal move >= `needed_move` (the move that pays `target_pct`).
  RISE_FALL   CALL/PUT on ticks (binary). win = strictly beyond the entry tick at expiry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict, model_validator

from app.models.schemas import Direction, Proposal

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class Product(StrEnum):
    RISE_FALL = "rise_fall"
    MULTIPLIER = "multiplier"
    ACCUMULATOR = "accumulator"
    TURBO = "turbo"
    VANILLA = "vanilla"


DIRECTIONAL = (Product.RISE_FALL, Product.MULTIPLIER, Product.TURBO, Product.VANILLA)


class ProductSpec(BaseModel):
    """Everything that defines a trade. Stored (as JSON) in the model artifact, so a model is only
    ever used with the spec it was trained for."""

    model_config = ConfigDict(frozen=True)

    product: Product = Product.MULTIPLIER
    horizon_ticks: int = 20  # label horizon N == expiry (rise/fall, turbo, vanilla) or hold cap
    duration_unit: str = "t"  # proposal duration unit for expiring products
    tick_seconds: float = 2.0  # seconds per tick of the traded symbol (R_100 = 2, 1HZ100V = 1)
    max_fee_pct: float = 0.05  # max (ask - stake)/stake tolerated; Rise/Fall is forced to 0
    tolerance: float = 0.10  # allowed relative deviation of live terms vs the trained assumption
    hold_grace_s: float = 2.0  # extra seconds beyond horizon*tick_seconds before a hold-cap sell
    assumed_fee_pct: float = 0.0  # research: fee as a fraction of stake (multiplier/turbo/vanilla)
    # --- rise/fall (research assumption only; live uses the proposal) ---
    payout_ratio: float = 1.95
    # --- multiplier ---
    multiplier: int = 20
    take_profit_pct: float = 0.5  # of stake (multiplier & turbo)
    stop_loss_pct: float = 0.5  # of stake, must be <= 1 (multiplier)
    # --- accumulator ---
    growth_rate: float = 0.01
    barrier_pct: float = 0.0006  # per-tick barrier as a FRACTION of spot (trained assumption)
    # --- turbo ---
    barrier_offset: float = 0.5  # knock-out distance in PRICE units
    # --- vanilla ---
    strike_offset: float = 0.0  # signed strike offset from spot in PRICE units (CALL: +, PUT: -)
    target_pct: float = 0.5  # profit target on the premium
    needed_move: float = (
        1.0  # terminal move (price units) that pays target_pct (trained assumption)
    )

    @model_validator(mode="after")
    def _sane(self) -> ProductSpec:
        if self.horizon_ticks < 1:
            raise ValueError("horizon_ticks must be >= 1")
        if not 0 < self.stop_loss_pct <= 1:
            raise ValueError("stop_loss_pct must be in (0, 1]: a loss can never exceed the stake")
        if self.take_profit_pct <= 0 or self.target_pct <= 0:
            raise ValueError("profit targets must be positive")
        if self.growth_rate <= 0 or self.barrier_pct <= 0 or self.barrier_offset <= 0:
            raise ValueError("growth_rate, barrier_pct and barrier_offset must be positive")
        return self

    def label_key(self) -> tuple[object, ...]:
        """The fields that define WHAT a model predicts; a model is only valid for an equal key."""
        p = self.product
        base: tuple[object, ...] = (p.value, self.horizon_ticks)
        if p is Product.MULTIPLIER:
            return (*base, self.multiplier, self.take_profit_pct, self.stop_loss_pct)
        if p is Product.ACCUMULATOR:
            return (*base, self.growth_rate, self.barrier_pct)
        if p is Product.TURBO:
            return (*base, self.barrier_offset, self.take_profit_pct)
        if p is Product.VANILLA:
            return (*base, self.needed_move)
        return base

    @property
    def fee_allowance(self) -> float:
        return 0.0 if self.product is Product.RISE_FALL else self.max_fee_pct

    @property
    def directional(self) -> bool:
        return self.product in DIRECTIONAL

    @property
    def expires(self) -> bool:
        return self.product in (Product.RISE_FALL, Product.TURBO, Product.VANILLA)

    @property
    def separate_bear_head(self) -> bool:
        """Multiplier/turbo/vanilla wins are not complementary, so bears need their own model."""
        return self.product in (Product.MULTIPLIER, Product.TURBO, Product.VANILLA)


# ---------------------------------------------------------------- contract types / requests
_CONTRACT_TYPES: dict[Product, dict[Direction, str]] = {
    Product.RISE_FALL: {Direction.CALL: "CALL", Direction.PUT: "PUT"},
    Product.MULTIPLIER: {Direction.CALL: "MULTUP", Direction.PUT: "MULTDOWN"},
    Product.TURBO: {Direction.CALL: "TURBOSLONG", Direction.PUT: "TURBOSSHORT"},
    Product.VANILLA: {Direction.CALL: "VANILLALONGCALL", Direction.PUT: "VANILLALONGPUT"},
    Product.ACCUMULATOR: {Direction.NEUTRAL: "ACCU"},
}


def directions(spec: ProductSpec) -> tuple[Direction, ...]:
    return tuple(_CONTRACT_TYPES[spec.product])


def product_for_contract_type(ctype: str) -> Product | None:
    for prod, mapping in _CONTRACT_TYPES.items():
        if ctype in mapping.values():
            return prod
    return None


def contract_type(spec: ProductSpec, direction: Direction) -> str:
    try:
        return _CONTRACT_TYPES[spec.product][direction]
    except KeyError:
        raise ValueError(f"{spec.product.value} has no {direction.value} contract") from None


def _money(x: float | Decimal) -> float:
    """Currency amounts are sent with cent precision, rounded DOWN (never more than intended)."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def _offset(value: float, sign: str) -> str:
    return f"{sign}{Decimal(str(value)):f}"


def accumulator_target(spec: ProductSpec, stake: Decimal) -> float:
    """Take-profit amount = growth after `horizon_ticks` surviving ticks."""
    growth = (1.0 + spec.growth_rate) ** spec.horizon_ticks - 1.0
    return _money(float(stake) * growth)


def proposal_params(spec: ProductSpec, direction: Direction, stake: Decimal) -> dict[str, Any]:
    """Product-specific proposal fields (symbol/currency/amount/basis are added by the caller)."""
    ctype = contract_type(spec, direction)
    p: dict[str, Any] = {"contract_type": ctype}
    if spec.product is Product.RISE_FALL:
        p.update(duration=spec.horizon_ticks, duration_unit=spec.duration_unit)
    elif spec.product is Product.MULTIPLIER:
        p.update(
            multiplier=spec.multiplier,
            limit_order={
                "take_profit": _money(float(stake) * spec.take_profit_pct),
                "stop_loss": _money(float(stake) * spec.stop_loss_pct),
            },
        )
    elif spec.product is Product.ACCUMULATOR:
        p.update(
            growth_rate=spec.growth_rate,
            limit_order={"take_profit": accumulator_target(spec, stake)},
        )
    elif spec.product is Product.TURBO:
        sign = "-" if direction is Direction.CALL else "+"
        p.update(
            duration=spec.horizon_ticks,
            duration_unit=spec.duration_unit,
            barrier=_offset(spec.barrier_offset, sign),
        )
    elif spec.product is Product.VANILLA:
        sign = "+" if direction is Direction.CALL else "-"
        p.update(
            duration=spec.horizon_ticks,
            duration_unit=spec.duration_unit,
            barrier=_offset(abs(spec.strike_offset), sign),
        )
    return p


# ---------------------------------------------------------------- payoff terms (edge gate input)
@dataclass(frozen=True)
class Terms:
    ok: bool
    win: float = 0.0  # net profit if the target is reached (currency)
    loss: float = 0.0  # net loss otherwise, valued conservatively (currency)
    reason: str = ""

    @property
    def break_even(self) -> float:
        return self.loss / (self.win + self.loss)


def _reject(reason: str) -> Terms:
    return Terms(False, reason=reason)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def assumed_terms(spec: ProductSpec) -> tuple[float, float]:
    """(win, loss) per unit stake for RESEARCH; the live bot uses `payoff_terms` on the proposal."""
    fee = spec.assumed_fee_pct
    if spec.product is Product.RISE_FALL:
        return spec.payout_ratio - 1.0, 1.0
    if spec.product is Product.MULTIPLIER:
        return spec.take_profit_pct - fee, spec.stop_loss_pct + fee
    if spec.product is Product.ACCUMULATOR:
        return (1.0 + spec.growth_rate) ** spec.horizon_ticks - 1.0, 1.0
    if spec.product is Product.TURBO:
        return spec.take_profit_pct - fee, 1.0
    return spec.target_pct - fee, 1.0  # vanilla


def assumed_break_even(spec: ProductSpec) -> float:
    w, l_ = assumed_terms(spec)
    return l_ / (w + l_)


def payoff_terms(
    spec: ProductSpec,
    direction: Direction,
    proposal: Proposal,
    stake: Decimal,
    spot: float | None,
) -> Terms:
    """Net win/loss from the ACTUAL proposal, and a check that the live contract still matches the
    terms the model was trained for (barrier distance, needed move, fees). Fails CLOSED."""
    ask = float(proposal.ask_price)
    stk = float(stake)
    if ask <= 0 or stk <= 0:
        return _reject("non-positive price")
    fee = max(ask - stk, 0.0)
    prod = spec.product
    # (Rise/Fall has no fees: a higher ask is slippage, bounded by the risk manager's ceiling.)
    if prod is not Product.RISE_FALL and fee > stk * spec.fee_allowance + 1e-9:
        return _reject(f"fee {fee:.4f} above allowance {spec.fee_allowance:.2%} of stake")
    tol = spec.tolerance

    if prod is Product.RISE_FALL:
        payout = float(proposal.payout)
        if payout <= ask:
            return _reject("payout must exceed price paid")
        return Terms(True, payout - ask, ask)

    if prod is Product.MULTIPLIER:
        tp = _money(stk * spec.take_profit_pct)
        sl = _money(stk * spec.stop_loss_pct)
        commission = _num(proposal.commission) or 0.0
        cost = max(fee, commission)  # commission's unit is unclear in the docs: assume the worse
        if cost > stk * spec.fee_allowance + 1e-9:
            return _reject(f"commission {cost:.4f} above allowance")
        win, loss = tp - cost, sl + cost
        if win <= 0:
            return _reject("take-profit does not cover fees")
        return Terms(True, win, loss)

    if prod is Product.ACCUMULATOR:
        actual = _num(proposal.barrier_pct_per_tick)
        if actual is None:
            return _reject("accumulator barrier size unavailable: cannot verify trained terms")
        if actual < spec.barrier_pct * (1.0 - tol):
            return _reject(f"barrier {actual:.6f} narrower than trained {spec.barrier_pct:.6f}")
        if proposal.max_ticks is not None and proposal.max_ticks < spec.horizon_ticks:
            return _reject(
                f"contract max {proposal.max_ticks} ticks < horizon {spec.horizon_ticks}"
            )
        return Terms(True, accumulator_target(spec, stake), ask)

    n = _num(proposal.contracts)
    barrier = _num(proposal.barrier_abs)
    if n is None or n <= 0 or barrier is None or spot is None:
        return _reject("contracts/barrier/spot unavailable: cannot verify trained terms")
    up = direction is Direction.CALL

    if prod is Product.TURBO:
        dist = abs(spot - barrier)
        if dist < spec.barrier_offset * (1.0 - tol):
            return _reject(f"knock-out distance {dist:.4f} < trained {spec.barrier_offset:.4f}")
        needed = ask * (1.0 + spec.take_profit_pct) / n - dist  # move that pays the target
        trained = spec.take_profit_pct * spec.barrier_offset
        if needed > trained * (1.0 + tol) + 1e-12:
            return _reject(f"target needs a {needed:.4f} move, trained on {trained:.4f}")
        return Terms(True, spec.take_profit_pct * ask, ask)

    # vanilla: intrinsic value at expiry must reach ask*(1+target)
    per_contract = ask * (1.0 + spec.target_pct) / n
    needed = (barrier + per_contract - spot) if up else (spot - (barrier - per_contract))
    if needed > spec.needed_move * (1.0 + tol) + 1e-12:
        return _reject(f"target needs a {needed:.4f} move, trained on {spec.needed_move:.4f}")
    return Terms(True, spec.target_pct * ask, ask)


def edge_gate(p: float | None, terms: Terms, margin: float) -> tuple[bool, float, float | None]:
    """Pass iff p >= L/(W+L) + margin. Returns (passes, break_even, edge)."""
    be = terms.break_even
    if p is None:
        return False, be, None
    return p >= be + margin - 1e-12, be, p - be


def expected_value(p: float, terms: Terms) -> float:
    return p * terms.win - (1.0 - p) * terms.loss


# ---------------------------------------------------------------- exit management
@dataclass(frozen=True)
class ExitPlan:
    """What the bot itself must do after the buy (server-side limit orders need nothing)."""

    max_hold_s: float | None  # sell at market after this long (products that never expire)
    take_profit_pct: float | None  # sell at market once profit >= pct * cost (turbos)
    cost: Decimal


def exit_plan(spec: ProductSpec, cost: Decimal) -> ExitPlan | None:
    if spec.product in (Product.MULTIPLIER, Product.ACCUMULATOR):
        return ExitPlan(spec.horizon_ticks * spec.tick_seconds + spec.hold_grace_s, None, cost)
    if spec.product is Product.TURBO:
        return ExitPlan(None, spec.take_profit_pct, cost)
    return None


# ---------------------------------------------------------------- research labels
def make_product_labels(
    prices: FloatArray, spec: ProductSpec
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """(win_bull, win_bear, tie) for t in [0, len - N). Vectorised first-passage simulation.

    `tie` is only used by Rise/Fall (both directions lose on equality). Ambiguity is always
    resolved AGAINST the trade (a simultaneous target and stop counts as a loss)."""
    n_h = spec.horizon_ticks
    m = len(prices) - n_h
    if m <= 0:
        e = np.zeros(0, dtype=bool)
        return e, e.copy(), e.copy()
    base = prices[:m]
    fut = sliding_window_view(prices[1:], n_h)[:m]  # (m, N): p[t+1 .. t+N]
    zeros = np.zeros(m, dtype=bool)

    if spec.product is Product.RISE_FALL:
        end = fut[:, -1]
        return end > base, end < base, end == base

    if spec.product is Product.ACCUMULATOR:
        prev = np.concatenate([base[:, None], fut[:, :-1]], axis=1)
        survive = (np.abs(fut / prev - 1.0) < spec.barrier_pct).all(axis=1)
        return survive, survive.copy(), zeros

    if spec.product is Product.VANILLA:
        move = fut[:, -1] - base
        return move >= spec.needed_move, -move >= spec.needed_move, zeros

    if spec.product is Product.MULTIPLIER:
        ret = spec.multiplier * (fut / base[:, None] - 1.0)  # bull P&L as a fraction of stake
        tp, sl = spec.take_profit_pct, spec.stop_loss_pct
        return (
            _first_before(ret >= tp, ret <= -sl),
            _first_before(-ret >= tp, -ret <= -sl),
            zeros,
        )

    # turbo: knock-out at `off`, target at take_profit_pct * off (return = move / off)
    off = spec.barrier_offset
    move = fut - base[:, None]
    tgt = spec.take_profit_pct * off
    return (
        _first_before(move >= tgt, move <= -off),
        _first_before(-move >= tgt, -move <= -off),
        zeros,
    )


def _first_before(win_hit: BoolArray, loss_hit: BoolArray) -> BoolArray:
    """True where the win condition occurs STRICTLY before the loss condition (or loss never)."""
    n = win_hit.shape[1]
    first_w = np.where(win_hit.any(axis=1), win_hit.argmax(axis=1), n)
    first_l = np.where(loss_hit.any(axis=1), loss_hit.argmax(axis=1), n)
    return np.asarray((first_w < n) & (first_w < first_l), dtype=bool)
