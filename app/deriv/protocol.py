"""Deriv WebSocket message builders/parsers.

Message shapes follow the Deriv API schemas (proposal, buy, proposal_open_contract, ticks,
ticks_history, balance, portfolio, ping, forget). See docs/API_NOTES.md for provenance and the
list of items that could not be confirmed against the current documentation.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from app.models.schemas import BalanceInfo, BuyResult, ContractUpdate, Proposal, Tick


class DerivError(Exception):
    def __init__(self, code: str, message: str, req_id: int | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.req_id = req_id


class RateLimitError(DerivError):
    pass


class ConnectionLost(Exception):
    """The socket closed while a request was in flight; the outcome of the request is unknown."""


class ProtocolError(Exception):
    """A response was missing fields the documentation says are present."""


_RATE_LIMIT_CODES = {"RateLimit", "RATE_LIMIT", "rate_limit", "TooManyRequests"}


def error_from_message(msg: dict[str, Any]) -> DerivError:
    err = msg.get("error") or {}
    code = str(err.get("code", "UnknownError"))
    message = str(err.get("message", ""))
    req_id = msg.get("req_id")
    cls = RateLimitError if code in _RATE_LIMIT_CODES else DerivError
    return cls(code, message, req_id if isinstance(req_id, int) else None)


def dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ProtocolError(f"not a decimal: {value!r}") from exc


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _opt_dec(value: Any) -> Decimal | None:
    return None if value is None else dec(value)


# ---- request builders ---------------------------------------------------------------------
def ping() -> dict[str, Any]:
    return {"ping": 1}


def balance() -> dict[str, Any]:
    return {"balance": 1}


def ticks(symbol: str, subscribe: bool = True) -> dict[str, Any]:
    return {"ticks": symbol, "subscribe": 1 if subscribe else 0}


def ticks_history(
    symbol: str, *, count: int, end: str | int = "latest", start: int | None = None
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "ticks_history": symbol,
        "end": str(end),  # schema: string matching ^(latest|[0-9]{1,10})$
        "count": count,
        "style": "ticks",
    }
    if start is not None:
        req["start"] = start
    return req


def proposal(
    *,
    symbol: str,
    amount: Decimal,
    currency: str,
    product_params: dict[str, Any],
) -> dict[str, Any]:
    """`product_params` comes from app.products.proposal_params (contract_type, duration,
    multiplier, growth_rate, barrier, limit_order ...)."""
    return {
        "proposal": 1,
        "amount": float(amount),  # schema: number
        "basis": "stake",
        "currency": currency,
        "symbol": symbol,
        **product_params,
    }


def sell(contract_id: int) -> dict[str, Any]:
    """Sell at market (price 0 = "sell at market" in the schema)."""
    return {"sell": contract_id, "price": 0}


def buy(
    proposal_id: str, max_price: Decimal, passthrough: dict[str, Any] | None = None
) -> dict[str, Any]:
    req: dict[str, Any] = {"buy": proposal_id, "price": float(max_price)}
    if passthrough:
        req["passthrough"] = passthrough
    return req


def proposal_open_contract(contract_id: int, subscribe: bool) -> dict[str, Any]:
    req: dict[str, Any] = {"proposal_open_contract": 1, "contract_id": contract_id}
    if subscribe:
        req["subscribe"] = 1
    return req


def portfolio() -> dict[str, Any]:
    return {"portfolio": 1}


def forget(subscription_id: str) -> dict[str, Any]:
    return {"forget": subscription_id}


# ---- response parsers ---------------------------------------------------------------------
def parse_tick(msg: dict[str, Any], received_at: float) -> Tick | None:
    body = msg.get("tick")
    if not isinstance(body, dict):
        return None
    try:
        return Tick(
            symbol=str(body["symbol"]),
            epoch=int(body["epoch"]),
            quote=float(body["quote"]),
            received_at=received_at,
            tick_id=None if body.get("id") is None else str(body["id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_history(msg: dict[str, Any]) -> list[tuple[int, float]]:
    hist = msg.get("history")
    if not isinstance(hist, dict):
        raise ProtocolError("ticks_history response without 'history'")
    times = hist.get("times") or []
    prices = hist.get("prices") or []
    if len(times) != len(prices):
        raise ProtocolError("ticks_history times/prices length mismatch")
    return [(int(t), float(p)) for t, p in zip(times, prices, strict=True)]


_BARRIER_IN_LONGCODE = re.compile(r"(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?)\s*%")


def parse_proposal(msg: dict[str, Any]) -> Proposal:
    body = msg.get("proposal")
    if not isinstance(body, dict):
        raise ProtocolError("proposal response without 'proposal'")
    try:
        vp = body.get("validation_params") or {}
        stake_vp = vp.get("stake") or {}
        details = body.get("contract_details") or {}
        pct = _num(details.get("tick_size_barrier_percentage"))
        if pct is None:  # the current API states it only in the longcode: "within the ± 0.06126%"
            m = _BARRIER_IN_LONGCODE.search(str(body.get("longcode", "")))
            pct = float(m.group(1)) if m else None
        max_ticks = vp.get("max_ticks", details.get("maximum_ticks"))
        return Proposal(
            proposal_id=str(body["id"]),
            ask_price=dec(body["ask_price"]),
            payout=dec(body.get("payout", 0)),  # multipliers/accumulators may carry no payout
            commission=_num(body.get("commission")),
            contracts=_num(body.get("display_number_of_contracts")),
            barrier_abs=_num(details.get("barrier")),
            # schema: "tick size barrier in percentage" => percent; convert to a fraction
            barrier_pct_per_tick=None if pct is None else pct / 100.0,
            max_ticks=None if max_ticks is None else int(max_ticks),
            spot=None if body.get("spot") is None else float(body["spot"]),
            longcode=str(body.get("longcode", "")),
            min_stake=_opt_dec(body.get("min_stake", stake_vp.get("min"))),
            max_stake=_opt_dec(body.get("max_stake", stake_vp.get("max"))),
        )
    except KeyError as exc:
        raise ProtocolError(f"proposal missing field {exc}") from exc


def parse_buy(msg: dict[str, Any]) -> BuyResult:
    body = msg.get("buy")
    if not isinstance(body, dict):
        raise ProtocolError("buy response without 'buy'")
    try:
        return BuyResult(
            contract_id=int(body["contract_id"]),
            buy_price=dec(body["buy_price"]),
            payout=dec(body.get("payout", 0)),
            balance_after=_opt_dec(body.get("balance_after")),
            purchase_time=int(body["purchase_time"]),
            start_time=None if body.get("start_time") is None else int(body["start_time"]),
            shortcode=str(body.get("shortcode", "")),
            transaction_id=None
            if body.get("transaction_id") is None
            else int(body["transaction_id"]),
        )
    except KeyError as exc:
        raise ProtocolError(f"buy missing field {exc}") from exc


def parse_contract(msg: dict[str, Any]) -> ContractUpdate | None:
    body = msg.get("proposal_open_contract")
    if not isinstance(body, dict) or "contract_id" not in body:
        return None  # Deriv returns an empty object for unknown contracts

    def _spot(*keys: str) -> float | None:
        for key in keys:
            if body.get(key) is not None:
                return float(body[key])
        return None

    return ContractUpdate(
        contract_id=int(body["contract_id"]),
        is_sold=bool(body.get("is_sold", 0)),
        is_expired=bool(body.get("is_expired", 0)),
        status=str(body.get("status", "open")),
        profit=_opt_dec(body.get("profit")),
        buy_price=_opt_dec(body.get("buy_price")),
        sell_price=_opt_dec(body.get("sell_price")),
        payout=_opt_dec(body.get("payout")),
        entry_spot=_spot("entry_spot", "entry_tick"),
        exit_spot=_spot("exit_tick", "sell_spot"),
        date_expiry=None if body.get("date_expiry") is None else int(body["date_expiry"]),
        symbol=None if body.get("underlying") is None else str(body["underlying"]),
        contract_type=None if body.get("contract_type") is None else str(body["contract_type"]),
        purchase_time=None if body.get("purchase_time") is None else int(body["purchase_time"]),
        bid_price=_opt_dec(body.get("bid_price")),
        valid_to_sell=bool(body.get("is_valid_to_sell", 0)),
    )


def parse_balance(msg: dict[str, Any]) -> BalanceInfo:
    body = msg.get("balance")
    if not isinstance(body, dict):
        raise ProtocolError("balance response without 'balance'")
    try:
        return BalanceInfo(
            balance=dec(body["balance"]),
            currency=str(body.get("currency", "")),
            loginid=None if body.get("loginid") is None else str(body["loginid"]),
        )
    except KeyError as exc:
        raise ProtocolError(f"balance missing field {exc}") from exc


def parse_portfolio(msg: dict[str, Any]) -> list[dict[str, Any]]:
    body = msg.get("portfolio")
    if not isinstance(body, dict):
        raise ProtocolError("portfolio response without 'portfolio'")
    contracts = body.get("contracts") or []
    return [c for c in contracts if isinstance(c, dict)]
