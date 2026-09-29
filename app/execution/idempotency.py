"""Matching helpers for ambiguous purchases (a `buy` whose outcome is unknown).

Signal-level duplicate protection is a persisted UNIQUE constraint on signals.signal_id
(see storage/migrations.py); this module only helps decide whether an unknown buy actually
produced a contract, by matching the broker's portfolio against the order we tried to place.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from typing import Any

from app.deriv.protocol import dec


def match_portfolio_contract(
    order: sqlite3.Row,
    portfolio: list[dict[str, Any]],
    known_contract_ids: set[int],
    *,
    sent_at: float,
    slack_s: float = 10.0,
) -> dict[str, Any] | None:
    """Find the unique portfolio entry that can only be this order's purchase."""
    stake = Decimal(str(order["stake"]))
    candidates: list[dict[str, Any]] = []
    for c in portfolio:
        try:
            cid = int(c["contract_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if cid in known_contract_ids:
            continue
        if str(c.get("symbol", c.get("underlying", ""))) != str(order["symbol"]):
            continue
        if str(c.get("contract_type", "")) != str(order["direction"]):
            continue
        try:
            if abs(dec(c.get("buy_price")) - stake) > Decimal("0.000001"):
                continue
        except Exception:
            continue
        purchased = c.get("purchase_time")
        if purchased is not None and float(purchased) < sent_at - slack_s:
            continue
        candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:  # ambiguous even after matching: take the earliest purchase
        return min(candidates, key=lambda c: float(c.get("purchase_time", 0)))
    return None
