"""Typed Proposal factory for tests."""

from __future__ import annotations

from decimal import Decimal

from app.models.schemas import Proposal


def make_proposal(
    ask: str = "10",
    *,
    payout: str = "0",
    spot: float | None = None,
    commission: float | None = None,
    contracts: float | None = None,
    barrier_abs: float | None = None,
    barrier_pct_per_tick: float | None = None,
    max_ticks: int | None = None,
) -> Proposal:
    return Proposal(
        proposal_id="p1",
        ask_price=Decimal(ask),
        payout=Decimal(payout),
        spot=spot,
        commission=commission,
        contracts=contracts,
        barrier_abs=barrier_abs,
        barrier_pct_per_tick=barrier_pct_per_tick,
        max_ticks=max_ticks,
    )
