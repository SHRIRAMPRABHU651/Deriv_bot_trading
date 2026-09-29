"""Pure risk arithmetic (Decimal for money, float for probabilities)."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def floor_to_precision(value: Decimal, precision: int) -> Decimal:
    """Round DOWN to `precision` decimals. Never rounds upward."""
    quantum = Decimal(1).scaleb(-precision)
    return value.quantize(quantum, rounding=ROUND_DOWN)


def compute_stake(
    balance: Decimal, stake_percent: Decimal, cap_percent: Decimal, precision: int
) -> Decimal:
    """stake = floor(min(balance * stake_percent, balance * cap_percent)) using CURRENT balance."""
    raw = min(balance * stake_percent, balance * cap_percent)
    return floor_to_precision(max(raw, Decimal(0)), precision)


def break_even_probability(payout_ratio: Decimal) -> float:
    """Break-even win probability = 1 / R where R = payout / stake (payout includes stake)."""
    if payout_ratio <= 0:
        raise ValueError("payout ratio must be positive")
    return float(Decimal(1) / payout_ratio)


def expected_value(probability: float, payout_ratio: Decimal) -> float:
    """EV per unit staked = p * R - 1."""
    return probability * float(payout_ratio) - 1.0


def edge_gate(
    probability: float | None, payout_ratio: Decimal, margin: float
) -> tuple[bool, float, float | None]:
    """Return (passes, break_even, edge). Passes iff p >= break_even + margin."""
    be = break_even_probability(payout_ratio)
    if probability is None:
        return False, be, None
    edge = probability - be
    return probability >= be + margin - 1e-12, be, edge
