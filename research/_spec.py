"""Shared CLI options describing the trade type (product) a model is trained for."""

from __future__ import annotations

import argparse
from typing import Any

from app.config import AppConfig
from app.products import Product, ProductSpec

# (flag, dest/field, type, help)
_FIELDS: list[tuple[str, str, type, str]] = [
    ("--horizon", "horizon_ticks", int, "N ticks: expiry (rise_fall/turbo/vanilla) or hold cap"),
    (
        "--tick-seconds",
        "tick_seconds",
        float,
        "seconds per tick of the symbol (R_100=2, 1HZ100V=1)",
    ),
    ("--assumed-fee-pct", "assumed_fee_pct", float, "research: fee as a fraction of stake"),
    ("--payout-ratio", "payout_ratio", float, "rise_fall: ASSUMED payout/stake (research only)"),
    ("--multiplier", "multiplier", int, "multiplier value"),
    ("--take-profit", "take_profit_pct", float, "multiplier/turbo take-profit, fraction of stake"),
    ("--stop-loss", "stop_loss_pct", float, "multiplier stop-loss, fraction of stake (<= 1)"),
    ("--growth-rate", "growth_rate", float, "accumulator growth rate per tick, e.g. 0.01"),
    ("--barrier-pct", "barrier_pct", float, "accumulator tick barrier as a fraction of spot"),
    ("--barrier-offset", "barrier_offset", float, "turbo knock-out distance (price units)"),
    ("--strike-offset", "strike_offset", float, "vanilla strike offset (price units)"),
    ("--target-pct", "target_pct", float, "vanilla profit target on the premium"),
    (
        "--needed-move",
        "needed_move",
        float,
        "vanilla: terminal move (price units) paying the target",
    ),
]


def add_spec_args(p: argparse.ArgumentParser, cfg: AppConfig) -> None:
    p.add_argument(
        "--product",
        choices=[x.value for x in Product],
        default=cfg.trading.product.product.value,
        help="trade type to model (default: config.yaml trading.product)",
    )
    for flag, dest, typ, help_ in _FIELDS:
        p.add_argument(flag, dest=f"spec_{dest}", type=typ, default=None, help=help_)


def spec_from_args(args: argparse.Namespace, cfg: AppConfig) -> ProductSpec:
    """Config's spec + CLI overrides (a different --product starts from the defaults)."""
    base = cfg.trading.product
    if args.product != base.product.value:
        base = ProductSpec(product=Product(args.product))
    updates: dict[str, Any] = {}
    for _, dest, _, _ in _FIELDS:
        value = getattr(args, f"spec_{dest}", None)
        if value is not None:
            updates[dest] = value
    return ProductSpec.model_validate({**base.model_dump(), **updates})
