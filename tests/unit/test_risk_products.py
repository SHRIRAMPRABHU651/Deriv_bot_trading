"""The RiskManager edge gate for every trade type, using ACTUAL proposal terms."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.models.schemas import Direction, OrderState, Proposal
from app.products import Product, ProductSpec
from app.risk.permit import BuyPermit
from tests.unit.prop_helper import make_proposal
from tests.unit.risk_harness import Harness, make_harness

D = Decimal


def spec(product: Product, **kw: object) -> ProductSpec:
    return ProductSpec.model_validate(
        {
            "product": product,
            "horizon_ticks": 5,
            "multiplier": 20,
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.5,
            **kw,
        }
    )


proposal = make_proposal


def authorize(
    h: Harness, p: float, prop: Proposal, direction: Direction = Direction.CALL
) -> tuple[BuyPermit | None, str]:
    sig = h.signal(p=p, direction=direction)
    dec = h.risk.evaluate(sig)
    assert dec.approved and dec.order_id, (dec.rule, dec.reason)
    permit, verdict = h.risk.authorize_buy(dec.order_id, sig, prop)
    return permit, verdict.rule


def test_multiplier_gate_uses_fee_adjusted_break_even() -> None:
    # tp = sl = 50% of stake; fee 0.2 on a 10 stake => W = 4.8, L = 5.2 => break-even 0.52
    s = spec(Product.MULTIPLIER)
    ok, _ = authorize(make_harness(spec=s), 0.55, proposal("10.2"))  # 0.52 + 0.03
    assert ok is not None and ok.max_price == D("10.2")
    none, rule = authorize(make_harness(spec=s), 0.55 - 1e-6, proposal("10.2"))
    assert none is None and rule == "edge_gate"
    none, rule = authorize(make_harness(spec=s), 0.9, proposal("10.6"))  # fee above allowance
    assert none is None and rule == "stale_quote"


def test_the_order_records_the_conservative_terms() -> None:
    h = make_harness(spec=spec(Product.MULTIPLIER))
    permit, _ = authorize(h, 0.9, proposal("10.2"))
    assert permit is not None
    row = h.repos.get_order(permit.order_id)
    assert row is not None
    assert row["contract_type"] == "MULTUP" and row["state"] == OrderState.PROPOSED.value
    assert row["win_amount"] == pytest.approx(4.8) and row["loss_amount"] == pytest.approx(5.2)
    assert row["break_even"] == pytest.approx(0.52)


def test_multiplier_bear_uses_multdown() -> None:
    h = make_harness(spec=spec(Product.MULTIPLIER))
    permit, _ = authorize(h, 0.9, proposal("10"), Direction.PUT)
    assert permit is not None
    row = h.repos.get_order(permit.order_id)
    assert row is not None and row["contract_type"] == "MULTDOWN"


def test_accumulator_gate_needs_survival_well_above_the_growth_breakeven() -> None:
    s = spec(Product.ACCUMULATOR, growth_rate=0.01, barrier_pct=0.0006)
    prop = proposal("10", barrier_pct_per_tick=0.0006, max_ticks=100)
    be = 10 / 10.51  # 0.9515
    ok, _ = authorize(make_harness(spec=s), be + 0.03 + 1e-6, prop, Direction.NEUTRAL)
    assert ok is not None
    none, rule = authorize(make_harness(spec=s), be + 0.03 - 1e-4, prop, Direction.NEUTRAL)
    assert none is None and rule == "edge_gate"
    none, rule = authorize(
        make_harness(spec=s), 0.999, proposal("10", barrier_pct_per_tick=0.0002), Direction.NEUTRAL
    )
    assert none is None and rule == "spec_mismatch"  # barrier narrower than the trained one


def test_turbo_and_vanilla_fail_closed_without_live_terms() -> None:
    t = spec(Product.TURBO, barrier_offset=0.5, take_profit_pct=0.5)
    good = proposal("10", spot=100.0, contracts=20.0, barrier_abs=99.5)
    ok, _ = authorize(make_harness(spec=t), 0.8, good)  # be 0.667 + 0.03
    assert ok is not None
    none, rule = authorize(make_harness(spec=t), 0.99, proposal("10"))
    assert none is None and rule == "spec_mismatch"
    v = spec(Product.VANILLA, needed_move=1.0, target_pct=0.5)
    vp = proposal("10", spot=100.0, contracts=15.0, barrier_abs=100.0)
    assert authorize(make_harness(spec=v), 0.8, vp)[0] is not None
    none, rule = authorize(
        make_harness(spec=v), 0.99, proposal("10", spot=100.0, contracts=3.0, barrier_abs=100.0)
    )
    assert none is None and rule == "spec_mismatch"


def test_a_signal_for_another_product_is_refused() -> None:
    h = make_harness(spec=spec(Product.MULTIPLIER))
    sig = h.signal(p=0.9, product="turbo")
    dec = h.risk.evaluate(sig)
    assert dec.approved and dec.order_id
    permit, verdict = h.risk.authorize_buy(dec.order_id, sig, proposal("10"))
    assert permit is None and verdict.rule == "product_mismatch"


def test_max_loss_stays_bounded_by_the_stake_for_exposure_purposes() -> None:
    # every product can lose at most its stake, so exposure accounting is unchanged
    for p in (Product.MULTIPLIER, Product.ACCUMULATOR, Product.TURBO, Product.VANILLA):
        h = make_harness(spec=spec(p), balance="1000")
        dec = h.risk.evaluate(
            h.signal(direction=Direction.NEUTRAL if p is Product.ACCUMULATOR else Direction.CALL)
        )
        assert dec.approved and dec.stake == D("10.00")
        assert h.risk.open_exposure() == D("10.00")
