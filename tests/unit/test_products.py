"""Trade types: requests, payoff terms, live-vs-trained checks, labels (vs brute force)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.models.schemas import Direction
from app.products import (
    Product,
    ProductSpec,
    accumulator_target,
    assumed_break_even,
    assumed_terms,
    contract_type,
    directions,
    edge_gate,
    exit_plan,
    expected_value,
    make_product_labels,
    payoff_terms,
    product_for_contract_type,
    proposal_params,
)
from tests.unit.prop_helper import make_proposal

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


prop = make_proposal


# ---------------------------------------------------------------- spec / contract types
def test_spec_rejects_impossible_terms() -> None:
    with pytest.raises(ValueError, match="stop_loss"):
        spec(Product.MULTIPLIER, stop_loss_pct=1.5)  # a loss can never exceed the stake
    with pytest.raises(ValueError):
        spec(Product.MULTIPLIER, take_profit_pct=0)
    with pytest.raises(ValueError):
        ProductSpec(horizon_ticks=0)
    with pytest.raises(ValueError):
        spec(Product.ACCUMULATOR, growth_rate=-0.01)


def test_contract_types_match_the_documented_names() -> None:
    assert contract_type(spec(Product.RISE_FALL), Direction.CALL) == "CALL"
    assert contract_type(spec(Product.MULTIPLIER), Direction.CALL) == "MULTUP"
    assert contract_type(spec(Product.MULTIPLIER), Direction.PUT) == "MULTDOWN"
    assert contract_type(spec(Product.TURBO), Direction.CALL) == "TURBOSLONG"
    assert contract_type(spec(Product.TURBO), Direction.PUT) == "TURBOSSHORT"
    assert contract_type(spec(Product.VANILLA), Direction.CALL) == "VANILLALONGCALL"
    assert contract_type(spec(Product.VANILLA), Direction.PUT) == "VANILLALONGPUT"
    assert contract_type(spec(Product.ACCUMULATOR), Direction.NEUTRAL) == "ACCU"
    assert directions(spec(Product.ACCUMULATOR)) == (Direction.NEUTRAL,)
    with pytest.raises(ValueError):
        contract_type(spec(Product.ACCUMULATOR), Direction.CALL)
    with pytest.raises(ValueError):
        contract_type(spec(Product.MULTIPLIER), Direction.NEUTRAL)
    assert product_for_contract_type("MULTDOWN") is Product.MULTIPLIER
    assert product_for_contract_type("ACCU") is Product.ACCUMULATOR
    assert product_for_contract_type("???") is None


def test_proposal_requests() -> None:
    m = proposal_params(spec(Product.MULTIPLIER, multiplier=20), Direction.PUT, D("10"))
    assert m == {
        "contract_type": "MULTDOWN",
        "multiplier": 20,
        "limit_order": {"take_profit": 5.0, "stop_loss": 5.0},
    }
    a = proposal_params(spec(Product.ACCUMULATOR, growth_rate=0.01), Direction.NEUTRAL, D("10"))
    assert a["contract_type"] == "ACCU" and a["growth_rate"] == 0.01 and "duration" not in a
    assert a["limit_order"] == {"take_profit": 0.51}  # 10 * (1.01^5 - 1) = 0.5101 -> floored
    t = proposal_params(spec(Product.TURBO, barrier_offset=0.5), Direction.CALL, D("10"))
    assert t["barrier"] == "-0.5" and t["duration"] == 5 and t["duration_unit"] == "t"
    assert (
        proposal_params(spec(Product.TURBO, barrier_offset=0.5), Direction.PUT, D("10"))["barrier"]
        == "+0.5"
    )
    v = proposal_params(spec(Product.VANILLA, strike_offset=0.3), Direction.PUT, D("10"))
    assert v["contract_type"] == "VANILLALONGPUT" and v["barrier"] == "-0.3"
    r = proposal_params(spec(Product.RISE_FALL), Direction.CALL, D("10"))
    assert r == {"contract_type": "CALL", "duration": 5, "duration_unit": "t"}


def test_currency_amounts_are_floored_never_rounded_up() -> None:
    p = proposal_params(spec(Product.MULTIPLIER, take_profit_pct=0.333), Direction.CALL, D("1.99"))
    assert p["limit_order"]["take_profit"] == 0.66  # 0.66267 -> 0.66
    assert accumulator_target(spec(Product.ACCUMULATOR), D("1")) == 0.05


# ---------------------------------------------------------------- research (assumed) terms
def test_assumed_terms_and_break_even() -> None:
    assert assumed_terms(spec(Product.RISE_FALL, payout_ratio=1.95)) == (0.95, 1.0)
    assert assumed_break_even(spec(Product.RISE_FALL, payout_ratio=1.95)) == pytest.approx(1 / 1.95)
    assert (
        assumed_break_even(spec(Product.MULTIPLIER, take_profit_pct=0.5, stop_loss_pct=0.5)) == 0.5
    )
    # a 1% fee on a symmetric multiplier moves break-even above 50%
    fee = spec(Product.MULTIPLIER, assumed_fee_pct=0.01)
    assert assumed_break_even(fee) == pytest.approx(0.51 / 1.0)
    w, lo = assumed_terms(spec(Product.ACCUMULATOR, growth_rate=0.01))
    assert lo == 1.0 and w == pytest.approx(1.01**5 - 1)
    assert assumed_break_even(spec(Product.TURBO, take_profit_pct=0.5)) == pytest.approx(1 / 1.5)
    assert assumed_break_even(spec(Product.VANILLA, target_pct=1.0)) == pytest.approx(0.5)


def test_edge_gate_with_generalised_break_even() -> None:
    from app.products import Terms

    t = Terms(True, win=4.8, loss=5.2)
    assert t.break_even == pytest.approx(0.52)
    assert edge_gate(0.55, t, 0.03)[0]  # exactly break-even + margin
    assert not edge_gate(0.55 - 1e-6, t, 0.03)[0]
    assert not edge_gate(None, t, 0.03)[0]
    assert expected_value(0.52, t) == pytest.approx(0.0)


# ---------------------------------------------------------------- live terms from the proposal
def test_rise_fall_terms_use_actual_payout() -> None:
    t = payoff_terms(
        spec(Product.RISE_FALL), Direction.CALL, prop("10", payout="19.5"), D("10"), 100
    )
    assert t.ok and (t.win, t.loss) == (9.5, 10.0) and t.break_even == pytest.approx(1 / 1.95)
    assert not payoff_terms(
        spec(Product.RISE_FALL), Direction.CALL, prop("10", payout="10"), D("10"), 100
    ).ok


def test_multiplier_terms_include_fees_and_reject_excess_fees() -> None:
    s = spec(Product.MULTIPLIER)
    t = payoff_terms(s, Direction.CALL, prop("10.2"), D("10"), 100)
    assert t.ok and t.win == pytest.approx(4.8) and t.loss == pytest.approx(5.2)
    assert t.break_even == pytest.approx(0.52)
    # commission reported separately is taken as an amount when that is the worse reading
    t2 = payoff_terms(s, Direction.CALL, prop("10", commission=0.3), D("10"), 100)
    assert t2.ok and t2.loss == pytest.approx(5.3)
    assert not payoff_terms(s, Direction.CALL, prop("10.6"), D("10"), 100).ok  # 6% > 5% allowance
    assert not payoff_terms(
        spec(Product.MULTIPLIER, take_profit_pct=0.01), Direction.CALL, prop("10.2"), D("10"), 100
    ).ok  # TP does not even cover the fee


def test_accumulator_terms_verify_the_trained_barrier() -> None:
    s = spec(Product.ACCUMULATOR, growth_rate=0.01, barrier_pct=0.0006)
    good = prop("10", barrier_pct_per_tick=0.0006, max_ticks=100)
    t = payoff_terms(s, Direction.NEUTRAL, good, D("10"), 100)
    assert t.ok and t.win == 0.51 and t.loss == 10.0
    assert t.break_even == pytest.approx(10 / 10.51)
    wider = prop("10", barrier_pct_per_tick=0.001)
    assert payoff_terms(s, Direction.NEUTRAL, wider, D("10"), 100).ok  # wider = safer
    narrow = prop("10", barrier_pct_per_tick=0.0003)
    r = payoff_terms(s, Direction.NEUTRAL, narrow, D("10"), 100)
    assert not r.ok and "narrower" in r.reason
    assert not payoff_terms(
        s, Direction.NEUTRAL, prop("10"), D("10"), 100
    ).ok  # unknown => fail closed
    assert not payoff_terms(
        s, Direction.NEUTRAL, prop("10", barrier_pct_per_tick=0.0006, max_ticks=3), D("10"), 100
    ).ok


def test_turbo_terms_verify_knockout_distance_and_target_move() -> None:
    s = spec(Product.TURBO, barrier_offset=0.5, take_profit_pct=0.5)
    ok = prop("10", contracts=20.0, barrier_abs=99.5)  # n = 10 / 0.5
    t = payoff_terms(s, Direction.CALL, ok, D("10"), 100.0)
    assert t.ok and t.win == pytest.approx(5.0) and t.loss == 10.0
    close_ko = prop("10", contracts=20.0, barrier_abs=99.9)  # knock-out only 0.1 away
    assert "knock-out" in payoff_terms(s, Direction.CALL, close_ko, D("10"), 100.0).reason
    few = prop("10", contracts=10.0, barrier_abs=99.5)  # needs a 1.0 move instead of 0.25
    assert "target needs" in payoff_terms(s, Direction.CALL, few, D("10"), 100.0).reason
    short = prop("10", contracts=20.0, barrier_abs=100.5)
    assert payoff_terms(s, Direction.PUT, short, D("10"), 100.0).ok
    assert not payoff_terms(s, Direction.CALL, prop("10"), D("10"), 100.0).ok  # missing data
    assert not payoff_terms(s, Direction.CALL, ok, D("10"), None).ok


def test_vanilla_terms_verify_the_needed_move() -> None:
    s = spec(Product.VANILLA, needed_move=1.0, target_pct=0.5)
    call = prop("10", contracts=15.0, barrier_abs=100.0)  # needs 10*1.5/15 = 1.0
    t = payoff_terms(s, Direction.CALL, call, D("10"), 100.0)
    assert t.ok and t.win == pytest.approx(5.0) and t.loss == 10.0
    put = prop("10", contracts=15.0, barrier_abs=100.0)
    assert payoff_terms(s, Direction.PUT, put, D("10"), 100.0).ok
    harder = prop("10", contracts=5.0, barrier_abs=100.0)  # needs 3.0
    assert not payoff_terms(s, Direction.CALL, harder, D("10"), 100.0).ok
    itm = prop("10", contracts=15.0, barrier_abs=99.0)  # in-the-money strike: easier
    assert payoff_terms(s, Direction.CALL, itm, D("10"), 100.0).ok


def test_exit_plans() -> None:
    m = exit_plan(spec(Product.MULTIPLIER, tick_seconds=2.0, hold_grace_s=2.0), D("10"))
    assert m is not None and m.max_hold_s == 5 * 2.0 + 2.0 and m.take_profit_pct is None
    a = exit_plan(spec(Product.ACCUMULATOR), D("10"))
    assert a is not None and a.max_hold_s is not None
    t = exit_plan(spec(Product.TURBO, take_profit_pct=0.5), D("10"))
    assert t is not None and t.take_profit_pct == 0.5 and t.max_hold_s is None
    assert exit_plan(spec(Product.VANILLA), D("10")) is None
    assert exit_plan(spec(Product.RISE_FALL), D("10")) is None


# ---------------------------------------------------------------- labels
def path(*future: float, base: float = 100.0) -> np.ndarray:
    return np.array([base, *future, future[-1]], dtype=np.float64)  # +1 so len > horizon


def labels(prices: np.ndarray, s: ProductSpec) -> tuple[bool, bool]:
    up, down, _ = make_product_labels(prices, s)
    return bool(up[0]), bool(down[0])


def test_multiplier_labels_target_before_stop() -> None:
    s = spec(
        Product.MULTIPLIER, horizon_ticks=3, multiplier=10, take_profit_pct=0.5, stop_loss_pct=0.5
    )
    assert labels(path(101, 106, 106), s) == (True, False)  # +6% * 10 = +60% >= TP first
    assert labels(path(94, 94, 94), s) == (False, True)  # bear target hit
    assert labels(path(106, 94, 94), s) == (True, False)  # TP first, then reversal: still a win
    assert labels(path(94, 106, 106), s) == (False, True)
    assert labels(path(101, 99, 101), s) == (False, False)  # timeout counts as a loss


def test_accumulator_labels_survival() -> None:
    s = spec(Product.ACCUMULATOR, horizon_ticks=3, barrier_pct=0.01)
    assert labels(path(100.5, 101.0, 100.6), s) == (True, True)
    assert labels(path(100.5, 102.5, 102.6), s) == (False, False)  # 2.4% tick move knocks out
    assert labels(path(99.0 - 0.5, 99.0, 99.2), s) == (False, False)  # first tick -1.5%


def test_turbo_labels_target_before_knockout() -> None:
    s = spec(Product.TURBO, horizon_ticks=3, barrier_offset=1.0, take_profit_pct=0.5)
    assert labels(path(100.6, 100.7, 100.7), s) == (True, False)  # +0.6 >= 0.5 target
    assert labels(path(99.4, 99.3, 99.3), s) == (False, True)
    assert labels(path(98.9, 100.9, 101.0), s) == (
        False,
        True,
    )  # bull knocked out first; bear target hit
    assert labels(path(100.2, 99.9, 100.1), s) == (False, False)


def test_vanilla_labels_terminal_move() -> None:
    s = spec(Product.VANILLA, horizon_ticks=3, needed_move=1.0)
    assert labels(path(99.0, 101.5, 101.0), s) == (True, False)  # only the END matters
    assert labels(path(101.5, 101.0, 98.9), s) == (False, True)
    assert labels(path(103.0, 103.0, 100.5), s) == (False, False)


def test_rise_fall_spec_labels_match_classic_labels() -> None:
    from app.ml.labels import make_labels

    rng = np.random.default_rng(1)
    p = 100 + np.cumsum(rng.normal(0, 0.1, 300))
    s = spec(Product.RISE_FALL, horizon_ticks=7)
    a = make_product_labels(p, s)
    b = make_labels(p, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def brute(prices: np.ndarray, s: ProductSpec, t: int) -> tuple[bool, bool]:
    """Straightforward per-row simulation used as the reference implementation."""
    n = s.horizon_ticks
    base = prices[t]
    fut = prices[t + 1 : t + 1 + n]
    if s.product is Product.ACCUMULATOR:
        prev = base
        for x in fut:
            if abs(x / prev - 1) >= s.barrier_pct:
                return False, False
            prev = x
        return True, True
    if s.product is Product.VANILLA:
        move = fut[-1] - base
        return move >= s.needed_move, -move >= s.needed_move
    out = []
    for sign in (1, -1):
        won = False
        for x in fut:
            if s.product is Product.MULTIPLIER:
                r = sign * s.multiplier * (x / base - 1)
                tgt, stop = s.take_profit_pct, -s.stop_loss_pct
            else:  # turbo
                r = sign * (x - base)
                tgt, stop = s.take_profit_pct * s.barrier_offset, -s.barrier_offset
            if r >= tgt:
                won = True
                break
            if r <= stop:
                break
        out.append(won)
    return out[0], out[1]


@pytest.mark.parametrize(
    "s",
    [
        spec(
            Product.MULTIPLIER,
            horizon_ticks=8,
            multiplier=200,
            take_profit_pct=0.06,
            stop_loss_pct=0.05,
        ),
        spec(Product.ACCUMULATOR, horizon_ticks=6, barrier_pct=0.0002),
        spec(Product.TURBO, horizon_ticks=8, barrier_offset=0.3, take_profit_pct=0.7),
        spec(Product.VANILLA, horizon_ticks=8, needed_move=0.2),
    ],
    ids=lambda s: s.product.value,
)
def test_vectorised_labels_equal_brute_force(s: ProductSpec) -> None:
    rng = np.random.default_rng(42)
    prices = 1000 + np.cumsum(rng.normal(0, 0.15, 600))
    up, down, _ = make_product_labels(prices, s)
    assert len(up) == len(prices) - s.horizon_ticks
    for t in range(0, len(up), 3):
        assert (bool(up[t]), bool(down[t])) == brute(prices, s, t), t
    assert 0 < up.mean() < 1  # the fixture is not degenerate


def test_labels_never_use_prices_beyond_the_horizon() -> None:
    s = spec(
        Product.MULTIPLIER,
        horizon_ticks=5,
        multiplier=200,
        take_profit_pct=0.06,
        stop_loss_pct=0.05,
    )
    rng = np.random.default_rng(3)
    p = 1000 + np.cumsum(rng.normal(0, 0.15, 200))
    base = make_product_labels(p, s)
    altered = p.copy()
    altered[120:] *= 3.0
    after = make_product_labels(altered, s)
    for a, b in zip(base[:2], after[:2], strict=True):
        assert np.array_equal(a[:114], b[:114])  # rows whose horizon ends before tick 120
