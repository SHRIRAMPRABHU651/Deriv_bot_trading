"""research.backtest: costs bite, signals never look ahead, and only a real edge is approved."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from research.backtest import STRATEGIES, Costs, Params, signals, simulate_trades, walk_forward

F = npt.NDArray[np.float64]


def series(n: int, phi: float, seed: int) -> tuple[npt.NDArray[np.int64], F]:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.0003, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    prices = 1.08 * np.exp(np.cumsum(r))
    return np.arange(n, dtype=np.int64) * 300 + 1_700_000_000, prices


def run(epochs: npt.NDArray[np.int64], prices: F) -> dict[str, bool]:
    reports = walk_forward(
        prices,
        epochs,
        Costs(),
        capital=20.0,
        risk_pct=0.01,
        max_stake_pct=0.25,
        min_stake=1.0,
        alpha=0.05,
        min_oos_trades=100,
    )
    return {r.strategy: r.approved for r in reports}


def test_round_trip_cost_is_a_fixed_fraction_of_price_whatever_the_multiplier() -> None:
    # commission and slippage both scale with the multiplier, so in PRICE terms the cost is fixed
    a, b = Costs(multiplier=100), Costs(multiplier=40)
    assert abs(a.round_trip / 100 - b.round_trip / 40) < 1e-12
    assert abs(a.round_trip - 0.054) < 1e-9


def test_signals_never_use_future_prices() -> None:
    _, prices = series(3000, 0.3, 4)
    for name in STRATEGIES:
        full = signals(prices, name, 20)
        for cut in (500, 1234, 2500):
            assert np.array_equal(signals(prices[:cut], name, 20), full[:cut]), name


def test_trades_never_lose_more_than_the_stake_and_never_overlap() -> None:
    _, prices = series(6000, 0.3, 5)
    trades = simulate_trades(prices, Params("breakout", 20, 1.0, 0.5, True), Costs())
    assert trades and all(t.ret >= -1.0 for t in trades)
    assert all(b.entry > a.exit for a, b in zip(trades, trades[1:], strict=False))


def test_random_walk_is_never_approved() -> None:
    epochs, prices = series(30_000, 0.0, 21)
    assert not any(run(epochs, prices).values())


def test_a_strong_planted_edge_is_found_out_of_sample() -> None:
    # phi=0.8 serial correlation is far stronger than any real market; weaker edges (phi=0.6)
    # barely clear the 5.4 %-of-stake round-trip cost, which is itself the finding.
    epochs, prices = series(30_000, 0.8, 9)
    approved = run(epochs, prices)
    assert approved["breakout"] and not approved["mean_reversion"]
