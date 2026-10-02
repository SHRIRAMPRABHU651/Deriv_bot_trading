"""What happens to an account that doubles its stake after every loss? (Simulation, not advice.)

python -m research.martingale_sim --capital 20 --target 10 --p-win 0.4 --payout-profit 1.43

Mimics the Deriv Bot "Candle Mine" logic exactly: stake starts at `initial`; after a WIN it resets;
after a LOSS the next stake = stake + the lost amount (i.e. doubles); a session ends when profit
reaches `target`, when the account cannot cover the next stake (ruin) or after `max_rounds`.

p-win and payout-profit must come from the real contract: for DIGITOVER 5, p-win = 0.4 and the
real payout comes from `python -m scripts.check_proposal`; the default 1.43 assumes Deriv's margin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimResult:
    sessions: int
    reached_target: float  # share of sessions that hit the profit target
    ruined: float  # share that could no longer afford the next stake
    unfinished: float
    mean_final_profit: float  # average profit/loss per session, in account currency
    mean_staked: float  # average total amount staked per session
    worst_stake: float  # largest single stake seen anywhere


def simulate(
    *,
    capital: float,
    target: float,
    p_win: float,
    payout_profit: float,
    initial: float = 1.0,
    loss_cap: float = 1000.0,
    max_rounds: int = 3000,
    sessions: int = 5000,
    seed: int = 1,
) -> SimResult:
    rng = np.random.default_rng(seed)
    equity = np.full(sessions, capital, dtype=np.float64)
    stake = np.full(sessions, initial, dtype=np.float64)
    live = np.ones(sessions, dtype=bool)
    hit = np.zeros(sessions, dtype=bool)
    ruined = np.zeros(sessions, dtype=bool)
    staked = np.zeros(sessions, dtype=np.float64)
    worst = initial
    for _ in range(max_rounds):
        if not live.any():
            break
        broke = live & (equity < stake)  # cannot cover the next stake
        ruined |= broke
        live &= ~broke
        win = rng.random(sessions) < p_win
        lost = live & ~win
        won = live & win
        staked += np.where(live, stake, 0.0)
        worst = max(worst, float(stake[live].max()) if live.any() else worst)
        equity = np.where(won, equity + stake * payout_profit, equity)
        equity = np.where(lost, equity - stake, equity)
        # Candle Mine: next stake = stake + loss on a loss; reset after a win or a huge loss
        stake = np.where(won, initial, stake)
        stake = np.where(lost, np.where(stake >= loss_cap, initial, stake * 2.0), stake)
        done = live & (equity - capital >= target)
        hit |= done
        live &= ~done
    return SimResult(
        sessions=sessions,
        reached_target=float(hit.mean()),
        ruined=float(ruined.mean()),
        unfinished=float((live & ~hit & ~ruined).mean()),
        mean_final_profit=float((equity - capital).mean()),
        mean_staked=float(staked.mean()),
        worst_stake=worst,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--target", type=float, default=10.0, help="profit that ends a session")
    ap.add_argument("--p-win", type=float, default=0.4)
    ap.add_argument(
        "--payout-profit", type=float, default=1.43, help="profit per 1 staked on a win"
    )
    ap.add_argument("--initial", type=float, default=1.0)
    ap.add_argument("--sessions", type=int, default=5000)
    ap.add_argument("--max-rounds", type=int, default=3000)
    a = ap.parse_args(argv)
    ev = a.p_win * a.payout_profit - (1.0 - a.p_win)
    print(f"expected profit per 1.00 staked: {ev:+.3f}  (a fair bet would be 0.000)")
    for cap in (a.capital, a.capital * 5, a.capital * 50):
        r = simulate(
            capital=cap,
            target=a.target,
            p_win=a.p_win,
            payout_profit=a.payout_profit,
            initial=a.initial,
            max_rounds=a.max_rounds,
            sessions=a.sessions,
        )
        print(
            f"capital {cap:>8.0f}: reach +{a.target:g} before ruin {r.reached_target:6.1%} | "
            f"ruined {r.ruined:6.1%} | unfinished {r.unfinished:5.1%} | "
            f"average result {r.mean_final_profit:+8.2f} | biggest stake {r.worst_stake:,.0f}"
        )
    print(
        "\nA high 'reach target' share does not mean the method works: the rare ruin costs "
        "more than\nthe frequent small wins gain. The average result is what matters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
