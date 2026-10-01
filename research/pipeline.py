"""One command, end to end: download -> randomness tests -> backtest -> charts for every market.

python -m research.pipeline --months 6 --granularity 300 --capital 20
python -m research.pipeline --skip-download          # reuse CSVs already in data/

Open reports/index.html afterwards. A market only becomes a candidate for DEMO validation when a
strategy passes out-of-sample after costs; "no strategy passed" is the usual and honest result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from app.ml.dataset import load_ticks_csv
from research import charts
from research.backtest import STRATEGIES, Costs, walk_forward
from research.download_universe import DEFAULT_SYMBOLS
from research.download_universe import run as download_run


def backtest_file(csv_path: Path, out_dir: Path, args: argparse.Namespace) -> list[str]:
    epochs, prices = load_ticks_csv(csv_path)
    if len(prices) < 2000:
        return []
    costs = Costs(args.multiplier, args.fee_per_multiplier, args.slippage_bps, args.horizon)
    reports = walk_forward(
        prices,
        epochs,
        costs,
        capital=args.capital,
        risk_pct=args.risk_pct,
        max_stake_pct=0.25,
        min_stake=args.min_stake,
        alpha=0.05,
        min_oos_trades=100,
        strategies=STRATEGIES,
    )
    (out_dir / f"backtest_{csv_path.stem}.json").write_text(
        json.dumps([{**asdict(r), "approved": r.approved} for r in reports], indent=2, default=str)
    )
    return [r.strategy for r in reports if r.approved]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--months", type=float, default=6.0)
    ap.add_argument("--granularity", type=int, default=300)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--min-stake", type=float, default=1.0)
    ap.add_argument("--multiplier", type=int, default=100)
    ap.add_argument("--fee-per-multiplier", type=float, default=0.00044)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--ws-url", default="wss://api.derivws.com/trading/v1/options/ws/public")
    a = ap.parse_args(argv)
    data_dir, out_dir = Path(a.data_dir), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.skip_download:
        asyncio.run(download_run(a.symbols, a.months, a.granularity, data_dir, a.ws_url))
    approved: dict[str, list[str]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        ok = backtest_file(csv_path, out_dir, a)
        approved[csv_path.stem] = ok
        print(f"{csv_path.stem:<22} {'APPROVED: ' + ', '.join(ok) if ok else 'no strategy passed'}")
    names = charts.build_all(data_dir, out_dir)
    print(f"\ncharts for {len(names)} markets -> {out_dir / 'index.html'}")
    if not any(approved.values()):
        print("No market/strategy passed out-of-sample after costs: do not trade these rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
