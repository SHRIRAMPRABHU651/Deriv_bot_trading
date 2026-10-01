"""Download ~6 months of bars for many markets (Deriv candles) into data/<SYMBOL>_<gran>s.csv.

python -m research.download_universe --months 6 --granularity 300
python -m research.download_universe --symbols frxEURUSD cryBTCUSD --months 6

Failures are reported per symbol and never stop the rest. The actual span received is printed:
if Deriv serves less history than requested, the report says so instead of pretending.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import load_config
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.deriv.websocket import DerivWebSocket
from research.download_ticks import DEFAULT_WS, download, write_csv

DEFAULT_SYMBOLS = (
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxAUDUSD",
    "frxXAUUSD",
    "cryBTCUSD",
    "cryETHUSD",
    "R_100",
)


def bars_for(months: float, granularity: int) -> int:
    return int(months * 30 * 86400 / granularity)


async def run(
    symbols: list[str], months: float, granularity: int, out_dir: Path, ws_url: str
) -> int:
    cfg = load_config()
    limiter = RateLimiter(cfg.deriv.requests_per_second, cfg.deriv.burst)

    async def url() -> str:
        return ws_url

    ws = DerivWebSocket(url, limiter, Backoff(), request_timeout=60.0)
    await ws.start()
    failures = 0
    try:
        for symbol in symbols:
            try:
                rows = await download(
                    ws, symbol, bars_for(months, granularity), granularity=granularity
                )
            except Exception as exc:  # report and continue with the next market
                print(f"{symbol:<12} FAILED: {type(exc).__name__}: {exc}")
                failures += 1
                continue
            if not rows:
                print(f"{symbol:<12} no data returned (symbol unavailable?)")
                failures += 1
                continue
            path = out_dir / f"{symbol}_{granularity}s.csv"
            write_csv(rows, path)
            span = (rows[-1][0] - rows[0][0]) / 86400
            note = (
                ""
                if span >= months * 30 * 0.9
                else f"  <-- only {span:.0f} of {months * 30:.0f} days"
            )
            print(f"{symbol:<12} {len(rows):>7} bars, {span:6.1f} days -> {path}{note}")
    finally:
        await ws.close()
    return 1 if failures == len(symbols) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--months", type=float, default=6.0)
    ap.add_argument("--granularity", type=int, default=300, help="seconds per bar (300 = 5 min)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--ws-url", default=DEFAULT_WS)
    a = ap.parse_args(argv)
    return asyncio.run(run(a.symbols, a.months, a.granularity, Path(a.out_dir), a.ws_url))


if __name__ == "__main__":
    raise SystemExit(main())
