"""Download historical ticks (public ticks_history) to CSV.

    python -m research.download_ticks --symbol R_100 --count 200000 --out data/R_100.csv

Pages backwards from "latest" in chunks, rate-limited and retried. ticks_history is a public
call, so no token is needed. The public WebSocket URL is configurable (`--ws-url`) because it
could not be confirmed against the live documentation from the build environment (see
docs/API_NOTES.md).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from pathlib import Path

from app.config import load_config
from app.deriv import protocol
from app.deriv.protocol import ConnectionLost, DerivError
from app.deriv.rate_limiter import RateLimiter
from app.deriv.reconnect import Backoff
from app.deriv.websocket import DerivWebSocket

DEFAULT_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
log = logging.getLogger("derivbot.research")


async def download(
    ws: DerivWebSocket, symbol: str, total: int, chunk: int = 5000
) -> list[tuple[int, float]]:
    """Walk backwards using `end` until `total` unique ticks are collected (or no progress)."""
    seen: dict[tuple[int, float], None] = {}
    end: str | int = "latest"
    stalls = 0
    while len(seen) < total and stalls < 3:
        try:
            msg = await ws.request(
                protocol.ticks_history(symbol, count=min(chunk, total - len(seen) + 1), end=end),
                timeout_s=60.0,
                safe_to_retry=True,
            )
        except (DerivError, ConnectionLost, TimeoutError) as exc:
            log.warning("history_chunk_failed", extra={"error": type(exc).__name__})
            stalls += 1
            await asyncio.sleep(1.0)
            continue
        rows = protocol.parse_history(msg)
        before = len(seen)
        for epoch, price in rows:
            seen[(epoch, price)] = None
        if not rows or len(seen) == before:
            stalls += 1
        else:
            stalls = 0
        if rows:
            end = min(e for e, _ in rows) - 1
    return sorted(seen)[-total:]


def write_csv(rows: list[tuple[int, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "quote"])
        writer.writerows(rows)


async def run(args: argparse.Namespace) -> int:
    cfg = load_config()
    limiter = RateLimiter(cfg.deriv.requests_per_second, cfg.deriv.burst)

    async def url() -> str:
        return str(args.ws_url)

    ws = DerivWebSocket(url, limiter, Backoff(), request_timeout=60.0)
    await ws.start()
    try:
        rows = await download(ws, args.symbol, args.count, args.chunk)
    finally:
        await ws.close()
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} ticks for {args.symbol} to {args.out}")
    return 0 if rows else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbol", default="R_100")
    p.add_argument("--count", type=int, default=100_000, help="number of ticks to fetch")
    p.add_argument("--chunk", type=int, default=5000, help="ticks per request (server max ~5000)")
    p.add_argument("--out", default="data/ticks.csv")
    p.add_argument("--ws-url", default=DEFAULT_WS)
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
