"""Analyse what the bot actually did: equity curve, per-trade results, win/loss shape, hold times.

python -m research.session_report --db data/derivbot.db --mode demo --out reports/session.html

Reads the bot's SQLite database READ-ONLY and writes one static HTML page:
cumulative profit curve, a bar per trade (green win / red loss), headline numbers (win rate,
average win vs average loss, expected profit per trade, longest losing streak, hold times) and the
last 60 trades with entry / exit prices and how long each stayed open.
"""

from __future__ import annotations

import argparse
import html
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from research.charts import STYLE, H, W, polyline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TradeRow:
    opened_at: datetime
    settled_at: datetime
    symbol: str
    direction: str
    product: str
    stake: float
    profit: float
    entry_spot: float | None
    exit_spot: float | None

    @property
    def seconds_open(self) -> float:
        return (self.settled_at - self.opened_at).total_seconds()

    @property
    def ret(self) -> float:
        return self.profit / self.stake if self.stake else 0.0


def load_trades(db_path: Path, mode: str) -> list[TradeRow]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT opened_at, settled_at, symbol, direction, product, stake, profit, entry_spot,"
            " exit_spot FROM trades WHERE mode=? AND settled_at IS NOT NULL AND profit IS NOT NULL"
            " ORDER BY settled_at, id",
            (mode,),
        ).fetchall()
    finally:
        con.close()
    return [
        TradeRow(
            datetime.fromisoformat(r["opened_at"]),
            datetime.fromisoformat(r["settled_at"]),
            r["symbol"],
            r["direction"],
            r["product"],
            float(r["stake"]),
            float(r["profit"]),
            r["entry_spot"],
            r["exit_spot"],
        )
        for r in rows
    ]


def longest_streak(flags: list[bool], target: bool) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f == target else 0
        best = max(best, run)
    return best


def headline(trades: list[TradeRow]) -> list[tuple[str, str]]:
    if not trades:
        return [("trades", "0 (nothing settled yet)")]
    profit = np.array([t.profit for t in trades])
    ret = np.array([t.ret for t in trades])
    wins = profit > 0
    flags = [bool(w) for w in wins]
    avg_win = float(profit[wins].mean()) if wins.any() else 0.0
    avg_loss = float(profit[~wins].mean()) if (~wins).any() else 0.0
    secs = np.array([t.seconds_open for t in trades])
    return [
        ("trades settled", f"{len(trades)}"),
        ("win rate", f"{wins.mean():.1%}"),
        ("average win / average loss", f"{avg_win:+.2f} / {avg_loss:+.2f}"),
        (
            "expected profit per trade",
            f"{ret.mean():+.2%} of stake (std error {ret.std() / max(len(ret) ** 0.5, 1):.2%})",
        ),
        ("total profit", f"{profit.sum():+.2f}"),
        (
            "longest win / loss streak",
            f"{longest_streak(flags, True)} / {longest_streak(flags, False)}",
        ),
        ("time open (median / max)", f"{np.median(secs):.1f} s / {secs.max():.1f} s"),
    ]


def trade_bars(trades: list[TradeRow], last: int = 200) -> str:
    sel = trades[-last:]
    if not sel:
        return ""
    peak = max(abs(t.ret) for t in sel) or 1.0
    bw = W / len(sel)
    mid = H / 2
    rects = "".join(
        f'<rect x="{i * bw:.1f}" y="{mid - max(t.ret, 0) / peak * (mid - 20):.1f}" '
        f'width="{max(bw - 1, 1):.1f}" height="{abs(t.ret) / peak * (mid - 20):.1f}" '
        f'fill="{"#3fb950" if t.profit > 0 else "#f85149"}"/>'
        for i, t in enumerate(sel)
    )
    return (
        f'<svg viewBox="0 0 {W} {H}">{rects}<line x1="0" x2="{W}" y1="{mid}" y2="{mid}" '
        f'stroke="#30363d"/><text x="4" y="14" fill="#8b949e" font-size="11">'
        f"return per trade, last {len(sel)} (max {peak:.1%})</text></svg>"
    )


def render(trades: list[TradeRow], mode: str) -> str:
    rows = "".join(
        f"<tr><td>{t.opened_at:%Y-%m-%d %H:%M:%S}</td><td>{html.escape(t.symbol)}</td>"
        f"<td>{html.escape(t.product)} {html.escape(t.direction)}</td><td>{t.stake:.2f}</td>"
        f"<td>{t.entry_spot if t.entry_spot is not None else '-'}</td>"
        f"<td>{t.exit_spot if t.exit_spot is not None else '-'}</td><td>{t.seconds_open:.1f}</td>"
        f'<td class="{"ok" if t.profit > 0 else "bad"}">{t.profit:+.2f}</td></tr>'
        for t in reversed(trades[-60:])
    )
    curve = np.cumsum([t.profit for t in trades]) if trades else np.zeros(2)
    stats = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in headline(trades)
    )
    head = (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<title>Session report</title><style>{STYLE}</style>"
        f"<h1>Session report ({html.escape(mode)})</h1>"
    )
    cum = polyline(curve.astype(np.float64), "#58a6ff")
    note = (
        "<p style='color:#8b949e'>Many small wins with rare total losses is how accumulators "
        "look: judge the expected profit per trade, not the win rate.</p>"
    )
    last = (
        "<table><tr><th>opened</th><th>symbol</th><th>type</th><th>stake</th><th>entry</th>"
        f"<th>exit</th><th>open s</th><th>profit</th></tr>{rows}</table>"
    )
    return (
        f"{head}<div class=card><h2>Cumulative profit</h2>{cum}</div>"
        f"<div class=card><h2>Every trade</h2>{trade_bars(trades)}</div>"
        f"<div class=card><h2>Numbers</h2><table>{stats}</table>{note}</div>"
        f"<div class=card><h2>Last trades (opened to closed)</h2>{last}</div>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", default="data/derivbot.db")
    ap.add_argument("--mode", default="demo", choices=["demo", "live"])
    ap.add_argument("--out", default="reports/session.html")
    a = ap.parse_args(argv)
    if not Path(a.db).exists():
        print(f"no database at {a.db}: run the bot first")
        return 1
    trades = load_trades(Path(a.db), a.mode)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(trades, a.mode), encoding="utf-8")
    print(f"{len(trades)} settled trades -> {out}")
    for k, v in headline(trades):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
