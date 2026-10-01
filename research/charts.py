"""Static HTML charts + evidence summary per market (inline SVG, no external libraries).

python -m research.charts --data-dir data --out reports

For every `data/<NAME>.csv` (epoch,quote) writes `reports/<NAME>.html` with the price chart, a
daily-volatility panel, key statistics, the randomness verdict and (when present) the backtest
verdict from `reports/backtest_<NAME>.json`; plus `reports/index.html` linking them all.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from app.ml.dataset import load_ticks_csv
from research.analyze import randomness_tests

FloatArray = npt.NDArray[np.float64]
W, H = 960, 260

STYLE = """
body{font:14px system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;
padding:16px;max-width:1000px}
a{color:#58a6ff} h1,h2{font-weight:600} table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #30363d;padding:5px 8px;text-align:left}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:12px 0}
.bad{color:#f85149}.ok{color:#3fb950}.warn{color:#d29922} svg{width:100%;height:auto}
"""


def downsample(x: FloatArray, points: int = 900) -> FloatArray:
    if len(x) <= points:
        return x
    idx = np.linspace(0, len(x) - 1, points).astype(int)
    return np.asarray(x[idx], dtype=np.float64)


def polyline(y: FloatArray, color: str) -> str:
    lo, hi = float(np.min(y)), float(np.max(y))
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i * W / max(len(y) - 1, 1):.1f},{H - 20 - (v - lo) / span * (H - 40):.1f}"
        for i, v in enumerate(y)
    )
    return (
        f'<svg viewBox="0 0 {W} {H}"><polyline fill="none" stroke="{color}" stroke-width="1.4" '
        f'points="{pts}"/><text x="4" y="14" fill="#8b949e" font-size="11">{hi:.5g}</text>'
        f'<text x="4" y="{H - 4}" fill="#8b949e" font-size="11">{lo:.5g}</text></svg>'
    )


def bars(y: FloatArray, color: str, label: str) -> str:
    hi = float(np.max(y)) or 1.0
    bw = W / max(len(y), 1)
    rects = "".join(
        f'<rect x="{i * bw:.1f}" y="{H - 20 - v / hi * (H - 40):.1f}" width="{max(bw - 1, 1):.1f}" '
        f'height="{v / hi * (H - 40):.1f}" fill="{color}"/>'
        for i, v in enumerate(y)
    )
    return (
        f'<svg viewBox="0 0 {W} {H}">{rects}<text x="4" y="14" fill="#8b949e" font-size="11">'
        f"{html.escape(label)} (max {hi:.2f})</text></svg>"
    )


def daily_volatility_bps(epochs: npt.NDArray[np.int64], prices: FloatArray) -> FloatArray:
    ret = np.abs(np.diff(np.log(prices))) * 1e4
    day = (epochs[1:] // 86400).astype(int)
    out = [float(ret[day == d].mean()) for d in np.unique(day)]
    return np.array(out, dtype=np.float64)


def fmt_day(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%d")


def render_symbol(
    name: str,
    epochs: npt.NDArray[np.int64],
    prices: FloatArray,
    backtest: list[dict[str, Any]] | None,
) -> str:
    rt = randomness_tests(prices, 0.05, epochs)
    ret = np.diff(np.log(prices))
    step = float(np.median(np.diff(epochs)))
    bars_per_year = 365 * 86400 / step
    peak = np.maximum.accumulate(prices)
    rows = [
        ("bars", f"{len(prices):,} (every {step:.0f} s)"),
        (
            "from / to",
            f"{fmt_day(int(epochs[0]))} -> {fmt_day(int(epochs[-1]))} "
            f"({(epochs[-1] - epochs[0]) / 86400:.0f} days)",
        ),
        ("price change", f"{prices[-1] / prices[0] - 1:+.2%}"),
        ("annualised volatility", f"{ret.std() * math.sqrt(bars_per_year):.1%}"),
        ("worst peak-to-trough", f"{float(np.max(1 - prices / peak)):.1%}"),
    ]
    stats_html = "".join(f"<tr><td>{k}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    rand = (
        '<span class="warn">structure found: ' + html.escape(", ".join(rt.significant)) + "</span>"
        if rt.significant
        else '<span class="bad">consistent with a random walk (no linear predictability)</span>'
    )
    bt_html = "<p>No backtest yet: run <code>python -m research.pipeline</code>.</p>"
    if backtest:
        lines = []
        for r in backtest:
            o = r["out_of_sample"]
            cls = "ok" if r["approved"] else "bad"
            lines.append(
                f"<tr><td>{html.escape(r['strategy'])}</td><td>{o['trades']}</td>"
                f"<td>{o['win_rate']:.1%}</td><td>{o['mean_ret']:+.2%}</td><td>{o['sharpe']:+.2f}</td>"
                f"<td>{o['max_drawdown']:.0%}</td>"
                f'<td class="{cls}">{"APPROVED" if r["approved"] else "REJECTED"}</td>'
                f"<td>{html.escape(r['reason'])}</td></tr>"
            )
        bt_html = (
            "<table><tr><th>strategy</th><th>OOS trades</th><th>win</th><th>mean/trade</th>"
            "<th>Sharpe</th><th>max DD</th><th>verdict</th><th>why</th></tr>"
            + "".join(lines)
            + "</table>"
        )
    return (
        f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<title>{html.escape(name)}</title><style>{STYLE}</style>"
        f"<h1>{html.escape(name)}</h1><p><a href='index.html'>all markets</a></p>"
        f"<div class=card><h2>Price</h2>{polyline(downsample(prices), '#58a6ff')}</div>"
        f"<div class=card><h2>Daily volatility</h2>"
        f"{bars(daily_volatility_bps(epochs, prices), '#d29922', 'mean |bar return|, bps')}</div>"
        f"<div class=card><h2>Statistics</h2><table>{stats_html}</table>"
        f"<p>Randomness tests: {rand}</p></div>"
        f"<div class=card><h2>Backtest (out-of-sample, after costs)</h2>{bt_html}</div>"
        "<p style='color:#8b949e'>Evidence report, not advice. "
        "Past data never guarantees profit.</p>"
    )


def render_index(entries: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td><a href='{html.escape(n)}.html'>{html.escape(n)}</a></td><td>{v}</td></tr>"
        for n, v in entries
    )
    return (
        f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<title>Markets</title><style>{STYLE}</style><h1>Markets</h1>"
        f"<div class=card><table><tr><th>market</th><th>verdict</th></tr>{rows}</table></div>"
    )


def verdict_of(backtest: list[dict[str, Any]] | None) -> str:
    if not backtest:
        return "not backtested"
    ok = [r["strategy"] for r in backtest if r["approved"]]
    return (
        f'<span class="ok">approved for demo validation: {html.escape(", ".join(ok))}</span>'
        if ok
        else '<span class="bad">no strategy passed out-of-sample after costs</span>'
    )


def build_all(data_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str]] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        name = csv_path.stem
        epochs, prices = load_ticks_csv(csv_path)
        if len(prices) < 200:
            continue
        bt_file = out_dir / f"backtest_{name}.json"
        backtest = json.loads(bt_file.read_text()) if bt_file.exists() else None
        (out_dir / f"{name}.html").write_text(
            render_symbol(name, epochs, prices, backtest), encoding="utf-8"
        )
        entries.append((name, verdict_of(backtest)))
    (out_dir / "index.html").write_text(render_index(entries), encoding="utf-8")
    return [n for n, _ in entries]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    names = build_all(Path(a.data_dir), Path(a.out))
    print(f"wrote {len(names)} market pages to {a.out}/index.html: {', '.join(names)}")
    return 0 if names else 1


if __name__ == "__main__":
    raise SystemExit(main())
