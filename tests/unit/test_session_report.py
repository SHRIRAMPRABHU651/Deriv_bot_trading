"""research.session_report reads the bot's own database and draws the evidence."""

from __future__ import annotations

from pathlib import Path

from app.storage.database import Database
from research.session_report import headline, load_trades, longest_streak, main, render


def seed(db_path: Path) -> None:
    db = Database(str(db_path))
    rows = [
        ("2026-10-01T10:00:00+00:00", "2026-10-01T10:00:06+00:00", "5", "0.15"),
        ("2026-10-01T10:00:10+00:00", "2026-10-01T10:00:16+00:00", "5", "0.15"),
        ("2026-10-01T10:00:20+00:00", "2026-10-01T10:00:26+00:00", "5", "-5"),
    ]
    for i, (o, s, stake, profit) in enumerate(rows):
        db.execute(
            "INSERT INTO trades(contract_id, order_id, mode, symbol, direction, stake, buy_price,"
            " state, profit, entry_spot, exit_spot, opened_at, settled_at, product)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                i + 1,
                f"o{i}",
                "demo",
                "R_100",
                "NEUTRAL",
                stake,
                stake,
                "WON",
                profit,
                628.0,
                628.1,
                o,
                s,
                "accumulator",
            ),
        )
    db.close()


def test_report_shows_wins_losses_and_hold_times(tmp_path: Path) -> None:
    seed(tmp_path / "bot.db")
    trades = load_trades(tmp_path / "bot.db", "demo")
    assert [round(t.seconds_open) for t in trades] == [6, 6, 6]
    lines = dict(headline(trades))
    assert lines["win rate"] == "66.7%" and lines["longest win / loss streak"] == "2 / 1"
    page = render(trades, "demo")
    assert "<polyline" in page and "accumulator NEUTRAL" in page and "-5.00" in page
    assert longest_streak([True, True, False, True], True) == 2


def test_cli_and_missing_database(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "none.db"), "--out", str(tmp_path / "r.html")]) == 1
    seed(tmp_path / "bot.db")
    assert main(["--db", str(tmp_path / "bot.db"), "--out", str(tmp_path / "r.html")]) == 0
    assert (tmp_path / "r.html").exists()
