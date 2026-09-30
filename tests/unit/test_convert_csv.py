from __future__ import annotations

from pathlib import Path

from research.convert_csv import convert, parse_time


def test_parse_time_accepts_seconds_millis_and_iso() -> None:
    assert parse_time("1700000000") == 1_700_000_000
    assert parse_time("1700000000000") == 1_700_000_000
    assert parse_time("2023-11-14 22:13:20") == 1_700_000_000
    assert parse_time("2023-11-14T22:13:20Z") == 1_700_000_000


def test_convert_sorts_dedupes_and_drops_bad_rows(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text(
        "ts,close\n1700000002,1.2\n1700000001,1.1\n1700000001,1.15\nbad,1\n1700000003,0\n"
    )
    out = tmp_path / "out.csv"
    n, skipped = convert(src, out, "ts", "close")
    assert (n, skipped) == (2, 2)
    assert out.read_text().splitlines() == ["epoch,quote", "1700000001,1.15", "1700000002,1.2"]
