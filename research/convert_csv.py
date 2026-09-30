"""Convert any price CSV (Hugging Face / Kaggle / broker export) into the `epoch,quote` file that
research.analyze and research.train_model read.

python -m research.convert_csv --in eurusd_1m.csv --time-col timestamp --price-col close \\
    --out data/eurusd.csv

--time-col accepts UNIX seconds, UNIX milliseconds or ISO/`YYYY-MM-DD HH:MM:SS` text (UTC assumed).
Rows are sorted, de-duplicated by time and non-positive / non-numeric prices dropped.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path


def parse_time(raw: str) -> int:
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp())
    return int(value / 1000) if value > 1e11 else int(value)  # ms -> s


def convert(src: Path, dst: Path, time_col: str, price_col: str) -> tuple[int, int]:
    rows: dict[int, float] = {}
    skipped = 0
    with src.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for col in (time_col, price_col):
            if reader.fieldnames is None or col not in reader.fieldnames:
                raise SystemExit(f"column {col!r} not found; columns: {reader.fieldnames}")
        for rec in reader:
            try:
                t, p = parse_time(rec[time_col]), float(rec[price_col])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if p <= 0:
                skipped += 1
                continue
            rows[t] = p
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as out:
        out.write("epoch,quote\n")
        for t in sorted(rows):
            out.write(f"{t},{rows[t]!r}\n")
    return len(rows), skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--time-col", required=True)
    p.add_argument("--price-col", default="close")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    n, skipped = convert(Path(a.src), Path(a.out), a.time_col, a.price_col)
    print(f"wrote {n} rows to {a.out} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
