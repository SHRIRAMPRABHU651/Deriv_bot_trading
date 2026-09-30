"""Delete the local SQLite database (asks for confirmation). Does NOT touch the broker.

python -m scripts.db_reset
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import load_config


def main() -> int:
    db = Path(load_config().app.db_path)
    files = [Path(str(db) + s) for s in ("", "-wal", "-shm")]
    existing = [f for f in files if f.exists()]
    if not existing:
        print(f"nothing to delete ({db})")
        return 0
    print("This deletes ALL local trade history, risk state (incl. kill switch / drawdown halt):")
    for f in existing:
        print("  ", f)
    if input("Type RESET to continue: ").strip() != "RESET":
        print("aborted")
        return 1
    for f in existing:
        f.unlink()
    print("deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
