"""Run the bot in DEMO mode with autostart, refusing to run without DEMO credentials.

python scripts/run_demo.py
"""

from __future__ import annotations

import os
import sys

from app.config import Settings

MESSAGE = (
    "Put your DEMO credentials in .env using DERIV_APP_ID, DERIV_DEMO_TOKEN and "
    "DERIV_DEMO_ACCOUNT_ID. Do not send the token in chat."
)


def main() -> int:
    settings = Settings()
    if not settings.demo_configured():
        print(MESSAGE, file=sys.stderr)
        return 2
    if settings.allow_live:
        print("note: ALLOW_LIVE=true is set; this script still starts in DEMO and never switches.")
    os.environ["AUTOSTART"] = "true"
    from app.main import run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
