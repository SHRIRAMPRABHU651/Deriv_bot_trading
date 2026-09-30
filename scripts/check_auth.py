"""Check your Deriv credentials WITHOUT trading (read-only REST call). Prints no secrets.

python scripts/check_auth.py            # DEMO credentials from .env
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import Settings, load_config
from app.deriv.auth import AuthError, DerivAuth
from app.models.schemas import Mode


def lint_value(name: str, value: str) -> list[str]:
    problems = []
    if not value:
        problems.append(f"{name} is empty")
        return problems
    if value != value.strip():
        problems.append(f"{name} has leading/trailing whitespace")
    if value[0] in "\"'" or value[-1] in "\"'":
        problems.append(f"{name} is wrapped in quotes (remove them)")
    if " " in value.strip():
        problems.append(f"{name} contains a space")
    return problems


async def main() -> int:
    s = Settings()
    cfg = load_config()
    token = s.token_for(Mode.DEMO)
    print(f"DERIV_APP_ID:          {len(s.deriv_app_id)} chars")
    print(f"DERIV_DEMO_TOKEN:      {len(token)} chars, starts with {token[:4]!r}")
    print(f"DERIV_DEMO_ACCOUNT_ID: {s.deriv_demo_account_id!r}")
    issues = (
        lint_value("DERIV_APP_ID", s.deriv_app_id)
        + lint_value("DERIV_DEMO_TOKEN", token)
        + lint_value("DERIV_DEMO_ACCOUNT_ID", s.deriv_demo_account_id)
    )
    if token and not token.startswith("pat_"):
        issues.append("DERIV_DEMO_TOKEN does not start with 'pat_' (is it the full token?)")
    for i in issues:
        print("PROBLEM:", i)
    if not s.demo_configured():
        print("Put DERIV_APP_ID, DERIV_DEMO_TOKEN and DERIV_DEMO_ACCOUNT_ID in .env.")
        return 2
    async with httpx.AsyncClient(timeout=15) as http:
        auth = DerivAuth(s, cfg.deriv.rest_base_url, http)
        try:
            accounts = await auth.list_accounts(Mode.DEMO)
            seen = [(a.account_id, a.account_type) for a in accounts]
            print(f"accounts visible to this token: {seen}")
            acct = await auth.verify_account(Mode.DEMO)
            print(f"OK: {acct.account_id} is a '{acct.account_type}' account. Credentials work.")
            return 0
        except (AuthError, httpx.HTTPError) as exc:
            print("FAILED:", exc)
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
