"""Show orders the bot still counts as open and what Deriv says about each (read-only by default).

python -m scripts.inspect_orders             # list them + Deriv's raw answer per contract
python -m scripts.inspect_orders --release   # bot STOPPED: mark them FAILED so trading can continue

Releasing only changes the local database: the order is marked FAILED ("released by operator") and
its result is NOT added to P&L. Your Deriv balance is unaffected and drawdown protection still
reads the real balance. Prints no secrets.
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from app.config import Settings, load_config
from app.deriv import protocol
from app.deriv.client import DerivClient
from app.models.schemas import Mode, OrderState
from app.storage.database import Database
from app.storage.repositories import Repositories


async def raw_status(client: DerivClient, contract_id: int) -> str:
    try:
        msg = await client.ws.request(
            protocol.proposal_open_contract(contract_id, subscribe=False), safe_to_retry=True
        )
    except protocol.DerivError as exc:
        return f"Deriv error: {exc}"
    return json.dumps(msg.get("proposal_open_contract", msg), indent=2, default=str)[:2500]


async def main(release: bool) -> int:
    cfg, settings = load_config(), Settings()
    repos = Repositories(Database(cfg.app.db_path))
    active = repos.active_orders(Mode.DEMO)
    if not active:
        print("no open orders in the database: nothing is blocking trading")
        return 0
    print(f"{len(active)} order(s) counted as open:")
    for o in active:
        print(
            f"  order {o['order_id'][:8]}  state={o['state']}  contract={o['contract_id']} "
            f"product={o['product']} stake={o['stake']}"
        )
    if settings.demo_configured():
        async with httpx.AsyncClient(timeout=15.0) as http:
            client = DerivClient(Mode.DEMO, settings, cfg, http)
            try:
                await client.verify_account()
                await client.connect()
                for o in active:
                    if o["contract_id"] is not None:
                        print(f"\nDeriv's answer for contract {o['contract_id']}:")
                        print(await raw_status(client, int(o["contract_id"])))
            except Exception as exc:
                print(f"could not ask Deriv: {exc}")
            finally:
                await client.close()
    if release:
        for o in active:
            repos.update_order(o["order_id"], state=OrderState.FAILED, error="released by operator")
        print(f"\nreleased {len(active)} order(s). Start the bot again.")
    else:
        print("\nTo free the slot: stop the bot, then run with --release")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main("--release" in sys.argv[1:])))
