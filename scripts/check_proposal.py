"""Ask Deriv for price proposals on the DEMO account. NEVER buys anything. Prints no secrets.

python scripts/check_proposal.py [SYMBOL]

Shows the exact request the bot sends for the configured product and Deriv's answer (or its
error message), so mismatches between the bot and the real API can be corrected quickly.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

import httpx

from app.config import Settings, load_config
from app.deriv import protocol
from app.deriv.client import DerivClient
from app.models.schemas import Direction, Mode
from app.products import proposal_params


async def main(symbol: str) -> int:
    settings, cfg = Settings(), load_config()
    spec = cfg.trading.product
    async with httpx.AsyncClient(timeout=15.0) as http:
        client = DerivClient(Mode.DEMO, settings, cfg, http)
        try:
            acct = await client.verify_account()
            print(f"account {acct.account_id} type={acct.account_type} balance={acct.balance}")
            await client.connect()
        except Exception as exc:
            print(f"cannot connect: {exc}")
            return 1
        stake = Decimal("1.00")
        print(f"product={spec.product.value} symbol={symbol} stake={stake}\n")
        code = 0
        try:
            for direction in (Direction.CALL, Direction.PUT):
                sent = protocol.proposal(
                    symbol=symbol,
                    amount=stake,
                    currency=cfg.trading.currency,
                    product_params=proposal_params(spec, direction, stake),
                )
                print(f"--- {direction.value}: request the bot builds\n{sent}")
                try:
                    p = await client.proposal(symbol, direction, stake)
                    print(f"OK ask={p.ask_price} payout={p.payout} spot={p.spot}")
                    print(f"   {p}")
                except protocol.DerivError as exc:
                    code = 2
                    print(f"DERIV REJECTED: {exc}")
                print()
        finally:
            await client.close()
    print(
        "Nothing was bought." + ("" if code == 0 else " Send this output back (no tokens in it).")
    )
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "R_100")))
