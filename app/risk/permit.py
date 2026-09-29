"""BuyPermit: proof that RiskManager approved this exact purchase.

The Deriv client refuses to send a `buy` without one. Permits can only be minted by
`RiskManager.authorize_buy` (which calls `_mint`); constructing one directly raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.models.schemas import Mode

_MINT_KEY = object()


class PermitError(PermissionError):
    pass


@dataclass(frozen=True)
class BuyPermit:
    order_id: str
    proposal_id: str
    max_price: Decimal
    mode: Mode
    _key: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._key is not _MINT_KEY:
            raise PermitError("BuyPermit can only be issued by RiskManager")


def _mint(order_id: str, proposal_id: str, max_price: Decimal, mode: Mode) -> BuyPermit:
    """Internal: only app.risk.manager may call this."""
    return BuyPermit(order_id, proposal_id, max_price, mode, _key=_MINT_KEY)
