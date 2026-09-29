"""Deterministic mock of the Deriv WebSocket + REST endpoints (no network access)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from decimal import Decimal
from typing import Any

import httpx
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed


class MockDeriv:
    def __init__(self) -> None:
        self.payout_ratio = Decimal("1.95")
        self.balance = Decimal("1000")
        self.currency = "USD"
        self.loginid = "DOT1"
        self.min_stake = Decimal("0.35")
        self.requests: list[dict[str, Any]] = []
        self.contracts: dict[int, dict[str, Any]] = {}
        self._next_contract = 1000
        self._next_prop = 1
        self.conns: set[ServerConnection] = set()
        self.tick_subs: list[tuple[ServerConnection, str, int]] = []
        self.contract_subs: list[tuple[ServerConnection, int, int]] = []
        self.server: Server | None = None
        self.port = 0
        # behaviours
        self.proposal_delay = 0.0
        self.buy_delay = 0.0
        self.drop_on_buy = False  # execute the buy, then kill the socket without answering
        self.swallow_buy = False  # execute the buy, never answer (client times out)
        self.fail_buy_without_contract = False  # never execute, never answer
        self.errors: dict[str, dict[str, str]] = {}  # msg key -> error payload (one-shot)
        self.rate_limit_next = 0
        self.ping_response = True
        self.auto_settle: tuple[float, bool] | None = None
        self.ask_price_override: Decimal | None = None
        self.last_quote = 100.0
        self._props: dict[str, dict[str, Any]] = {}

    # ---- lifecycle --------------------------------------------------------------------------
    async def start(self) -> str:
        self.server = await serve(self._handler, "127.0.0.1", 0)
        self.port = next(iter(self.server.sockets)).getsockname()[1]
        return self.url

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws?otp=TESTOTP"

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            with contextlib.suppress(Exception):
                await self.server.wait_closed()

    async def drop_all(self) -> None:
        for c in list(self.conns):
            await c.close()

    # ---- helpers ----------------------------------------------------------------------------
    async def _send(self, ws: ServerConnection, msg: dict[str, Any]) -> None:
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(msg))

    def _error(self, req: dict[str, Any], code: str, message: str) -> dict[str, Any]:
        return {
            "error": {"code": code, "message": message},
            "echo_req": req,
            "req_id": req.get("req_id"),
        }

    def _poc(self, cid: int) -> dict[str, Any]:
        c = self.contracts[cid]
        return {
            "contract_id": cid,
            "is_sold": int(c["sold"]),
            "is_expired": int(c["sold"]),
            "status": c["status"],
            "profit": float(c["profit"]) if c["sold"] else 0.0,
            "buy_price": float(c["buy_price"]),
            "sell_price": float(c["sell_price"]) if c["sold"] else None,
            "payout": float(c["payout"]),
            "underlying": c["symbol"],
            "contract_type": c["type"],
            "purchase_time": c["purchase_time"],
            "date_expiry": c["purchase_time"] + 20,
            "entry_spot": self.last_quote,
            "exit_tick": self.last_quote if c["sold"] else None,
        }

    def open_contract_ids(self) -> list[int]:
        return [cid for cid, c in self.contracts.items() if not c["sold"]]

    # ---- server-side actions used by tests --------------------------------------------------
    async def push_tick(self, symbol: str, quote: float, epoch: int | None = None) -> None:
        self.last_quote = quote
        ep = epoch if epoch is not None else int(time.time())
        for ws, sym, req_id in list(self.tick_subs):
            if sym == symbol:
                await self._send(
                    ws,
                    {
                        "msg_type": "tick",
                        "tick": {"symbol": symbol, "epoch": ep, "quote": quote, "id": f"t{ep}"},
                        "subscription": {"id": f"sub-tick-{req_id}"},
                        "req_id": req_id,
                    },
                )

    async def settle(self, cid: int, win: bool, *, notify: bool = True) -> None:
        c = self.contracts[cid]
        if c["sold"]:
            return
        c["sold"] = True
        c["status"] = "won" if win else "lost"
        c["sell_price"] = c["payout"] if win else Decimal(0)
        c["profit"] = c["payout"] - c["buy_price"] if win else -c["buy_price"]
        if win:
            self.balance += c["payout"]
        for ws, sub_cid, req_id in list(self.contract_subs if notify else []):
            if sub_cid == cid:
                await self._send(
                    ws,
                    {
                        "msg_type": "proposal_open_contract",
                        "proposal_open_contract": self._poc(cid),
                        "subscription": {"id": f"sub-poc-{req_id}"},
                        "req_id": req_id,
                    },
                )

    def add_external_contract(self, symbol: str = "R_100", stake: str = "5") -> int:
        cid = self._new_contract(symbol, "CALL", Decimal(stake), Decimal(stake) * self.payout_ratio)
        return cid

    def _new_contract(self, symbol: str, ctype: str, price: Decimal, payout: Decimal) -> int:
        cid = self._next_contract
        self._next_contract += 1
        self.contracts[cid] = {
            "symbol": symbol,
            "type": ctype,
            "buy_price": price,
            "payout": payout,
            "sold": False,
            "status": "open",
            "profit": Decimal(0),
            "sell_price": Decimal(0),
            "purchase_time": int(time.time()),
        }
        self.balance -= price
        return cid

    # ---- protocol ---------------------------------------------------------------------------
    async def _handler(self, ws: ServerConnection) -> None:
        self.conns.add(ws)
        try:
            async for raw in ws:
                req = json.loads(raw)
                self.requests.append(req)
                await self._dispatch(ws, req)
        except ConnectionClosed:
            pass
        finally:
            self.conns.discard(ws)
            self.tick_subs = [t for t in self.tick_subs if t[0] is not ws]
            self.contract_subs = [t for t in self.contract_subs if t[0] is not ws]

    async def _dispatch(self, ws: ServerConnection, req: dict[str, Any]) -> None:
        rid = req.get("req_id")
        key = next((k for k in req if k not in ("req_id", "passthrough")), "")
        if self.rate_limit_next > 0 and key != "ping":
            self.rate_limit_next -= 1
            await self._send(ws, self._error(req, "RateLimit", "too many requests"))
            return
        if key in self.errors:
            await self._send(ws, self._error(req, **self.errors.pop(key)))
            return
        if key == "ping":
            if self.ping_response:
                await self._send(ws, {"ping": "pong", "msg_type": "ping", "req_id": rid})
        elif key == "balance":
            await self._send(
                ws,
                {
                    "msg_type": "balance",
                    "req_id": rid,
                    "balance": {
                        "balance": float(self.balance),
                        "currency": self.currency,
                        "loginid": self.loginid,
                    },
                },
            )
        elif key == "ticks":
            symbol = str(req["ticks"])
            self.tick_subs.append((ws, symbol, int(rid or 0)))
            await self._send(
                ws,
                {
                    "msg_type": "tick",
                    "req_id": rid,
                    "subscription": {"id": f"sub-tick-{rid}"},
                    "tick": {
                        "symbol": symbol,
                        "epoch": int(time.time()),
                        "quote": self.last_quote,
                        "id": "first",
                    },
                },
            )
        elif key == "ticks_history":
            n = int(req.get("count", 10))
            latest = 1_700_100_000
            end_req = req.get("end", "latest")
            end = latest if end_req == "latest" else min(int(end_req), latest)
            times = list(range(end - n + 1, end + 1))
            await self._send(
                ws,
                {
                    "msg_type": "history",
                    "req_id": rid,
                    "history": {"times": times, "prices": [100.0 + 0.01 * (t % 97) for t in times]},
                },
            )
        elif key == "proposal":
            if self.proposal_delay:
                await asyncio.sleep(self.proposal_delay)
            stake = Decimal(str(req["amount"]))
            ask = self.ask_price_override if self.ask_price_override is not None else stake
            pid = f"prop-{self._next_prop}"
            self._next_prop += 1
            self._props[pid] = {
                "ask": ask,
                "payout": stake * self.payout_ratio,
                "symbol": req["symbol"],
                "type": req["contract_type"],
            }
            await self._send(
                ws,
                {
                    "msg_type": "proposal",
                    "req_id": rid,
                    "proposal": {
                        "id": pid,
                        "ask_price": float(ask),
                        "payout": float(stake * self.payout_ratio),
                        "spot": self.last_quote,
                        "longcode": "mock rise/fall",
                        "min_stake": float(self.min_stake),
                        "max_stake": 1000,
                    },
                },
            )
        elif key == "buy":
            await self._buy(ws, req)
        elif key == "proposal_open_contract":
            cid = int(req["contract_id"])
            if cid not in self.contracts:
                await self._send(
                    ws,
                    {
                        "msg_type": "proposal_open_contract",
                        "req_id": rid,
                        "proposal_open_contract": {},
                    },
                )
                return
            if req.get("subscribe"):
                self.contract_subs.append((ws, cid, int(rid or 0)))
            msg: dict[str, Any] = {
                "msg_type": "proposal_open_contract",
                "req_id": rid,
                "proposal_open_contract": self._poc(cid),
            }
            if req.get("subscribe"):
                msg["subscription"] = {"id": f"sub-poc-{rid}"}
            await self._send(ws, msg)
        elif key == "portfolio":
            contracts = [
                {
                    "contract_id": cid,
                    "symbol": c["symbol"],
                    "contract_type": c["type"],
                    "buy_price": float(c["buy_price"]),
                    "payout": float(c["payout"]),
                    "purchase_time": c["purchase_time"],
                }
                for cid, c in self.contracts.items()
                if not c["sold"]
            ]
            await self._send(
                ws, {"msg_type": "portfolio", "req_id": rid, "portfolio": {"contracts": contracts}}
            )
        elif key == "forget":
            await self._send(ws, {"msg_type": "forget", "req_id": rid, "forget": 1})
        else:
            await self._send(ws, self._error(req, "UnrecognisedRequest", f"unknown {key}"))

    async def _buy(self, ws: ServerConnection, req: dict[str, Any]) -> None:
        rid = req.get("req_id")
        if self.buy_delay:
            await asyncio.sleep(self.buy_delay)
        if self.fail_buy_without_contract:
            return
        prop = self._props.get(str(req["buy"]))
        if prop is None:
            await self._send(ws, self._error(req, "InvalidContractProposal", "unknown proposal"))
            return
        if Decimal(str(req["price"])) < prop["ask"]:
            await self._send(
                ws,
                self._error(
                    req,
                    "ContractBuyValidationError",
                    "price moved: ask above maximum purchase price",
                ),
            )
            return
        cid = self._new_contract(prop["symbol"], prop["type"], prop["ask"], prop["payout"])
        if self.drop_on_buy:
            await ws.close()
            return
        if self.swallow_buy:
            return
        await self._send(
            ws,
            {
                "msg_type": "buy",
                "req_id": rid,
                "buy": {
                    "contract_id": cid,
                    "buy_price": float(prop["ask"]),
                    "payout": float(prop["payout"]),
                    "purchase_time": int(time.time()),
                    "start_time": int(time.time()),
                    "balance_after": float(self.balance),
                    "shortcode": "MOCK",
                    "transaction_id": cid + 1,
                },
            },
        )
        if self.auto_settle is not None:
            delay, win = self.auto_settle
            asyncio.get_running_loop().call_later(
                delay, lambda: asyncio.ensure_future(self.settle(cid, win))
            )


def make_http(
    mock: MockDeriv, *, account_type: str = "demo", account_id: str = "DOT1"
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    """httpx client whose transport answers the REST endpoints (accounts list + OTP)."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("Deriv-App-ID") is None or not request.headers.get(
            "Authorization", ""
        ).startswith("Bearer "):
            return httpx.Response(401, json={"error": "auth"})
        if request.method == "GET" and request.url.path == "/trading/v1/options/accounts":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "account_id": account_id,
                            "account_type": account_type,
                            "balance": float(mock.balance),
                            "currency": "USD",
                            "status": "active",
                        }
                    ]
                },
            )
        if request.method == "POST" and request.url.path.endswith("/otp"):
            return httpx.Response(200, json={"data": {"url": mock.url}})
        return httpx.Response(404, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls
