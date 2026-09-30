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
        self.reject_legacy_symbol = False  # real API: proposal has no `symbol` field
        self.ask_price_override: Decimal | None = None
        self.last_quote = 100.0
        self._props: dict[str, dict[str, Any]] = {}
        self.commission = Decimal(0)
        self.accu_barrier_pct = 0.0006  # fraction of spot per tick
        self.turbo_contracts: float | None = None  # None => stake / knock-out distance
        self.vanilla_contracts = 15.0
        self.sells: list[int] = []

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
            "profit": float(c["profit"]) if c["sold"] else float(c["profit_open"]),
            "bid_price": float(c["buy_price"] + c["profit_open"]) if not c["sold"] else None,
            "is_valid_to_sell": 0 if c["sold"] else 1,
            "limit_order": c["limit_order"],
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
            "profit_open": Decimal(0),
            "limit_order": None,
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
            if req.get("style") == "candles":
                gran = int(req.get("granularity", 60))
                epochs = [(t // gran) * gran for t in range(end - n * gran + 1, end + 1, gran)]
                await self._send(
                    ws,
                    {
                        "msg_type": "candles",
                        "req_id": rid,
                        "candles": [
                            {
                                "epoch": e,
                                "open": 1.0,
                                "high": 1.1,
                                "low": 0.9,
                                "close": 1.0 + 1e-4 * (e % 97),
                            }
                            for e in epochs
                        ],
                    },
                )
                return
            await self._send(
                ws,
                {
                    "msg_type": "history",
                    "req_id": rid,
                    "history": {"times": times, "prices": [100.0 + 0.01 * (t % 97) for t in times]},
                },
            )
        elif key == "proposal":
            await self._proposal(ws, req)
        elif key == "buy":
            await self._buy(ws, req)
        elif key == "sell":
            await self._sell(ws, req)
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

    # ---- multi-product support -------------------------------------------------------------
    async def _proposal(self, ws: ServerConnection, req: dict[str, Any]) -> None:
        rid = req.get("req_id")
        if self.proposal_delay:
            await asyncio.sleep(self.proposal_delay)
        if self.reject_legacy_symbol and "symbol" in req:
            await self._send(
                ws,
                self._error(
                    req,
                    "InputValidationFailed",
                    "Input validation failed: Properties not allowed: symbol.",
                ),
            )
            return
        stake = Decimal(str(req["amount"]))
        ctype = str(req["contract_type"])
        spot = self.last_quote
        ask = self.ask_price_override if self.ask_price_override is not None else stake
        payout = stake * self.payout_ratio if ctype in ("CALL", "PUT") else Decimal(0)
        body: dict[str, Any] = {
            "longcode": f"mock {ctype}",
            "spot": spot,
            "min_stake": 0.35,
            "max_stake": 1000,
        }
        details: dict[str, Any] = {}
        if ctype in ("MULTUP", "MULTDOWN"):
            ask = ask + self.commission
            body["commission"] = float(self.commission)
        elif ctype == "ACCU":
            details["tick_size_barrier_percentage"] = str(self.accu_barrier_pct * 100)
            details["maximum_ticks"] = 100
            body["validation_params"] = {"max_ticks": 100}
        elif ctype.startswith("TURBOS"):
            off = abs(float(str(req["barrier"])))
            barrier = spot - off if ctype == "TURBOSLONG" else spot + off
            details["barrier"] = f"{barrier:.5f}"
            body["display_number_of_contracts"] = str(
                self.turbo_contracts if self.turbo_contracts else float(ask) / off
            )
        elif ctype.startswith("VANILLA"):
            off = float(str(req["barrier"]))  # signed offset
            details["barrier"] = f"{spot + off:.5f}"
            body["display_number_of_contracts"] = str(self.vanilla_contracts)
        if details:
            body["contract_details"] = details
        pid = f"prop-{self._next_prop}"
        self._next_prop += 1
        self._props[pid] = {
            "ask": ask,
            "payout": payout,
            "symbol": req.get("symbol", req.get("underlying_symbol")),
            "type": ctype,
            "limit_order": req.get("limit_order"),
        }
        body.update({"id": pid, "ask_price": float(ask), "payout": float(payout)})
        await self._send(ws, {"msg_type": "proposal", "req_id": rid, "proposal": body})

    async def close_with_profit(self, cid: int, profit: float, *, notify: bool = True) -> None:
        """Close a contract with an explicit profit (take-profit / stop-loss / knock-out)."""
        c = self.contracts[cid]
        if c["sold"]:
            return
        p = Decimal(str(profit))
        c["sold"] = True
        c["status"] = "won" if p > 0 else "lost"
        c["profit"] = p
        c["sell_price"] = c["buy_price"] + p
        self.balance += max(c["buy_price"] + p, Decimal(0))
        if notify:
            await self.push_contract(cid)

    def set_profit(self, cid: int, profit: float) -> None:
        """Change an open contract's current profit (bid = buy price + profit)."""
        self.contracts[cid]["profit_open"] = Decimal(str(profit))

    async def push_contract(self, cid: int) -> None:
        for ws, sub_cid, req_id in list(self.contract_subs):
            if sub_cid == cid:
                await self._send(
                    ws,
                    {
                        "msg_type": "proposal_open_contract",
                        "req_id": req_id,
                        "proposal_open_contract": self._poc(cid),
                        "subscription": {"id": f"sub-poc-{req_id}"},
                    },
                )

    async def _sell(self, ws: ServerConnection, req: dict[str, Any]) -> None:
        rid = req.get("req_id")
        cid = int(req["sell"])
        c = self.contracts.get(cid)
        if c is None or c["sold"]:
            await self._send(ws, self._error(req, "ContractSellError", "already sold"))
            return
        self.sells.append(cid)
        proceeds = c["buy_price"] + c["profit_open"]
        c["sold"] = True
        c["status"] = "sold"
        c["sell_price"] = proceeds
        c["profit"] = c["profit_open"]
        self.balance += proceeds
        await self._send(
            ws,
            {
                "msg_type": "sell",
                "req_id": rid,
                "sell": {
                    "contract_id": cid,
                    "sold_for": float(proceeds),
                    "balance_after": float(self.balance),
                },
            },
        )
        await self.push_contract(cid)

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
        self.contracts[cid]["limit_order"] = prop.get("limit_order")
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
