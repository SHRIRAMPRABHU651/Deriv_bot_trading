"""Repositories: every trading row carries `mode`; Decimal money is stored as TEXT."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from app.models.schemas import ACTIVE_STATES, Mode, OrderState, Severity, Signal
from app.storage.database import Database, utc_iso


def _coerce(value: Any) -> Any:
    if isinstance(value, OrderState):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    return value


class Repositories:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- bot_state (global, mode independent: kill switch, model halt, ...) --------------
    def get_bot_state(self, key: str, default: str | None = None) -> str | None:
        row = self.db.query_one("SELECT value FROM bot_state WHERE key=?", (key,))
        return str(row["value"]) if row else default

    def set_bot_state(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO bot_state(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, utc_iso()),
        )

    # ---- risk_state (per mode) -----------------------------------------------------------
    def get_risk(self, mode: Mode, key: str, default: str | None = None) -> str | None:
        row = self.db.query_one(
            "SELECT value FROM risk_state WHERE mode=? AND key=?", (mode.value, key)
        )
        return str(row["value"]) if row else default

    def set_risk(self, mode: Mode, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO risk_state(mode,key,value,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(mode,key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (mode.value, key, value, utc_iso()),
        )

    def all_risk(self, mode: Mode) -> dict[str, str]:
        rows = self.db.query("SELECT key,value FROM risk_state WHERE mode=?", (mode.value,))
        return {str(r["key"]): str(r["value"]) for r in rows}

    # ---- account snapshots -----------------------------------------------------------------
    def add_snapshot(
        self, mode: Mode, account_id: str, balance: Decimal, currency: str, ts: datetime
    ) -> None:
        self.db.execute(
            "INSERT INTO account_snapshots(mode,account_id,balance,currency,ts) VALUES(?,?,?,?,?)",
            (mode.value, account_id, str(balance), currency, utc_iso(ts)),
        )

    # ---- signals ---------------------------------------------------------------------------
    def insert_signal(self, mode: Mode, signal: Signal, ts: datetime) -> bool:
        """Persist a signal. Returns False when the deterministic signal_id already exists."""
        try:
            self.db.execute(
                "INSERT INTO signals(signal_id,mode,symbol,direction,strategy,strategy_version,"
                "model_version,probability,tick_epoch,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    signal.signal_id,
                    mode.value,
                    signal.symbol,
                    signal.direction.value,
                    signal.strategy,
                    signal.strategy_version,
                    signal.model_version,
                    signal.probability,
                    signal.tick_epoch,
                    utc_iso(ts),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def set_signal_status(self, signal_id: str, status: str, reason: str | None = None) -> None:
        self.db.execute(
            "UPDATE signals SET status=?, reject_reason=? WHERE signal_id=?",
            (status, reason, signal_id),
        )

    # ---- orders ----------------------------------------------------------------------------
    def insert_order(
        self,
        *,
        order_id: str,
        signal: Signal,
        mode: Mode,
        stake: Decimal,
        break_even: float | None,
        ts: datetime,
    ) -> None:
        now = utc_iso(ts)
        self.db.execute(
            "INSERT INTO orders(order_id,signal_id,mode,symbol,direction,stake,state,probability,"
            "break_even,created_at,updated_at,product) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                order_id,
                signal.signal_id,
                mode.value,
                signal.symbol,
                signal.direction.value,
                str(stake),
                OrderState.APPROVED.value,
                signal.probability,
                break_even,
                now,
                now,
                signal.product,
            ),
        )

    def update_order(self, order_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        values = tuple(_coerce(v) for v in fields.values())
        self.db.execute(f"UPDATE orders SET {cols} WHERE order_id=?", (*values, order_id))

    def get_order(self, order_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM orders WHERE order_id=?", (order_id,))

    def get_order_by_contract(self, contract_id: int) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM orders WHERE contract_id=?", (contract_id,))

    def active_orders(self, mode: Mode) -> list[sqlite3.Row]:
        marks = ",".join("?" for _ in ACTIVE_STATES)
        return self.db.query(
            f"SELECT * FROM orders WHERE mode=? AND state IN ({marks}) ORDER BY id",
            (mode.value, *[s.value for s in ACTIVE_STATES]),
        )

    def orders_in_state(self, mode: Mode, state: OrderState) -> list[sqlite3.Row]:
        return self.db.query(
            "SELECT * FROM orders WHERE mode=? AND state=? ORDER BY id", (mode.value, state.value)
        )

    # ---- trades ----------------------------------------------------------------------------
    def insert_trade(
        self,
        *,
        contract_id: int,
        order_id: str,
        mode: Mode,
        symbol: str,
        direction: str,
        stake: Decimal,
        buy_price: Decimal,
        payout: Decimal | None,
        probability: float | None,
        break_even: float | None,
        ts: datetime,
        product: str = "rise_fall",
    ) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO trades(contract_id,order_id,mode,symbol,direction,stake,"
            "buy_price,payout,state,probability,break_even,opened_at,product) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                contract_id,
                order_id,
                mode.value,
                symbol,
                direction,
                str(stake),
                str(buy_price),
                None if payout is None else str(payout),
                OrderState.OPEN.value,
                probability,
                break_even,
                utc_iso(ts),
                product,
            ),
        )

    def settle_trade(
        self,
        contract_id: int,
        state: OrderState,
        profit: Decimal,
        entry_spot: float | None,
        exit_spot: float | None,
        ts: datetime,
        *,
        target_hit: bool | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE trades SET state=?, profit=?, entry_spot=?, exit_spot=?, settled_at=?, "
            "target_hit=? WHERE contract_id=?",
            (
                state.value,
                str(profit),
                entry_spot,
                exit_spot,
                utc_iso(ts),
                None if target_hit is None else int(target_hit),
                contract_id,
            ),
        )

    def settled_trades(
        self, mode: Mode, *, since: datetime | None = None, limit: int | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM trades WHERE mode=? AND settled_at IS NOT NULL"
        params: list[Any] = [mode.value]
        if since is not None:
            sql += " AND settled_at>=?"
            params.append(utc_iso(since))
        sql += " ORDER BY settled_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.db.query(sql, tuple(params))

    def recent_trades(self, mode: Mode, limit: int = 20) -> list[sqlite3.Row]:
        return self.db.query(
            "SELECT * FROM trades WHERE mode=? ORDER BY id DESC LIMIT ?", (mode.value, limit)
        )

    # ---- risk events -----------------------------------------------------------------------
    def add_risk_event(
        self,
        mode: Mode,
        severity: Severity,
        rule: str,
        message: str,
        *,
        symbol: str | None = None,
        signal_id: str | None = None,
        details: dict[str, object] | None = None,
        ts: datetime | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO risk_events(ts,mode,severity,rule,message,symbol,signal_id,details) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                utc_iso(ts),
                mode.value,
                severity.value,
                rule,
                message,
                symbol,
                signal_id,
                json.dumps(details or {}, default=str),
            ),
        )

    def recent_risk_events(self, limit: int = 30) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM risk_events ORDER BY id DESC LIMIT ?", (limit,))

    def recent_signals(self, mode: Mode, limit: int = 30) -> list[sqlite3.Row]:
        return self.db.query(
            "SELECT * FROM signals WHERE mode=? ORDER BY id DESC LIMIT ?", (mode.value, limit)
        )

    # ---- latency ---------------------------------------------------------------------------
    def add_latency(self, mode: Mode, kind: str, ms: float, ts: datetime) -> None:
        self.db.execute(
            "INSERT INTO latency_samples(ts,mode,kind,ms) VALUES(?,?,?,?)",
            (utc_iso(ts), mode.value, kind, ms),
        )

    def latency_samples(self, mode: Mode, kind: str, since: datetime) -> list[float]:
        rows = self.db.query(
            "SELECT ms FROM latency_samples WHERE mode=? AND kind=? AND ts>=?",
            (mode.value, kind, utc_iso(since)),
        )
        return [float(r["ms"]) for r in rows]

    # ---- ticks (optional) ------------------------------------------------------------------
    def add_tick(self, symbol: str, epoch: int, quote: float) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO ticks(symbol,epoch,quote) VALUES(?,?,?)", (symbol, epoch, quote)
        )

    # ---- model runs / reconciliation -------------------------------------------------------
    def add_model_run(self, version: str, status: str, metrics: dict[str, object]) -> None:
        self.db.execute(
            "INSERT INTO model_runs(ts,model_version,status,metrics) VALUES(?,?,?,?)",
            (utc_iso(), version, status, json.dumps(metrics, default=str)),
        )

    def add_reconciliation(
        self, mode: Mode, kind: str, detail: str, contract_id: int | None = None
    ) -> None:
        self.db.execute(
            "INSERT INTO reconciliation_events(ts,mode,kind,contract_id,detail) VALUES(?,?,?,?,?)",
            (utc_iso(), mode.value, kind, contract_id, detail),
        )

    # ---- retention -------------------------------------------------------------------------
    def cleanup(self, now: datetime, ticks_days: int, latency_days: int) -> tuple[int, int]:
        """Delete old ticks and latency samples only. Trades/risk events are never deleted."""
        ticks_cutoff = int((now - timedelta(days=ticks_days)).astimezone(UTC).timestamp())
        n_ticks = self.db.execute("DELETE FROM ticks WHERE epoch<?", (ticks_cutoff,)).rowcount
        n_lat = self.db.execute(
            "DELETE FROM latency_samples WHERE ts<?", (utc_iso(now - timedelta(days=latency_days)),)
        ).rowcount
        return n_ticks, n_lat
