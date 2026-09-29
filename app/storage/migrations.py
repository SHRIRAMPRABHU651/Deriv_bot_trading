"""SQLite schema and forward-only migrations (tracked with PRAGMA user_version)."""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # ---- v1: initial schema -------------------------------------------------------------
    """
    CREATE TABLE bot_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE risk_state (
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (mode, key)
    );
    CREATE TABLE account_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        account_id TEXT NOT NULL,
        balance TEXT NOT NULL,
        currency TEXT NOT NULL,
        ts TEXT NOT NULL
    );
    CREATE INDEX idx_snapshots_mode_ts ON account_snapshots(mode, ts);
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL UNIQUE,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        strategy TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        model_version TEXT,
        probability REAL,
        tick_epoch INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'NEW',
        reject_reason TEXT
    );
    CREATE INDEX idx_signals_mode_created ON signals(mode, created_at);
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT NOT NULL UNIQUE,
        signal_id TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        stake TEXT NOT NULL,
        max_price TEXT,
        proposal_id TEXT,
        ask_price TEXT,
        payout TEXT,
        state TEXT NOT NULL,
        contract_id INTEGER,
        error TEXT,
        probability REAL,
        break_even REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        order_sent_at REAL,
        confirmed_at REAL
    );
    CREATE INDEX idx_orders_mode_state ON orders(mode, state);
    CREATE INDEX idx_orders_contract ON orders(contract_id);
    CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_id INTEGER NOT NULL UNIQUE,
        order_id TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        stake TEXT NOT NULL,
        buy_price TEXT NOT NULL,
        payout TEXT,
        state TEXT NOT NULL,
        profit TEXT,
        probability REAL,
        break_even REAL,
        entry_spot REAL,
        exit_spot REAL,
        opened_at TEXT NOT NULL,
        settled_at TEXT
    );
    CREATE INDEX idx_trades_mode_settled ON trades(mode, settled_at);
    CREATE TABLE risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        severity TEXT NOT NULL,
        rule TEXT NOT NULL,
        message TEXT NOT NULL,
        symbol TEXT,
        signal_id TEXT,
        details TEXT
    );
    CREATE INDEX idx_risk_events_ts ON risk_events(ts);
    CREATE TABLE latency_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        kind TEXT NOT NULL,
        ms REAL NOT NULL
    );
    CREATE INDEX idx_latency_ts ON latency_samples(ts);
    CREATE TABLE ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        quote REAL NOT NULL,
        UNIQUE (symbol, epoch, quote)
    );
    CREATE INDEX idx_ticks_symbol_epoch ON ticks(symbol, epoch);
    CREATE TABLE model_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        model_version TEXT NOT NULL,
        status TEXT NOT NULL,
        metrics TEXT NOT NULL
    );
    CREATE TABLE reconciliation_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('demo','live')),
        kind TEXT NOT NULL,
        contract_id INTEGER,
        detail TEXT NOT NULL
    );
    """,
]


def migrate(conn: sqlite3.Connection) -> None:
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for version, script in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        conn.execute("BEGIN")
        try:
            for statement in (s.strip() for s in script.split(";")):
                if statement:
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
