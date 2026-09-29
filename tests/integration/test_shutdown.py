"""Graceful shutdown: in-process ordering and real SIGINT/SIGTERM against a running server."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from app.controller import Controller
from tests.integration.test_executor import mk_signal, orders
from tests.mocks.deriv_mock import MockDeriv

REPO = Path(__file__).resolve().parents[2]


async def test_shutdown_leaves_no_tasks_and_closes_database(
    controller: Controller, mock: MockDeriv
) -> None:
    await controller.start()
    await controller.process_signal_now(mk_signal())
    assert orders(controller)
    db_path = controller.config.app.db_path
    await controller.shutdown()
    await asyncio.sleep(0.05)
    leftovers = [
        t.get_name()
        for t in asyncio.all_tasks()
        if not t.done()
        and t is not asyncio.current_task()
        and t.get_name().startswith(("tick-", "signal-", "housekeeping", "deriv-ws"))
    ]
    assert leftovers == []
    with pytest.raises(sqlite3.ProgrammingError):
        controller.db.query("SELECT 1")  # closed cleanly
    # state persisted BEFORE the database closed
    conn = sqlite3.connect(db_path)
    try:
        stopped = conn.execute("SELECT COUNT(*) FROM risk_events WHERE rule='bot_stopped'")
        assert stopped.fetchone()[0] >= 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    assert not controller.running


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_real_process_exits_cleanly_on_signal(tmp_path: Path, sig: signal.Signals) -> None:
    port = _free_port()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"app:\n  port: {port}\n  db_path: {tmp_path / 'db' / 'bot.db'}\n"
        f"  log_dir: {tmp_path / 'logs'}\n  model_dir: {tmp_path / 'model'}\n"
        f"  allowed_origins: ['http://127.0.0.1:{port}']\n"
    )
    env = {
        **os.environ,
        "DERIVBOT_CONFIG": str(cfg),
        "DASHBOARD_TOKEN": "t" * 24,
        "AUTOSTART": "false",
        "PYTHONPATH": str(REPO),
    }
    for var in ("DERIV_DEMO_TOKEN", "DERIV_LIVE_TOKEN", "ALLOW_LIVE"):
        env.pop(var, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 20
        up = False
        while time.time() < deadline and proc.poll() is None:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1).status_code == 200:
                    up = True
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        assert up, "server did not start"
        status = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=2).json()
        assert status["mode"] == "demo" and status["running"] is False
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=15)
        assert "Application shutdown complete" in out
    finally:
        if proc.poll() is None:
            proc.kill()
    db = tmp_path / "db" / "bot.db"
    assert db.exists()
    assert not Path(str(db) + "-wal").exists()  # WAL checkpointed => connection closed cleanly
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
