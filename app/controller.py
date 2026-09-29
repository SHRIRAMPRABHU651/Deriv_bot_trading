"""Controller: owns the lifecycle and wires
   WebSocket -> bounded tick queue -> strategy -> bounded signal queue -> risk -> executor.

Tick reception never waits on proposal/buy: signals are handed to worker tasks through a queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from app.alerts.telegram import TelegramAlerter
from app.clock import Clock, SystemClock
from app.config import AppConfig, Settings
from app.deriv.auth import AuthError
from app.deriv.client import DerivClient
from app.deriv.protocol import ConnectionLost, DerivError
from app.execution.executor import Executor
from app.execution.reconciliation import Reconciler
from app.execution.settlement import SettlementTracker
from app.metrics.latency import FeedHealth, LatencyTracker
from app.ml.artifacts import ArtifactError
from app.ml.features import FEATURE_VERSION
from app.ml.predict import ModelPredictor
from app.models.schemas import Mode, ModelStatus, Severity, Signal, Tick
from app.risk.limits import compute_stake
from app.risk.manager import ModelInfo, RiskManager
from app.risk.state import AccountState
from app.storage.database import Database
from app.storage.repositories import Repositories
from app.strategy.base import Strategy
from app.strategy.ema import EmaStrategy
from app.strategy.ml_strategy import MLStrategy

log = logging.getLogger("derivbot.controller")

ClientFactory = Callable[[Mode], DerivClient]


class ControllerError(Exception):
    pass


class StartError(ControllerError):
    pass


class ModeSwitchError(ControllerError):
    pass


@dataclass
class _Nonce:
    value: str
    expires: float


class Controller:
    def __init__(
        self,
        settings: Settings,
        config: AppConfig,
        *,
        clock: Clock | None = None,
        http: httpx.AsyncClient | None = None,
        client_factory: ClientFactory | None = None,
        alerter: TelegramAlerter | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.clock = clock or SystemClock()
        self._own_http = http is None
        self.http = http or httpx.AsyncClient(timeout=15)
        self._client_factory = client_factory or (
            lambda mode: DerivClient(mode, self.settings, self.config, self.http)
        )
        self.alerter = alerter or TelegramAlerter(settings)
        self.db = db or Database(config.app.db_path)
        self.repos = Repositories(self.db)

        # Always DEMO at process start; LIVE is never persisted.
        self.account = AccountState(mode=Mode.DEMO, account_id=settings.deriv_demo_account_id)
        self.feed = FeedHealth()
        self.latency = LatencyTracker(self.repos)
        self.predictor: ModelPredictor | None = None
        self.model_error: str | None = None
        self.running = False
        self.trading_enabled = False
        self.risk = RiskManager(
            self.repos,
            config,
            self.clock,
            account=self.account,
            feed=self.feed,
            latency=self.latency,
            model_getter=self.model_info,
            running_getter=lambda: self.running,
            alert=self.alerter.notify,
        )
        self.strategy: Strategy = self._build_strategy()
        self.client: DerivClient | None = None
        self.tracker: SettlementTracker | None = None
        self.reconciler: Reconciler | None = None
        self.executor: Executor | None = None

        self._tick_q: asyncio.Queue[Tick] = asyncio.Queue(config.trading.tick_queue_size)
        self._signal_q: asyncio.Queue[Signal] = asyncio.Queue(config.trading.signal_queue_size)
        self._tasks: list[asyncio.Task[None]] = []
        self._lifecycle = asyncio.Lock()
        self._nonces: dict[str, _Nonce] = {}
        self.recent_ticks: dict[str, deque[tuple[int, float]]] = {}
        self.counters = {
            "ticks": 0,
            "ticks_dropped": 0,
            "signals_generated": 0,
            "signals_dropped": 0,
            "worker_errors": 0,
            "stale_feed_events": 0,
        }
        self._feed_was_stale = False
        self._consecutive_worker_errors = 0
        self.started_at: float | None = None
        self.reload_model()

    # ---- model ------------------------------------------------------------------------------
    def model_info(self) -> ModelInfo | None:
        p = self.predictor
        return None if p is None else ModelInfo(p.version, p.status, p.meta.feature_version)

    def reload_model(self) -> bool:
        """(Re)load the artifact. Any problem => no model => NO TRADING."""
        try:
            loaded = ModelPredictor.load(self.config.app.model_dir, FEATURE_VERSION)
            wanted = self.config.trading.product
            if loaded.spec.label_key() != wanted.label_key():
                raise ValueError(
                    f"model was trained for {loaded.spec.label_key()}, "
                    f"configured trade terms are {wanted.label_key()}"
                )
            self.predictor = loaded
            self.model_error = None
        except (ArtifactError, OSError, KeyError, ValueError) as exc:
            self.predictor = None
            self.model_error = f"{type(exc).__name__}: {exc}"
        if isinstance(self.strategy, MLStrategy):
            self.strategy.set_predictor(self.predictor)
        self.repos.add_risk_event(
            self.account.mode,
            Severity.INFO if self.predictor else Severity.WARNING,
            "model_loaded" if self.predictor else "model_unavailable",
            self.model_error or f"model {self.predictor.version if self.predictor else ''} loaded",
        )
        return self.predictor is not None

    def _build_strategy(self) -> Strategy:
        s = self.config.strategy
        if s.name == "ema":
            return EmaStrategy(
                s.ema_fast,
                s.ema_slow,
                self.config.trading.product.horizon_ticks,
                s.version,
                self.config.trading.product.product.value,
            )
        return MLStrategy(
            s.version,
            margin=self.config.ml.edge_margin,
            horizon=self.config.trading.product.horizon_ticks,
        )

    # ---- mode -------------------------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        return self.account.mode

    def open_trade_count(self) -> int:
        return len(self.repos.active_orders(self.mode))

    # ---- lifecycle --------------------------------------------------------------------------
    async def start(self) -> None:
        async with self._lifecycle:
            if self.running:
                return
            if self.risk.kill_active():
                raise StartError("kill switch is active; clear it deliberately before starting")
            mode = self.mode
            client = self._client_factory(mode)
            try:
                acct = await client.verify_account()  # REST first: clear errors for bad creds
                await client.connect()
                self.account.verified_type = acct.account_type
                if not self.account.type_matches_mode():
                    raise StartError(
                        f"account type {acct.account_type!r} does not match mode {mode.value}"
                    )
                self.account.account_id = acct.account_id
                bal = await client.get_balance()
            except (
                AuthError,
                DerivError,
                ConnectionLost,
                TimeoutError,
                httpx.HTTPError,
                OSError,
            ) as exc:
                await client.close()
                raise StartError(f"cannot start: {exc}") from exc
            except StartError:
                await client.close()
                raise
            self.client = client
            self.risk.update_balance(bal.balance, bal.currency)
            self.tracker = SettlementTracker(
                client,
                self.repos,
                self.risk,
                self.config,
                self.clock,
                self.latency,
                refresh_balance=self.refresh_balance,
                model_getter=self.model_info,
            )
            self.reconciler = Reconciler(client, self.repos, self.risk, self.tracker, self.clock)
            self.executor = Executor(
                client,
                self.risk,
                self.repos,
                self.config,
                self.clock,
                self.latency,
                self.tracker,
                self.reconciler,
                refresh_balance=self.refresh_balance,
            )
            # Wire reconnect handling BEFORE trading is enabled.
            client.ws.on_disconnected = self._on_disconnected
            client.ws.on_connected = self._on_connected
            report = await self.reconciler.startup()
            if not report.ok:
                await self._teardown_client()
                raise StartError("startup reconciliation failed; trading not enabled")
            self.feed.reset()
            self.feed.watch(self.config.trading.symbols, self.clock.time())
            for symbol in self.config.trading.symbols:
                await client.subscribe_ticks(symbol, self.on_tick)
            self.trading_enabled = self.predictor is not None
            self.running = True
            self.started_at = self.clock.time()
            self._tasks = [
                asyncio.create_task(self._tick_worker(), name="tick-worker"),
                asyncio.create_task(self._housekeeping(), name="housekeeping"),
                *[
                    asyncio.create_task(self._signal_worker(i), name=f"signal-worker-{i}")
                    for i in range(self.config.trading.execution_workers)
                ],
            ]
            self.repos.add_risk_event(
                mode,
                Severity.INFO,
                "bot_started",
                "trading enabled"
                if self.trading_enabled
                else "started WITHOUT a model: no trading",
            )
            self.alerter.notify("startup", f"bot started in {mode.value} mode")

    async def stop(self) -> None:
        """Graceful stop. Order: disable trading, stop signals, stop executor, reconcile,
        persist, close websocket (the database is closed by shutdown())."""
        async with self._lifecycle:
            if not self.running and self.client is None:
                return
            self.running = False  # 1. no new trades can pass risk from here on
            self.trading_enabled = False
            await self._cancel_tasks()  # 2. stop tick/signal processing
            if self.executor is not None:  # 3. stop executor
                await self.executor.close()
            if self.reconciler is not None:  # 4. resolve anything still unsettled
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.reconciler.reconcile_pending(), 10.0)
            with contextlib.suppress(Exception):  # 5. persist final balance snapshot
                await asyncio.wait_for(self.refresh_balance(), 5.0)
            await self._teardown_client()  # 6. close websocket
            self.repos.add_risk_event(self.mode, Severity.INFO, "bot_stopped", "bot stopped")
            self.alerter.notify("shutdown", "bot stopped")

    async def shutdown(self) -> None:
        await self.stop()
        if self._own_http:
            await self.http.aclose()
        self.db.close()  # 7. close DB

    async def _teardown_client(self) -> None:
        if self.tracker is not None:
            await self.tracker.close()
        if self.client is not None:
            await self.client.close()
        self.client = self.tracker = self.reconciler = self.executor = None

    async def _cancel_tasks(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        for q in (self._tick_q, self._signal_q):
            while not q.empty():
                q.get_nowait()

    # ---- reconnect hooks --------------------------------------------------------------------
    def _on_disconnected(self) -> None:
        if self.tracker is not None:
            self.tracker.mark_all_pending()
        self.repos.add_risk_event(
            self.mode, Severity.WARNING, "ws_disconnected", "websocket lost; open contracts pending"
        )

    async def _on_connected(self, is_reconnect: bool) -> None:
        if not is_reconnect or self.client is None:
            return
        await self.client.ws.resubscribe_all()
        with contextlib.suppress(Exception):
            await self.refresh_balance()
        if self.reconciler is not None:
            with contextlib.suppress(Exception):
                await self.reconciler.reconcile_pending()

    # ---- data path --------------------------------------------------------------------------
    def on_tick(self, tick: Tick) -> None:
        """Called synchronously by the websocket receiver: must be O(1) and never block."""
        self.counters["ticks"] += 1
        self.feed.on_tick(tick.symbol, tick.received_at)
        try:
            self._tick_q.put_nowait(tick)
        except asyncio.QueueFull:
            # Policy: latest information wins. Drop the OLDEST tick, then reset that symbol's
            # rolling state because its window now has a gap.
            with contextlib.suppress(asyncio.QueueEmpty):
                dropped = self._tick_q.get_nowait()
                self.strategy.reset(dropped.symbol)
            self.counters["ticks_dropped"] += 1
            log.warning("tick_queue_full", extra={"event": "tick_queue_full"})
            with contextlib.suppress(asyncio.QueueFull):
                self._tick_q.put_nowait(tick)

    async def process_tick(self, tick: Tick) -> Signal | None:
        """tick -> per-symbol state -> prediction -> signal -> (queue for risk/execution)."""
        buf = self.recent_ticks.setdefault(tick.symbol, deque(maxlen=300))
        buf.append((tick.epoch, tick.quote))
        if self.config.retention.store_ticks:
            self.repos.add_tick(tick.symbol, tick.epoch, tick.quote)
        signal = await self.strategy.on_tick(tick.symbol, tick)
        if signal is None:
            return None
        if self.clock.time() - tick.received_at > self.config.trading.max_signal_age_s:
            self.counters["signals_dropped"] += 1  # never trade on a stale tick
            return None
        self.counters["signals_generated"] += 1
        self._enqueue_signal(signal)
        return signal

    def _enqueue_signal(self, signal: Signal) -> None:
        try:
            self._signal_q.put_nowait(signal)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._signal_q.get_nowait()  # drop the oldest signal, keep the newest
            self.counters["signals_dropped"] += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._signal_q.put_nowait(signal)

    async def _tick_worker(self) -> None:
        while True:
            tick = await self._tick_q.get()
            try:
                await self.process_tick(tick)
            except Exception:
                log.exception("tick_processing_failed", extra={"event": "tick_error"})

    async def _signal_worker(self, index: int) -> None:
        while True:
            signal = await self._signal_q.get()
            executor = self.executor
            if executor is None:
                continue
            try:
                await executor.process(signal)
                self._consecutive_worker_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                self.counters["worker_errors"] += 1
                self._consecutive_worker_errors += 1
                log.exception(
                    "execution_failed",
                    extra={"event": "execution_error", "signal_id": signal.signal_id},
                )
                self.repos.add_risk_event(
                    self.mode,
                    Severity.WARNING,
                    "execution_error",
                    "unexpected execution error",
                    symbol=signal.symbol,
                    signal_id=signal.signal_id,
                )
                if self._consecutive_worker_errors >= 3:
                    self.alerter.notify("execution_failures", "repeated execution failures")

    async def process_signal_now(self, signal: Signal) -> None:
        """Synchronous variant used by tests: run a signal through risk + executor."""
        if self.executor is None:
            raise ControllerError("not started")
        await self.executor.process(signal)

    # ---- housekeeping -----------------------------------------------------------------------
    async def refresh_balance(self) -> None:
        if self.client is None:
            return
        bal = await self.client.get_balance()
        self.risk.update_balance(bal.balance, bal.currency)

    async def _housekeeping(self) -> None:
        last_balance = last_cleanup = time.monotonic()
        while True:
            await asyncio.sleep(1.0)
            now = self.clock.time()
            stale = self.feed.any_stale(now, self.config.risk.feed_stale_after_s)
            if stale and not self._feed_was_stale:
                self.counters["stale_feed_events"] += 1
                self.repos.add_risk_event(
                    self.mode, Severity.CRITICAL, "stale_feed", "no ticks: new trades halted"
                )
                self.alerter.notify("stale_feed", "market feed is stale")
            elif not stale and self._feed_was_stale:
                self.repos.add_risk_event(
                    self.mode, Severity.INFO, "feed_recovered", "ticks resumed"
                )
            self._feed_was_stale = stale
            mono = time.monotonic()
            if mono - last_balance > 30:
                last_balance = mono
                with contextlib.suppress(Exception):
                    await self.refresh_balance()
            if mono - last_cleanup > self.config.retention.cleanup_interval_s:
                last_cleanup = mono
                self.repos.cleanup(
                    self.clock.now(),
                    self.config.retention.ticks_days,
                    self.config.retention.latency_days,
                )

    # ---- kill switch ------------------------------------------------------------------------
    def kill(self, reason: str = "manual") -> None:
        """Stop new trades immediately, cancel queued work, persist, alert."""
        self.risk.activate_kill(reason)
        for q in (self._signal_q,):
            while not q.empty():
                q.get_nowait()
        if self.executor is not None:
            for task in self.executor.pending_tasks():
                task.cancel()

    def clear_kill(self) -> None:
        if self.running:
            raise ControllerError("stop the bot before clearing the kill switch")
        self.risk.clear_kill()

    def reset_drawdown_halt(self) -> None:
        if self.running:
            raise ControllerError("stop the bot before resetting the drawdown halt")
        bal = self.account.balance
        if bal is None:
            raise ControllerError("no verified balance; start the bot once to refresh it")
        self.risk.state.reset_drawdown(self.mode, bal)
        self.repos.add_risk_event(
            self.mode,
            Severity.WARNING,
            "drawdown_reset",
            "drawdown halt reset manually; HWM re-based",
        )

    # ---- mode switching (DEMO -> LIVE) ------------------------------------------------------
    def _deny_live(self, reason: str) -> ModeSwitchError:
        self.repos.add_risk_event(self.mode, Severity.CRITICAL, "live_switch_denied", reason)
        self.alerter.notify("live_switch_denied", reason)
        return ModeSwitchError(reason)

    def _live_static_checks(self) -> None:
        s = self.settings
        if not s.allow_live:
            raise self._deny_live("ALLOW_LIVE is not true")
        if not s.token_for(Mode.LIVE):
            raise self._deny_live("DERIV_LIVE_TOKEN is not set")
        if not s.deriv_live_account_id:
            raise self._deny_live("DERIV_LIVE_ACCOUNT_ID is not set")
        if self.running:
            raise self._deny_live("bot must be stopped before switching mode")
        if self.open_trade_count() > 0:
            raise self._deny_live("open trades exist")
        if self.risk.kill_active():
            raise self._deny_live("kill switch is active")

    async def prepare_live_switch(self) -> dict[str, Any]:
        """Step 1: gather everything the confirmation screen must show; issue a one-time nonce."""
        self._live_static_checks()
        from app.deriv.auth import DerivAuth

        auth = DerivAuth(self.settings, self.config.deriv.rest_base_url, self.http)
        try:
            acct = await auth.verify_account(Mode.LIVE)
        except (AuthError, httpx.HTTPError) as exc:
            raise self._deny_live(f"live account could not be verified: {exc}") from exc
        prof = self.config.risk.live
        bal = acct.balance or Decimal(0)
        info = self.model_info()
        nonce = secrets.token_urlsafe(24)
        self._nonces[nonce] = _Nonce(nonce, time.monotonic() + 120)
        return {
            "nonce": nonce,
            "account_id": acct.account_id,
            "account_type": acct.account_type,
            "balance": str(bal),
            "proposed_stake": str(
                compute_stake(
                    bal,
                    prof.stake_percent,
                    prof.stake_cap_percent,
                    self.config.trading.stake_precision,
                )
            ),
            "risk_limits": prof.model_dump(mode="json"),
            "drawdown": str(self.risk.drawdown()),
            "daily_pnl": str(self.risk.state.daily_pnl(Mode.LIVE)),
            "weekly_pnl": str(self.risk.state.weekly_pnl(Mode.LIVE)),
            "open_exposure": str(self.risk.open_exposure()),
            "model_status": info.status.value if info else "NO MODEL",
            "edge_gate": {
                "edge_margin": self.config.ml.edge_margin,
                "live_requires": ModelStatus.PROMOTABLE.value,
                "passes": bool(info and info.status is ModelStatus.PROMOTABLE),
            },
            "must_type": "LIVE",
        }

    async def switch_mode(
        self, target: Mode, *, typed: str = "", second_confirm: bool = False, nonce: str = ""
    ) -> Mode:
        if target is Mode.DEMO:
            if self.running:
                raise ModeSwitchError("stop the bot before switching mode")
            if self.open_trade_count() > 0:
                raise ModeSwitchError("open trades exist")
            self._set_mode(Mode.DEMO)
            return self.mode
        # ---- LIVE: every gate must pass, otherwise remain DEMO ----
        self._live_static_checks()
        if typed != "LIVE":
            raise self._deny_live("confirmation text must be exactly LIVE")
        entry = self._nonces.pop(nonce, None)
        if entry is None or entry.expires < time.monotonic():
            raise self._deny_live(
                "confirmation expired or missing; open the confirmation screen again"
            )
        if not second_confirm:
            raise self._deny_live("explicit second confirmation missing")
        from app.deriv.auth import DerivAuth

        auth = DerivAuth(self.settings, self.config.deriv.rest_base_url, self.http)
        try:
            acct = await auth.verify_account(Mode.LIVE)  # server re-validation: account is REAL
        except (AuthError, httpx.HTTPError) as exc:
            raise self._deny_live(f"live account verification failed: {exc}") from exc
        if acct.account_type != "real":
            raise self._deny_live(f"account type is {acct.account_type!r}, not real")
        self._set_mode(Mode.LIVE)  # in-memory only; NEVER persisted across restart
        self.account.verified_type = None  # re-verified on start()
        self.repos.add_risk_event(
            Mode.LIVE, Severity.CRITICAL, "mode_live", "switched to LIVE (in memory)"
        )
        self.alerter.notify("mode_live", "switched to LIVE mode")
        return self.mode

    def _set_mode(self, mode: Mode) -> None:
        self.account.mode = mode
        self.account.account_id = self.settings.account_id_for(mode)
        self.account.balance = None
        self.account.balance_ts = None
        self.account.verified_type = None
        self.feed.reset()

    # ---- status for the dashboard -----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        mode = self.mode
        now = self.clock.time()
        st = self.risk.state
        prof = self.risk.profile
        settled = self.repos.settled_trades(mode, limit=500)
        wins = sum(1 for r in settled if Decimal(str(r["profit"])) > 0)
        bes = [float(r["break_even"]) for r in settled if r["break_even"] is not None]
        info = self.model_info()
        probs: dict[str, dict[str, float]] = {}
        if isinstance(self.strategy, MLStrategy):
            probs = {
                s: {d.value: p for d, p in v.items()}
                for s, v in self.strategy.last_probability.items()
            }
        halts: list[str] = []
        if self.risk.kill_active():
            halts.append("KILL_SWITCH")
        if st.drawdown_halt(mode):
            halts.append("DRAWDOWN_HALT")
        if st.daily_halt(mode):
            halts.append("DAILY_LOSS_HALT")
        if st.consecutive_losses(mode) >= prof.max_consecutive_losses:
            halts.append("CONSECUTIVE_LOSSES")
        if self.risk.model_halted(info):
            halts.append("MODEL_HALT")
        feed_age = {s: self.feed.age(s, now) for s in self.config.trading.symbols}
        return {
            "mode": mode.value,
            "product": self.config.trading.product.model_dump(mode="json"),
            "running": self.running,
            "trading_enabled": self.trading_enabled and self.predictor is not None,
            "account": {
                "id": self.account.account_id,
                "type": self.account.verified_type,
                "balance": None if self.account.balance is None else str(self.account.balance),
                "balance_ts": self.account.balance_ts,
                "currency": self.account.currency,
            },
            "open_trades": self.open_trade_count(),
            "open_exposure": str(self.risk.open_exposure()),
            "daily_pnl": str(st.daily_pnl(mode)),
            "weekly_pnl": str(st.weekly_pnl(mode)),
            "drawdown": str(self.risk.drawdown()),
            "hwm": None if st.hwm(mode) is None else str(st.hwm(mode)),
            "consecutive_losses": st.consecutive_losses(mode),
            "daily_trades": st.daily_trades(mode),
            "win_rate": wins / len(settled) if settled else None,
            "break_even_win_rate": sum(bes) / len(bes) if bes else None,
            "settled_trades": len(settled),
            "probabilities": probs,
            "edge_margin": self.config.ml.edge_margin,
            "model": {
                "status": info.status.value if info else "NO_MODEL",
                "version": info.version if info else None,
                "error": self.model_error,
            },
            "risk_status": "HALTED" if halts else "OK",
            "halts": halts,
            "feed": {
                "stale": self.running
                and self.feed.any_stale(now, self.config.risk.feed_stale_after_s),
                "age_s": feed_age,
                "reconnects": self.client.ws.reconnect_count if self.client else 0,
                "disconnects": self.client.ws.disconnect_count if self.client else 0,
            },
            "latency": self.latency.daily_stats(mode, self.clock.now()),
            "counters": dict(self.counters),
            "executor": dict(self.executor.stats) if self.executor else {},
            "limits": prof.model_dump(mode="json"),
            "live_allowed": self.settings.live_permitted(),
            "telegram": self.alerter.configured,
        }

    def dashboard_lists(self) -> dict[str, Any]:
        mode = self.mode

        def rows(rs: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in rs]

        return {
            "signals": rows(self.repos.recent_signals(mode, 15)),
            "trades": rows(self.repos.recent_trades(mode, 15)),
            "risk_events": rows(self.repos.recent_risk_events(20)),
            "ticks": {s: list(v)[-120:] for s, v in self.recent_ticks.items()},
        }


AsyncFn = Callable[[], Awaitable[None]]
