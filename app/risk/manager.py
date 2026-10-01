"""RiskManager: the ONLY gateway between strategy signals and Deriv purchases.

Two gates:
  evaluate(signal)            -> RiskDecision   (all rules; reserves exposure via an APPROVED order)
  authorize_buy(order, prop)  -> BuyPermit|None (edge gate with the ACTUAL proposal payout)
The Deriv client refuses to buy without a permit minted here.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.clock import Clock
from app.config import AppConfig, RiskProfile
from app.metrics.latency import FeedHealth, LatencyTracker
from app.models.schemas import (
    Mode,
    ModelStatus,
    OrderState,
    Proposal,
    RiskDecision,
    Severity,
    Signal,
)
from app.products import contract_type, edge_gate, payoff_terms
from app.risk.limits import compute_stake
from app.risk.permit import BuyPermit, _mint
from app.risk.state import AccountState, RiskStateStore
from app.storage.repositories import Repositories
from app.strategy.probe import PROBE_NAME

log = logging.getLogger("derivbot.risk")

KILL_KEY = "kill_switch"
MODEL_HALT_KEY = "model_halt"


@dataclass(frozen=True)
class ModelInfo:
    version: str
    status: ModelStatus
    feature_version: str


AlertFn = Callable[[str, str], None]


def _target_hit(row: Any) -> bool:
    """The event the model predicts (falls back to profit > 0 for legacy/adopted rows)."""
    if row["target_hit"] is not None:
        return bool(row["target_hit"])
    return Decimal(str(row["profit"])) > 0


class RiskManager:
    def __init__(
        self,
        repos: Repositories,
        config: AppConfig,
        clock: Clock,
        *,
        account: AccountState,
        feed: FeedHealth,
        latency: LatencyTracker,
        model_getter: Callable[[], ModelInfo | None],
        running_getter: Callable[[], bool],
        alert: AlertFn | None = None,
        probe_getter: Callable[[], bool] | None = None,
    ) -> None:
        self._probe_enabled = probe_getter or (lambda: False)
        self._repos = repos
        self._cfg = config
        self._clock = clock
        self.account = account
        self._feed = feed
        self._latency = latency
        self._model = model_getter
        self._running = running_getter
        self._alert = alert
        self.state = RiskStateStore(repos, clock, config.tz)

    # ---- helpers ----------------------------------------------------------------------------
    def _is_probe(self, signal: Signal) -> bool:
        return signal.strategy == PROBE_NAME

    def _probe_allowed(self) -> bool:
        """The model/edge gates may be skipped ONLY for the probe, ONLY in DEMO, ONLY if enabled."""
        return self._probe_enabled() and self.mode is Mode.DEMO

    @property
    def mode(self) -> Mode:
        return self.account.mode

    @property
    def profile(self) -> RiskProfile:
        return self._cfg.risk.profile(self.mode)

    def _event(
        self,
        severity: Severity,
        rule: str,
        message: str,
        *,
        signal: Signal | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self._repos.add_risk_event(
            self.mode,
            severity,
            rule,
            message,
            symbol=signal.symbol if signal else None,
            signal_id=signal.signal_id if signal else None,
            details=details,
            ts=self._clock.now(),
        )
        if severity is Severity.CRITICAL and self._alert is not None:
            self._alert(rule, message)

    def _reject(
        self, signal: Signal, rule: str, message: str, details: dict[str, object] | None = None
    ) -> RiskDecision:
        self._event(Severity.WARNING, rule, message, signal=signal, details=details)
        self._repos.set_signal_status(signal.signal_id, "REJECTED", f"{rule}: {message}")
        return RiskDecision(approved=False, rule=rule, reason=message, details=details or {})

    # ---- kill switch / model halt -----------------------------------------------------------
    def kill_active(self) -> bool:
        return self._repos.get_bot_state(KILL_KEY, "0") == "1"

    def activate_kill(self, reason: str) -> None:
        self._repos.set_bot_state(KILL_KEY, "1")
        self._event(Severity.CRITICAL, "kill_switch", f"kill switch activated: {reason}")

    def clear_kill(self) -> None:
        """Caller (controller/API) must have verified: bot stopped + authenticated confirmation."""
        self._repos.set_bot_state(KILL_KEY, "0")
        self._event(Severity.WARNING, "kill_switch_cleared", "kill switch cleared deliberately")

    def model_halted(self, model: ModelInfo | None) -> bool:
        halted = self._repos.get_bot_state(MODEL_HALT_KEY, "")
        return bool(halted) and model is not None and halted == model.version

    def review_available(self) -> bool:
        prof = self.profile
        st = self.state
        halted = (
            st.consecutive_losses(self.mode) >= prof.max_consecutive_losses
            or st.daily_losses(self.mode) >= prof.max_losses_per_day
        )
        return halted and st.reviews_today(self.mode) < prof.max_reviews_per_day

    def resume_after_review(self, note: str) -> None:
        """Human review of a losing streak. Re-opens the loss-count rules for the rest of the day,
        a limited number of times per day. Daily/weekly money limits, drawdown and kill switch are
        NOT touched. Raises ValueError when no review is needed or allowed."""
        prof = self.profile
        if self.state.reviews_today(self.mode) >= prof.max_reviews_per_day:
            raise ValueError(
                f"already reviewed {prof.max_reviews_per_day} times today; stop for the day"
            )
        if not self.review_available():
            raise ValueError("no loss-count halt is active, nothing to review")
        self.state.note_review(self.mode)
        self._event(
            Severity.WARNING,
            "review_resume",
            f"loss-count halt reviewed and re-opened by operator: {note[:200]}",
        )

    # ---- balance / drawdown -----------------------------------------------------------------
    def update_balance(self, balance: Decimal, currency: str = "") -> None:
        """Feed a VERIFIED balance. Updates HWM and enforces the persistent drawdown halt."""
        mode = self.mode
        self.account.balance = balance
        self.account.balance_ts = self._clock.time()
        if currency:
            self.account.currency = currency
        self._repos.add_snapshot(
            mode, self.account.account_id, balance, currency, self._clock.now()
        )
        self.state.observe_balance(mode, balance)
        self._check_drawdown(balance)

    def drawdown(self, balance: Decimal | None = None) -> Decimal:
        bal = balance if balance is not None else self.account.balance
        hwm = self.state.hwm(self.mode)
        if bal is None or hwm is None or hwm <= 0:
            return Decimal(0)
        return (hwm - bal) / hwm

    def _check_drawdown(self, balance: Decimal) -> None:
        if self.state.drawdown_halt(self.mode):
            return
        dd = self.drawdown(balance)
        if dd >= self.profile.max_drawdown_percent:
            self.state.set_drawdown_halt(self.mode, True)
            self._event(
                Severity.CRITICAL,
                "drawdown_halt",
                f"drawdown {dd:.4f} >= {self.profile.max_drawdown_percent}; manual reset required",
                details={"drawdown": str(dd), "hwm": str(self.state.hwm(self.mode))},
            )

    # ---- exposure ---------------------------------------------------------------------------
    def open_exposure(self) -> Decimal:
        return sum(
            (Decimal(str(o["stake"])) for o in self._repos.active_orders(self.mode)), Decimal(0)
        )

    def _group_of(self, symbol: str) -> str | None:
        for name, members in self._cfg.risk.correlation_groups.items():
            if symbol in members:
                return name
        return None

    def group_exposure(self, symbol: str) -> Decimal:
        group = self._group_of(symbol)
        if group is None:
            return sum(
                (
                    Decimal(str(o["stake"]))
                    for o in self._repos.active_orders(self.mode)
                    if o["symbol"] == symbol
                ),
                Decimal(0),
            )
        members = set(self._cfg.risk.correlation_groups[group])
        return sum(
            (
                Decimal(str(o["stake"]))
                for o in self._repos.active_orders(self.mode)
                if o["symbol"] in members
            ),
            Decimal(0),
        )

    # ---- gate 1: signal ---------------------------------------------------------------------
    def evaluate(self, signal: Signal) -> RiskDecision:  # flat, ordered rule list by design
        mode = self.mode
        prof = self.profile
        now = self._clock.time()
        self.state.roll(mode)

        if not self._repos.insert_signal(mode, signal, self._clock.now()):
            self._event(Severity.WARNING, "duplicate_signal", "signal already seen", signal=signal)
            return RiskDecision(False, "duplicate_signal", "signal already seen")

        if self.kill_active():
            return self._reject(signal, "kill_switch", "kill switch is active")
        if not self._running():
            return self._reject(signal, "bot_stopped", "bot is not running")
        if not self.account.type_matches_mode():
            return self._reject(
                signal,
                "account_unverified",
                f"account type '{self.account.verified_type}' not verified for mode {mode.value}",
            )
        if self.state.drawdown_halt(mode):
            return self._reject(
                signal, "drawdown_halt", "drawdown halt active (manual reset needed)"
            )

        if self._is_probe(signal):
            if not self._probe_allowed():
                return self._reject(signal, "probe_not_allowed", "probe signals are DEMO-only")
            model = None
        else:
            model = self._model()
        if model is None and not self._is_probe(signal):
            return self._reject(signal, "model_unavailable", "no validated model loaded")
        allowed = (
            {ModelStatus.PROMOTABLE}
            if mode is Mode.LIVE
            else {ModelStatus.DEMO_VALIDATING, ModelStatus.PROMOTABLE}
        )
        if model is not None:
            if model.status not in allowed:
                return self._reject(
                    signal,
                    "model_status",
                    f"model status {model.status.value} may not trade in {mode.value}",
                )
            if signal.model_version != model.version:
                return self._reject(signal, "model_version", "signal model version != loaded model")
            if self.model_halted(model):
                return self._reject(
                    signal, "model_halt", "model halted by rolling performance monitor"
                )
            if signal.probability is None:
                return self._reject(
                    signal, "no_probability", "signal has no calibrated probability"
                )

        if self._feed.is_stale(signal.symbol, now, self._cfg.risk.feed_stale_after_s):
            return self._reject(signal, "stale_feed", "market feed is stale")
        if self._cfg.risk.latency_halt_enabled and self._cfg.risk.max_latency_ms > 0:
            p95 = self._latency.recent_p95("order_to_confirmation")
            if p95 is not None and p95 > self._cfg.risk.max_latency_ms:
                return self._reject(
                    signal, "high_latency", f"order confirmation p95 {p95:.0f}ms too high"
                )

        balance = self.account.balance
        if balance is None or self.account.balance_ts is None:
            return self._reject(signal, "balance_unknown", "no verified balance")
        if now - self.account.balance_ts > self._cfg.risk.balance_max_age_s:
            return self._reject(signal, "balance_stale", "balance older than allowed")
        self._check_drawdown(balance)
        if self.state.drawdown_halt(mode):
            return self._reject(signal, "drawdown_halt", "drawdown limit reached")

        if self.state.consecutive_losses(mode) >= prof.max_consecutive_losses:
            return self._reject(signal, "consecutive_losses", "max consecutive losses reached")

        if self.state.daily_losses(mode) >= prof.max_losses_per_day:
            return self._reject(
                signal,
                "daily_loss_count",
                f"{prof.max_losses_per_day} losing trades today: review the trades to continue",
            )

        # Daily / weekly loss (realized + open worst-case exposure + this stake)
        day_ref = self.state.day_start_balance(mode) or balance
        week_ref = self.state.week_start_balance(mode) or balance
        daily_limit = day_ref * prof.daily_loss_percent
        weekly_limit = week_ref * prof.weekly_loss_percent
        realized_day_loss = max(Decimal(0), -self.state.daily_pnl(mode))
        realized_week_loss = max(Decimal(0), -self.state.weekly_pnl(mode))
        if self.state.daily_halt(mode) or realized_day_loss >= daily_limit:
            if not self.state.daily_halt(mode):
                self.state.set_daily_halt(mode, True)
                self._event(Severity.CRITICAL, "daily_loss_halt", "daily loss limit reached")
            return self._reject(signal, "daily_loss_limit", "daily loss limit reached")
        if realized_week_loss >= weekly_limit:
            return self._reject(signal, "weekly_loss_limit", "weekly loss limit reached")

        if self.state.daily_trades(mode) >= prof.max_trades_per_day:
            return self._reject(signal, "max_trades_per_day", "daily trade count reached")
        last = self.state.last_trade_ts(mode)
        if last is not None and now - last < prof.cooldown_seconds:
            return self._reject(signal, "cooldown", "cooldown between trades")

        active = self._repos.active_orders(mode)
        if len(active) >= prof.max_open_trades:
            return self._reject(signal, "max_open_trades", "max open trades reached")
        if (
            sum(1 for o in active if o["symbol"] == signal.symbol)
            >= prof.max_open_trades_per_symbol
        ):
            return self._reject(signal, "max_open_per_symbol", "max open trades for symbol reached")

        stake = compute_stake(
            balance, prof.stake_percent, prof.stake_cap_percent, self._cfg.trading.stake_precision
        )
        if stake < self._cfg.trading.min_stake or stake <= 0:
            return self._reject(
                signal,
                "min_stake",
                f"computed stake {stake} below minimum {self._cfg.trading.min_stake}",
                {"stake": str(stake), "balance": str(balance)},
            )

        exposure = self.open_exposure()
        if realized_day_loss + exposure + stake > daily_limit:
            return self._reject(
                signal,
                "daily_loss_exposure",
                "realized loss + open exposure + stake would exceed daily loss limit",
                {
                    "realized": str(realized_day_loss),
                    "exposure": str(exposure),
                    "stake": str(stake),
                },
            )
        if realized_week_loss + exposure + stake > weekly_limit:
            return self._reject(
                signal, "weekly_loss_exposure", "weekly loss limit would be exceeded"
            )
        if exposure + stake > balance * prof.max_total_exposure_percent:
            return self._reject(
                signal, "max_exposure", "total open exposure limit would be exceeded"
            )
        if self.group_exposure(signal.symbol) + stake > balance * prof.max_group_exposure_percent:
            return self._reject(
                signal, "group_exposure", "correlated-symbol exposure limit would be exceeded"
            )

        order_id = uuid.uuid4().hex
        with self._repos.db.transaction():
            self._repos.insert_order(
                order_id=order_id,
                signal=signal,
                mode=mode,
                stake=stake,
                break_even=None,
                ts=self._clock.now(),
            )
            self.state.note_trade_started(mode)
            self._repos.set_signal_status(signal.signal_id, "APPROVED")
        return RiskDecision(True, stake=stake, order_id=order_id)

    # ---- gate 2: proposal -------------------------------------------------------------------
    def authorize_buy(
        self, order_id: str, signal: Signal, proposal: Proposal
    ) -> tuple[BuyPermit | None, RiskDecision]:
        """Edge gate on the ACTUAL proposal payout + stale-quote protection. Mints the permit."""
        mode = self.mode
        row = self._repos.get_order(order_id)

        def deny(
            rule: str, message: str, details: dict[str, object] | None = None
        ) -> tuple[None, RiskDecision]:
            self._event(Severity.WARNING, rule, message, signal=signal, details=details)
            self._repos.set_signal_status(signal.signal_id, "REJECTED", f"{rule}: {message}")
            if row is not None:
                self._repos.update_order(
                    order_id, state=OrderState.FAILED, error=f"{rule}: {message}"
                )
                self.state.refund_trade(mode)
            return None, RiskDecision(False, rule, message, details=details or {})

        if row is None or row["state"] != OrderState.APPROVED.value or row["mode"] != mode.value:
            return deny("order_state", "order is not in APPROVED state for this mode")
        if self.kill_active():
            return deny("kill_switch", "kill switch is active")
        if not self._running():
            return deny("bot_stopped", "bot stopped before purchase")
        probe = self._is_probe(signal)
        if probe and not self._probe_allowed():
            return deny("probe_not_allowed", "probe signals are DEMO-only")
        stake = Decimal(str(row["stake"]))
        spec = self._cfg.trading.product
        if signal.product != spec.product.value:
            return deny(
                "product_mismatch",
                f"signal product {signal.product} != configured {spec.product.value}",
            )
        slip = self._cfg.trading.max_price_slippage_percent
        # Never chase a quote: the price may exceed the stake only by the product's fee allowance.
        ceiling = stake * (Decimal(1) + slip + Decimal(str(spec.fee_allowance)))
        if proposal.ask_price > ceiling:
            return deny(
                "stale_quote",
                f"ask {proposal.ask_price} above max acceptable {ceiling}",
                {"ask": str(proposal.ask_price), "max": str(ceiling)},
            )
        max_price = min(proposal.ask_price * (Decimal(1) + slip), ceiling)
        if proposal.min_stake is not None and stake < proposal.min_stake:
            return deny("proposal_min_stake", f"stake below broker minimum {proposal.min_stake}")
        if proposal.max_stake is not None and stake > proposal.max_stake:
            return deny("proposal_max_stake", f"stake above broker maximum {proposal.max_stake}")

        # Win/loss amounts come from the ACTUAL proposal; the live contract must still match the
        # terms the model was trained for (fails closed).
        spot = proposal.spot if proposal.spot is not None else signal.entry_price
        terms = payoff_terms(spec, signal.direction, proposal, stake, spot)
        if not terms.ok:
            return deny("spec_mismatch", terms.reason, {"product": spec.product.value})
        ok, be, edge = edge_gate(signal.probability, terms, self._cfg.ml.edge_margin)
        if not ok and not probe:  # the probe deliberately has no edge (DEMO pipeline test only)
            return deny(
                "edge_gate",
                f"p={signal.probability} < break-even {be:.4f} + margin {self._cfg.ml.edge_margin}",
                {
                    "break_even": be,
                    "probability": signal.probability,
                    "edge": edge,
                    "win": terms.win,
                    "loss": terms.loss,
                },
            )
        self._repos.update_order(
            order_id,
            state=OrderState.PROPOSED,
            proposal_id=proposal.proposal_id,
            ask_price=proposal.ask_price,
            payout=proposal.payout,
            max_price=max_price,
            break_even=be,
            contract_type=contract_type(spec, signal.direction),
            win_amount=terms.win,
            loss_amount=terms.loss,
        )
        return _mint(order_id, proposal.proposal_id, max_price, mode), RiskDecision(
            True, stake=stake, order_id=order_id, details={"break_even": be}
        )

    # ---- lifecycle hooks --------------------------------------------------------------------
    def on_purchase(self) -> None:
        self.state.note_purchase(self.mode, self._clock.time())

    def on_settlement(self, profit: Decimal) -> None:
        mode = self.mode
        prof = self.profile
        self.state.apply_settlement(mode, profit)
        if self.state.consecutive_losses(mode) >= prof.max_consecutive_losses:
            self._event(
                Severity.CRITICAL, "consecutive_losses_halt", "max consecutive losses reached"
            )
        ref = self.state.day_start_balance(mode) or self.account.balance
        limit_hit = ref is not None and -self.state.daily_pnl(mode) >= ref * prof.daily_loss_percent
        if limit_hit and not self.state.daily_halt(mode):
            self.state.set_daily_halt(mode, True)
            self._event(Severity.CRITICAL, "daily_loss_halt", "daily loss limit reached")

    def check_model_performance(self, model: ModelInfo | None) -> bool:
        """Rolling-window monitor. Returns True if the model was halted."""
        if model is None:
            return False
        from app.ml.statistics import rolling_edge_check

        window = self._cfg.ml.monitor_window
        rows = self._repos.settled_trades(self.mode, limit=window)
        if len(rows) < window:
            return False
        wins = sum(1 for r in rows if _target_hit(r))
        bes = [float(r["break_even"]) for r in rows if r["break_even"] is not None]
        if not bes:
            return False
        halted = rolling_edge_check(
            wins, len(rows), sum(bes) / len(bes), self._cfg.ml.monitor_alpha
        )
        if halted:
            self._repos.set_bot_state(MODEL_HALT_KEY, model.version)
            self._event(
                Severity.CRITICAL,
                "model_performance_halt",
                f"rolling {window}-trade win rate materially below break-even; "
                "manual review needed",
                details={"wins": wins, "n": len(rows)},
            )
        return halted
