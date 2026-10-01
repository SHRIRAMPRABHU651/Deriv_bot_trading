"""RiskManager: every rule, with exact-boundary / boundary-epsilon / boundary+epsilon cases."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.models.schemas import Mode, ModelStatus, OrderState
from app.risk.limits import (
    break_even_probability,
    compute_stake,
    expected_value,
    floor_to_precision,
)
from app.risk.manager import ModelInfo
from app.risk.permit import BuyPermit, PermitError
from tests.conftest import count_rows
from tests.unit.risk_harness import Harness, make_harness, proposal

D = Decimal


def rules(h: Harness) -> list[str]:
    return [r["rule"] for r in h.repos.recent_risk_events(200)]


def approve(h: Harness, symbol: str = "R_100") -> str:
    dec = h.risk.evaluate(h.signal(symbol))
    assert dec.approved, (dec.rule, dec.reason)
    assert dec.order_id is not None
    return dec.order_id


# ---------------------------------------------------------------- stake sizing
def test_stake_is_percentage_of_current_balance() -> None:
    h = make_harness(balance="1000")
    dec = h.risk.evaluate(h.signal())
    assert dec.approved and dec.stake == D("10.00")


def test_stake_uses_current_balance_not_starting_balance() -> None:
    h = make_harness(balance="1000")
    h.set_balance("950")  # within the drawdown limit
    dec = h.risk.evaluate(h.signal())
    assert dec.stake == D("9.50")


def test_stake_rounds_down_never_up() -> None:
    assert floor_to_precision(D("1.2399999"), 2) == D("1.23")
    assert floor_to_precision(D("1.239"), 2) == D("1.23")
    assert compute_stake(D("123.456"), D("0.01"), D("0.01"), 2) == D("1.23")
    h = make_harness(balance="123.456")
    dec = h.risk.evaluate(h.signal())
    assert dec.stake == D("1.23")


def test_stake_capped_by_cap_percent() -> None:
    assert compute_stake(D("1000"), D("0.05"), D("0.01"), 2) == D("10.00")


@pytest.mark.parametrize(
    ("balance", "ok"),
    [("35", True), ("35.01", True), ("34.99", False), ("34", False), ("10", False)],
)
def test_minimum_stake_boundary_no_rounding_up(balance: str, ok: bool) -> None:
    h = make_harness(balance=balance)
    dec = h.risk.evaluate(h.signal())
    assert dec.approved is ok
    if not ok:
        assert dec.rule == "min_stake"  # rejected, NOT silently raised to the minimum
        assert h.repos.active_orders(Mode.DEMO) == []


def test_stale_balance_rejected_and_fresh_balance_accepted() -> None:
    h = make_harness()
    h.clock.advance(61)
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "balance_stale"
    h.set_balance("1000")
    assert h.risk.evaluate(h.signal()).approved


# ---------------------------------------------------------------- daily loss
@pytest.mark.parametrize(
    ("loss", "ok"),
    [("19.99", True), ("20", True), ("20.01", False)],
)
def test_daily_loss_projection_boundary(loss: str, ok: bool) -> None:
    # limit = 3% of 1000 = 30; stake = 10 => realized + stake must be <= 30
    h = make_harness()
    h.risk.on_settlement(-D(loss))
    dec = h.risk.evaluate(h.signal())
    assert dec.approved is ok
    if not ok:
        assert dec.rule == "daily_loss_exposure"


def test_daily_loss_limit_exactly_reached_halts_and_persists() -> None:
    h = make_harness()
    h.risk.on_settlement(D("-30"))
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "daily_loss_limit"
    assert h.risk.state.daily_halt(Mode.DEMO)
    assert not h.new_risk().evaluate(h.signal()).approved  # survives a restart


def test_daily_loss_just_below_limit_is_not_a_halt_but_blocks_new_stake() -> None:
    h = make_harness()
    h.risk.on_settlement(D("-29.99"))
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "daily_loss_exposure"
    assert not h.risk.state.daily_halt(Mode.DEMO)


@pytest.mark.parametrize(("open_stake", "ok"), [("10", True), ("10.01", False), ("9.99", True)])
def test_daily_loss_counts_open_exposure(open_stake: str, ok: bool) -> None:
    h = make_harness()
    h.risk.on_settlement(-D("10"))  # realized 10
    h.add_active_order("R_50", open_stake)  # worst-case open loss
    dec = h.risk.evaluate(h.signal("R_100"))  # + new stake 10  => 10 + open + 10 vs 30
    assert dec.approved is ok
    if not ok:
        assert dec.rule == "daily_loss_exposure"


def test_docs_example_realized_plus_open_exposure_rejects() -> None:
    # realized 25 (2.5%) + open 10 (1%) => a new trade must be rejected under a 3% limit
    h = make_harness()
    h.risk.on_settlement(D("-25"))
    h.add_active_order("R_50", "10")
    assert not h.risk.evaluate(h.signal()).approved


# ---------------------------------------------------------------- weekly loss
@pytest.mark.parametrize(("loss", "ok"), [("49.99", True), ("50", True), ("50.01", False)])
def test_weekly_loss_projection_boundary(loss: str, ok: bool) -> None:
    h = make_harness(profile={"daily_loss_percent": D("0.9")})  # isolate the weekly rule
    h.risk.on_settlement(-D(loss))
    dec = h.risk.evaluate(h.signal())
    assert dec.approved is ok
    if not ok:
        assert dec.rule == "weekly_loss_exposure"


def test_weekly_limit_reached() -> None:
    h = make_harness(profile={"daily_loss_percent": D("0.9")})
    h.risk.on_settlement(D("-60"))
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "weekly_loss_limit"


# ---------------------------------------------------------------- drawdown / HWM
def test_drawdown_boundary_and_persistence() -> None:
    h = make_harness(balance="1000")
    h.set_balance("900.01")  # 9.999% => below the 10% limit
    assert not h.risk.state.drawdown_halt(Mode.DEMO)
    assert h.risk.evaluate(h.signal()).approved
    h.set_balance("900")  # exactly 10%
    assert h.risk.state.drawdown_halt(Mode.DEMO)
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "drawdown_halt"
    # persistent: new process, two days later, even a recovered balance does not clear it
    h.clock.advance(2 * 86400)
    h.set_balance("1000")
    assert h.risk.state.drawdown_halt(Mode.DEMO)
    assert not h.new_risk().evaluate(h.signal()).approved
    assert any(
        r["severity"] == "CRITICAL" and r["rule"] == "drawdown_halt"
        for r in h.repos.recent_risk_events(50)
    )


def test_drawdown_manual_reset_rebases_hwm() -> None:
    h = make_harness(balance="1000")
    h.set_balance("850")
    assert h.risk.state.drawdown_halt(Mode.DEMO)
    h.risk.state.reset_drawdown(Mode.DEMO, D("850"))
    assert not h.risk.state.drawdown_halt(Mode.DEMO)
    assert h.risk.state.hwm(Mode.DEMO) == D("850")
    assert h.risk.evaluate(h.signal()).approved


def test_high_water_mark_only_rises_from_verified_balances() -> None:
    h = make_harness(balance="1000")
    h.set_balance("1200")
    assert h.risk.state.hwm(Mode.DEMO) == D("1200")
    h.set_balance("1100")
    assert h.risk.state.hwm(Mode.DEMO) == D("1200")
    assert h.risk.drawdown() == (D("1200") - D("1100")) / D("1200")


# ---------------------------------------------------------------- counts / exposure
def test_max_trades_per_day_and_reset_next_day() -> None:
    h = make_harness(profile={"max_trades_per_day": 2, "max_open_trades_per_symbol": 10})
    approve(h)
    approve(h)
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "max_trades_per_day"
    h.clock.advance(86400)
    h.set_balance("1000")
    assert h.risk.evaluate(h.signal()).approved


def test_max_open_trades_and_per_symbol() -> None:
    h = make_harness(profile={"max_open_trades": 2})
    approve(h, "R_100")
    dec = h.risk.evaluate(h.signal("R_100"))
    assert not dec.approved and dec.rule == "max_open_per_symbol"
    approve(h, "R_50")
    dec = h.risk.evaluate(h.signal("R_25"))
    assert not dec.approved and dec.rule == "max_open_trades"


def test_reconciliation_pending_still_counts_as_open_exposure() -> None:
    h = make_harness(profile={"max_open_trades": 1})
    h.add_active_order("R_100", "10", OrderState.RECONCILIATION_PENDING)
    assert h.risk.open_exposure() == D("10")
    assert h.risk.evaluate(h.signal("R_50")).rule == "max_open_trades"


@pytest.mark.parametrize(("pct", "third_ok"), [("0.03", True), ("0.0299", False)])
def test_total_exposure_boundary(pct: str, third_ok: bool) -> None:
    h = make_harness(profile={"max_total_exposure_percent": D(pct), "max_open_trades": 5})
    approve(h, "R_100")
    approve(h, "frxEURUSD")
    dec = h.risk.evaluate(h.signal("frxGBPUSD"))  # 30 <= 30 when pct=3%
    assert dec.approved is third_ok
    if not third_ok:
        assert dec.rule == "max_exposure"


def test_correlated_symbol_group_exposure() -> None:
    h = make_harness(profile={"max_group_exposure_percent": D("0.02"), "max_open_trades": 9})
    approve(h, "R_100")
    approve(h, "R_50")  # 20 <= 20
    dec = h.risk.evaluate(h.signal("R_75"))  # 30 > 20 (same group)
    assert not dec.approved and dec.rule == "group_exposure"
    assert h.risk.evaluate(h.signal("frxEURUSD")).approved  # uncorrelated symbol is fine


def test_cooldown_boundary() -> None:
    h = make_harness(profile={"cooldown_seconds": 30.0})
    approve(h, "R_100")
    h.risk.on_purchase()
    h.clock.advance(29.9)
    h.set_balance("1000")
    dec = h.risk.evaluate(h.signal("R_50"))
    assert not dec.approved and dec.rule == "cooldown"
    h.clock.advance(0.1)
    h.set_balance("1000")
    assert h.risk.evaluate(h.signal("R_50")).approved


# ---------------------------------------------------------------- consecutive losses
def test_consecutive_losses_halt_win_resets_and_persists() -> None:
    h = make_harness()
    h.risk.on_settlement(D("-1"))
    h.risk.on_settlement(D("-1"))
    assert h.risk.evaluate(h.signal()).approved  # only 2 losses
    h.risk.on_settlement(D("2"))  # a win resets the streak
    assert h.risk.state.consecutive_losses(Mode.DEMO) == 0
    for _ in range(3):
        h.risk.on_settlement(D("-1"))
    dec = h.risk.evaluate(h.signal("R_50"))
    assert not dec.approved and dec.rule == "consecutive_losses"
    assert h.new_risk().state.consecutive_losses(Mode.DEMO) == 3  # persists through restart
    assert any(r["rule"] == "consecutive_losses_halt" for r in h.repos.recent_risk_events(50))


# ---------------------------------------------------------------- resets
def test_daily_reset_at_midnight_clears_daily_state_only() -> None:
    h = make_harness(start=datetime(2026, 9, 30, 23, 59, 0, tzinfo=UTC))
    h.risk.on_settlement(D("-30"))  # halts the day
    h.risk.state.note_trade_started(Mode.DEMO)
    assert h.risk.state.daily_halt(Mode.DEMO)
    h.clock.advance(120)  # -> Oct 1
    st = h.risk.state
    assert st.daily_pnl(Mode.DEMO) == 0
    assert st.daily_trades(Mode.DEMO) == 0
    assert st.consecutive_losses(Mode.DEMO) == 0
    assert not st.daily_halt(Mode.DEMO)
    assert st.weekly_pnl(Mode.DEMO) == D("-30")  # same ISO week (Wed -> Thu): not reset


def test_weekly_reset_on_monday_boundary() -> None:
    h = make_harness(start=datetime(2026, 10, 4, 23, 59, 59, tzinfo=UTC))  # Sunday
    h.risk.on_settlement(D("-5"))
    assert h.risk.state.weekly_pnl(Mode.DEMO) == D("-5")
    h.clock.advance(1)  # Monday 00:00:00 UTC
    assert h.risk.state.weekly_pnl(Mode.DEMO) == 0


def test_day_boundary_uses_configured_timezone() -> None:
    from zoneinfo import ZoneInfo

    h = make_harness(start=datetime(2026, 9, 30, 14, 30, 0, tzinfo=UTC))
    h.config.app.timezone = "Asia/Tokyo"
    st = type(h.risk.state)(h.repos, h.clock, ZoneInfo("Asia/Tokyo"))
    st.roll(Mode.DEMO)
    st.note_trade_started(Mode.DEMO)
    h.clock.advance(1800)  # 15:00 UTC == 00:00 JST next day
    assert st.daily_trades(Mode.DEMO) == 0


# ---------------------------------------------------------------- kill switch
def test_kill_switch_blocks_persists_and_clears_deliberately() -> None:
    h = make_harness()
    h.risk.activate_kill("test")
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "kill_switch"
    assert h.new_risk().kill_active()  # persisted
    assert any(
        r["severity"] == "CRITICAL" and r["rule"] == "kill_switch"
        for r in h.repos.recent_risk_events(10)
    )
    h.risk.clear_kill()
    assert h.risk.evaluate(h.signal()).approved


# ---------------------------------------------------------------- feed / latency / bot / account
def test_stale_feed_halts_and_recovers() -> None:
    h = make_harness(risk_cfg={"feed_stale_after_s": 10.0})
    sig = h.signal()
    h.clock.advance(11)
    h.set_balance("1000")
    dec = h.risk.evaluate(sig)
    assert not dec.approved and dec.rule == "stale_feed"
    h.feed.on_tick("R_100", h.clock.time())
    assert h.risk.evaluate(h.signal()).approved


def test_high_latency_halt_only_when_enabled() -> None:
    h = make_harness(risk_cfg={"latency_halt_enabled": True, "max_latency_ms": 100.0})
    for _ in range(6):
        h.latency.record(Mode.DEMO, "order_to_confirmation", 500.0, h.clock.now())
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "high_latency"
    h2 = make_harness(risk_cfg={"latency_halt_enabled": False, "max_latency_ms": 100.0})
    for _ in range(6):
        h2.latency.record(Mode.DEMO, "order_to_confirmation", 500.0, h2.clock.now())
    assert h2.risk.evaluate(h2.signal()).approved


def test_bot_stopped_and_account_unverified() -> None:
    h = make_harness()
    h.flags["running"] = False
    assert h.risk.evaluate(h.signal()).rule == "bot_stopped"
    h.flags["running"] = True
    h.account.verified_type = "real"  # DEMO mode but the API says REAL account
    assert h.risk.evaluate(h.signal()).rule == "account_unverified"
    h.account.verified_type = None
    assert h.risk.evaluate(h.signal()).rule == "account_unverified"


# ---------------------------------------------------------------- model gates
def test_model_unavailable_status_version_and_probability() -> None:
    h = make_harness()
    h.flags["model"] = None
    assert h.risk.evaluate(h.signal()).rule == "model_unavailable"
    h.flags["model"] = ModelInfo("m1", ModelStatus.BACKTESTED, "v1")
    assert h.risk.evaluate(h.signal()).rule == "model_status"
    h.flags["model"] = ModelInfo("m1", ModelStatus.DEMO_VALIDATING, "v1")
    assert h.risk.evaluate(h.signal(model_version="other")).rule == "model_version"
    assert h.risk.evaluate(h.signal(p=None)).rule == "no_probability"


def test_live_requires_promotable_model_and_uses_live_profile() -> None:
    h = make_harness(mode=Mode.LIVE, model_status=ModelStatus.DEMO_VALIDATING)
    assert h.risk.evaluate(h.signal()).rule == "model_status"
    h.flags["model"] = ModelInfo("m1", ModelStatus.PROMOTABLE, "v1")
    dec = h.risk.evaluate(h.signal())
    assert dec.approved and dec.stake == D("5.00")  # LIVE default 0.5%


def test_live_default_limits_are_stricter_than_demo() -> None:
    h = make_harness()
    demo, live = h.config.risk.demo, h.config.risk.live
    assert live.stake_cap_percent < demo.stake_cap_percent
    assert live.daily_loss_percent < demo.daily_loss_percent
    assert demo.stake_cap_percent == D("0.01") and live.stake_cap_percent == D("0.005")
    assert demo.daily_loss_percent == D("0.03") and live.daily_loss_percent == D("0.01")
    assert demo.max_drawdown_percent == D("0.10")


# ---------------------------------------------------------------- duplicates
def test_duplicate_signal_rejected_and_persisted_across_restart() -> None:
    h = make_harness()
    sig = h.signal(sid="fixed-id")
    assert h.risk.evaluate(sig).approved
    dec = h.risk.evaluate(sig)
    assert not dec.approved and dec.rule == "duplicate_signal"
    assert not h.new_risk().evaluate(sig).approved
    assert count_rows(h.db, "signals") == 1


def test_every_rejection_is_stored_as_risk_event() -> None:
    h = make_harness()
    h.risk.activate_kill("x")
    for _ in range(3):
        h.risk.evaluate(h.signal())
    assert rules(h).count("kill_switch") >= 3 + 1  # 3 rejections + the activation itself


# ---------------------------------------------------------------- edge gate (proposal stage)
def _edge(h: Harness, p: float, payout: str = "19.5", ask: str = "10") -> BuyPermit | None:
    sig = h.signal(p=p)
    dec = h.risk.evaluate(sig)
    assert dec.approved and dec.order_id
    permit, _ = h.risk.authorize_buy(dec.order_id, sig, proposal(payout=payout, ask=ask))
    return permit


def test_edge_gate_boundary_uses_dynamic_break_even() -> None:
    be = break_even_probability(D("1.95"))
    thr = be + 0.03
    assert _edge(make_harness(), thr) is not None  # exact boundary passes
    assert _edge(make_harness(), thr + 1e-6) is not None
    assert _edge(make_harness(), thr - 1e-6) is None
    assert expected_value(0.6, D("1.95")) == pytest.approx(0.17)


def test_edge_gate_follows_the_actual_payout_not_a_hardcoded_number() -> None:
    # payout ratio 1.5 => break-even 0.6667, threshold 0.6967
    assert _edge(make_harness(), 0.70, payout="15") is not None
    assert _edge(make_harness(), 0.69, payout="15") is None
    # a rich payout lowers the bar
    assert _edge(make_harness(), 0.54, payout="20") is not None  # BE 0.5 -> threshold 0.53
    assert _edge(make_harness(), 0.52, payout="20") is None


def test_bad_payout_rejected() -> None:
    assert _edge(make_harness(), 0.9, payout="10") is None


@pytest.mark.parametrize(("ask", "ok"), [("10", True), ("10.01", False)])
def test_stale_quote_protection(ask: str, ok: bool) -> None:
    h = make_harness()
    permit = _edge(h, 0.9, ask=ask)
    assert (permit is not None) is ok
    if permit is not None:
        assert permit.max_price == D("10")
    else:
        assert "stale_quote" in rules(h)


def test_slippage_allowance_is_configurable_but_bounded() -> None:
    for ask, ok in (("10.10", True), ("10.11", False)):
        h = make_harness()
        h.config.trading.max_price_slippage_percent = D("0.01")
        assert (_edge(h, 0.9, payout="19.6", ask=ask) is not None) is ok


def test_authorize_is_single_use_and_refunds_trade_slot_on_denial() -> None:
    h = make_harness(profile={"max_trades_per_day": 1, "max_open_trades_per_symbol": 5})
    sig = h.signal(p=0.9)
    dec = h.risk.evaluate(sig)
    assert dec.order_id
    p1, _ = h.risk.authorize_buy(dec.order_id, sig, proposal())
    assert p1 is not None
    p2, d2 = h.risk.authorize_buy(dec.order_id, sig, proposal())
    assert p2 is None and d2.rule == "order_state"
    # a denied proposal returns the daily trade slot
    h2 = make_harness(profile={"max_trades_per_day": 1})
    s2 = h2.signal(p=0.9)
    d = h2.risk.evaluate(s2)
    assert d.order_id
    assert h2.risk.authorize_buy(d.order_id, s2, proposal(ask="11"))[0] is None
    assert h2.risk.state.daily_trades(Mode.DEMO) == 0


def test_kill_between_risk_and_buy_blocks_the_purchase() -> None:
    h = make_harness()
    sig = h.signal(p=0.9)
    dec = h.risk.evaluate(sig)
    assert dec.order_id
    h.risk.activate_kill("mid-flight")
    permit, verdict = h.risk.authorize_buy(dec.order_id, sig, proposal())
    assert permit is None and verdict.rule == "kill_switch"


def test_buy_permit_cannot_be_forged() -> None:
    with pytest.raises(PermitError):
        BuyPermit("o", "p", D("1"), Mode.DEMO)
    with pytest.raises(PermitError):
        BuyPermit("o", "p", D("1"), Mode.DEMO, _key=object())


# ---------------------------------------------------------------- rolling model monitor
def _settled(h: Harness, wins: int, n: int, be: float = 0.5128) -> None:
    for i in range(n):
        sig = h.signal()
        oid = f"m-{i}"
        h.repos.insert_order(
            order_id=oid, signal=sig, mode=Mode.DEMO, stake=D("1"), break_even=be, ts=h.clock.now()
        )
        h.repos.insert_trade(
            contract_id=10_000 + i,
            order_id=oid,
            mode=Mode.DEMO,
            symbol="R_100",
            direction="CALL",
            stake=D("1"),
            buy_price=D("1"),
            payout=D("1.95"),
            probability=0.6,
            break_even=be,
            ts=h.clock.now(),
        )
        h.repos.update_order(oid, state=OrderState.WON if i < wins else OrderState.LOST)
        h.repos.settle_trade(
            10_000 + i,
            OrderState.WON if i < wins else OrderState.LOST,
            D("0.95") if i < wins else D("-1"),
            None,
            None,
            h.clock.now(),
        )
        h.clock.advance(1)


def test_rolling_monitor_halts_on_material_underperformance_and_new_model_clears() -> None:
    h = make_harness()
    model = h.flags["model"]
    _settled(h, wins=30, n=100)
    assert h.risk.check_model_performance(model) is True
    dec = h.risk.evaluate(h.signal())
    assert not dec.approved and dec.rule == "model_halt"
    h.flags["model"] = ModelInfo("m2", ModelStatus.DEMO_VALIDATING, "v1")  # manual re-promotion
    h.set_balance("1000")
    assert h.risk.evaluate(h.signal(model_version="m2")).approved


def test_rolling_monitor_does_not_halt_normal_variance() -> None:
    h = make_harness()
    _settled(h, wins=52, n=100)
    assert h.risk.check_model_performance(h.flags["model"]) is False


# ---------------------------------------------------------------- losses per day + review
def test_three_losses_in_a_day_stop_trading_until_a_review() -> None:
    h = make_harness()
    for pnl in ("-1", "2", "-1", "-1", "2", "-1"):  # 4 losses, never 3 in a row
        h.risk.on_settlement(D(pnl))
    dec = h.risk.evaluate(h.signal("R_50"))
    assert not dec.approved and dec.rule == "daily_loss_count"
    assert h.risk.review_available()

    h.risk.resume_after_review("spreads were wide; moved to a calmer symbol")
    assert h.risk.evaluate(h.signal("R_75")).approved
    assert any(r["rule"] == "review_resume" for r in h.repos.recent_risk_events(50))


def test_review_is_limited_per_day_and_never_clears_money_halts() -> None:
    import pytest

    h = make_harness()
    cap = h.risk.profile.max_reviews_per_day
    for i in range(cap):
        for _ in range(3):
            h.risk.on_settlement(D("-0.1"))
        h.risk.resume_after_review(f"review {i}")
    for _ in range(3):
        h.risk.on_settlement(D("-0.1"))
    with pytest.raises(ValueError, match="stop for the day"):
        h.risk.resume_after_review("one more")
    assert h.risk.evaluate(h.signal("R_50")).rule in {"consecutive_losses", "daily_loss_count"}

    h2 = make_harness()
    with pytest.raises(ValueError, match="nothing to review"):
        h2.risk.resume_after_review("nothing happened")
    h2.risk.on_settlement(D("-30"))  # breaks the daily MONEY limit
    h2.risk.on_settlement(D("-1"))
    h2.risk.on_settlement(D("-1"))
    h2.risk.resume_after_review("reviewed")
    assert h2.risk.state.daily_halt(Mode.DEMO)  # money halt untouched
    assert not h2.risk.evaluate(h2.signal("R_50")).approved


def test_next_day_resets_the_loss_count_and_reviews() -> None:
    h = make_harness(start=datetime(2026, 9, 30, 23, 59, 0, tzinfo=UTC))
    for _ in range(3):
        h.risk.on_settlement(D("-0.1"))
    h.risk.resume_after_review("ok")
    h.clock.advance(120)
    assert h.risk.state.daily_losses(Mode.DEMO) == 0
    assert h.risk.state.reviews_today(Mode.DEMO) == 0
