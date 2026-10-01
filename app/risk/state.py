"""Persistent risk state. The database is authoritative; in-memory copies are never trusted."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.clock import Clock
from app.models.schemas import Mode
from app.storage.repositories import Repositories


@dataclass
class AccountState:
    """Latest verified account facts for the active mode (updated by the controller)."""

    mode: Mode = Mode.DEMO
    account_id: str = ""
    balance: Decimal | None = None
    balance_ts: float | None = None  # epoch seconds of the last verified balance
    currency: str = ""
    verified_type: str | None = None  # "demo" | "real" once the API confirmed it

    def type_matches_mode(self) -> bool:
        expected = "demo" if self.mode is Mode.DEMO else "real"
        return self.verified_type == expected


def _dec(value: str | None, default: Decimal = Decimal(0)) -> Decimal:
    return default if value in (None, "") else Decimal(str(value))


class RiskStateStore:
    def __init__(self, repos: Repositories, clock: Clock, tz: ZoneInfo) -> None:
        self._r = repos
        self._clock = clock
        self._tz = tz

    # ---- calendar helpers -------------------------------------------------------------------
    def today(self) -> date:
        return self._clock.now().astimezone(self._tz).date()

    def week_start(self) -> date:
        d = self.today()
        return d - timedelta(days=d.weekday())  # Monday

    # ---- rollover ---------------------------------------------------------------------------
    def roll(self, mode: Mode) -> None:
        """Reset daily/weekly counters if the calendar moved. Called on EVERY state access."""
        day = self.today().isoformat()
        week = self.week_start().isoformat()
        stored_day = self._r.get_risk(mode, "day")
        stored_week = self._r.get_risk(mode, "week")
        if stored_day == day and stored_week == week:
            return
        with self._r.db.transaction():
            if stored_day != day:
                self._r.set_risk(mode, "day", day)
                self._r.set_risk(mode, "daily_pnl", "0")
                self._r.set_risk(mode, "daily_trades", "0")
                self._r.set_risk(mode, "consecutive_losses", "0")
                self._r.set_risk(mode, "daily_losses", "0")
                self._r.set_risk(mode, "reviews_today", "0")
                self._r.set_risk(mode, "daily_halt", "0")
                self._r.set_risk(mode, "day_start_balance", "")
            if stored_week != week:
                self._r.set_risk(mode, "week", week)
                self._r.set_risk(mode, "weekly_pnl", "0")
                self._r.set_risk(mode, "week_start_balance", "")

    # ---- reads (always roll first) ----------------------------------------------------------
    def daily_pnl(self, mode: Mode) -> Decimal:
        self.roll(mode)
        return _dec(self._r.get_risk(mode, "daily_pnl"))

    def weekly_pnl(self, mode: Mode) -> Decimal:
        self.roll(mode)
        return _dec(self._r.get_risk(mode, "weekly_pnl"))

    def daily_trades(self, mode: Mode) -> int:
        self.roll(mode)
        return int(self._r.get_risk(mode, "daily_trades", "0") or 0)

    def consecutive_losses(self, mode: Mode) -> int:
        self.roll(mode)
        return int(self._r.get_risk(mode, "consecutive_losses", "0") or 0)

    def daily_losses(self, mode: Mode) -> int:
        self.roll(mode)
        return int(self._r.get_risk(mode, "daily_losses", "0") or 0)

    def reviews_today(self, mode: Mode) -> int:
        self.roll(mode)
        return int(self._r.get_risk(mode, "reviews_today", "0") or 0)

    def daily_halt(self, mode: Mode) -> bool:
        self.roll(mode)
        return self._r.get_risk(mode, "daily_halt", "0") == "1"

    def drawdown_halt(self, mode: Mode) -> bool:
        return self._r.get_risk(mode, "drawdown_halt", "0") == "1"

    def hwm(self, mode: Mode) -> Decimal | None:
        v = self._r.get_risk(mode, "hwm")
        return None if v in (None, "") else Decimal(str(v))

    def day_start_balance(self, mode: Mode) -> Decimal | None:
        self.roll(mode)
        v = self._r.get_risk(mode, "day_start_balance")
        return None if v in (None, "") else Decimal(str(v))

    def week_start_balance(self, mode: Mode) -> Decimal | None:
        self.roll(mode)
        v = self._r.get_risk(mode, "week_start_balance")
        return None if v in (None, "") else Decimal(str(v))

    def last_trade_ts(self, mode: Mode) -> float | None:
        v = self._r.get_risk(mode, "last_trade_ts")
        return None if v in (None, "") else float(v)

    # ---- writes -----------------------------------------------------------------------------
    def set_daily_halt(self, mode: Mode, value: bool) -> None:
        self._r.set_risk(mode, "daily_halt", "1" if value else "0")

    def set_drawdown_halt(self, mode: Mode, value: bool) -> None:
        self._r.set_risk(mode, "drawdown_halt", "1" if value else "0")

    def note_trade_started(self, mode: Mode) -> None:
        """Count an approved order toward the daily trade limit."""
        self.roll(mode)
        self._r.set_risk(mode, "daily_trades", str(self.daily_trades(mode) + 1))

    def refund_trade(self, mode: Mode) -> None:
        """An approved order never reached Deriv (rejected at proposal gate): refund the slot."""
        self.roll(mode)
        self._r.set_risk(mode, "daily_trades", str(max(0, self.daily_trades(mode) - 1)))

    def note_purchase(self, mode: Mode, ts: float) -> None:
        self._r.set_risk(mode, "last_trade_ts", str(ts))

    def apply_settlement(self, mode: Mode, profit: Decimal) -> None:
        self.roll(mode)
        with self._r.db.transaction():
            self._r.set_risk(mode, "daily_pnl", str(self.daily_pnl(mode) + profit))
            self._r.set_risk(mode, "weekly_pnl", str(self.weekly_pnl(mode) + profit))
            if profit < 0:
                self._r.set_risk(mode, "consecutive_losses", str(self.consecutive_losses(mode) + 1))
                self._r.set_risk(mode, "daily_losses", str(self.daily_losses(mode) + 1))
            elif profit > 0:
                self._r.set_risk(mode, "consecutive_losses", "0")

    def observe_balance(self, mode: Mode, balance: Decimal) -> None:
        """Record a VERIFIED balance: seeds day/week start balances, raises the high-water mark."""
        self.roll(mode)
        with self._r.db.transaction():
            if self.day_start_balance(mode) is None:
                self._r.set_risk(mode, "day_start_balance", str(balance))
            if self.week_start_balance(mode) is None:
                self._r.set_risk(mode, "week_start_balance", str(balance))
            hwm = self.hwm(mode)
            if hwm is None or balance > hwm:
                self._r.set_risk(mode, "hwm", str(balance))

    def reset_drawdown(self, mode: Mode, new_hwm: Decimal) -> None:
        """Deliberate manual reset of the persistent drawdown halt."""
        with self._r.db.transaction():
            self._r.set_risk(mode, "drawdown_halt", "0")
            self._r.set_risk(mode, "hwm", str(new_hwm))

    def reset_streak(self, mode: Mode) -> None:
        self._r.set_risk(mode, "consecutive_losses", "0")

    def note_review(self, mode: Mode) -> None:
        """A human reviewed the losing trades: re-open the day's loss allowance (not the money
        limits: daily/weekly loss and drawdown halts are untouched)."""
        self.roll(mode)
        with self._r.db.transaction():
            self._r.set_risk(mode, "consecutive_losses", "0")
            self._r.set_risk(mode, "daily_losses", "0")
            self._r.set_risk(mode, "reviews_today", str(self.reviews_today(mode) + 1))

    def snapshot(self, mode: Mode) -> dict[str, str]:
        self.roll(mode)
        return self._r.all_risk(mode)

    @staticmethod
    def parse_iso(ts: str) -> datetime:
        return datetime.fromisoformat(ts)
