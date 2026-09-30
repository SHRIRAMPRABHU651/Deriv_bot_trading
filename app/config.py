"""Configuration: secrets from the environment (.env), everything else from config.yaml."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.schemas import Mode
from app.products import ProductSpec


class Settings(BaseSettings):
    """Secrets and switches read from the environment / .env. Never logged."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    deriv_app_id: str = ""
    deriv_demo_token: SecretStr = SecretStr("")
    deriv_demo_account_id: str = ""
    deriv_live_token: SecretStr = SecretStr("")
    deriv_live_account_id: str = ""
    allow_live: bool = False
    dashboard_token: SecretStr = SecretStr("")
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""
    demo_probe: bool = False  # DEMO-only pipeline probe (see app/strategy/probe.py)

    def token_for(self, mode: Mode) -> str:
        token = self.deriv_demo_token if mode is Mode.DEMO else self.deriv_live_token
        return token.get_secret_value()

    def account_id_for(self, mode: Mode) -> str:
        return self.deriv_demo_account_id if mode is Mode.DEMO else self.deriv_live_account_id

    def demo_configured(self) -> bool:
        return bool(self.deriv_app_id and self.token_for(Mode.DEMO) and self.deriv_demo_account_id)

    def live_credentials_present(self) -> bool:
        return bool(self.deriv_app_id and self.token_for(Mode.LIVE) and self.deriv_live_account_id)

    def live_permitted(self) -> bool:
        """LIVE is impossible unless ALLOW_LIVE=true AND live credentials exist."""
        return self.allow_live and self.live_credentials_present()


class RiskProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    stake_percent: Decimal
    stake_cap_percent: Decimal
    daily_loss_percent: Decimal
    weekly_loss_percent: Decimal
    max_drawdown_percent: Decimal = Decimal("0.10")
    max_consecutive_losses: int = 3
    max_trades_per_day: int = 200
    max_open_trades: int = 2
    max_open_trades_per_symbol: int = 1
    max_total_exposure_percent: Decimal
    max_group_exposure_percent: Decimal
    cooldown_seconds: float = 5.0


def _demo_profile() -> RiskProfile:
    return RiskProfile(
        stake_percent=Decimal("0.01"),
        stake_cap_percent=Decimal("0.01"),
        daily_loss_percent=Decimal("0.03"),
        weekly_loss_percent=Decimal("0.06"),
        max_total_exposure_percent=Decimal("0.03"),
        max_group_exposure_percent=Decimal("0.02"),
    )


def _live_profile() -> RiskProfile:
    return RiskProfile(
        stake_percent=Decimal("0.005"),
        stake_cap_percent=Decimal("0.005"),
        daily_loss_percent=Decimal("0.01"),
        weekly_loss_percent=Decimal("0.03"),
        max_open_trades=1,
        max_trades_per_day=50,
        max_total_exposure_percent=Decimal("0.01"),
        max_group_exposure_percent=Decimal("0.01"),
        cooldown_seconds=10.0,
    )


class RiskConfig(BaseModel):
    balance_max_age_s: float = 60.0
    feed_stale_after_s: float = 10.0
    max_latency_ms: float = 2000.0
    latency_halt_enabled: bool = False
    correlation_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {"volatility_indices": ["R_10", "R_25", "R_50", "R_75", "R_100"]}
    )
    demo: RiskProfile = Field(default_factory=_demo_profile)
    live: RiskProfile = Field(default_factory=_live_profile)

    def profile(self, mode: Mode) -> RiskProfile:
        return self.demo if mode is Mode.DEMO else self.live


class AppSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "data/derivbot.db"
    log_dir: str = "logs"
    model_dir: str = "model"
    timezone: str = "UTC"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:8000", "http://localhost:8000"]
    )

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        ZoneInfo(v)
        return v


class DerivSection(BaseModel):
    rest_base_url: str = "https://api.derivws.com"
    requests_per_second: float = 5.0
    burst: int = 10
    ping_interval_s: float = 15.0
    ping_timeout_s: float = 10.0
    request_timeout_s: float = 10.0
    proposal_timeout_s: float = 5.0
    buy_timeout_s: float = 8.0
    settlement_timeout_s: float = 120.0


class TradingSection(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["R_100"])
    currency: str = "USD"
    product: ProductSpec = Field(default_factory=ProductSpec)
    min_stake: Decimal = Decimal("0.35")
    stake_precision: int = 2
    max_signal_age_s: float = 2.0
    max_price_slippage_percent: Decimal = Decimal("0")
    execution_workers: int = 2
    tick_queue_size: int = 256
    signal_queue_size: int = 32

    @property
    def duration_ticks(self) -> int:
        return self.product.horizon_ticks


class StrategySection(BaseModel):
    name: str = "ml"
    version: str = "1"
    ema_fast: int = 5
    ema_slow: int = 20


class ProbeSection(BaseModel):
    """DEMO-only pipeline test mode: trades without a model to exercise buy/settle end to end."""

    enabled: bool = False
    interval_ticks: int = Field(default=10, ge=1)


class MLSection(BaseModel):
    edge_margin: float = 0.03
    alpha: float = 0.05
    min_validation_trades: int = 300
    min_demo_trades: int = 1000
    monitor_window: int = 100
    monitor_alpha: float = 0.05
    feature_version: str = "v1"


class RetentionSection(BaseModel):
    ticks_days: int = 7
    latency_days: int = 30
    store_ticks: bool = False
    cleanup_interval_s: float = 3600.0


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    deriv: DerivSection = Field(default_factory=DerivSection)
    trading: TradingSection = Field(default_factory=TradingSection)
    strategy: StrategySection = Field(default_factory=StrategySection)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLSection = Field(default_factory=MLSection)
    probe: ProbeSection = Field(default_factory=ProbeSection)
    retention: RetentionSection = Field(default_factory=RetentionSection)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.app.timezone)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load config.yaml (or the given path); fall back to safe defaults when absent."""
    candidate = Path(path) if path else Path(os.environ.get("DERIVBOT_CONFIG", "config.yaml"))
    if not candidate.exists():
        return AppConfig()
    raw: Any = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{candidate}: top level must be a mapping")
    return AppConfig.model_validate(raw)
