"""
================================================================================
CONFIG MODELS
================================================================================
Pydantic v2 models for backtest and live/testnet configurations.

Models:
    PathsConfig: Directory paths (data_dir, logs_dir, etc.)
    LoggingConfig: Logging level and format
    BotConfig: Bot name + strategy params (flat, extra="allow")
    CostsConfig: Trading costs (slippage, fees)
    FundingConfig: Funding rate model (fixed, historical, ema_ou)
    MetricsConfig: Backtest metrics to report
    BacktestDataConfig: Backtest data source (start_date, end_date, symbol)
    BacktestConfig: Root config for backtest.yaml

    ExchangeConfig: Live exchange settings (name, market_type)
    RiskConfig: Live risk limits (max drawdown, consecutive errors)
    LiveConfig: Root config for live.yaml and testnet.yaml (testnet is a flag, not a separate model)
================================================================================
"""

from __future__ import annotations

import warnings
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from lighthouse.domain.timeframe import Timeframe
from lighthouse.notifications.notifier import ALERT_CATEGORIES


############ SHARED MODELS ############

class PathsConfig(BaseModel):
    """Directory paths for data, logs, state, results."""
    model_config = ConfigDict(extra="forbid")
    
    data_dir: str = "backtest_data"
    logs_dir: str = "logs"
    state_dir: str | None = Field(default="state", description="Directory for the live-state SQLite store (runtime/state_store.py); null disables state persistence")
    results_dir: str | None = Field(default="results", description="Reserved: result-file writing not yet implemented")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    model_config = ConfigDict(extra="forbid")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class BotConfig(BaseModel):
    """Bot instance config (flat, allows extra strategy params)."""
    model_config = ConfigDict(extra="allow")  # Strategy params go here
    
    name: str = Field(..., description="Bot class name (e.g. 'crypto_bot')")
    symbol: str = Field(..., description="Trading symbol (ccxt format, e.g. 'BTC/USDT:USDT')")
    quote_currency: str = Field(..., description="Currency equity is denominated in, e.g. USDT / USD")
    timeframe: str = Field(..., description="Candle timeframe (e.g. '1h', '4h')")
    summary_interval_seconds: int | None = Field(default=None, ge=1, description="Live-only: seconds between periodic open-position/P&L summary alerts; null disables")

    @field_validator("timeframe")
    @classmethod
    def _check_timeframe(cls, value: str) -> str:
        """Validate at load time via Timeframe.parse(); stays a str so convert.py remains the sole translation layer."""
        Timeframe.parse(value)
        return value


############ BACKTEST-SPECIFIC MODELS ############

class CostsConfig(BaseModel):
    """Trading costs for backtest simulation (fed into BacktestExchange)."""
    model_config = ConfigDict(extra="forbid")
    
    spread: float = Field(default=0.0005, ge=0.0)
    taker_fee: float = Field(default=0.0006, ge=0.0)
    maker_fee: float = Field(default=0.0002, ge=0.0)
    slippage: float = Field(default=0.0001, ge=0.0)
    latency_bars: int = Field(default=1, ge=1, description="Must be >=1; 0 would cause look-ahead bias")
    partial_fill_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    limit_fill_policy: Literal["through", "touch"] = "through"


class FundingConfigFixed(BaseModel):
    """Fixed funding rate: overrides the instrument's base_funding_rate."""
    model_config = ConfigDict(extra="forbid")
    
    model: Literal["fixed"] = "fixed"
    rate: float = Field(default=0.0001, description="Overrides InstrumentSpec.base_funding_rate")


class FundingConfigHistorical(BaseModel):
    """Historical funding rates, replayed from <data_dir>/funding/<symbol>.csv."""
    model_config = ConfigDict(extra="forbid")
    
    model: Literal["historical"] = "historical"
    symbol: str = Field(..., description="Funding-rate CSV filename stem, e.g. '.BTCFR8H' (as saved by BaseFundingFetcher)")


class FundingConfigEmaOU(BaseModel):
    """EMA trend + Ornstein-Uhlenbeck noise funding model (matches FundingModel_EMA_OU fields)."""
    model_config = ConfigDict(extra="forbid")
    
    model: Literal["ema_ou"] = "ema_ou"
    ema_period: int = Field(default=120, ge=1)
    basis_sensitivity: float = 0.5
    ou_theta: float = Field(default=0.1, ge=0.0)
    ou_sigma: float = Field(default=0.05, ge=0.0)


# Union type for funding (discriminated by 'model' field)
FundingConfig = Annotated[
    Union[FundingConfigFixed, FundingConfigHistorical, FundingConfigEmaOU],
    Field(discriminator="model")
]


class MetricsConfig(BaseModel):
    """Metrics reporting configuration (for Reporter)."""
    model_config = ConfigDict(extra="forbid")
    
    metrics_timeframe: str = "1d"
    risk_free_rate: float = Field(default=0.0, ge=-1.0, le=1.0)
    sortino_target: float | None = Field(default=None, ge=-1.0, le=1.0)


class BacktestDataConfig(BaseModel):
    """Backtest data source and account settings."""
    model_config = ConfigDict(extra="forbid")
    
    start_date: str | None = None
    end_date: str | None = None
    symbol_datasource: str = Field(..., description="Symbol in CSV files (e.g. 'BTCUSDT')")
    initial_balance: float = Field(default=10_000.0, description="Starting capital in quote currency")
    warmup_bars: int = Field(default=0, description="Bars to advance before on_tick() (let indicators accumulate)")


class BacktestConfig(BaseModel):
    """Root config for backtest.yaml."""
    model_config = ConfigDict(extra="forbid")
    
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backtest: BacktestDataConfig
    costs: CostsConfig = Field(default_factory=CostsConfig)
    funding: FundingConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    bots: list[BotConfig] = Field(..., min_length=1)


############ LIVE/TESTNET-SPECIFIC MODELS ############

class ExchangeConfig(BaseModel):
    """Live exchange configuration."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(..., description="Exchange name (e.g. 'phemex')")
    market_type: Literal["spot", "swap", "margin", "stock"] = "swap"


class AlertingConfig(BaseModel):
    """
    Live alerting settings. Single-channel for now (Telegram) — the shape already
    supports adding more `channel` literal values later without a breaking change
    (see TODO_end_vision.md item 11 for the deferred pluggable multi-backend registry).
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    channel: Literal["telegram"] = "telegram"
    categories: dict[str, bool] = Field(
        default_factory=lambda: {key: True for key in ALERT_CATEGORIES},
        description="Per-category on/off toggle; all default to True",
    )

    @model_validator(mode="after")
    def _check_known_categories(self) -> "AlertingConfig":
        unknown = set(self.categories) - set(ALERT_CATEGORIES)
        if unknown:
            raise ValueError(
                f"Unknown alerting categor{'y' if len(unknown) == 1 else 'ies'}: {sorted(unknown)}. "
                f"Valid categories: {sorted(ALERT_CATEGORIES)}"
            )
        return self


class RiskConfig(BaseModel):
    """Live risk management settings."""
    model_config = ConfigDict(extra="forbid")
    
    max_drawdown_pct: float = Field(default=5.0, gt=0.0, le=100.0)
    drawdown_quote_currency: str = "USDT"
    drawdown_check_interval: int = Field(default=60, ge=1)
    consecutive_errors_limit: int = Field(default=5, ge=1)
    max_bot_restarts: int = Field(default=3, ge=0, description="Watchdog auto-restart attempts before giving up; 0 disables auto-restart")
    restart_backoff_seconds: float = Field(default=30.0, gt=0.0, description="Initial delay before the first restart attempt")
    restart_backoff_max_seconds: float = Field(default=600.0, gt=0.0, description="Cap on the exponential backoff delay between restart attempts")
    restart_healthy_reset_seconds: float = Field(default=3600.0, gt=0.0, description="Sustained healthy uptime after a restart before the attempt counter resets")

    @model_validator(mode="after")
    def _check_backoff_bounds(self) -> "RiskConfig":
        if self.restart_backoff_max_seconds < self.restart_backoff_seconds:
            raise ValueError(
                "restart_backoff_max_seconds must be >= restart_backoff_seconds "
                f"(got {self.restart_backoff_max_seconds} < {self.restart_backoff_seconds})"
            )
        return self


class LiveConfig(BaseModel):
    """
    Root config for both live.yaml and testnet.yaml.

    Testnet is a flag, not a separate model/subclass (a testnet run has the
    identical shape to a live run) — set testnet: true and provide
    LIGHTHOUSE_<EXCHANGE>_TESTNET_API_KEY/SECRET instead of the live ones.
    """
    model_config = ConfigDict(extra="forbid")
    
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    exchange: ExchangeConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    bots: list[BotConfig] = Field(..., min_length=1)
    testnet: bool = False

    @model_validator(mode="after")
    def _disarm_watchdog_without_persistence(self) -> "LiveConfig":
        """A watchdog rebuild with no persisted state silently disarms the entry-failure
        backoff (ADR 0002) and same-bar re-entry guard (ADR 0003); refuse that combination
        by forcing the restart budget to zero instead of leaving guards quietly undone."""
        if self.paths.state_dir is None and self.risk.max_bot_restarts != 0:
            warnings.warn(
                "paths.state_dir is null (state persistence disabled) but risk.max_bot_restarts="
                f"{self.risk.max_bot_restarts}; forcing max_bot_restarts to 0 because a watchdog "
                "restart without persisted state would lose entry-failure and re-entry guards.",
                stacklevel=2,
            )
            self.risk.max_bot_restarts = 0
        return self
