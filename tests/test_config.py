"""
================================================================================
CONFIG TESTS
================================================================================

Tests for config module: pydantic models and YAML loaders.

Tests:
    test_load_backtest_config_example() : load examples/backtest.yaml
    test_load_live_config_example() : load examples/live.yaml
    test_load_testnet_config_example() : load examples/testnet.yaml (testnet: true flag)
    test_config_to_run_config() : convert BacktestConfig to run_config dict
    test_config_to_live_context() : convert LiveConfig to context dict
================================================================================
"""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError
from lighthouse.config.loader import (
    load_backtest_config,
    load_live_config,
)
from lighthouse.config.convert import config_to_run_config, config_to_live_context
from lighthouse.config.models import AlertingConfig, BotConfig, ExchangeConfig, LiveConfig, PathsConfig, RiskConfig


# Locate examples directory
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class TestBacktestConfig:
    """Test backtest config loading and validation."""

    def test_load_backtest_config_example(self):
        """Load and validate examples/backtest.yaml."""
        config_path = EXAMPLES_DIR / "backtest.yaml"
        config = load_backtest_config(config_path)

        # Verify top-level structure
        assert config.paths.data_dir == "backtest_data/ohlcv"
        assert config.logging.level == "INFO"

        # Verify backtest section
        assert config.backtest.symbol_datasource == "BTCUSDT"
        assert config.backtest.initial_balance == 10000.0
        assert config.backtest.warmup_bars == 50

        # Verify costs
        assert config.costs.spread == 0.0005
        assert config.costs.taker_fee == 0.0006

        # Verify funding
        assert config.funding.model == "fixed"
        assert config.funding.rate == 0.0001

        # Verify metrics
        assert config.metrics.metrics_timeframe == "1d"

        # Verify bots
        assert len(config.bots) == 1
        assert config.bots[0].name == "crypto_bot"
        assert config.bots[0].symbol == "BTC/USDT:USDT"
        assert config.bots[0].timeframe == "1h"
        # Strategy extras
        assert config.bots[0].direction == "both"
        assert config.bots[0].risk_pct == 0.01

    def test_config_to_run_config(self):
        """Test conversion of BacktestConfig pydantic -> run_config dict."""
        config_path = EXAMPLES_DIR / "backtest.yaml"
        config = load_backtest_config(config_path)
        
        # Convert to run_config dict
        run_config = config_to_run_config(config, config_path.parent)
        
        # Verify structure
        assert "data_dir" in run_config
        assert "symbol_datasource" in run_config
        assert "timeframe" in run_config
        assert "symbol" in run_config
        assert "initial_balance" in run_config
        assert "backtest_config" in run_config
        assert "metrics_config" in run_config
        assert "bot_spec" in run_config
        assert "warmup_bars" in run_config
        
        # Verify values
        assert run_config["symbol_datasource"] == "BTCUSDT"
        assert run_config["symbol"] == "BTC/USDT:USDT"
        assert run_config["initial_balance"] == 10000.0
        assert run_config["warmup_bars"] == 50
        assert run_config["timeframe"] == "1h"
        assert run_config["data_dir"] == (config_path.parent / "backtest_data/ohlcv").resolve()
        
        # Verify bot_spec structure
        assert run_config["bot_spec"]["class"].__name__ == "CryptoBot"
        assert run_config["bot_spec"]["config"]["name"] == "crypto_bot"
        assert run_config["bot_spec"]["config"]["direction"] == "both"
        assert run_config["bot_spec"]["config"]["risk_pct"] == 0.01
        
        # Verify backtest_config (BacktestConfig dataclass)
        assert run_config["backtest_config"].spread == 0.0005
        assert run_config["backtest_config"].taker_fee == 0.0006
        assert run_config["backtest_config"].funding_model is None  # attached post-data-load

        # "fixed" funding overrides the instrument's base_funding_rate
        assert run_config["instrument"].base_funding_rate == 0.0001
        
        # Verify metrics_config (MetricsConfig dataclass)
        assert run_config["metrics_config"].metrics_timeframe == "1d"
        assert run_config["metrics_config"].risk_free_rate == 0.0

    def test_config_to_run_config_rejects_malformed_metrics_timeframe(self):
        """config_to_run_config fails fast on a malformed metrics_timeframe (no field_validator on MetricsConfig)."""
        config_path = EXAMPLES_DIR / "backtest.yaml"
        config = load_backtest_config(config_path)
        bad_config = config.model_copy(
            update={"metrics": config.metrics.model_copy(update={"metrics_timeframe": "bogus"})}
        )
        with pytest.raises(ValueError):
            config_to_run_config(bad_config, config_path.parent)


class TestLiveConfig:
    """Test live config loading and validation."""

    def test_load_live_config_example(self):
        """Load and validate examples/live.yaml."""
        config_path = EXAMPLES_DIR / "live.yaml"
        config = load_live_config(config_path)

        # Verify top-level structure
        assert config.paths.logs_dir == "logs"
        assert config.paths.state_dir == "state"
        assert config.logging.level == "INFO"

        # Verify exchange
        assert config.exchange.name == "phemex"
        assert config.exchange.market_type == "swap"
        assert config.testnet is False

        # Verify risk
        assert config.risk.max_drawdown_pct == 5.0
        assert config.risk.drawdown_quote_currency == "USDT"
        assert config.risk.drawdown_check_interval == 60

        # Verify bots
        assert len(config.bots) == 1
        assert config.bots[0].name == "crypto_bot"
        assert config.bots[0].symbol == "BTC/USDT:USDT"
        assert config.bots[0].risk_pct == 0.01

    def test_config_to_live_context(self):
        """Test conversion of LiveConfig pydantic -> context dict."""
        config_path = EXAMPLES_DIR / "live.yaml"
        config = load_live_config(config_path)
        
        # Convert to context dict
        context = config_to_live_context(config, config_path.parent)
        
        # Verify structure
        assert "exchange_name" in context
        assert "market_type" in context
        assert "testnet" in context
        assert "risk_settings" in context
        assert "bot_spec" in context
        assert "logging_level" in context
        
        # Verify values
        assert context["exchange_name"] == "phemex"
        assert context["market_type"] == "swap"
        assert context["testnet"] is False
        
        # Verify risk_settings
        assert context["risk_settings"]["max_drawdown_pct"] == 5.0
        assert context["risk_settings"]["quote_currency"] == "USDT"
        assert context["risk_settings"]["check_interval"] == 60
        assert context["risk_settings"]["consecutive_limit"] == 5
        assert context["risk_settings"]["max_bot_restarts"] == 3
        assert context["risk_settings"]["restart_backoff_seconds"] == 30.0
        assert context["risk_settings"]["restart_backoff_max_seconds"] == 600.0
        assert context["risk_settings"]["restart_healthy_reset_seconds"] == 3600.0
        
        # Verify bot_spec
        assert context["bot_spec"]["class"].__name__ == "CryptoBot"
        assert context["bot_spec"]["config"]["name"] == "crypto_bot"
        assert context["bot_spec"]["config"]["direction"] == "both"
        
        # Verify logging level
        assert context["logging_level"] == "INFO"

        # Verify state_dir, resolved relative to config_dir like logs_dir already is
        assert context["state_dir"] == (config_path.parent / "state").resolve()

        # Verify alerting_settings (alerting disabled by default when omitted from YAML)
        assert context["alerting_settings"]["enabled"] is False
        assert context["alerting_settings"]["channel"] == "telegram"
        assert isinstance(context["alerting_settings"]["categories"], dict)

    def test_config_to_live_context_state_dir_none_disables_persistence(self):
        """paths.state_dir: null passes state_dir=None through (persistence opt-out)."""
        config_path = EXAMPLES_DIR / "live.yaml"
        config = load_live_config(config_path)
        config = config.model_copy(update={"paths": config.paths.model_copy(update={"state_dir": None})})

        context = config_to_live_context(config, config_path.parent)

        assert context["state_dir"] is None


class TestTestnetConfig:
    """Test testnet config loading (testnet is a flag on LiveConfig, not a separate model)."""

    def test_load_testnet_config_example(self):
        """Load and validate examples/testnet.yaml."""
        config_path = EXAMPLES_DIR / "testnet.yaml"
        config = load_live_config(config_path)

        # Same structure as live, but testnet flag set
        assert config.exchange.name == "phemex"
        assert config.risk.drawdown_quote_currency == "USDT"
        assert len(config.bots) == 1
        assert config.testnet is True

    def test_config_to_live_context_testnet_flag(self):
        """Converting testnet.yaml surfaces testnet=True in the context dict."""
        config_path = EXAMPLES_DIR / "testnet.yaml"
        config = load_live_config(config_path)
        context = config_to_live_context(config, config_path.parent)

        assert context["testnet"] is True


class TestRiskConfigWatchdog:
    """Test the watchdog auto-restart fields on RiskConfig."""

    def test_defaults_when_omitted(self):
        """Watchdog fields default sensibly when not specified in YAML."""
        risk = RiskConfig()

        assert risk.max_bot_restarts == 3
        assert risk.restart_backoff_seconds == 30.0
        assert risk.restart_backoff_max_seconds == 600.0
        assert risk.restart_healthy_reset_seconds == 3600.0

    def test_max_bot_restarts_zero_allowed(self):
        """0 is a valid value — it disables auto-restart."""
        risk = RiskConfig(max_bot_restarts=0)
        assert risk.max_bot_restarts == 0

    def test_max_bot_restarts_negative_rejected(self):
        """Negative restart counts are invalid."""
        with pytest.raises(ValidationError):
            RiskConfig(max_bot_restarts=-1)

    def test_backoff_max_below_backoff_rejected(self):
        """restart_backoff_max_seconds must be >= restart_backoff_seconds."""
        with pytest.raises(ValidationError):
            RiskConfig(restart_backoff_seconds=100.0, restart_backoff_max_seconds=50.0)

    def test_backoff_max_equal_backoff_allowed(self):
        """Equal values are the boundary case and should be accepted."""
        risk = RiskConfig(restart_backoff_seconds=100.0, restart_backoff_max_seconds=100.0)
        assert risk.restart_backoff_max_seconds == 100.0


class TestWatchdogRequiresPersistence:
    """LiveConfig forces risk.max_bot_restarts to 0 when paths.state_dir is null (issue #27)."""

    def _make_live_config(self, *, state_dir: str | None, max_bot_restarts: int) -> LiveConfig:
        return LiveConfig(
            paths=PathsConfig(state_dir=state_dir),
            exchange=ExchangeConfig(name="test"),
            risk=RiskConfig(max_bot_restarts=max_bot_restarts),
            bots=[BotConfig(name="test_bot", symbol="BTC/USDT", quote_currency="USDT", timeframe="1h")],
        )

    # paths.state_dir: null coerces a non-zero restart budget down to 0, with a warning naming both settings
    def test_persistence_disabled_forces_restart_budget_to_zero(self):
        with pytest.warns(UserWarning, match="state_dir.*max_bot_restarts"):
            config = self._make_live_config(state_dir=None, max_bot_restarts=3)

        assert config.risk.max_bot_restarts == 0

    # no spurious warning when the restart budget is already 0
    def test_persistence_disabled_with_restart_budget_already_zero_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = self._make_live_config(state_dir=None, max_bot_restarts=0)

        assert config.risk.max_bot_restarts == 0

    # paths.state_dir set (persistence on) never touches a non-zero restart budget
    def test_persistence_enabled_leaves_nonzero_restart_budget_untouched(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = self._make_live_config(state_dir="state", max_bot_restarts=3)

        assert config.risk.max_bot_restarts == 3

    # paths.state_dir set (persistence on) with an already-zero restart budget stays untouched and silent
    def test_persistence_enabled_leaves_zero_restart_budget_untouched(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = self._make_live_config(state_dir="state", max_bot_restarts=0)

        assert config.risk.max_bot_restarts == 0


class TestAlertingConfig:
    """Test the AlertingConfig model (live-only alerting settings)."""

    def test_defaults(self):
        """Disabled by default, telegram channel, all categories True."""
        alerting = AlertingConfig()

        assert alerting.enabled is False
        assert alerting.channel == "telegram"
        assert alerting.categories["startup"] is True
        assert alerting.categories["halt"] is True
        assert all(alerting.categories.values())

    def test_per_category_override(self):
        """Individual categories can be disabled while others stay default."""
        alerting = AlertingConfig(categories={**AlertingConfig().categories, "summary": False})
        assert alerting.categories["summary"] is False
        assert alerting.categories["halt"] is True

    def test_unknown_category_rejected(self):
        """A typo'd/unknown category key raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown alerting categor"):
            AlertingConfig(categories={"not_a_real_category": True})

    def test_live_config_alerting_defaults(self):
        """LiveConfig.alerting defaults to a disabled AlertingConfig when omitted from YAML."""
        config_path = EXAMPLES_DIR / "live.yaml"
        config = load_live_config(config_path)
        assert config.alerting.enabled is False


class TestBotConfigSummaryInterval:
    """Test BotConfig.summary_interval_seconds (per-bot periodic summary opt-in)."""

    def test_defaults_to_none(self):
        """Omitted -> None, periodic summary disabled for that bot."""
        bot = BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="1h")
        assert bot.summary_interval_seconds is None

    def test_positive_value_accepted(self):
        bot = BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="1h", summary_interval_seconds=300)
        assert bot.summary_interval_seconds == 300

    def test_zero_rejected(self):
        """0 is invalid — ge=1 boundary."""
        with pytest.raises(ValidationError):
            BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="1h", summary_interval_seconds=0)


class TestBotConfigTimeframe:
    """Test BotConfig.timeframe validation (delegates to Timeframe.parse via a field_validator)."""

    def test_valid_timeframe_accepted(self):
        """A well-formed timeframe stays a plain str on the model."""
        bot = BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="4h")
        assert bot.timeframe == "4h"

    def test_malformed_timeframe_rejected(self):
        """A YAML typo surfaces as a ValidationError at load time, not deep in the engine."""
        with pytest.raises(ValidationError):
            BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="bogus")

    def test_calendar_month_rejected(self):
        """Calendar-month notation ("1M") is rejected, same as Timeframe.parse."""
        with pytest.raises(ValidationError):
            BotConfig(name="crypto_bot", symbol="BTC/USDT:USDT", quote_currency="USDT", timeframe="1M")

