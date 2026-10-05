"""
================================================================================
CONFIG CONVERSION
================================================================================
Translate validated pydantic config models into the plain dicts/dataclasses
the runtime and engine layers already expect. Pydantic models are a pure
validation boundary — nothing downstream of this module knows pydantic exists.

Functions:
    config_to_run_config()  : BacktestConfig (pydantic) -> run_config dict for run_single()
    config_to_live_context(): LiveConfig (pydantic) -> context dict for runtime.live.main()
    build_funding_model()   : FundingConfig (pydantic) -> BaseFundingModel instance
================================================================================
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from lighthouse.bots.registry import resolve_bot
from lighthouse.config.models import BacktestConfig as BacktestConfigModel
from lighthouse.config.models import FundingConfig, FundingConfigFixed, FundingConfigHistorical
from lighthouse.config.models import LiveConfig as LiveConfigModel
from lighthouse.domain.instruments import PERP_FUTURES
from lighthouse.domain.timeframe import Timeframe
from lighthouse.exchanges.backtest.config import BacktestConfig, MetricsConfig
from lighthouse.exchanges.backtest.funding_models.base import BaseFundingModel
from lighthouse.exchanges.backtest.funding_models.ema_ou import FundingModel_EMA_OU
from lighthouse.exchanges.backtest.funding_models.fixed import FundingModel_Fixed
from lighthouse.exchanges.backtest.funding_models.historical import FundingModel_Historical


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    """Resolve a config path relative to the config file's own directory, not the CWD."""
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_funding_model(funding: FundingConfig, data_dir: Path, bar_timestamps: pd.DatetimeIndex) -> BaseFundingModel:
    """
    Instantiate the funding model selected by `funding.model`.

    Args:
        funding: validated FundingConfig union (fixed / historical / ema_ou)
        data_dir: resolved backtest data directory (funding CSVs live under <data_dir>/funding/)
        bar_timestamps: OHLCV DataFrame's DatetimeIndex, required to align historical rates

    Returns:
        A BaseFundingModel instance ready for BacktestConfig.funding_model.
    """
    if isinstance(funding, FundingConfigFixed):
        return FundingModel_Fixed()
    if isinstance(funding, FundingConfigHistorical):
        path = data_dir / "funding" / f"{funding.symbol}.csv"
        return FundingModel_Historical.from_csv(path, bar_timestamps)
    return FundingModel_EMA_OU(
        ema_period=funding.ema_period,
        basis_sensitivity=funding.basis_sensitivity,
        ou_theta=funding.ou_theta,
        ou_sigma=funding.ou_sigma,
    )


def config_to_run_config(config: BacktestConfigModel, config_dir: Path) -> dict[str, Any]:
    """
    Convert a validated BacktestConfig (pydantic) into a run_config dict for run_single().

    Args:
        config: BacktestConfigModel loaded from YAML
        config_dir: directory containing the YAML file (paths resolve relative to this, not the CWD)

    Returns:
        Dict with keys: data_dir, symbol_datasource, timeframe, start_date, end_date,
                       symbol, initial_balance, instrument, backtest_config, metrics_config,
                       bot_spec, warmup_bars, funding_config, logs_dir, logging_level.
        backtest_config.funding_model is left as None — the historical model needs the
        OHLCV DataFrame's timestamps, so it's attached later via build_funding_model()
        once the data is loaded (see runtime/backtest.py run()).
    """
    bot_name = config.bots[0].name
    bot_class = resolve_bot(bot_name)
    bot_config_dict = config.bots[0].model_dump()

    # "fixed" funding overrides the instrument's base rate; other models supply their own.
    instrument = (
        replace(PERP_FUTURES, base_funding_rate=config.funding.rate)
        if isinstance(config.funding, FundingConfigFixed)
        else PERP_FUTURES
    )

    backtest_cfg = BacktestConfig(
        spread                = config.costs.spread,
        taker_fee             = config.costs.taker_fee,
        maker_fee             = config.costs.maker_fee,
        slippage              = config.costs.slippage,
        latency_bars          = config.costs.latency_bars,
        partial_fill_fraction = config.costs.partial_fill_fraction,
        funding_model         = None,  # attached post-data-load, see docstring above
        limit_fill_policy     = config.costs.limit_fill_policy,
    )

    metrics_cfg = MetricsConfig(
        metrics_timeframe = str(Timeframe.parse(config.metrics.metrics_timeframe)),
        risk_free_rate    = config.metrics.risk_free_rate,
        sortino_target    = config.metrics.sortino_target,
    )

    return {
        "data_dir":          _resolve_path(config_dir, config.paths.data_dir),
        "logs_dir":          _resolve_path(config_dir, config.paths.logs_dir),
        "logging_level":     config.logging.level,
        "symbol_datasource": config.backtest.symbol_datasource,
        "timeframe":         config.bots[0].timeframe,
        "start_date":        config.backtest.start_date,
        "end_date":          config.backtest.end_date,
        "symbol":            config.bots[0].symbol,
        "initial_balance":   config.backtest.initial_balance,
        "instrument":        instrument,
        "backtest_config":   backtest_cfg,
        "metrics_config":    metrics_cfg,
        "funding_config":    config.funding,
        "bot_spec": {
            "class":  bot_class,
            "config": bot_config_dict,
        },
        "warmup_bars":       config.backtest.warmup_bars,
    }


def config_to_live_context(config: LiveConfigModel, config_dir: Path) -> dict[str, Any]:
    """
    Convert a validated LiveConfig (pydantic) into a context dict for runtime.live.main().

    Args:
        config: LiveConfigModel loaded from YAML (live.yaml or testnet.yaml)
        config_dir: directory containing the YAML file (paths resolve relative to this, not the CWD)

    Returns:
        Dict with keys: exchange_name, market_type, testnet, risk_settings, bot_spec,
                       logging_level, logs_dir, state_dir, alerting_settings.
    """
    if len(config.bots) != 1:
        raise ValueError(f"Live trading requires exactly 1 bot; got {len(config.bots)}")

    bot_name = config.bots[0].name
    bot_class = resolve_bot(bot_name)
    bot_config_dict = config.bots[0].model_dump()

    return {
        "exchange_name": config.exchange.name,
        "market_type": config.exchange.market_type,
        "testnet": config.testnet,
        "risk_settings": {
            "max_drawdown_pct": config.risk.max_drawdown_pct,
            "quote_currency": config.risk.drawdown_quote_currency,
            "check_interval": config.risk.drawdown_check_interval,
            "consecutive_limit": config.risk.consecutive_errors_limit,
            "max_bot_restarts": config.risk.max_bot_restarts,
            "restart_backoff_seconds": config.risk.restart_backoff_seconds,
            "restart_backoff_max_seconds": config.risk.restart_backoff_max_seconds,
            "restart_healthy_reset_seconds": config.risk.restart_healthy_reset_seconds,
        },
        "bot_spec": {
            "class": bot_class,
            "config": bot_config_dict,
        },
        "logging_level": config.logging.level,
        "logs_dir": _resolve_path(config_dir, config.paths.logs_dir),
        "state_dir": (
            _resolve_path(config_dir, config.paths.state_dir)
            if config.paths.state_dir is not None
            else None
        ),
        "alerting_settings": {
            "enabled": config.alerting.enabled,
            "channel": config.alerting.channel,
            "categories": config.alerting.categories,
        },
    }
