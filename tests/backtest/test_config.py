"""
================================================================================
UNIT TESTS FOR exchanges/backtest/config.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_config.py -v

To run a specific test function:
    pytest tests/backtest/test_config.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from lighthouse.exchanges.backtest.config import BacktestConfig, MetricsConfig
from lighthouse.exchanges.backtest.funding_models.ema_ou import FundingModel_EMA_OU

################### HELPERS ##########################

def make_config(**overrides) -> BacktestConfig:
    """Return a valid BacktestConfig with sensible defaults, allowing field overrides."""
    defaults = dict(
        spread=0.001,
        taker_fee=0.006,
        maker_fee=0.002,
        slippage=0.001,
        latency_bars=1,
        partial_fill_fraction=1.0,
        funding_model=FundingModel_EMA_OU(),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)

################### TESTS ##########################

# ── Construction ────────────────────────────────────────────────────────

# latency_bars = 0 is rejected
def test_latency_bars_zero_raises():
    with pytest.raises(ValueError, match="latency_bars must be >= 1"):
        make_config(latency_bars=0)

# latency_bars = -1 is also rejected
def test_latency_bars_negative_raises():
    with pytest.raises(ValueError, match="latency_bars must be >= 1"):
        make_config(latency_bars=-1)

# latency_bars = 1 is the minimum valid value
def test_latency_bars_one_is_valid():
    make_config(latency_bars=1)

# valid config with all fields constructs without error
def test_valid_config_constructs():
    make_config()


# ── MetricsConfig ────────────────────────────────────────────────────────

# default MetricsConfig: "1d" resampling, no risk-free rate
def test_metrics_config_defaults():
    mc = MetricsConfig()

    assert mc.metrics_timeframe == "1d"
    assert mc.risk_free_rate == 0.0
    assert mc.sortino_target is None


# a malformed metrics_timeframe raises via _parse_timeframe
def test_metrics_config_invalid_timeframe_raises():
    with pytest.raises(ValueError, match="unrecognized timeframe"):
        MetricsConfig(metrics_timeframe="bogus")


# risk_free_rate outside [-1.0, 1.0] raises
def test_metrics_config_risk_free_rate_out_of_bounds_raises():
    with pytest.raises(ValueError, match="risk_free_rate"):
        MetricsConfig(risk_free_rate=1.5)


# sortino_target outside [-1.0, 1.0] raises
def test_metrics_config_sortino_target_out_of_bounds_raises():
    with pytest.raises(ValueError, match="sortino_target"):
        MetricsConfig(sortino_target=-2.0)


# valid overrides construct without error
def test_metrics_config_valid_overrides_construct():
    mc = MetricsConfig(
        metrics_timeframe="4h", risk_free_rate=0.02,
        sortino_target=0.01,
    )

    assert mc.metrics_timeframe == "4h"
    assert mc.risk_free_rate == 0.02
    assert mc.sortino_target == 0.01
