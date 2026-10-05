"""
================================================================================
UNIT TESTS FOR exchanges/backtest/funding_models/

Covers:
    ema_ou.py : FundingModel_EMA_OU
    historical.py : FundingModel_Historical

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_funding_models.py -v

To run a specific test function:
    pytest tests/backtest/test_funding_models.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import numpy as np
import pandas as pd
from lighthouse.exchanges.backtest.funding_models.ema_ou import FundingModel_EMA_OU
from lighthouse.exchanges.backtest.funding_models.historical import FundingModel_Historical

################### TESTS ##########################

# ── FundingModel_EMA_OU.compute_rate() ────────────────────────────────────────────────────────

# EMA initializes to the first close price on the first call
def test_ema_ou_ema_initializes_to_first_close():
    model = FundingModel_EMA_OU(ema_period=10, ou_sigma=0.0)

    model.compute_rate(bar_idx=0, close_price=500.0, base_funding_rate=0.01)

    # After first call, EMA should equal the first close price
    assert model._ema == pytest.approx(500.0)


# rate is positive when price is above EMA (long trend → longs pay)
def test_ema_ou_positive_rate_when_price_above_ema():
    model = FundingModel_EMA_OU(ema_period=100, basis_sensitivity=1.0, ou_sigma=0.0)

    # Seed EMA well below the next close so basis is clearly positive
    model._ema = 100.0
    rate = model.compute_rate(bar_idx=1, close_price=120.0, base_funding_rate=0.01)

    assert rate > 0.0


# ou_sigma=0 produces the same result on repeated calls with the same inputs
def test_ema_ou_deterministic_with_zero_noise():
    model_a = FundingModel_EMA_OU(ema_period=10, ou_sigma=0.0)
    model_b = FundingModel_EMA_OU(ema_period=10, ou_sigma=0.0)

    rate_a = model_a.compute_rate(bar_idx=0, close_price=100.0, base_funding_rate=0.01)
    rate_b = model_b.compute_rate(bar_idx=0, close_price=100.0, base_funding_rate=0.01)

    assert rate_a == pytest.approx(rate_b)


# ── FundingModel_Historical fixtures ──────────────────────────────────────────

def _make_bar_timestamps(n: int = 10, freq: str = "1h") -> pd.DatetimeIndex:
    """Return n UTC hourly bar timestamps starting at 2024-01-01 00:00."""
    return pd.date_range("2024-01-01 00:00", periods=n, freq=freq, tz="UTC")


def _make_rates(timestamps: pd.DatetimeIndex, values: list[float]) -> pd.Series:
    """Build a UTC-indexed funding rate Series from parallel lists."""
    return pd.Series(values, index=pd.DatetimeIndex(timestamps, tz="UTC"))


# ── FundingModel_Historical.compute_rate() ────────────────────────────────────

# rate is returned exactly when the bar timestamp matches a funding timestamp
def test_historical_exact_timestamp_match():
    bars  = _make_bar_timestamps(8)
    # rates must cover the full bar range; bars[7] (07:00) is the last bar
    rates = _make_rates([bars[0], bars[4], bars[7]], [0.0001, 0.0002, 0.0002])
    model = FundingModel_Historical(rates, bars)

    assert model.compute_rate(bar_idx=0, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0001)
    assert model.compute_rate(bar_idx=4, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0002)


# bars between two settlements carry the earlier (forward-filled) rate
def test_historical_forward_fill_between_settlements():
    bars  = _make_bar_timestamps(8)
    # rates must cover the full bar range; bars[7] (07:00) anchors the end
    rates = _make_rates([bars[0], bars[4], bars[7]], [0.0001, 0.0002, 0.0002])
    model = FundingModel_Historical(rates, bars)

    # bars 1, 2, 3 should carry the rate published at bar 0
    for idx in [1, 2, 3]:
        assert model.compute_rate(bar_idx=idx, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0001)

    # bars 5, 6, 7 should carry the rate published at bar 4
    for idx in [5, 6, 7]:
        assert model.compute_rate(bar_idx=idx, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0002)


# raises ValueError when funding data starts after the first bar (strict coverage enforced)
def test_historical_funding_data_starts_after_first_bar_raises():
    bars  = _make_bar_timestamps(8)
    # Rates only start at bar 4 — does not cover bars 0-3, which is now a hard error
    rates = _make_rates([bars[4], bars[7]], [0.0002, 0.0002])

    with pytest.raises(ValueError, match="funding data starts at"):
        FundingModel_Historical(rates, bars)

# raises ValueError when funding data ends before the last bar (strict coverage enforced)
def test_historical_funding_data_ends_before_last_bar_raises():
    bars  = _make_bar_timestamps(8)
    # Rates only cover bars 0-4 — does not cover bar 7, which is now a hard error
    rates = _make_rates([bars[0], bars[4]], [0.0001, 0.0002])

    with pytest.raises(ValueError, match="funding data ends at"):
        FundingModel_Historical(rates, bars)


# model handles a series with a single unique rate value spanning all bars
def test_historical_single_rate_covers_all_bars():
    bars  = _make_bar_timestamps(5)
    # Anchor start and end with the same value so forward-fill produces a uniform rate
    rates = _make_rates([bars[0], bars[-1]], [0.00015, 0.00015])
    model = FundingModel_Historical(rates, bars)

    for idx in range(5):
        assert model.compute_rate(bar_idx=idx, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.00015)


# negative funding rates are returned correctly
def test_historical_negative_rate():
    bars  = _make_bar_timestamps(4)
    rates = _make_rates([bars[0], bars[-1]], [-0.0003, -0.0003])
    model = FundingModel_Historical(rates, bars)

    assert model.compute_rate(bar_idx=0, close_price=100.0, base_funding_rate=0.0) == pytest.approx(-0.0003)


# close_price argument is unused and does not affect the returned rate
def test_historical_close_price_is_ignored():
    bars  = _make_bar_timestamps(4)
    rates = _make_rates([bars[0], bars[-1]], [0.0001, 0.0001])
    model = FundingModel_Historical(rates, bars)

    rate_a = model.compute_rate(bar_idx=0, close_price=100.0,  base_funding_rate=0.0)
    rate_b = model.compute_rate(bar_idx=0, close_price=50000.0, base_funding_rate=0.0)

    assert rate_a == pytest.approx(rate_b)


# ── FundingModel_Historical.__init__() validation ─────────────────────────────

# timezone-naive rates index raises ValueError
def test_historical_naive_rates_index_raises():
    bars         = _make_bar_timestamps(4)
    naive_index  = pd.date_range("2024-01-01", periods=2, freq="8h")  # no tz
    naive_rates  = pd.Series([0.0001, 0.0002], index=naive_index)

    with pytest.raises(ValueError, match="timezone-aware"):
        FundingModel_Historical(naive_rates, bars)


# timezone-naive bar_timestamps raises ValueError
def test_historical_naive_bar_timestamps_raises():
    naive_bars = pd.date_range("2024-01-01", periods=4, freq="1h")    # no tz
    rates      = _make_rates(
        pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC"),
        [0.0001, 0.0002],
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        FundingModel_Historical(rates, naive_bars)


# ── FundingModel_Historical.from_csv() ────────────────────────────────────────

# from_csv loads a well-formed CSV and returns the correct rates
def test_historical_from_csv_loads_correctly(tmp_path):
    bars = _make_bar_timestamps(8)

    csv_path = tmp_path / "BTCUSD.csv"
    csv_path.write_text(
        "timestamp,funding_rate\n"
        "2024-01-01 00:00:00+00:00,0.0001\n"
        "2024-01-01 04:00:00+00:00,0.0002\n"
        "2024-01-01 07:00:00+00:00,0.0002\n"
    )

    model = FundingModel_Historical.from_csv(csv_path, bars)

    assert model.compute_rate(bar_idx=0, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0001)
    # bar at 04:00 (bar_idx=4) picks up the second rate
    assert model.compute_rate(bar_idx=4, close_price=100.0, base_funding_rate=0.0) == pytest.approx(0.0002)


# from_csv raises FileNotFoundError for a missing file
def test_historical_from_csv_missing_file(tmp_path):
    bars = _make_bar_timestamps(4)

    with pytest.raises(FileNotFoundError):
        FundingModel_Historical.from_csv(tmp_path / "nonexistent.csv", bars)


# from_csv raises ValueError when the 'funding_rate' column is absent
def test_historical_from_csv_missing_column(tmp_path):
    bars = _make_bar_timestamps(4)

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,wrong_column\n2024-01-01 00:00:00+00:00,0.0001\n")

    with pytest.raises(ValueError, match="funding_rate"):
        FundingModel_Historical.from_csv(csv_path, bars)
