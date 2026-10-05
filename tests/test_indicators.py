"""
================================================================================
UNIT TESTS FOR INDICATORS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_indicators.py -v

To run a specific test function:
    pytest tests/test_indicators.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import math
import pandas as pd
from lighthouse.domain.indicators import (
    swing_high,
    swing_low,
    atr,
    atr_current,
    atr_pct,
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_pct_b,
    rolling_volume_avg,
    rolling_volume_avg_current,
    volume_ratio_current,
    sma,
    ema,
    rsi,
)

################### TESTS ##########################

# ── swing_high ────────────────────────────────────────────────────────────────

# normal case; returns the highest high over the lookback window
# ohlcv_df: high[i] = 101 + i → last 5 highs are 146, 147, 148, 149, 150 → max = 150.0
def test_swing_high(ohlcv_df):
    result = swing_high(ohlcv_df, window=5)
    assert result == 150.0

# window equals full dataframe length; still returns the global high
def test_swing_high_full_window(ohlcv_df):
    result = swing_high(ohlcv_df, window=50)
    assert result == 150.0

# window larger than dataframe length; raises ValueError
def test_swing_high_window_too_large(ohlcv_df):
    with pytest.raises(ValueError):
        swing_high(ohlcv_df, window=51)

# wrong column names; raises ValueError
def test_swing_high_missing_column(bad_columns_df):
    with pytest.raises(ValueError):
        swing_high(bad_columns_df, window=1)


# ── swing_low ─────────────────────────────────────────────────────────────────

# normal case; returns the lowest low over the lookback window
# ohlcv_df: low[i] = 99 + i → last 5 lows are 144, 145, 146, 147, 148 → min = 144.0
def test_swing_low(ohlcv_df):
    result = swing_low(ohlcv_df, window=5)
    assert result == 144.0

# window equals full dataframe length; returns the global low
def test_swing_low_full_window(ohlcv_df):
    result = swing_low(ohlcv_df, window=50)
    assert result == 99.0

# window larger than dataframe length; raises ValueError
def test_swing_low_window_too_large(ohlcv_df):
    with pytest.raises(ValueError):
        swing_low(ohlcv_df, window=51)

# wrong column names; raises ValueError
def test_swing_low_missing_column(bad_columns_df):
    with pytest.raises(ValueError):
        swing_low(bad_columns_df, window=1)


# ── atr ───────────────────────────────────────────────────────────────────────

# normal case; TR is constant 2.0, so all valid ATR values equal exactly 2.0
def test_atr_last_value(ohlcv_df):
    result = atr(ohlcv_df, period=14)
    assert (result.dropna() == 2.0).all()

# values before min_periods are NaN
def test_atr_nan_before_period(ohlcv_df):
    result = atr(ohlcv_df, period=14)
    assert result.iloc[12] != result.iloc[12]  # NaN != NaN is True

# first non-NaN value (index 13, the 14th bar) is already 2.0; no warm-up drift
def test_atr_first_valid_value(ohlcv_df):
    result = atr(ohlcv_df, period=14)
    assert result.iloc[13] == 2.0

# wrong column names; raises ValueError
def test_atr_missing_column(bad_columns_df):
    with pytest.raises(ValueError):
        atr(bad_columns_df, period=14)


# ── atr_current ───────────────────────────────────────────────────────────────

# normal case; returns the last ATR value as a scalar
def test_atr_current(ohlcv_df):
    result = atr_current(ohlcv_df, period=14)
    assert result == 2.0

# returns a float, not a Series
def test_atr_current_is_float(ohlcv_df):
    result = atr_current(ohlcv_df, period=14)
    assert isinstance(result, float)


# ── atr_pct ───────────────────────────────────────────────────────────────────

# normal case; atr_current=2.0, close[-1]=149.0 → (2.0/149.0)*100
def test_atr_pct(ohlcv_df):
    result = atr_pct(ohlcv_df, period=14)
    assert result == pytest.approx((2.0 / 149.0) * 100)

# returns a float, not a Series
def test_atr_pct_is_float(ohlcv_df):
    result = atr_pct(ohlcv_df, period=14)
    assert isinstance(result, float)


# ── bollinger_bands ─────────────────────────────────────────────────────────────

# ohlcv_df with period=3:
#   last 3 closes = [147, 148, 149]
#   middle = 148.0 (exact)
#   std (ddof=0) = sqrt(2/3)
#   upper = 148 + 2*sqrt(2/3), lower = 148 - 2*sqrt(2/3)
_STD3 = math.sqrt(2 / 3)  # population std of [147, 148, 149]

# returns a tuple of three Series
def test_bollinger_bands_returns_tuple(ohlcv_df):
    result = bollinger_bands(ohlcv_df, period=3)
    assert isinstance(result, tuple)
    assert len(result) == 3

# middle band equals the SMA; for a linear series middle[-1] = mean([147,148,149]) = 148.0
def test_bollinger_bands_middle(ohlcv_df):
    middle, _, _ = bollinger_bands(ohlcv_df, period=3)
    assert middle.iloc[-1] == pytest.approx(148.0)

# upper band is above middle, lower is below
def test_bollinger_bands_ordering(ohlcv_df):
    middle, upper, lower = bollinger_bands(ohlcv_df, period=3)
    assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

# exact upper and lower values
def test_bollinger_bands_upper_lower(ohlcv_df):
    _, upper, lower = bollinger_bands(ohlcv_df, period=3)
    assert upper.iloc[-1] == pytest.approx(148.0 + 2 * _STD3)
    assert lower.iloc[-1] == pytest.approx(148.0 - 2 * _STD3)

# first period-1 values are NaN
def test_bollinger_bands_nan_before_period(ohlcv_df):
    middle, _, _ = bollinger_bands(ohlcv_df, period=3)
    assert middle.iloc[1] != middle.iloc[1]  # NaN != NaN


# ── bollinger_bandwidth ──────────────────────────────────────────────────────────

# normal case; bandwidth = (upper - lower) / middle = 4*sqrt(2/3) / 148
def test_bollinger_bandwidth(ohlcv_df):
    result = bollinger_bandwidth(ohlcv_df, period=3)
    assert result.iloc[-1] == pytest.approx(4 * _STD3 / 148.0)

# bandwidth is always positive where defined
def test_bollinger_bandwidth_positive(ohlcv_df):
    result = bollinger_bandwidth(ohlcv_df, period=3).dropna()
    assert (result > 0).all()


# ── bollinger_pct_b ─────────────────────────────────────────────────────────────

# normal case; %B = (close - lower) / (upper - lower)
# close[-1]=149, lower=148-2*sqrt(2/3) → %B = (1 + 2*sqrt(2/3)) / (4*sqrt(2/3))
def test_bollinger_pct_b(ohlcv_df):
    result = bollinger_pct_b(ohlcv_df, period=3)
    expected = (1.0 + 2 * _STD3) / (4 * _STD3)
    assert result.iloc[-1] == pytest.approx(expected)

# for a rising series the last close is above the middle band, so %B > 0.5
def test_bollinger_pct_b_above_midpoint(ohlcv_df):
    result = bollinger_pct_b(ohlcv_df, period=3)
    assert result.iloc[-1] > 0.5


# ── rolling_volume_avg ──────────────────────────────────────────────────────────

# normal case; volume is constant 1000.0, so all valid rolling averages equal 1000.0
def test_rolling_volume_avg(ohlcv_df):
    result = rolling_volume_avg(ohlcv_df, period=10)
    assert (result.dropna() == 1000.0).all()

# first period-1 values are NaN
def test_rolling_volume_avg_nan_before_period(ohlcv_df):
    result = rolling_volume_avg(ohlcv_df, period=10)
    assert result.iloc[8] != result.iloc[8]  # NaN != NaN

# first valid value (index 9, the 10th bar) is 1000.0; no warm-up drift
def test_rolling_volume_avg_first_valid(ohlcv_df):
    result = rolling_volume_avg(ohlcv_df, period=10)
    assert result.iloc[9] == 1000.0


# ── rolling_volume_avg_current ──────────────────────────────────────────────

# normal case; constant volume → rolling average is always 1000.0
def test_rolling_volume_avg_current(ohlcv_df):
    result = rolling_volume_avg_current(ohlcv_df, period=10)
    assert result == 1000.0

# returns a float, not a Series
def test_rolling_volume_avg_current_is_float(ohlcv_df):
    result = rolling_volume_avg_current(ohlcv_df, period=10)
    assert isinstance(result, float)


# ── volume_ratio_current ─────────────────────────────────────────────────────────

# normal case; current volume = rolling average → ratio = 1.0
def test_volume_ratio_current(ohlcv_df):
    result = volume_ratio_current(ohlcv_df, period=10)
    assert result == 1.0

# returns a float, not a Series
def test_volume_ratio_current_is_float(ohlcv_df):
    result = volume_ratio_current(ohlcv_df, period=10)
    assert isinstance(result, float)

# zero average volume returns NaN
def test_volume_ratio_current_zero_avg():
    df = pd.DataFrame({
        "open":   [100.0] * 5,
        "high":   [101.0] * 5,
        "low":    [99.0]  * 5,
        "close":  [100.0] * 5,
        "volume": [0.0]   * 5,
    })
    result = volume_ratio_current(df, period=5)
    assert result != result  # NaN != NaN


# ── sma ────────────────────────────────────────────────────────────────────────

# ohlcv_df: close[i] = 100 + i. SMA(period=3)[i] = mean([i-2, i-1, i] closes) = (100+i) - 1.
# So all valid SMA values equal close - 1.0.

# last value: mean([147, 148, 149]) = 148.0
def test_sma_last_value(ohlcv_df):
    result = sma(ohlcv_df, period=3)
    assert result.iloc[-1] == 148.0

# all non-NaN values equal close - 1.0 (arithmetic property of the linear series)
def test_sma_all_values(ohlcv_df):
    result   = sma(ohlcv_df, period=3).dropna()
    expected = ohlcv_df["close"].iloc[2:] - 1.0
    assert (result.values == expected.values).all()

# first period-1 values are NaN
def test_sma_nan_before_period(ohlcv_df):
    result = sma(ohlcv_df, period=3)
    assert result.iloc[1] != result.iloc[1]  # NaN != NaN

# works on a non-default column; high[i] = close[i] + 1, so SMA of high = close + 0
def test_sma_high_column(ohlcv_df):
    result = sma(ohlcv_df, period=3, column="high")
    assert result.iloc[-1] == 149.0  # mean([148, 149, 150]) = 149.0


# ── ema ────────────────────────────────────────────────────────────────────────

# ohlcv_df: close[i] = 100 + i. With alpha=0.5 (span=3), steady-state EMA[i] = 99 + i.
# After 50 bars the transient (100 - 99) * 0.5^49 ≈ 1.78e-15 is negligible.

# no NaN values; ewm with adjust=False produces a value for every bar
def test_ema_no_nan(ohlcv_df):
    result = ema(ohlcv_df, period=3)
    assert result.isna().sum() == 0

# last value converges to 148.0 (steady-state = close[-1] - (span-1)/2 = 149 - 1)
def test_ema_last_value(ohlcv_df):
    result = ema(ohlcv_df, period=3)
    assert result.iloc[-1] == pytest.approx(148.0, abs=1e-6)

# series is monotonically increasing (tracking a rising price series)
def test_ema_monotonically_increasing(ohlcv_df):
    result = ema(ohlcv_df, period=3)
    assert (result.diff().iloc[1:] > 0).all()

# works on a non-default column; high[i] = close[i] + 1, so EMA of high ≈ 149.0
def test_ema_high_column(ohlcv_df):
    result = ema(ohlcv_df, period=3, column="high")
    assert result.iloc[-1] == pytest.approx(149.0, abs=1e-6)


# ── rsi ────────────────────────────────────────────────────────────────────────

# rsi_df uses a +2/-1 alternating pattern: bullish bias, avg_gain > avg_loss.
# ohlcv_df is NOT used here because a monotonically rising series gives avg_loss=0,
# which gets replaced with NaN, producing NaN RSI values throughout.

# NaN for bars before the warmup period; delta[0] is NaN so the first 14 non-NaN
# delta values appear at indices 1-14, making index 13 the last NaN output
def test_rsi_nan_before_warmup(rsi_df):
    result = rsi(rsi_df, period=14)
    assert result.iloc[13] != result.iloc[13]  # NaN != NaN

# no NaN values after warmup is complete
def test_rsi_no_nan_after_warmup(rsi_df):
    result = rsi(rsi_df, period=14)
    assert result.iloc[14:].isna().sum() == 0

# all valid values are within the valid RSI range [0, 100]
def test_rsi_valid_range(rsi_df):
    result = rsi(rsi_df, period=14).dropna()
    assert ((result >= 0) & (result <= 100)).all()

# bullish bias (+2 gain, -1 loss per cycle) means avg_gain > avg_loss → RSI > 50
def test_rsi_bullish_bias(rsi_df):
    result = rsi(rsi_df, period=14).dropna()
    assert (result > 50).all()

# returns a Series of the same length as the input
def test_rsi_series_length(rsi_df):
    result = rsi(rsi_df, period=14)
    assert len(result) == len(rsi_df)
