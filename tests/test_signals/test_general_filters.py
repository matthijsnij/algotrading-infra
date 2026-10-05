"""
================================================================================
UNIT TESTS FOR SIGNALS/GENERAL_FILTERS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_signals/test_general_filters.py -v

To run a specific test function:
    pytest tests/test_signals/test_general_filters.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from datetime import time
from lighthouse.domain.signals.general_filters import (
    is_volume_above_average,
    is_atr_above_threshold,
    is_atr_below_threshold,
    is_rsi_above,
    is_rsi_below,
    is_rsi_in_range,
    is_price_above_ema,
    is_price_below_ema,
    is_price_above_sma,
    is_price_below_sma,
    is_sma_above_sma,
    is_ema_above_ema,
    is_price_above_upper_band,
    is_price_below_lower_band,
    is_price_inside_bands,
    is_price_above_swing_high,
    is_price_below_swing_low,
    is_within_session,
    is_in_named_session,
)

################### TESTS ##########################

# ── is_volume_above_average ───────────────────────────────────────────────────

# ohlcv_df: constant volume 1000.0, rolling avg = 1000.0 for any period.
# The check is: current_volume >= avg * multiplier

# normal case; volume equals average, multiplier=1.0 (default) → True (uses >=)
def test_is_volume_above_average_equal(ohlcv_df):
    assert is_volume_above_average(ohlcv_df, period=10) is True

# multiplier below 1.0; volume easily exceeds the scaled average → True
def test_is_volume_above_average_low_multiplier(ohlcv_df):
    assert is_volume_above_average(ohlcv_df, period=10, multiplier=0.5) is True

# multiplier above 1.0; volume is below the scaled average → False
def test_is_volume_above_average_high_multiplier(ohlcv_df):
    assert is_volume_above_average(ohlcv_df, period=10, multiplier=1.5) is False

# multiplier exactly 1.0; boundary — volume equals threshold exactly → True (>=)
def test_is_volume_above_average_boundary(ohlcv_df):
    assert is_volume_above_average(ohlcv_df, period=10, multiplier=1.0) is True

# returns a bool, not a numpy bool or other truthy type
def test_is_volume_above_average_returns_bool(ohlcv_df):
    result = is_volume_above_average(ohlcv_df, period=10)
    assert isinstance(result, bool)


# ── is_atr_above_threshold ─────────────────────────────────────────────────────────

# ohlcv_df: ATR converges to exactly 2.0 for any period.
# The check is: current_atr > threshold  (strict)

# ATR above threshold → True
def test_is_atr_above_threshold_true(ohlcv_df):
    assert is_atr_above_threshold(ohlcv_df, threshold=1.0, period=14) is True

# ATR below threshold → False
def test_is_atr_above_threshold_false(ohlcv_df):
    assert is_atr_above_threshold(ohlcv_df, threshold=3.0, period=14) is False

# ATR exactly equals threshold; strict > means False at the boundary
def test_is_atr_above_threshold_boundary(ohlcv_df):
    assert is_atr_above_threshold(ohlcv_df, threshold=2.0, period=14) is False

# returns a bool
def test_is_atr_above_threshold_returns_bool(ohlcv_df):
    assert isinstance(is_atr_above_threshold(ohlcv_df, threshold=1.0, period=14), bool)


# ── is_atr_below_threshold ─────────────────────────────────────────────────────────

# ohlcv_df: ATR converges to exactly 2.0 for any period.
# The check is: current_atr < threshold  (strict)

# ATR below threshold → True
def test_is_atr_below_threshold_true(ohlcv_df):
    assert is_atr_below_threshold(ohlcv_df, threshold=3.0, period=14) is True

# ATR above threshold → False
def test_is_atr_below_threshold_false(ohlcv_df):
    assert is_atr_below_threshold(ohlcv_df, threshold=1.0, period=14) is False

# ATR exactly equals threshold; strict < means False at the boundary
def test_is_atr_below_threshold_boundary(ohlcv_df):
    assert is_atr_below_threshold(ohlcv_df, threshold=2.0, period=14) is False

# returns a bool
def test_is_atr_below_threshold_returns_bool(ohlcv_df):
    assert isinstance(is_atr_below_threshold(ohlcv_df, threshold=3.0, period=14), bool)


# ── is_rsi_above ──────────────────────────────────────────────────────────────────

# rsi_df (+2/-1 pattern) produces RSI > 50 at all valid bars.
# ohlcv_df is not used here — monotonically rising series gives NaN RSI.
# The check is: current_rsi > level  (strict)

# RSI > 50 and level=50 → True
def test_is_rsi_above_true(rsi_df):
    assert is_rsi_above(rsi_df, period=14, level=50.0) is True

# level=100; RSI never reaches 100 when losses are present → False
def test_is_rsi_above_false(rsi_df):
    assert is_rsi_above(rsi_df, period=14, level=100.0) is False

# level=0; RSI always >= 0 → True regardless of market
def test_is_rsi_above_zero_level(rsi_df):
    assert is_rsi_above(rsi_df, period=14, level=0.0) is True

# returns a bool
def test_is_rsi_above_returns_bool(rsi_df):
    assert isinstance(is_rsi_above(rsi_df, period=14, level=50.0), bool)


# ── is_rsi_below ──────────────────────────────────────────────────────────────────

# The check is: current_rsi < level  (strict <)

# RSI > 50 and level=100 → True
def test_is_rsi_below_true(rsi_df):
    assert is_rsi_below(rsi_df, period=14, level=100.0) is True

# RSI > 50 and level=50; strict < means False
def test_is_rsi_below_false(rsi_df):
    assert is_rsi_below(rsi_df, period=14, level=50.0) is False

# level=0; RSI is always > 0, so strict < 0 is always False
def test_is_rsi_below_zero_level(rsi_df):
    assert is_rsi_below(rsi_df, period=14, level=0.0) is False

# returns a bool
def test_is_rsi_below_returns_bool(rsi_df):
    assert isinstance(is_rsi_below(rsi_df, period=14, level=100.0), bool)


# ── is_rsi_in_range ──────────────────────────────────────────────────────────────

# The check is: lower <= current_rsi <= upper  (both inclusive)

# RSI > 50; range (50, 100) covers it → True
def test_is_rsi_in_range_true(rsi_df):
    assert is_rsi_in_range(rsi_df, period=14, lower=50.0, upper=100.0) is True

# RSI > 50; range (0, 50) excludes it → False
def test_is_rsi_in_range_false(rsi_df):
    assert is_rsi_in_range(rsi_df, period=14, lower=0.0, upper=50.0) is False

# full range (0, 100) always contains any valid RSI → True
def test_is_rsi_in_range_full_range(rsi_df):
    assert is_rsi_in_range(rsi_df, period=14, lower=0.0, upper=100.0) is True

# returns a bool
def test_is_rsi_in_range_returns_bool(rsi_df):
    assert isinstance(is_rsi_in_range(rsi_df, period=14, lower=0.0, upper=100.0), bool)


# ── Trend filter helpers ──────────────────────────────────────────────────────

# ohlcv_df is a monotonically rising series: close[i] = 100 + i.
# For any period, EMA and SMA both lag behind close, so close > EMA and close > SMA.
# For tests that require close < EMA/SMA, a small falling DataFrame is used inline.

@pytest.fixture
def falling_df():
    """10-bar falling price series: close goes 109 → 100. EMA/SMA lag above close."""
    n     = 10
    close = [109.0 - i for i in range(n)]
    return pd.DataFrame({
        "open":   [c + 0.5 for c in close],
        "high":   [c + 1.0 for c in close],
        "low":    [c - 1.0 for c in close],
        "close":  close,
        "volume": [1000.0] * n,
    })


# ── is_price_above_ema ───────────────────────────────────────────────────────────

# rising series; close leads EMA → True
def test_is_price_above_ema_true(ohlcv_df):
    assert is_price_above_ema(ohlcv_df, period=3) is True

# falling series; close trails below EMA → False
def test_is_price_above_ema_false(falling_df):
    assert is_price_above_ema(falling_df, period=3) is False

# returns a bool
def test_is_price_above_ema_returns_bool(ohlcv_df):
    assert isinstance(is_price_above_ema(ohlcv_df, period=3), bool)


# ── is_price_below_ema ───────────────────────────────────────────────────────────

# falling series; close trails below EMA → True
def test_is_price_below_ema_true(falling_df):
    assert is_price_below_ema(falling_df, period=3) is True

# rising series; close leads EMA → False
def test_is_price_below_ema_false(ohlcv_df):
    assert is_price_below_ema(ohlcv_df, period=3) is False

# returns a bool
def test_is_price_below_ema_returns_bool(ohlcv_df):
    assert isinstance(is_price_below_ema(ohlcv_df, period=3), bool)


# ── is_price_above_sma ───────────────────────────────────────────────────────────

# ohlcv_df: close[-1]=149, SMA(3)[-1]=148 → close > SMA

# rising series; close leads SMA → True
def test_is_price_above_sma_true(ohlcv_df):
    assert is_price_above_sma(ohlcv_df, period=3) is True

# falling series; close trails below SMA → False
def test_is_price_above_sma_false(falling_df):
    assert is_price_above_sma(falling_df, period=3) is False

# returns a bool
def test_is_price_above_sma_returns_bool(ohlcv_df):
    assert isinstance(is_price_above_sma(ohlcv_df, period=3), bool)


# ── is_price_below_sma ───────────────────────────────────────────────────────────

# falling series; close trails below SMA → True
def test_is_price_below_sma_true(falling_df):
    assert is_price_below_sma(falling_df, period=3) is True

# rising series; close leads SMA → False
def test_is_price_below_sma_false(ohlcv_df):
    assert is_price_below_sma(ohlcv_df, period=3) is False

# returns a bool
def test_is_price_below_sma_returns_bool(ohlcv_df):
    assert isinstance(is_price_below_sma(ohlcv_df, period=3), bool)


# ── is_sma_above_sma ─────────────────────────────────────────────────────────────

# ohlcv_df: rising series → shorter-period SMA is closer to current price (higher).
# fast=3: SMA[-1]=148.0, slow=10: SMA[-1]=144.5 → fast > slow

# fast SMA above slow SMA (golden cross condition) → True
def test_is_sma_above_sma_true(ohlcv_df):
    assert is_sma_above_sma(ohlcv_df, fast_period=3, slow_period=10) is True

# fast and slow swapped; slow SMA above fast SMA → False
def test_is_sma_above_sma_false(ohlcv_df):
    assert is_sma_above_sma(ohlcv_df, fast_period=10, slow_period=3) is False

# returns a bool
def test_is_sma_above_sma_returns_bool(ohlcv_df):
    assert isinstance(is_sma_above_sma(ohlcv_df, fast_period=3, slow_period=10), bool)


# ── is_ema_above_ema ─────────────────────────────────────────────────────────────

# ohlcv_df: fast EMA lags less than slow EMA on a rising series → fast > slow.

# fast EMA above slow EMA → True
def test_is_ema_above_ema_true(ohlcv_df):
    assert is_ema_above_ema(ohlcv_df, fast_period=3, slow_period=10) is True

# fast and slow swapped → False
def test_is_ema_above_ema_false(ohlcv_df):
    assert is_ema_above_ema(ohlcv_df, fast_period=10, slow_period=3) is False

# returns a bool
def test_is_ema_above_ema_returns_bool(ohlcv_df):
    assert isinstance(is_ema_above_ema(ohlcv_df, fast_period=3, slow_period=10), bool)


# ── is_price_above_upper_band ─────────────────────────────────────────────────────

# ohlcv_df period=3: close[-1]=149, middle=148, std≈√(2/3)≈0.8165
#   upper ≈ 149.63 → close inside bands with std_dev=2.0
#   upper ≈ 148.001 with std_dev=0.001 → close (149) is above the tiny upper band

# normal bands; close is inside → False
def test_is_price_above_upper_band_false(ohlcv_df):
    assert is_price_above_upper_band(ohlcv_df, period=3, std_dev=2.0) is False

# bands shrunk to near-zero; close breaks above upper band → True
def test_is_price_above_upper_band_true(ohlcv_df):
    assert is_price_above_upper_band(ohlcv_df, period=3, std_dev=0.001) is True

# returns a bool
def test_is_price_above_upper_band_returns_bool(ohlcv_df):
    assert isinstance(is_price_above_upper_band(ohlcv_df, period=3), bool)


# ── is_price_below_lower_band ────────────────────────────────────────────────────

# For any uniform step-1 series, close[-1] is 1 unit from the SMA and
# 2*√(2/3)≈1.63 > 1 so close always sits inside normal bands.
# To get close below lower band: shrink bands via std_dev=0.001 on falling_df.
#   falling_df period=3: middle=101, lower≈100.9992 → close=100 < lower → True

# normal bands; close is inside → False
def test_is_price_below_lower_band_false(ohlcv_df):
    assert is_price_below_lower_band(ohlcv_df, period=3, std_dev=2.0) is False

# bands shrunk to near-zero on falling series; close breaks below lower band → True
def test_is_price_below_lower_band_true(falling_df):
    assert is_price_below_lower_band(falling_df, period=3, std_dev=0.001) is True

# returns a bool
def test_is_price_below_lower_band_returns_bool(ohlcv_df):
    assert isinstance(is_price_below_lower_band(ohlcv_df, period=3), bool)


# ── is_price_inside_bands ─────────────────────────────────────────────────────────

# normal bands; close is inside → True
def test_is_price_inside_bands_true(ohlcv_df):
    assert is_price_inside_bands(ohlcv_df, period=3, std_dev=2.0) is True

# bands shrunk to near-zero; close is outside → False
def test_is_price_inside_bands_false(ohlcv_df):
    assert is_price_inside_bands(ohlcv_df, period=3, std_dev=0.001) is False

# returns a bool
def test_is_price_inside_bands_returns_bool(ohlcv_df):
    assert isinstance(is_price_inside_bands(ohlcv_df, period=3), bool)


# ── is_price_above_swing_high ─────────────────────────────────────────────────────

# The function excludes the last bar when computing swing_high to avoid lookahead bias.
# check is strict: close[-1] > swing_high
#
# ohlcv_df: high[i] = 101+i. Excluding last bar, swing_high(window=5) = high[48] = 149.
# close[-1] = 149. Strict > means 149 > 149 → False.
#
# For True: a DataFrame whose last bar close spikes above the prior swing high.
#   bars 0-8 alternate close 100/102 → swing_high(window=5) of bars[4..8] = 103
#   last bar close = 110 > 103 → True

# close equals previous swing high (strict >) → False
def test_is_price_above_swing_high_false(ohlcv_df):
    assert is_price_above_swing_high(ohlcv_df, window=5) is False

# last bar spikes above previous swing high → True
def test_is_price_above_swing_high_true():
    closes = [100.0, 102.0] * 4 + [100.0, 110.0]  # 10 bars, last = 110
    df = pd.DataFrame({
        "open":   [c - 0.5 for c in closes],
        "high":   [c + 1.0 for c in closes],
        "low":    [c - 1.0 for c in closes],
        "close":  closes,
        "volume": [1000.0] * len(closes),
    })
    assert is_price_above_swing_high(df, window=5) is True

# returns a bool
def test_is_price_above_swing_high_returns_bool(ohlcv_df):
    assert isinstance(is_price_above_swing_high(ohlcv_df, window=5), bool)


# ── is_price_below_swing_low ─────────────────────────────────────────────────────

# ohlcv_df: low[i] = 99+i. Excluding last bar, swing_low(window=5) = low[44] = 143.
# close[-1] = 149. 149 < 143 → False.
#
# For True: last bar crashes below the prior swing low.
#   bars 0-8 alternate close 100/102 → swing_low(window=5) of bars[4..8] = 99
#   last bar close = 85 < 99 → True

# close is well above previous swing low → False
def test_is_price_below_swing_low_false(ohlcv_df):
    assert is_price_below_swing_low(ohlcv_df, window=5) is False

# last bar crashes below previous swing low → True
def test_is_price_below_swing_low_true():
    closes = [100.0, 102.0] * 4 + [100.0, 85.0]  # 10 bars, last = 85
    df = pd.DataFrame({
        "open":   [c - 0.5 for c in closes],
        "high":   [c + 1.0 for c in closes],
        "low":    [c - 1.0 for c in closes],
        "close":  closes,
        "volume": [1000.0] * len(closes),
    })
    assert is_price_below_swing_low(df, window=5) is True

# returns a bool
def test_is_price_below_swing_low_returns_bool(ohlcv_df):
    assert isinstance(is_price_below_swing_low(ohlcv_df, window=5), bool)


# ── Session filter helpers ──────────────────────────────────────────────────────

def _session_df(ts: str) -> pd.DataFrame:
    """
    Build a minimal 1-row OHLCV DataFrame whose index is the given timestamp string.
    Timestamp may be tz-naive (e.g. '2024-01-15 10:30:00') or
    tz-aware (e.g. '2024-01-15 10:30:00+00:00').
    """
    index = pd.DatetimeIndex([pd.Timestamp(ts)])
    return pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=index,
    )


# ── is_within_session ────────────────────────────────────────────────────────────

# Uses UTC timestamps and tz="UTC" to avoid DST complications.
# Session window [09:00, 17:00), check is: session_open <= bar_time < session_close

_OPEN  = time(9, 0)
_CLOSE = time(17, 0)

# bar at 10:30 UTC is inside [09:00, 17:00) → True
def test_is_within_session_true():
    df = _session_df("2024-01-15 10:30:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is True

# bar at 08:00 UTC is before session open → False
def test_is_within_session_false_before():
    df = _session_df("2024-01-15 08:00:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is False

# bar at 18:00 UTC is after session close → False
def test_is_within_session_false_after():
    df = _session_df("2024-01-15 18:00:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is False

# exactly at session_open is inclusive → True
def test_is_within_session_boundary_open():
    df = _session_df("2024-01-15 09:00:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is True

# exactly at session_close is exclusive → False
def test_is_within_session_boundary_close():
    df = _session_df("2024-01-15 17:00:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is False

# tz-aware timestamp is converted to target tz before comparison
# 10:30 UTC+00:00 == 10:30 UTC → inside session → True
def test_is_within_session_tz_aware():
    df = _session_df("2024-01-15 10:30:00+00:00")
    assert is_within_session(df, _OPEN, _CLOSE, tz="UTC") is True

# midnight-wrapping session: open=22:00, close=02:00
# 23:00 is after open → True; 01:00 is before close → True; 10:00 is neither → False
def test_is_within_session_midnight_wrap_inside_before_midnight():
    df = _session_df("2024-01-15 23:00:00")
    assert is_within_session(df, time(22, 0), time(2, 0), tz="UTC") is True

def test_is_within_session_midnight_wrap_inside_after_midnight():
    df = _session_df("2024-01-15 01:00:00")
    assert is_within_session(df, time(22, 0), time(2, 0), tz="UTC") is True

def test_is_within_session_midnight_wrap_outside():
    df = _session_df("2024-01-15 10:00:00")
    assert is_within_session(df, time(22, 0), time(2, 0), tz="UTC") is False

# returns a bool
def test_is_within_session_returns_bool():
    df = _session_df("2024-01-15 10:30:00")
    assert isinstance(is_within_session(df, _OPEN, _CLOSE, tz="UTC"), bool)


# ── is_in_named_session ──────────────────────────────────────────────────────────

# London session: 08:00-17:00 Europe/London.
# January has no DST, so Europe/London == UTC+00:00.
# 10:00 UTC → 10:00 London → inside London session → True
# 07:00 UTC → 07:00 London → outside London session → False

# bar within London session → True
def test_is_in_named_session_true():
    df = _session_df("2024-01-15 10:00:00")
    assert is_in_named_session(df, "london") is True

# bar outside London session → False
def test_is_in_named_session_false():
    df = _session_df("2024-01-15 07:00:00")
    assert is_in_named_session(df, "london") is False

# unknown session name raises ValueError
def test_is_in_named_session_unknown():
    df = _session_df("2024-01-15 10:00:00")
    with pytest.raises(ValueError):
        is_in_named_session(df, "invalid_session")

# returns a bool
def test_is_in_named_session_returns_bool():
    df = _session_df("2024-01-15 10:00:00")
    assert isinstance(is_in_named_session(df, "london"), bool)
