"""
================================================================================
UNIT TESTS FOR SIGNALS/EXAMPLE.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_signals/test_example.py -v

To run a specific test function:
    pytest tests/test_signals/test_example.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.domain.signals.example import is_long_breakout, is_short_breakout

################### TESTS ##########################

def _make_ranging_df(bar8_close: float) -> pd.DataFrame:
    """
    Build a 9-bar OHLCV DataFrame for breakout tests.

    Bars 0-7 alternate between close=100 and close=102 (ranging, no trend).
    Bar 8 (the last row, iloc[-1]) has the given close, creating the
    breakout or crash scenario under test.

    With window=5 and df.iloc[:-1] (bars 0-7):
        swing_high = max(high[3..7]) = max(103, 101, 103, 101, 103) = 103.0
        swing_low  = min(low[3..7])  = min(101,  99, 101,  99, 101) =  99.0
    """
    closes = [100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0,
              bar8_close]
    idx = pd.date_range("2024-01-01", periods=9, freq="1h")
    df = pd.DataFrame(
        {
            "open":   [c - 0.5 for c in closes],
            "high":   [c + 1.0 for c in closes],
            "low":    [c - 1.0 for c in closes],
            "close":  closes,
            "volume": [1000.0] * 9,
        },
        index=idx,
    )
    return df


# ── is_long_breakout ────────────────────────────────────────────────────────

# no breakout; close[-1]=149 equals swing_high=149, strict > is False
def test_is_long_breakout_false_when_no_breakout(ohlcv_df):
    assert is_long_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=0.0) is False

# no breakout; positive buffer makes condition even harder to satisfy
def test_is_long_breakout_false_with_buffer(ohlcv_df):
    assert is_long_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=1.0) is False

# breakout; close[-1]=115 > swing_high=103, no ATR buffer applied
def test_is_long_breakout_true_on_breakout():
    df = _make_ranging_df(bar8_close=115.0)
    assert is_long_breakout(df, window=5, atr_period=3, atr_buffer_mult=0.0) is True

# breakout; 115 clears swing_high=103 by 12, well above any ATR buffer
def test_is_long_breakout_true_with_buffer():
    df = _make_ranging_df(bar8_close=115.0)
    assert is_long_breakout(df, window=5, atr_period=3, atr_buffer_mult=1.0) is True

# return type; must be bool
def test_is_long_breakout_returns_bool(ohlcv_df):
    result = is_long_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=0.0)
    assert isinstance(result, bool)


# ── is_short_breakout ────────────────────────────────────────────────────────

# no breakout; close[-1]=149 is above swing_low=143
def test_is_short_breakout_false_when_no_breakout(ohlcv_df):
    assert is_short_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=0.0) is False

# no breakout; positive buffer makes condition even harder to satisfy
def test_is_short_breakout_false_with_buffer(ohlcv_df):
    assert is_short_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=1.0) is False

# breakout; close[-1]=85 < swing_low=99, no ATR buffer applied
def test_is_short_breakout_true_on_breakout():
    df = _make_ranging_df(bar8_close=85.0)
    assert is_short_breakout(df, window=5, atr_period=3, atr_buffer_mult=0.0) is True

# breakout; 85 clears swing_low=99 by 14, well below any ATR buffer
def test_is_short_breakout_true_with_buffer():
    df = _make_ranging_df(bar8_close=85.0)
    assert is_short_breakout(df, window=5, atr_period=3, atr_buffer_mult=1.0) is True

# return type; must be bool
def test_is_short_breakout_returns_bool(ohlcv_df):
    result = is_short_breakout(ohlcv_df, window=5, atr_period=14, atr_buffer_mult=0.0)
    assert isinstance(result, bool)
