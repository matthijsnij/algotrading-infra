"""
================================================================================
FIXED DATA FOR UNIT TESTS 
================================================================================
"""

################## IMPORTS ##################

import pytest
import pandas as pd
import logging
from typing import Callable
from lighthouse.utils.logging import init_logging
from lighthouse.domain.exchange_types import NormalizedOrder

################## SETUP ##################

init_logging("test")

############### FIXTURES ##################

@pytest.fixture(autouse=True)
def reset_logger_state():
    """
    Reinitialize logger before each test to ensure clean state.
    
    Prevents logger state pollution from previous tests (e.g. tests that
    explicitly reset _initialized). After each test, also cleans up.
    """
    init_logging("test")
    yield
    # Teardown: close and remove all handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """
    Standard OHLCV DataFrame for indicator tests.

    50 bars of a steadily rising price series:
        close[i]  = 100 + i          (100.0, 101.0, ..., 149.0)
        high[i]   = close[i] + 1.0
        low[i]    = close[i] - 1.0
        open_[i]  = close[i] - 0.5
        volume[i] = 1000.0

    Properties that make assertions exact:
    - True Range is constant 2.0 → ATR converges to exactly 2.0
    - All bars are gains, no losses → avg_loss=0 → RSI returns NaN (use rsi_df for RSI tests)
    - Volume is constant → rolling average is always 1000.0, ratio always 1.0
    - SMA / EMA of close[i] = 100 + i can be computed by hand for any window
    - swing_high(window) = high[-1] = 150.0 (monotonically rising)
    - swing_low(window)  = low[-window] (monotonically rising)
    """
    n      = 50
    close  = [100.0 + i for i in range(n)]
    high   = [c + 1.0 for c in close]
    low    = [c - 1.0 for c in close]
    open_  = [c - 0.5 for c in close]
    volume = [1000.0] * n

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })


@pytest.fixture
def rsi_df() -> pd.DataFrame:
    """
    DataFrame for RSI tests. Alternates gains and losses so avg_loss > 0
    and RSI produces a finite value rather than NaN.

    50 bars with a repeating +2 / -1 pattern:
        close = [100, 102, 101, 103, 102, ...]

    With this pattern every two bars net +1, so the series trends up
    while always having non-zero losses.
    """
    n      = 50
    close  = [100.0]
    for i in range(1, n):
        close.append(close[-1] + 2.0 if i % 2 == 1 else close[-1] - 1.0)
    high   = [c + 0.5 for c in close]
    low    = [c - 0.5 for c in close]
    open_  = [c - 0.25 for c in close]
    volume = [1000.0] * n

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })


@pytest.fixture
def bad_columns_df() -> pd.DataFrame:
    """
    DataFrame with wrong column names for testing ValueError on missing columns.
    Contains 'Open', 'High', 'Low', 'Close', 'Volume' (capitalised) instead of
    the expected lowercase names.
    """
    return pd.DataFrame({
        "Open":   [100.0],
        "High":   [101.0],
        "Low":    [99.0],
        "Close":  [100.5],
        "Volume": [1000.0],
    })


@pytest.fixture
def weekday_daily_df() -> pd.DataFrame:
    """
    15 daily bars, Mon-Fri only, 3 weeks starting 2024-01-01 (a Monday).

    pd.date_range(freq="B") emits business days only, so the index has exactly
    two gaps: Fri 2024-01-05 -> Mon 2024-01-08 and Fri 2024-01-12 -> Mon 2024-01-15,
    each a 3-calendar-day delta instead of the modal 1-day delta. No other gaps.

    Timestamped at 21:00 UTC (= 16:00 EST session close) so that the local date
    in America/New_York matches the trading day, not the previous day.

    close[i] = 100 + i, high/low/open offsets and volume match ohlcv_df's pattern
    so aggregation math (e.g. SMA) can be computed by hand.
    """
    n     = 15
    idx   = pd.date_range("2024-01-01 21:00", periods=n, freq="B", tz="UTC", name="timestamp")
    close = [100.0 + i for i in range(n)]
    high  = [c + 1.0 for c in close]
    low   = [c - 1.0 for c in close]
    open_ = [c - 0.5 for c in close]
    volume = [1000.0] * n

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=idx)


@pytest.fixture
def weekday_hourly_df() -> pd.DataFrame:
    """
    35 hourly bars: 5 weekdays (Mon 2024-01-08 .. Fri 2024-01-12) x 7 bars/day,
    covering the 14:30-20:30 UTC session (America/New_York 09:30-16:00 EST in
    January, before DST). Built session-by-session so only in-session hours are
    included, giving two gap shapes automatically:
      - overnight: day N's 20:30 bar -> day N+1's 14:30 bar (18h delta)
      - weekend:   Friday's 20:30 bar -> Monday's 14:30 bar (66h delta)
    versus the modal 1h delta within a session.

    close[i] = 100 + i, high/low/open offsets and volume match ohlcv_df's pattern.
    """
    session_days = pd.date_range("2024-01-08", periods=5, freq="B", tz="UTC")
    timestamps = [
        day + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(hours=h)
        for day in session_days
        for h in range(7)
    ]
    idx   = pd.DatetimeIndex(timestamps, name="timestamp")
    n     = len(idx)
    close = [100.0 + i for i in range(n)]
    high  = [c + 1.0 for c in close]
    low   = [c - 1.0 for c in close]
    open_ = [c - 0.5 for c in close]
    volume = [1000.0] * n

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=idx)


@pytest.fixture
def make_order() -> Callable[..., NormalizedOrder]:
    """
    Factory for NormalizedOrder test fixtures.

    Returns a callable that builds a NormalizedOrder with sensible defaults
    (id="order-1", status="open", symbol="BTC/USDT:USDT", side="buy",
    type="market", size=0.01), overridable via keyword arguments, e.g.
    make_order(id="abc", type="stop", stop_price=100.0).
    """
    def _make_order(**overrides) -> NormalizedOrder:
        defaults = dict(
            id="order-1",
            status="open",
            symbol="BTC/USDT:USDT",
            side="buy",
            type="market",
            size=0.01,
        )
        defaults.update(overrides)
        return NormalizedOrder(**defaults)
    return _make_order

