"""
================================================================================
UNIT TESTS FOR exchanges/backtest/data_window.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_data_window.py -v

To run a specific test function:
    pytest tests/backtest/test_data_window.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.data_window import DataWindow
from lighthouse.exchanges.backtest.state import BacktestState
from lighthouse.exchanges.backtest.config import BacktestConfig
from lighthouse.domain.instruments import InstrumentSpec
from lighthouse.domain.calendars import TradingCalendar, CALENDAR_24_7, CALENDAR_WEEKDAY
from lighthouse.domain.timeframe import Timeframe

################### HELPERS ##########################

def make_df() -> pd.DataFrame:
    """
    8 bars at 1h frequency covering exactly two complete 4h periods.

    First 4h period (bars 0-3, 00:00-04:00):
        open=100, high=110, low=95, close=108, volume=700
    Second 4h period (bars 4-7, 04:00-08:00):
        open=108, high=115, low=105, volume=1100

    Known values enable exact aggregation assertions.
    """
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 00:00", periods=8, freq="1h", tz="UTC"),
        "open":      [100, 102, 104, 103,  108, 109, 107, 110],
        "high":      [105, 107, 106, 110,  112, 113, 111, 115],
        "low":       [ 95,  98, 101, 100,  106, 107, 105, 108],
        "close":     [102, 104, 103, 108,  109, 107, 110, 112],
        "volume":    [100, 200, 150, 250,  300, 200, 250, 350],
    })


def make_state(cursor: int = 0, calendar: TradingCalendar = None) -> BacktestState:
    """Return a BacktestState at the given cursor position with 1h base timeframe."""
    config = BacktestConfig(
        spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
        latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
    )
    instrument = InstrumentSpec(
        quote_currency="USDT", can_short=True, leverage=1.0,
        maintenance_margin=0.0, has_liquidation=False,
        base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
    )
    state = BacktestState(
        df=make_df(), ts_col="timestamp", symbol="BTC/USDT",
        instrument=instrument, base_timeframe="1h",
        initial_balance=10_000.0, config=config, calendar=calendar,
    )
    state.cursor = cursor
    return state


def make_window(cursor: int = 0) -> DataWindow:
    """Return a DataWindow at the given cursor."""
    return DataWindow(make_state(cursor))


def make_custom_window(df: pd.DataFrame, base_timeframe: str, calendar: TradingCalendar, cursor: int) -> DataWindow:
    """Return a DataWindow over an arbitrary df/base_timeframe/calendar (for calendar-aware tests)."""
    config = BacktestConfig(
        spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
        latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
    )
    instrument = InstrumentSpec(
        quote_currency="USDT", can_short=True, leverage=1.0,
        maintenance_margin=0.0, has_liquidation=False,
        base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
    )
    state = BacktestState(
        df=df.reset_index(drop=False), ts_col=df.index.name, symbol="AAPL",
        instrument=instrument, base_timeframe=base_timeframe,
        initial_balance=10_000.0, config=config, calendar=calendar,
    )
    state.cursor = cursor
    return DataWindow(state)

################### TESTS ##########################

# ── get_window() — base timeframe pass-through ────────────────────────────────────────────────────────

# returns exactly 'limit' bars ending at cursor (no future bars)
def test_get_window_base_returns_limit_bars():
    dw = make_window(cursor=5)

    result = dw.get_window(limit=3, timeframe="1h")

    assert len(result) == 3
    assert result.index[-1] == pd.Timestamp("2024-01-01 05:00", tz="UTC")
    assert result.index[0]  == pd.Timestamp("2024-01-01 03:00", tz="UTC")


# returns fewer than limit when not enough history (cursor near start)
def test_get_window_base_fewer_bars_near_start():
    dw = make_window(cursor=1)

    result = dw.get_window(limit=5, timeframe="1h")

    assert len(result) == 2  # only bars 0 and 1 exist

# ── get_window() — coarser timeframe aggregation ────────────────────────────────────────────────────────

# raises when requesting a finer timeframe than base
def test_get_window_finer_timeframe_raises():
    dw = make_window(cursor=3)

    with pytest.raises(ValueError, match="not a whole multiple of"):
        dw.get_window(limit=1, timeframe="15m")  # 15m < 1h


# raises when requesting a coarser timeframe that isn't a whole multiple of base (bug fix:
# a plain seconds comparison (5400s >= 3600s) would have wrongly accepted this)
def test_get_window_non_multiple_timeframe_raises():
    dw = make_window(cursor=3)

    with pytest.raises(ValueError, match="not a whole multiple of"):
        dw.get_window(limit=1, timeframe="90m")  # 90m is coarser than 1h but not a clean multiple


# OHLCV aggregation is correct: open=first, high=max, low=min, close=last, volume=sum
def test_get_window_aggregation_correct():
    dw = make_window(cursor=3)  # last bar of first 4h period → boundary

    result = dw.get_window(limit=1, timeframe="4h")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["open"]   == pytest.approx(100.0)   # first bar's open
    assert row["high"]   == pytest.approx(110.0)   # max of [105,107,106,110]
    assert row["low"]    == pytest.approx(95.0)    # min of [95,98,101,100]
    assert row["close"]  == pytest.approx(108.0)   # last bar's close
    assert row["volume"] == pytest.approx(700.0)   # sum of [100,200,150,250]


# includes the incomplete last bucket (as the current/forming period) when cursor is
# mid-period — per ADR 0001, the last row is always the current period, complete or not
def test_get_window_includes_incomplete_bucket_as_current():
    dw = make_window(cursor=5)  # 05:00 is mid-period (4h period 04:00-08:00 not complete)

    result = dw.get_window(limit=5, timeframe="4h")

    # First complete 4h period (bars 0-3) plus the still-forming second period (bars 4-5)
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert result.index[1] == pd.Timestamp("2024-01-01 04:00", tz="UTC")
    last_row = result.iloc[-1]
    assert last_row["open"]   == pytest.approx(108.0)  # bar4's open
    assert last_row["high"]   == pytest.approx(113.0)  # max of [112, 113]
    assert last_row["low"]    == pytest.approx(106.0)  # min of [106, 107]
    assert last_row["close"]  == pytest.approx(107.0)  # bar5's close
    assert last_row["volume"] == pytest.approx(500.0)  # sum of [300, 200]


# includes completed bucket when cursor is at period boundary
def test_get_window_includes_complete_bucket_at_boundary():
    dw = make_window(cursor=7)  # 07:00 is the last bar of the second 4h period

    result = dw.get_window(limit=5, timeframe="4h")

    assert len(result) == 2  # both complete 4h periods returned

# ── is_signal_bar_close() ────────────────────────────────────────────────────────

# always True when timeframe == base_timeframe
def test_is_signal_bar_close_base_timeframe():
    dw = make_window(cursor=2)

    assert dw.is_signal_bar_close("1h") is True


# True at a period boundary (last 1h bar of a 4h period)
def test_is_signal_bar_close_at_boundary():
    dw = make_window(cursor=3)  # 03:00: next bar is 04:00 which starts a new 4h period

    assert dw.is_signal_bar_close("4h") is True


# False mid-period (not the last 1h bar of a 4h period)
def test_is_signal_bar_close_mid_period():
    dw = make_window(cursor=1)  # 01:00: next bar is 02:00, still in the same 4h period

    assert dw.is_signal_bar_close("4h") is False

# ── fetch_ohlcv() ────────────────────────────────────────────────────────

# returns list-of-lists in [timestamp_ms, open, high, low, close, volume] format
def test_fetch_ohlcv_format():
    dw = make_window(cursor=0)

    result = dw.fetch_ohlcv("BTC/USDT", timeframe=Timeframe.parse("1h"), limit=1)

    assert len(result)    == 1
    row = result[0]
    assert len(row)       == 6
    assert isinstance(row[0], int)    # timestamp in ms
    assert row[0]         == int(pd.Timestamp("2024-01-01 00:00", tz="UTC").timestamp() * 1000)
    assert row[1]         == pytest.approx(100.0)  # open
    assert row[4]         == pytest.approx(102.0)  # close

# ── calendar-aware regression tests ──────────────────────────────────────────────

# explicit CALENDAR_24_7 gives byte-identical results to the implicit default (backward compat)
def test_get_window_explicit_24_7_calendar_matches_default():
    default_result = DataWindow(make_state(cursor=7)).get_window(limit=5, timeframe="4h")
    explicit_result = DataWindow(make_state(cursor=7, calendar=CALENDAR_24_7)).get_window(limit=5, timeframe="4h")

    pd.testing.assert_frame_equal(default_result, explicit_result)


# silent-zero-trade regression: 1h equities (14:30-20:30 UTC session) + "1d" signal
# now correctly reports a bar close at the last in-session hour of the day, since
# next-row detection compares against the actual next bar instead of guessing a bar
# would exist at 23:00 UTC (which never exists for this session)
def test_is_signal_bar_close_hourly_equities_daily_signal_ticks(weekday_hourly_df):
    dw = make_custom_window(weekday_hourly_df, base_timeframe="1h", calendar=CALENDAR_WEEKDAY, cursor=6)  # Monday 20:30, last bar of the day

    assert dw.is_signal_bar_close("1d") is True


# mid-day hour is not a "1d" signal close under the same session
def test_is_signal_bar_close_hourly_equities_daily_signal_mid_day(weekday_hourly_df):
    dw = make_custom_window(weekday_hourly_df, base_timeframe="1h", calendar=CALENDAR_WEEKDAY, cursor=3)  # Monday 17:30, mid-session

    assert dw.is_signal_bar_close("1d") is False


# session-split aggregation: a "1d" get_window over 1h equities bars produces one
# row per trading day (not merged across the overnight/weekend gap)
def test_get_window_session_split_daily_from_hourly(weekday_hourly_df):
    dw = make_custom_window(weekday_hourly_df, base_timeframe="1h", calendar=CALENDAR_WEEKDAY, cursor=13)  # Tuesday's last bar

    result = dw.get_window(limit=5, timeframe="1d")

    assert len(result) == 2  # Monday + Tuesday, each a complete session
    assert result.index[0].date().isoformat() == "2024-01-08"  # Monday
    assert result.index[1].date().isoformat() == "2024-01-09"  # Tuesday
