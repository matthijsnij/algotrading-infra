"""
================================================================================
UNIT TESTS FOR exchanges/backtest/exchange.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_exchange.py -v

To run a specific test function:
    pytest tests/backtest/test_exchange.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.exchange import BacktestExchange
from lighthouse.exchanges.backtest.config import BacktestConfig, MetricsConfig
from lighthouse.domain.instruments import InstrumentSpec
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition

################### HELPERS ##########################

def make_df(n_bars: int = 10) -> pd.DataFrame:
    """Return a valid OHLCV DataFrame with a named DatetimeIndex for BacktestExchange."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n_bars)],
        "high":   [101.0 + i for i in range(n_bars)],
        "low":    [ 99.0 + i for i in range(n_bars)],
        "close":  [100.0 + i for i in range(n_bars)],
        "volume": [1000.0]   * n_bars,
    }, index=idx)


def make_config() -> BacktestConfig:
    """Return a minimal valid BacktestConfig."""
    return BacktestConfig(
        spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
        latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
    )


def make_instrument(calendar: str = "24/7") -> InstrumentSpec:
    """Return a minimal InstrumentSpec."""
    return InstrumentSpec(
        quote_currency="USDT", can_short=True, leverage=10.0,
        maintenance_margin=0.05, has_liquidation=False,
        base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
        calendar=calendar,
    )


def make_weekday_df(n_bars: int = 10) -> pd.DataFrame:
    """n_bars daily bars, Mon-Fri only (freq='B'), starting 2024-01-08 (a Monday).
    With n_bars=10 this spans two trading weeks and includes one Fri->Mon gap.
    Timestamped at 21:00 UTC (= 16:00 EST session close) so local weekday is correct."""
    idx = pd.date_range("2024-01-08 21:00", periods=n_bars, freq="B", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n_bars)],
        "high":   [101.0 + i for i in range(n_bars)],
        "low":    [ 99.0 + i for i in range(n_bars)],
        "close":  [100.0 + i for i in range(n_bars)],
        "volume": [1000.0]   * n_bars,
    }, index=idx)


def make_exchange(n_bars: int = 5) -> BacktestExchange:
    """Return a ready-to-use BacktestExchange with n_bars of data."""
    return BacktestExchange(
        df              = make_df(n_bars),
        symbol          = "BTC/USDT",
        initial_balance = 10_000.0,
        instrument      = make_instrument(),
        base_timeframe  = "1h",
        config          = make_config(),
    )

################### TESTS ##########################

# ── Constructor validation ────────────────────────────────────────────────────────

# raises when df does not have a DatetimeIndex
def test_constructor_raises_non_datetime_index():
    df = make_df().reset_index(drop=True)  # integer index

    with pytest.raises(ValueError, match="DatetimeIndex"):
        BacktestExchange(
            df=df, symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        )


# raises when df has a timezone-naive DatetimeIndex
def test_constructor_raises_naive_datetime_index():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", name="timestamp")  # no tz
    df = pd.DataFrame({
        "open": [100.0] * 5, "high": [101.0] * 5,
        "low":  [ 99.0] * 5, "close": [100.0] * 5, "volume": [1000.0] * 5,
    }, index=idx)

    with pytest.raises(ValueError, match="timezone-aware"):
        BacktestExchange(
            df=df, symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        )


# raises when df has a non-uniform time delta between rows
def test_constructor_raises_non_uniform_time_delta():
    idx = pd.DatetimeIndex([
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 03:00",  # gap: 2h instead of 1h
        "2024-01-01 04:00",
    ], tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 4, "high": [101.0] * 4,
        "low":  [ 99.0] * 4, "close": [100.0] * 4, "volume": [1000.0] * 4,
    }, index=idx)

    with pytest.raises(ValueError, match="Illegal gap"):
        BacktestExchange(
            df=df, symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        )


# calendar="weekday" accepts a gappy Mon-Fri df (weekend gap is structurally legal)
def test_constructor_weekday_calendar_accepts_gappy_df():
    exchange = BacktestExchange(
        df=make_weekday_df(), symbol="AAPL", initial_balance=10_000.0,
        instrument=make_instrument(calendar="weekday"), base_timeframe="1d", config=make_config(),
    )

    assert exchange._state.calendar.name == "weekday"


# calendar="weekday" still rejects a hole inside a single trading day
def test_constructor_weekday_calendar_rejects_in_session_hole():
    idx = pd.DatetimeIndex([
        "2024-01-08 14:30", "2024-01-08 15:30", "2024-01-08 17:30",  # 16:30 bar missing, same day
    ], tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 3, "high": [101.0] * 3,
        "low":  [ 99.0] * 3, "close": [100.0] * 3, "volume": [1000.0] * 3,
    }, index=idx)

    with pytest.raises(ValueError, match="Illegal gap"):
        BacktestExchange(
            df=df, symbol="AAPL", initial_balance=10_000.0,
            instrument=make_instrument(calendar="weekday"), base_timeframe="1h", config=make_config(),
        )


# duplicate timestamps raise regardless of calendar
def test_constructor_raises_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 3, "high": [101.0] * 3,
        "low":  [ 99.0] * 3, "close": [100.0] * 3, "volume": [1000.0] * 3,
    }, index=idx)

    with pytest.raises(ValueError, match="duplicate"):
        BacktestExchange(
            df=df, symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        )


# out-of-order timestamps raise regardless of calendar
def test_constructor_raises_out_of_order_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 02:00"], tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 3, "high": [101.0] * 3,
        "low":  [ 99.0] * 3, "close": [100.0] * 3, "volume": [1000.0] * 3,
    }, index=idx)

    with pytest.raises(ValueError, match="sorted ascending"):
        BacktestExchange(
            df=df, symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        )


# raises when config is None
def test_constructor_raises_config_none():
    with pytest.raises(Exception):
        BacktestExchange(
            df=make_df(), symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1h", config=None,
        )


# __init__ wires each sub-engine to the shared BacktestState, and wires
# LiquidationEngine's close_position callback to FillEngine.close_position
def test_constructor_wires_sub_engines_to_shared_state():
    exchange = make_exchange(n_bars=3)

    assert exchange._fills._s is exchange._state
    assert exchange._book._s is exchange._state
    assert exchange._book._fills is exchange._fills
    assert exchange._funding._s is exchange._state
    assert exchange._liq._s is exchange._state
    assert exchange._liq._close_position == exchange._fills.close_position
    assert exchange._data._s is exchange._state
    assert exchange._reporter._s is exchange._state


# ── metrics_config threading ────────────────────────────────────────────────

# metrics_config defaults to MetricsConfig() when omitted
def test_constructor_metrics_config_defaults():
    exchange = make_exchange()

    assert exchange._state.metrics_config.metrics_timeframe == "1d"
    assert exchange._state.metrics_config.risk_free_rate == 0.0


# a supplied metrics_config is threaded through to the shared state unchanged
def test_constructor_metrics_config_threaded_through():
    mc = MetricsConfig(metrics_timeframe="1h", risk_free_rate=0.02)
    exchange = BacktestExchange(
        df=make_df(), symbol="BTC/USDT", initial_balance=10_000.0,
        instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        metrics_config=mc,
    )

    assert exchange._state.metrics_config is mc


# metrics_timeframe coarser than base_timeframe is valid (downsampling)
def test_constructor_metrics_timeframe_coarser_than_base_is_valid():
    BacktestExchange(
        df=make_df(), symbol="BTC/USDT", initial_balance=10_000.0,
        instrument=make_instrument(), base_timeframe="1h", config=make_config(),
        metrics_config=MetricsConfig(metrics_timeframe="1d"),
    )


# metrics_timeframe finer than base_timeframe raises (would upsample, fabricating data)
def test_constructor_metrics_timeframe_finer_than_base_raises():
    with pytest.raises(ValueError, match="metrics_timeframe"):
        BacktestExchange(
            df=make_df(), symbol="BTC/USDT", initial_balance=10_000.0,
            instrument=make_instrument(), base_timeframe="1d", config=make_config(),
            metrics_config=MetricsConfig(metrics_timeframe="1h"),
        )

# ── advance() ────────────────────────────────────────────────────────

# returns True while bars remain, False when end-of-data is reached
def test_advance_returns_false_at_end_of_data():
    exchange = make_exchange(n_bars=3)

    assert exchange.advance() is True   # bar 0
    assert exchange.advance() is True   # bar 1
    assert exchange.advance() is True   # bar 2
    assert exchange.advance() is False  # end-of-data


# equity curve grows by one entry per advance() call
def test_advance_appends_equity_curve_each_bar():
    exchange = make_exchange(n_bars=4)

    for _ in range(4):
        exchange.advance()

    assert len(exchange.equity_curve) == 4

# ── fetch_open_positions() ──────────────────────────────────

# no open position, returns empty list
def test_fetch_open_positions_flat_returns_empty():
    exchange = make_exchange(n_bars=3)
    exchange.advance()

    assert exchange.fetch_open_positions("BTC/USDT") == []

# internal position dict is converted to a NormalizedPosition at the boundary
# (internal extras like entry_price/mark_price/leverage are not exposed)
def test_fetch_open_positions_returns_normalized_position():
    exchange = make_exchange(n_bars=3)
    exchange.advance()

    exchange._state.position = {
        "symbol":      "BTC/USDT",
        "side":        "long",
        "size":        1.0,
        "entry_price": 100.0,
        "mark_price":  100.0,
        "leverage":    10.0,
    }

    assert exchange.fetch_open_positions("BTC/USDT") == [
        NormalizedPosition(symbol="BTC/USDT", side=Side.LONG, size=1.0)
    ]


# ── get_equity() ──────────────────────────────────────────────

# flat (no position): equity equals balance total, symbols/quote_currency args ignored
def test_get_equity_flat_returns_balance_total():
    exchange = make_exchange(n_bars=3)
    exchange.advance()

    assert exchange.get_equity(["BTC/USDT"], "USDT") == exchange._state.balance["total"]


# long position: equity = balance total + (mark - entry) * size
def test_get_equity_long_position_adds_unrealized_pnl():
    exchange = make_exchange(n_bars=3)
    exchange.advance()
    s = exchange._state
    s.position = {
        "symbol":      "BTC/USDT",
        "side":        "long",
        "size":        2.0,
        "entry_price": 100.0,
        "mark_price":  110.0,
        "leverage":    10.0,
    }

    expected = s.balance["total"] + (110.0 - 100.0) * 2.0
    assert exchange.get_equity(["BTC/USDT"], "USDT") == expected


# short position: equity = balance total + (entry - mark) * size
def test_get_equity_short_position_adds_unrealized_pnl():
    exchange = make_exchange(n_bars=3)
    exchange.advance()
    s = exchange._state
    s.position = {
        "symbol":      "BTC/USDT",
        "side":        "short",
        "size":        1.5,
        "entry_price": 100.0,
        "mark_price":  90.0,
        "leverage":    10.0,
    }

    expected = s.balance["total"] + (100.0 - 90.0) * 1.5
    assert exchange.get_equity(["BTC/USDT"], "USDT") == expected


# ── finalize() ────────────────────────────────────────────────────────

# force-closes an open position at the last bar's close, recorded in trade log
def test_finalize_closes_open_position():
    exchange = make_exchange(n_bars=3)
    while exchange.advance():
        pass  # advance to last bar

    # Inject an open position directly into state
    s = exchange._state
    s.position = {
        "symbol":      "BTC/USDT",
        "side":        "long",
        "size":        1.0,
        "entry_price": 100.0,
        "mark_price":  100.0,
        "leverage":    10.0,
    }
    s.open_trade = {
        "symbol":       "BTC/USDT",
        "side":         "long",
        "entry_price":  100.0,
        "size":         1.0,
        "entry_bar":    0,
        "entry_ts":     s.get_bar_timestamp(0),
        "entry_fee":    0.0,
        "funding_paid": 0.0,
    }

    exchange.finalize()

    assert s.position                        is None
    assert len(exchange.trade_log)           == 1
    assert exchange.trade_log[0]["close_reason"] == "finalize"


# idempotent: calling finalize() twice does not double-close or raise
def test_finalize_is_idempotent():
    exchange = make_exchange(n_bars=3)
    while exchange.advance():
        pass

    exchange.finalize()
    exchange.finalize()  # should not raise or cause any side effects

    assert exchange._state.finalized is True
