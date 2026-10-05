"""
================================================================================
UNIT TESTS FOR DATA.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_data.py -v

To run a specific test function:
    pytest tests/test_data.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from unittest.mock import MagicMock
from lighthouse.execution.data import get_ohlcv_df, get_bid_ask, get_spot_equity

################### HELPERS ##########################

def _make_raw_ohlcv(n: int = 3) -> list:
    """
    Return a minimal ccxt-style raw OHLCV list with n rows.
    Timestamps are in milliseconds (Unix epoch).
    """
    return [
        [1_700_000_000_000 + i * 60_000, 100.0, 101.0, 99.0, 100.5, 1000.0]
        for i in range(n)
    ]

def _make_exchange(raw: list) -> MagicMock:
    """Return a mock exchange whose fetch_ohlcv returns raw."""
    exchange = MagicMock()
    exchange.fetch_ohlcv.return_value = raw
    return exchange

################### TESTS ##########################

# ── get_ohlcv_df ────────────────────────────────────────────────────────

# normal case; correct columns returned
def test_get_ohlcv_df_columns():
    exchange = _make_exchange(_make_raw_ohlcv())
    df = get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]

# per ADR 0001, 'limit' means closed bars: the exchange is asked for limit+1
# rows so the caller gets limit closed bars plus the current forming bar
def test_get_ohlcv_df_requests_limit_plus_one():
    exchange = _make_exchange(_make_raw_ohlcv(n=4))
    get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)
    _, kwargs = exchange.fetch_ohlcv.call_args
    assert kwargs["limit"] == 4

# normal case; index is a DatetimeTZDtype in UTC
def test_get_ohlcv_df_index_is_utc_datetime():
    exchange = _make_exchange(_make_raw_ohlcv())
    df = get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"

# normal case; index name is set correctly
def test_get_ohlcv_df_index_name():
    exchange = _make_exchange(_make_raw_ohlcv())
    df = get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)
    assert df.index.name == "timestamp (UTC)"

# exchange raises an exception; must raise ValueError
def test_get_ohlcv_df_exchange_error():
    exchange = MagicMock()
    exchange.fetch_ohlcv.side_effect = RuntimeError("network error")
    with pytest.raises(ValueError):
        get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)

# fetch_ohlcv returns a non-list; must raise ValueError
def test_get_ohlcv_df_non_list_response():
    exchange = MagicMock()
    exchange.fetch_ohlcv.return_value = "not a list"
    with pytest.raises(ValueError):
        get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=3)

# fetch_ohlcv returns rows with unparseable timestamps; must raise ValueError
def test_get_ohlcv_df_bad_timestamps():
    raw = [["not-a-timestamp", 100.0, 101.0, 99.0, 100.5, 1000.0]]
    exchange = _make_exchange(raw)
    with pytest.raises(ValueError):
        get_ohlcv_df(exchange, symbol="BTC/USD", timeframe="1m", limit=1)


# ── get_bid_ask ────────────────────────────────────────────────────────

# normal case; all keys present in result
def test_get_bid_ask_keys():
    exchange = MagicMock()
    exchange.fetch_bid_ask.return_value = {"bid": 100.0, "ask": 101.0}
    result = get_bid_ask(exchange, symbol="BTC/USD")
    assert set(result.keys()) == {"symbol", "bid", "ask", "spread"}

# normal case; spread is ask minus bid
def test_get_bid_ask_spread():
    exchange = MagicMock()
    exchange.fetch_bid_ask.return_value = {"bid": 100.0, "ask": 101.5}
    result = get_bid_ask(exchange, symbol="BTC/USD")
    assert result["spread"] == pytest.approx(1.5)

# normal case; bid and ask are cast to float
def test_get_bid_ask_values_are_float():
    exchange = MagicMock()
    exchange.fetch_bid_ask.return_value = {"bid": "100.0", "ask": "101.0"}
    result = get_bid_ask(exchange, symbol="BTC/USD")
    assert isinstance(result["bid"], float)
    assert isinstance(result["ask"], float)

# bid is None; must raise ValueError
def test_get_bid_ask_none_bid():
    exchange = MagicMock()
    exchange.fetch_bid_ask.return_value = {"bid": None, "ask": 101.0}
    with pytest.raises(ValueError):
        get_bid_ask(exchange, symbol="BTC/USD")

# ask is None; must raise ValueError
def test_get_bid_ask_none_ask():
    exchange = MagicMock()
    exchange.fetch_bid_ask.return_value = {"bid": 100.0, "ask": None}
    with pytest.raises(ValueError):
        get_bid_ask(exchange, symbol="BTC/USD")

# exchange raises an exception; must raise ValueError
def test_get_bid_ask_exchange_error():
    exchange = MagicMock()
    exchange.fetch_bid_ask.side_effect = RuntimeError("network error")
    with pytest.raises(ValueError):
        get_bid_ask(exchange, symbol="BTC/USD")


# ── get_spot_equity ────────────────────────────────────────────────────────

def _make_equity_exchange(usdt: float = 1000.0, holdings: dict = None, prices: dict = None) -> MagicMock:
    """
    Return a mock exchange for get_spot_equity tests.

    usdt:     total USDT cash balance (example quote currency)
    holdings: dict of {currency: total_held}, e.g. {"BTC": 0.5}
    prices:   dict of {symbol: (bid, ask)}, e.g. {"BTC/USDT": (49000.0, 51000.0)}
    """
    holdings = holdings or {}
    prices = prices or {}

    balance = {"USDT": {"free": usdt, "used": 0.0, "total": usdt}}
    for currency, total in holdings.items():
        balance[currency] = {"free": total, "used": 0.0, "total": total}

    exchange = MagicMock()
    exchange.fetch_balance.return_value = balance

    def _fetch_bid_ask(symbol):
        if symbol in prices:
            bid, ask = prices[symbol]
            return {"bid": bid, "ask": ask}
        raise ValueError(f"No price configured for {symbol}")

    exchange.fetch_bid_ask.side_effect = _fetch_bid_ask
    return exchange


# normal case; no base asset holdings —> returns USDT cash only
def test_get_spot_equity_cash_only():
    exchange = _make_equity_exchange(usdt=5000.0)
    result = get_spot_equity(exchange, symbols=["BTC/USDT"], quote_currency="USDT")
    assert result == 5000.0

# normal case; BTC held —> equity = USDT + BTC * mid_price
def test_get_spot_equity_with_holdings():
    exchange = _make_equity_exchange(
        usdt=1000.0,
        holdings={"BTC": 0.5},
        prices={"BTC/USDT": (49000.0, 51000.0)},   # mid = 50000.0
    )
    result = get_spot_equity(exchange, symbols=["BTC/USDT"], quote_currency="USDT")
    assert result == 26000.0  # 1000.0 + 0.5 * 50000.0

# normal case; multiple base assets held —> all valued and summed
def test_get_spot_equity_multiple_assets():
    exchange = _make_equity_exchange(
        usdt=500.0,
        holdings={"BTC": 0.1, "ETH": 2.0},
        prices={
            "BTC/USDT": (49000.0, 51000.0),   # mid = 50000.0
            "ETH/USDT": (2900.0, 3100.0),      # mid = 3000.0
        },
    )
    result = get_spot_equity(exchange, symbols=["BTC/USDT", "ETH/USDT"], quote_currency="USDT")
    assert result == 11500.0  # 500.0 + 0.1 * 50000.0 + 2.0 * 3000.0

# futures-style symbol "BTC/USDT:USDT" is parsed to base "BTC" correctly
def test_get_spot_equity_futures_style_symbol():
    exchange = _make_equity_exchange(
        usdt=0.0,
        holdings={"BTC": 1.0},
        prices={"BTC/USDT": (49000.0, 51000.0)},
    )
    result = get_spot_equity(exchange, symbols=["BTC/USDT:USDT"], quote_currency="USDT")
    assert result == 50000.0  # 0.0 + 1.0 * 50000.0

# zero holdings —> mid price fetch is skipped entirely, returns USDT cash only
def test_get_spot_equity_zero_holdings_skips_price_fetch():
    exchange = _make_equity_exchange(usdt=2000.0, holdings={"BTC": 0.0})
    get_spot_equity(exchange, symbols=["BTC/USDT"], quote_currency="USDT")
    exchange.fetch_bid_ask.assert_not_called()

# duplicate symbols with the same base —> mid price fetched only once
def test_get_spot_equity_duplicate_symbols_priced_once():
    exchange = _make_equity_exchange(
        usdt=0.0,
        holdings={"BTC": 1.0},
        prices={"BTC/USDT": (49000.0, 51000.0)},
    )
    get_spot_equity(exchange, symbols=["BTC/USDT", "BTC/USDT"], quote_currency="USDT")
    assert exchange.fetch_bid_ask.call_count == 1

# quote currency not in balance —> equity starts at 0.0 (graceful default)
def test_get_spot_equity_unknown_quote_currency_returns_zero():
    exchange = _make_equity_exchange(usdt=1000.0)   # only USDT in balance
    result = get_spot_equity(exchange, symbols=[], quote_currency="EUR")
    assert result == 0.0

# pricing fails for one asset —> that asset excluded, rest of equity returned
def test_get_spot_equity_pricing_failure_excludes_asset():
    exchange = _make_equity_exchange(
        usdt=1000.0,
        holdings={"BTC": 1.0, "ETH": 2.0},
        prices={"ETH/USDT": (2900.0, 3100.0)},   # BTC/USDT not configured → raises
    )
    result = get_spot_equity(exchange, symbols=["BTC/USDT", "ETH/USDT"], quote_currency="USDT")
    # BTC excluded due to pricing failure; ETH mid = 3000.0
    assert result == 7000.0  # 1000.0 + 2.0 * 3000.0
