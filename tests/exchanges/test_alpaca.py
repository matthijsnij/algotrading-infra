"""
================================================================================
UNIT TESTS FOR EXCHANGES/LIVE/ALPACA.PY

To run all tests in this file:
    pytest tests/exchanges/test_alpaca.py -v
================================================================================
"""

################### IMPORTS ##########################

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderStatus, OrderType, PositionSide

from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition
from lighthouse.domain.timeframe import Timeframe
from lighthouse.exchanges.live.alpaca import AlpacaExchange

################### HELPERS ##########################

def _make_exchange(trading_client: MagicMock, data_client: MagicMock | None = None, testnet: bool = False) -> AlpacaExchange:
    """Build an AlpacaExchange with the SDK clients replaced by the given mocks."""
    data_client = data_client if data_client is not None else MagicMock()
    with patch("lighthouse.exchanges.live.alpaca.TradingClient", return_value=trading_client) as trading_ctor, \
         patch("lighthouse.exchanges.live.alpaca.StockHistoricalDataClient", return_value=data_client):
        exchange = AlpacaExchange(credentials={"api_key": "k", "api_secret": "s"}, market_type="stock", testnet=testnet)
    exchange._trading_ctor_mock = trading_ctor  # stash for assertions on init call args
    return exchange


def _make_raw_order(**overrides) -> MagicMock:
    """Build a MagicMock resembling an alpaca-py Order model with sensible defaults."""
    order = MagicMock()
    order.id = overrides.get("id", "order-1")
    order.status = overrides.get("status", OrderStatus.NEW)
    order.symbol = overrides.get("symbol", "AAPL")
    order.side = overrides.get("side", AlpacaOrderSide.BUY)
    order.type = overrides.get("type", OrderType.MARKET)
    order.qty = overrides.get("qty", "1")
    order.limit_price = overrides.get("limit_price", None)
    order.stop_price = overrides.get("stop_price", None)
    order.filled_qty = overrides.get("filled_qty", None)
    order.filled_avg_price = overrides.get("filled_avg_price", None)
    order.filled_at = overrides.get("filled_at", None)
    order.submitted_at = overrides.get("submitted_at", datetime(2024, 1, 1, tzinfo=timezone.utc))
    return order

################### TESTS ##########################

# ── __init__ ─────────────────────────────────────────────────────

# rejects any market_type other than "stock"
def test_init_rejects_unsupported_market_type():
    with pytest.raises(ValueError):
        AlpacaExchange(credentials={"api_key": "k", "api_secret": "s"}, market_type="swap")

# testnet flag routes to Alpaca's paper trading endpoint via TradingClient(paper=...)
def test_init_routes_testnet_to_paper_trading():
    exchange = _make_exchange(MagicMock(), testnet=True)

    _, kwargs = exchange._trading_ctor_mock.call_args
    assert kwargs["paper"] is True

def test_init_routes_live_to_live_trading():
    exchange = _make_exchange(MagicMock(), testnet=False)

    _, kwargs = exchange._trading_ctor_mock.call_args
    assert kwargs["paper"] is False

# ── fetch_ohlcv ─────────────────────────────────────────────────────

# native tickers pass through untouched; bars requested desc-first are returned chronological
def test_fetch_ohlcv_returns_native_ticker_bars_in_chronological_order():
    bar_old = MagicMock(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc), open=1.0, high=2.0, low=0.5, close=1.5, volume=100.0)
    bar_new = MagicMock(timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc), open=1.5, high=2.5, low=1.0, close=2.0, volume=200.0)
    data_client = MagicMock()
    data_client.get_stock_bars.return_value = MagicMock(data={"AAPL": [bar_new, bar_old]})  # desc order from request
    exchange = _make_exchange(MagicMock(), data_client=data_client)

    result = exchange.fetch_ohlcv("AAPL", Timeframe.parse("1d"), 2)

    assert result == [
        [int(bar_old.timestamp.timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 100.0],
        [int(bar_new.timestamp.timestamp() * 1000), 1.5, 2.5, 1.0, 2.0, 200.0],
    ]

# ── fetch_bid_ask ─────────────────────────────────────────────────────

def test_fetch_bid_ask_returns_native_ticker_quote():
    quote = MagicMock(bid_price=100.0, ask_price=101.0)
    data_client = MagicMock()
    data_client.get_stock_latest_quote.return_value = {"AAPL": quote}
    exchange = _make_exchange(MagicMock(), data_client=data_client)

    result = exchange.fetch_bid_ask("AAPL")

    assert result == {"symbol": "AAPL", "bid": 100.0, "ask": 101.0}

# ── fetch_balance / get_equity ─────────────────────────────────────────────────────

# cash + equity from the account are exposed as free/used/total under "USD"
def test_fetch_balance_maps_account_cash_and_equity():
    client = MagicMock()
    client.get_account.return_value = MagicMock(cash="1000.0", equity="1500.0")
    exchange = _make_exchange(client)

    result = exchange.fetch_balance()

    assert result == {"USD": {"free": 1000.0, "used": 500.0, "total": 1500.0}}

# equity is Alpaca's account equity field directly -- no mark-to-market calculation needed
def test_get_equity_returns_account_equity_directly():
    client = MagicMock()
    client.get_account.return_value = MagicMock(equity="2500.0")
    exchange = _make_exchange(client)

    result = exchange.get_equity(["AAPL"], "USD")

    assert result == 2500.0

# ── fetch_open_positions ─────────────────────────────────────────────────────

def test_fetch_open_positions_returns_normalized_position():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(symbol="AAPL", side=PositionSide.LONG, qty="10")
    exchange = _make_exchange(client)

    result = exchange.fetch_open_positions("AAPL")

    assert result == [NormalizedPosition(symbol="AAPL", side=Side.LONG, size=10.0)]

# no position for the symbol -> Alpaca raises APIError, translated to an empty list
def test_fetch_open_positions_no_position_returns_empty_list():
    client = MagicMock()
    client.get_open_position.side_effect = APIError("position does not exist")
    exchange = _make_exchange(client)

    result = exchange.fetch_open_positions("AAPL")

    assert result == []

# ── orders: whole-share rounding ─────────────────────────────────────────────────────

# market orders keep fractional size as requested
def test_place_buy_market_order_keeps_fractional_size():
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order(qty="0.5")
    exchange = _make_exchange(client)

    exchange.place_buy_market_order("AAPL", 0.5)

    request = client.submit_order.call_args[0][0]
    assert request.qty == 0.5

# limit orders round fractional size to the nearest whole share
def test_place_buy_limit_order_rounds_to_whole_shares():
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order(type=OrderType.LIMIT, qty="2", limit_price="10.0")
    exchange = _make_exchange(client)

    exchange.place_buy_limit_order("AAPL", 2.4, 10.0)

    request = client.submit_order.call_args[0][0]
    assert request.qty == 2.0

# stop orders round fractional size to the nearest whole share, minimum 1
def test_place_sell_stop_order_rounds_up_to_minimum_one_share():
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order(type=OrderType.STOP, qty="1", side=AlpacaOrderSide.SELL, stop_price="9.0")
    exchange = _make_exchange(client)

    exchange.place_sell_stop_order("AAPL", 0.2, 9.0)

    request = client.submit_order.call_args[0][0]
    assert request.qty == 1.0

# ── orders: response mapping ─────────────────────────────────────────────────────

def test_place_buy_limit_order_returns_normalized_order_with_price():
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order(type=OrderType.LIMIT, qty="1", limit_price="10.0")
    exchange = _make_exchange(client)

    result = exchange.place_buy_limit_order("AAPL", 1.0, 10.0)

    assert result.price == 10.0
    assert result.size == 1.0

def test_place_buy_stop_limit_order_returns_normalized_order_with_both_prices():
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order(
        type=OrderType.STOP_LIMIT, qty="1", limit_price="10.0", stop_price="9.5"
    )
    exchange = _make_exchange(client)

    result = exchange.place_buy_stop_limit_order("AAPL", 1.0, 9.5, 10.0)

    assert result.price == 10.0
    assert result.stop_price == 9.5

# ── order management ─────────────────────────────────────────────────────

# open orders omit fill/timestamp/fee fields (they haven't filled yet)
def test_fetch_open_orders_omits_fill_fields():
    client = MagicMock()
    client.get_orders.return_value = [_make_raw_order(type=OrderType.LIMIT, qty="1", limit_price="10.0")]
    exchange = _make_exchange(client)

    result = exchange.fetch_open_orders("AAPL")

    assert result[0].price == 10.0
    assert result[0].filled is None
    assert result[0].fill_price is None
    assert result[0].timestamp is None

# fetch_order populates filled/fill_price/timestamp from the raw order
def test_fetch_order_populates_fill_fields():
    filled_at = datetime(2024, 1, 3, tzinfo=timezone.utc)
    client = MagicMock()
    client.get_order_by_id.return_value = _make_raw_order(
        status=OrderStatus.FILLED, qty="1", filled_qty="1", filled_avg_price="12.5", filled_at=filled_at,
    )
    exchange = _make_exchange(client)

    result = exchange.fetch_order("order-1", "AAPL")

    assert result.filled == 1.0
    assert result.fill_price == 12.5
    assert result.timestamp == int(filled_at.timestamp() * 1000)

# Alpaca has no per-symbol bulk cancel endpoint: cancel_all_orders cancels each open order individually
def test_cancel_all_orders_cancels_each_open_order_individually():
    client = MagicMock()
    client.get_orders.return_value = [
        _make_raw_order(id="order-1", qty="1"),
        _make_raw_order(id="order-2", qty="1"),
    ]
    exchange = _make_exchange(client)

    exchange.cancel_all_orders("AAPL")

    assert client.cancel_order_by_id.call_count == 2
    client.cancel_order_by_id.assert_any_call("order-1")
    client.cancel_order_by_id.assert_any_call("order-2")
