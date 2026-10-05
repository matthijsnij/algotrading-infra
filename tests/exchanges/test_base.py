"""
================================================================================
UNIT TESTS FOR EXCHANGES/BASE.PY

To run all tests in this file:
    pytest tests/exchanges/test_base.py -v
================================================================================
"""

################### IMPORTS ##########################

import pytest
from datetime import datetime, timezone

from lighthouse.exchanges.base import BaseExchange

################### HELPERS ##########################

def _make_dummy(supports_testnet: bool) -> type:
    """Build a minimal concrete BaseExchange subclass with all abstract methods stubbed."""
    methods = [
        "fetch_ohlcv", "fetch_bid_ask", "fetch_balance", "fetch_open_positions", "get_equity",
        "place_buy_market_order", "place_sell_market_order",
        "place_buy_limit_order", "place_sell_limit_order",
        "place_buy_stop_order", "place_sell_stop_order",
        "place_buy_stop_limit_order", "place_sell_stop_limit_order",
        "cancel_order", "cancel_all_orders", "fetch_open_orders", "fetch_order",
    ]
    namespace = {name: (lambda self, *a, **kw: None) for name in methods}
    namespace["supports_testnet"] = supports_testnet
    return type("DummyExchange", (BaseExchange,), namespace)


################### TESTS ##########################

# testnet=False always succeeds regardless of supports_testnet
def test_init_testnet_false_always_succeeds():
    for supports_testnet in (True, False):
        exchange = _make_dummy(supports_testnet)()
        assert exchange.testnet is False

# testnet=True succeeds when the subclass declares support
def test_init_testnet_true_succeeds_when_supported():
    exchange = _make_dummy(True)(testnet=True)
    assert exchange.testnet is True

# testnet=True raises NotImplementedError when the subclass does not declare support
def test_init_testnet_true_raises_when_unsupported():
    with pytest.raises(NotImplementedError, match="DummyExchange"):
        _make_dummy(False)(testnet=True)

# default supports_testnet on BaseExchange itself is False
def test_supports_testnet_default_is_false():
    assert BaseExchange.supports_testnet is False

# default fetch_funding_payments() returns None (no override needed for non-funding exchanges)
def test_fetch_funding_payments_default_returns_none():
    exchange = _make_dummy(False)()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert exchange.fetch_funding_payments("BTC/USDT", start, end) is None

# get_equity is abstract -> a subclass missing it cannot be instantiated
def test_get_equity_is_abstract():
    namespace = {
        name: (lambda self, *a, **kw: None)
        for name in [
            "fetch_ohlcv", "fetch_bid_ask", "fetch_balance", "fetch_open_positions",
            "place_buy_market_order", "place_sell_market_order",
            "place_buy_limit_order", "place_sell_limit_order",
            "place_buy_stop_order", "place_sell_stop_order",
            "place_buy_stop_limit_order", "place_sell_stop_limit_order",
            "cancel_order", "cancel_all_orders", "fetch_open_orders", "fetch_order",
        ]
    }
    incomplete = type("IncompleteExchange", (BaseExchange,), namespace)

    with pytest.raises(TypeError, match="get_equity"):
        incomplete()
