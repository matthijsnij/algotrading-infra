"""
================================================================================
UNIT TESTS FOR EXCHANGES/FACTORY.PY

To run all tests in this file:
    pytest tests/exchanges/test_factory.py -v
================================================================================
"""

################### IMPORTS ##########################

from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from lighthouse.exchanges.base import BaseExchange
from lighthouse.exchanges.factory import create_exchange, _EXCHANGE_REGISTRY

################### TESTS ##########################

# create_exchange("phemex", testnet=True) succeeds without hitting the real network
def test_create_exchange_phemex_testnet_succeeds():
    with patch("lighthouse.exchanges.live.phemex.ccxt.phemex", return_value=MagicMock()):
        exchange = create_exchange(
            "phemex", credentials={"api_key": "k", "api_secret": "s"}, market_type="swap", testnet=True,
        )
    assert exchange.testnet is True

# unknown exchange name still raises before any capability check
def test_create_exchange_unknown_name_raises():
    with pytest.raises(ValueError, match="not supported"):
        create_exchange("nonexistent", credentials={}, market_type="swap", testnet=False)

# a registered exchange without testnet support raises ValueError naming the exchange,
# before any client/credential work happens (registry monkeypatched, no ccxt call made)
def test_create_exchange_testnet_unsupported_raises(monkeypatch):
    class DummyExchange(BaseExchange):
        supports_testnet: ClassVar[bool] = False

        def __init__(self, credentials, market_type="swap", testnet=False):
            super().__init__(testnet=testnet)

        def fetch_ohlcv(self, symbol, timeframe, limit): pass
        def fetch_bid_ask(self, symbol): pass
        def fetch_balance(self): pass
        def fetch_open_positions(self, symbol): pass
        def place_buy_market_order(self, *a, **kw): pass
        def place_sell_market_order(self, *a, **kw): pass
        def place_buy_limit_order(self, *a, **kw): pass
        def place_sell_limit_order(self, *a, **kw): pass
        def place_buy_stop_order(self, *a, **kw): pass
        def place_sell_stop_order(self, *a, **kw): pass
        def place_buy_stop_limit_order(self, *a, **kw): pass
        def place_sell_stop_limit_order(self, *a, **kw): pass
        def cancel_order(self, *a, **kw): pass
        def cancel_all_orders(self, *a, **kw): pass
        def fetch_open_orders(self, *a, **kw): pass
        def fetch_order(self, *a, **kw): pass

    monkeypatch.setitem(_EXCHANGE_REGISTRY, "dummy", DummyExchange)

    with pytest.raises(ValueError, match="dummy"):
        create_exchange("dummy", credentials={}, market_type="swap", testnet=True)
