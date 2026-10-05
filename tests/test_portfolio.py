"""
================================================================================
UNIT TESTS FOR PORTFOLIO.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_portfolio.py -v

To run a specific test function:
    pytest tests/test_portfolio.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

from unittest.mock import MagicMock, patch

from lighthouse.runtime.portfolio import Portfolio

################### TESTS ##########################

# ── register_exchange() / exchange_equity() ──────────────────────────────────

# exchange_equity() calls exchange.get_equity() with the registered symbols/quote_currency
def test_exchange_equity_calls_get_equity():
    portfolio = Portfolio(quote_currency="USDT")
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    portfolio.register_exchange("phemex", exchange, ["BTC/USDT:USDT"])

    result = portfolio.exchange_equity("phemex")

    exchange.get_equity.assert_called_once_with(["BTC/USDT:USDT"], "USDT")
    assert result == 1000.0

# register_exchange() defaults quote_currency to the portfolio's own quote_currency
def test_register_exchange_defaults_quote_currency():
    portfolio = Portfolio(quote_currency="USDT")
    exchange = MagicMock()
    exchange.get_equity.return_value = 500.0
    portfolio.register_exchange("phemex", exchange, ["BTC/USDT:USDT"])

    portfolio.exchange_equity("phemex")

    assert exchange.get_equity.call_args.args[1] == "USDT"

# exchange_equity() raises for a name that was never registered
def test_exchange_equity_unknown_name_raises():
    portfolio = Portfolio(quote_currency="USDT")
    try:
        portfolio.exchange_equity("nonexistent")
        assert False, "expected KeyError"
    except KeyError:
        pass

# ── total_equity() ───────────────────────────────────────────────────────────

# total_equity() sums exchange_equity() across all registered exchanges
def test_total_equity_sums_across_exchanges():
    portfolio = Portfolio(quote_currency="USDT")
    exchange_a = MagicMock()
    exchange_a.get_equity.return_value = 1000.0
    exchange_b = MagicMock()
    exchange_b.get_equity.return_value = 2000.0
    portfolio.register_exchange("phemex", exchange_a, ["BTC/USDT:USDT"])
    portfolio.register_exchange("otherexchange", exchange_b, ["ETH/USDT:USDT"])

    result = portfolio.total_equity()

    assert result == 3000.0

# ── TTL cache ─────────────────────────────────────────────────────────────────

# a second call within ttl_seconds returns the cached value without refetching
def test_ttl_cache_avoids_refetch_within_ttl():
    portfolio = Portfolio(quote_currency="USDT", ttl_seconds=60.0)
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    portfolio.register_exchange("phemex", exchange, ["BTC/USDT:USDT"])

    portfolio.exchange_equity("phemex")
    portfolio.exchange_equity("phemex")

    exchange.get_equity.assert_called_once()

# once ttl_seconds has elapsed, the next call refetches
def test_ttl_cache_refetches_after_ttl_expires():
    portfolio = Portfolio(quote_currency="USDT", ttl_seconds=5.0)
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    portfolio.register_exchange("phemex", exchange, ["BTC/USDT:USDT"])

    with patch("lighthouse.runtime.portfolio.time.monotonic", side_effect=[0.0, 10.0]):
        portfolio.exchange_equity("phemex")
        portfolio.exchange_equity("phemex")

    assert exchange.get_equity.call_count == 2
