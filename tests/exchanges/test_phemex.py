"""
================================================================================
UNIT TESTS FOR EXCHANGES/LIVE/PHEMEX.PY

To run all tests in this file:
    pytest tests/exchanges/test_phemex.py -v
================================================================================
"""

################### IMPORTS ##########################

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition
from lighthouse.exchanges.live.phemex import PhemexExchange

################### HELPERS ##########################

def _make_exchange(client: MagicMock, market_type: str = "swap") -> PhemexExchange:
    """Build a PhemexExchange with the ccxt client replaced by the given mock."""
    with patch("lighthouse.exchanges.live.phemex.ccxt.phemex", return_value=client):
        return PhemexExchange(credentials={"api_key": "k", "api_secret": "s"}, market_type=market_type)

################### TESTS ##########################

# ── fetch_open_positions ─────────────────────────────────────────────

# ccxt positions become NormalizedPosition; ccxt "buy" side normalized to Side.LONG,
# zero-contract and contract-less entries filtered out
def test_fetch_open_positions_returns_normalized_positions():
    client = MagicMock()
    client.fetchPositions.return_value = [
        {"symbol": "BTC/USDT:USDT", "side": "buy", "contracts": 0.5},
        {"symbol": "BTC/USDT:USDT", "side": "sell", "contracts": 0},      # zero size, excluded
        {"symbol": "BTC/USDT:USDT", "side": "sell", "contracts": None},  # no size, excluded
    ]
    exchange = _make_exchange(client)

    result = exchange.fetch_open_positions("BTC/USDT:USDT")

    assert result == [
        NormalizedPosition(symbol="BTC/USDT:USDT", side=Side.LONG, size=0.5)
    ]

# fetchFundingHistory unsupported -> None (no funding concept for this client)
def test_fetch_funding_payments_unsupported_returns_none():
    client = MagicMock()
    client.has = {"fetchFundingHistory": False}
    exchange = _make_exchange(client)

    result = exchange.fetch_funding_payments(
        "BTC/USDT:USDT",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert result is None
    client.fetchFundingHistory.assert_not_called()

# sums amounts within window as-is (sign convention unverified, see fetch_funding_payments comment)
def test_fetch_funding_payments_sums_amounts_in_window():
    client = MagicMock()
    client.has = {"fetchFundingHistory": True}
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end_ms = int(end.timestamp() * 1000)
    client.fetchFundingHistory.return_value = [
        {"amount": -1.5, "timestamp": end_ms - 1000},
        {"amount": 0.5, "timestamp": end_ms - 500},
        {"amount": 100.0, "timestamp": end_ms + 1000},  # after window, excluded
    ]
    exchange = _make_exchange(client)

    result = exchange.fetch_funding_payments("BTC/USDT:USDT", start, end)

    assert result == -1.0
    client.fetchFundingHistory.assert_called_once_with(
        "BTC/USDT:USDT", since=int(start.timestamp() * 1000)
    )

# any exception from ccxt is swallowed -> None, never breaks the close flow
def test_fetch_funding_payments_returns_none_on_error():
    client = MagicMock()
    client.has = {"fetchFundingHistory": True}
    client.fetchFundingHistory.side_effect = Exception("network error")
    exchange = _make_exchange(client)

    result = exchange.fetch_funding_payments(
        "BTC/USDT:USDT",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert result is None

# ── get_equity ─────────────────────────────────────────────

# market_type is stored on the instance for get_equity to route on
def test_market_type_stored_on_init():
    client = MagicMock()
    exchange = _make_exchange(client, market_type="spot")

    assert exchange.market_type == "spot"

# spot: quote cash balance plus mid-price mark-to-market of held base assets
def test_get_equity_spot_marks_to_market_holdings():
    client = MagicMock()
    client.fetchBalance.return_value = {
        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
        "BTC":  {"free": 0.5, "used": 0.0, "total": 0.5},
        "total": {"USDT": 1000.0, "BTC": 0.5},
    }
    client.fetchOrderBook.return_value = {"bids": [[100.0, 1]], "asks": [[102.0, 1]]}
    exchange = _make_exchange(client, market_type="spot")

    result = exchange.get_equity(["BTC/USDT"], "USDT")

    assert result == 1000.0 + 0.5 * 101.0

# spot: an asset that fails to price is excluded rather than raising
def test_get_equity_spot_pricing_failure_excludes_asset():
    client = MagicMock()
    client.fetchBalance.return_value = {
        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
        "BTC":  {"free": 0.5, "used": 0.0, "total": 0.5},
        "total": {"USDT": 1000.0, "BTC": 0.5},
    }
    client.fetchOrderBook.return_value = {"bids": [], "asks": []}
    exchange = _make_exchange(client, market_type="spot")

    result = exchange.get_equity(["BTC/USDT"], "USDT")

    assert result == 1000.0

# swap: quote wallet balance plus unrealized PnL summed across open positions
def test_get_equity_swap_adds_unrealized_pnl():
    client = MagicMock()
    client.fetchBalance.return_value = {
        "USDT": {"free": 500.0, "used": 100.0, "total": 600.0},
        "total": {"USDT": 600.0},
    }
    client.fetchPositions.return_value = [
        {"symbol": "BTC/USDT:USDT", "unrealizedPnl": 25.0},
        {"symbol": "ETH/USDT:USDT", "unrealizedPnl": -5.0},
        {"symbol": "SOL/USDT:USDT", "unrealizedPnl": None},
    ]
    exchange = _make_exchange(client, market_type="swap")

    result = exchange.get_equity(["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"], "USDT")

    assert result == 620.0
    client.fetchPositions.assert_called_once_with(
        ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    )
