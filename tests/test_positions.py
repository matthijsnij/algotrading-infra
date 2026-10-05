"""
================================================================================
UNIT TESTS FOR POSITION.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_position.py -v

To run a specific test function:
    pytest tests/test_position.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from unittest.mock import MagicMock, patch
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition
from lighthouse.execution.positions import is_position_closed, OrderCancelledError

################### TESTS ##########################

# ── is_position_closed ────────────────────────────────────────────────────────

# order still open; returns False without checking position
def test_is_position_closed_order_still_open():
    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = [{"id": "order-1"}]
    exchange.fetch_open_positions.return_value = []
    result = is_position_closed(exchange, order_id="order-1", symbol="BTC/USD", side=Side.LONG)
    assert result is False

# order gone and position gone; returns True
def test_is_position_closed_returns_true():
    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = []
    exchange.fetch_open_positions.return_value = []
    result = is_position_closed(exchange, order_id="order-1", symbol="BTC/USD", side=Side.LONG)
    assert result is True

# order gone but position still open; raises OrderCancelledError
def test_is_position_closed_raises_on_cancelled_order():
    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = []
    exchange.fetch_open_positions.return_value = [
        NormalizedPosition(symbol="BTC/USD", side=Side.LONG, size=0.01)
    ]
    with pytest.raises(OrderCancelledError):
        is_position_closed(exchange, order_id="order-1", symbol="BTC/USD", side=Side.LONG)

# unexpected exchange exception; returns False and does not propagate
def test_is_position_closed_exchange_error_returns_false():
    exchange = MagicMock()
    exchange.fetch_open_orders.side_effect = RuntimeError("network error")
    result = is_position_closed(exchange, order_id="order-1", symbol="BTC/USD", side=Side.LONG)
    assert result is False
