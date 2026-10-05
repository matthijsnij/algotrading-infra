"""
================================================================================
UNIT TESTS FOR domain/exchange_types.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_exchange_types.py -v

To run a specific test function:
    pytest tests/test_exchange_types.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import dataclasses
import pytest

from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import Fee, NormalizedOrder, NormalizedPosition

################### TESTS ##########################

# ── Fee ───────────────────────────────────────────────────────────────────────

# construction stores cost and currency
def test_fee_construction():
    fee = Fee(cost=0.25, currency="USDT")
    assert fee.cost == 0.25
    assert fee.currency == "USDT"

# frozen: assigning a field raises
def test_fee_is_frozen():
    fee = Fee(cost=0.25, currency="USDT")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fee.cost = 1.0

# ── NormalizedPosition ────────────────────────────────────────────────────────

# construction stores symbol, side, size
def test_position_construction():
    pos = NormalizedPosition(symbol="BTC/USDT:USDT", side=Side.LONG, size=0.5)
    assert pos.symbol == "BTC/USDT:USDT"
    assert pos.side is Side.LONG
    assert pos.size == 0.5

# frozen: assigning a field raises
def test_position_is_frozen():
    pos = NormalizedPosition(symbol="BTC/USDT:USDT", side=Side.LONG, size=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pos.size = 1.0

# value equality: two positions with identical fields compare equal
def test_position_equality():
    a = NormalizedPosition(symbol="BTC/USDT:USDT", side=Side.SHORT, size=0.5)
    b = NormalizedPosition(symbol="BTC/USDT:USDT", side=Side.SHORT, size=0.5)
    assert a == b

# ── NormalizedOrder ───────────────────────────────────────────────────────────

# minimal construction: optional fields default to None
def test_order_defaults():
    order = NormalizedOrder(
        id="o-1", status="open", symbol="BTC/USDT:USDT",
        side="buy", type="market", size=0.1,
    )
    assert order.price is None
    assert order.stop_price is None
    assert order.filled is None
    assert order.fill_price is None
    assert order.timestamp is None
    assert order.fee is None

# full construction: all fields stored, including nested Fee
def test_order_full_construction():
    fee = Fee(cost=0.05, currency="USDT")
    order = NormalizedOrder(
        id="o-2", status="closed", symbol="BTC/USDT:USDT",
        side="sell", type="limit", size=0.1,
        price=100.0, stop_price=95.0, filled=0.1,
        fill_price=100.5, timestamp=1_700_000_000_000, fee=fee,
    )
    assert order.price == 100.0
    assert order.stop_price == 95.0
    assert order.filled == 0.1
    assert order.fill_price == 100.5
    assert order.timestamp == 1_700_000_000_000
    assert order.fee == fee

# frozen: assigning a field raises
def test_order_is_frozen():
    order = NormalizedOrder(
        id="o-1", status="open", symbol="BTC/USDT:USDT",
        side="buy", type="market", size=0.1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.status = "closed"
