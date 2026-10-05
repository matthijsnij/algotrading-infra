"""
================================================================================
UNIT TESTS FOR ORDERS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_orders.py -v

To run a specific test function:
    pytest tests/test_orders.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

from datetime import datetime, timezone
from lighthouse.execution.orders import parse_order_result
from lighthouse.domain.exchange_types import Fee

################### TESTS ##########################

# ── parse_order_result ────────────────────────────────────────────────────────

# order_data is None, all fields return None
def test_parse_order_result_none_input():
    fill_price, fee, fill_time = parse_order_result(None)

    assert fill_price is None
    assert fee        is None
    assert fill_time  is None


# all fields present, returns correct values with timestamp converted from ms to UTC datetime
def test_parse_order_result_all_fields_present(make_order):
    order_data = make_order(
        fill_price=50000.0,
        fee=Fee(cost=1.5, currency="USDT"),
        timestamp=1700000000000,  # 1700000000.0 seconds UTC
    )

    fill_price, fee, fill_time = parse_order_result(order_data)

    assert fill_price == 50000.0
    assert fee        == Fee(cost=1.5, currency="USDT")
    assert fill_time  == datetime.fromtimestamp(1700000000.0, tz=timezone.utc)
    assert fill_time.tzinfo == timezone.utc


# timestamp missing, fill_time is None, other fields still parsed
def test_parse_order_result_missing_timestamp(make_order):
    order_data = make_order(
        fill_price=50000.0,
        fee=Fee(cost=1.5, currency="USDT"),
    )

    fill_price, fee, fill_time = parse_order_result(order_data)

    assert fill_price == 50000.0
    assert fee        == Fee(cost=1.5, currency="USDT")
    assert fill_time  is None


# fill_price missing, fill_price is None, other fields still parsed
def test_parse_order_result_missing_fill_price(make_order):
    order_data = make_order(
        fee=Fee(cost=1.5, currency="USDT"),
        timestamp=1700000000000,
    )

    fill_price, fee, fill_time = parse_order_result(order_data)

    assert fill_price is None
    assert fee        == Fee(cost=1.5, currency="USDT")
    assert fill_time  == datetime.fromtimestamp(1700000000.0, tz=timezone.utc)
