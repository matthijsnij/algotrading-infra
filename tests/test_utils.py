"""
================================================================================
UNIT TESTS FOR UTILS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_utils.py -v

To run a specific test function:
    pytest tests/test_utils.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from datetime import datetime, timezone
from lighthouse.utils.time import to_utc_datetime, to_epoch_ms

################### TESTS ##########################

# ── to_utc_datetime() ────────────────────────────────────────────────────────

# ISO string without timezone info → UTC is attached
def test_to_utc_datetime_naive_string():
    result = to_utc_datetime("2024-01-01")
    assert result.tzinfo == timezone.utc
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 1


# ISO string with UTC offset → timezone preserved, not overwritten
def test_to_utc_datetime_string_with_timezone():
    result = to_utc_datetime("2024-06-15T08:00:00+00:00")
    assert result.tzinfo is not None
    assert result.hour == 8
    assert result.day == 15


# Naive datetime object → UTC is attached
def test_to_utc_datetime_naive_datetime():
    dt = datetime(2024, 3, 1, 12, 0, 0)
    result = to_utc_datetime(dt)
    assert result.tzinfo == timezone.utc
    assert result.hour == 12


# UTC-aware datetime object → passed through unchanged
def test_to_utc_datetime_aware_datetime_passthrough():
    dt = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = to_utc_datetime(dt)
    assert result == dt
    assert result.tzinfo == timezone.utc

# ── to_epoch_ms() ───────────────────────────────────────────────────────────

# Known date → exact millisecond value (2024-01-01 00:00:00 UTC = 1704067200000 ms)
def test_to_epoch_ms_known_date():
    result = to_epoch_ms("2024-01-01")
    assert result == 1704067200000


# Returns an integer
def test_to_epoch_ms_returns_int():
    result = to_epoch_ms("2024-01-01")
    assert isinstance(result, int)


# Round-trip: convert to ms, back to datetime, same result
def test_to_epoch_ms_round_trip():
    original = "2024-06-15T08:00:00"
    ms = to_epoch_ms(original)
    recovered = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    expected = to_utc_datetime(original)
    assert recovered == expected


# Later dates produce larger ms values
def test_to_epoch_ms_ordering():
    assert to_epoch_ms("2024-01-01") < to_epoch_ms("2024-12-31")
