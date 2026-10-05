"""
================================================================================
UNIT TESTS FOR TIMEFRAME.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_timeframe.py -v

To run a specific test function:
    pytest tests/test_timeframe.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest

from lighthouse.domain.timeframe import Timeframe, TimeframeUnit

################### TESTS ##########################

# ── Timeframe.parse: valid forms ─────────────────────────────────────────────

# standard minute/hour/day/week units all parse
def test_parse_valid_units():
    assert Timeframe.parse("5m") == Timeframe(5, TimeframeUnit.MINUTE)
    assert Timeframe.parse("4h") == Timeframe(4, TimeframeUnit.HOUR)
    assert Timeframe.parse("1d") == Timeframe(1, TimeframeUnit.DAY)
    assert Timeframe.parse("2w") == Timeframe(2, TimeframeUnit.WEEK)

# amounts outside the old hardcoded set (e.g. 6h, 12h) now parse -- open format, not a closed enum
def test_parse_valid_non_hardcoded_amounts():
    assert Timeframe.parse("6h") == Timeframe(6, TimeframeUnit.HOUR)
    assert Timeframe.parse("90m") == Timeframe(90, TimeframeUnit.MINUTE)

# ── Timeframe.parse: rejected forms ──────────────────────────────────────────

# unrecognized unit letters or malformed strings raise
def test_parse_invalid_format_raises():
    with pytest.raises(ValueError):
        Timeframe.parse("bogus")
    with pytest.raises(ValueError):
        Timeframe.parse("5x")
    with pytest.raises(ValueError):
        Timeframe.parse("m5")

# uppercase unit letters are rejected -- parsing is case-sensitive, lowercase only
def test_parse_uppercase_unit_raises():
    with pytest.raises(ValueError):
        Timeframe.parse("5H")

# calendar months ("1M" in ccxt notation) are rejected with a message naming the reason explicitly
def test_parse_calendar_month_raises_with_explicit_message():
    with pytest.raises(ValueError, match="calendar month"):
        Timeframe.parse("1M")

# ── str ───────────────────────────────────────────────────────────────────────

# __str__ preserves the parsed form rather than normalizing to an equivalent smaller unit
def test_str_preserves_parsed_form():
    assert str(Timeframe.parse("60m")) == "60m"
    assert str(Timeframe.parse("1h")) == "1h"

# ── seconds ───────────────────────────────────────────────────────────────────

def test_seconds():
    assert Timeframe.parse("1m").seconds == 60
    assert Timeframe.parse("4h").seconds == 14_400
    assert Timeframe.parse("1d").seconds == 86_400
    assert Timeframe.parse("2w").seconds == 1_209_600

# ── pandas_freq ───────────────────────────────────────────────────────────────

def test_pandas_freq():
    assert Timeframe.parse("1m").pandas_freq == "1min"
    assert Timeframe.parse("4h").pandas_freq == "4h"
    assert Timeframe.parse("1d").pandas_freq == "1D"
    assert Timeframe.parse("2w").pandas_freq == "2W"

# ── equality / hashing delegate to .seconds ──────────────────────────────────

# equal seconds compare equal even across different amount/unit combinations
def test_eq_delegates_to_seconds():
    assert Timeframe.parse("60m") == Timeframe.parse("1h")
    assert Timeframe.parse("24h") == Timeframe.parse("1d")

# different seconds compare unequal
def test_eq_false_for_different_seconds():
    assert Timeframe.parse("30m") != Timeframe.parse("1h")

# comparing against a non-Timeframe never raises
def test_eq_with_non_timeframe_is_false():
    assert Timeframe.parse("1h") != "1h"

# equal-seconds timeframes hash the same, so they collapse in a set
def test_hash_matches_for_equal_seconds():
    assert len({Timeframe.parse("60m"), Timeframe.parse("1h")}) == 1

# ── ordering delegates to .seconds ───────────────────────────────────────────

def test_lt_delegates_to_seconds():
    assert Timeframe.parse("4h") < Timeframe.parse("1d")
    assert not (Timeframe.parse("1d") < Timeframe.parse("4h"))

# ── is_multiple_of ────────────────────────────────────────────────────────────

# an exact multiple returns True
def test_is_multiple_of_true():
    assert Timeframe.parse("1h").is_multiple_of(Timeframe.parse("30m"))
    assert Timeframe.parse("1d").is_multiple_of(Timeframe.parse("4h"))

# a non-multiple returns False (this is the data_window.py bug: 90m signal on a 1h base)
def test_is_multiple_of_false():
    assert not Timeframe.parse("90m").is_multiple_of(Timeframe.parse("1h"))

# ── attributes ────────────────────────────────────────────────────────────────

def test_amount_and_unit_attributes():
    tf = Timeframe.parse("4h")
    assert tf.amount == 4
    assert tf.unit == TimeframeUnit.HOUR

# ── immutability ──────────────────────────────────────────────────────────────

# Timeframe is frozen -- attribute mutation raises
def test_timeframe_is_frozen():
    tf = Timeframe.parse("1h")
    with pytest.raises(Exception):
        tf.amount = 2
