"""
================================================================================
UNIT TESTS FOR ENUMS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_enums.py -v

To run a specific test function:
    pytest tests/test_enums.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from lighthouse.domain.enums import Side, normalize_side

################### TESTS ##########################

# ── normalize_side ────────────────────────────────────────────────────────

# "long" input; already normalized
def test_normalize_side_long():
    assert normalize_side("long") == Side.LONG

# "buy" input; normalized to Side.LONG
def test_normalize_side_buy():
    assert normalize_side("buy") == Side.LONG

# "short" input; already normalized
def test_normalize_side_short():
    assert normalize_side("short") == Side.SHORT

# "sell" input; normalized to Side.SHORT
def test_normalize_side_sell():
    assert normalize_side("sell") == Side.SHORT

# uppercase input; case-insensitive normalization
def test_normalize_side_case_insensitive():
    assert normalize_side("LONG") == Side.LONG
    assert normalize_side("BUY") == Side.LONG
    assert normalize_side("SHORT") == Side.SHORT
    assert normalize_side("SELL") == Side.SHORT

# unknown input; raises ValueError
def test_normalize_side_unknown():
    with pytest.raises(ValueError):
        normalize_side("flat")
