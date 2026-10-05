"""
================================================================================
UNIT TESTS FOR SIGNALS/TRIGGERS.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_signals/test_triggers.py -v

To run a specific test function:
    pytest tests/test_signals/test_triggers.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.domain.signals.triggers import became_true
from lighthouse.domain.signals.general_filters import is_atr_above_threshold

################### TESTS ##########################

def _make_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "open":   [c - 0.5 for c in closes],
            "high":   [c + 1.0 for c in closes],
            "low":    [c - 1.0 for c in closes],
            "close":  closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


# rising edge; predicate false one bar earlier, true on the latest bar
def test_became_true_true_on_rising_edge():
    df = _make_df([1.0, 2.0, 3.0])
    assert became_true(lambda d: bool(d["close"].iloc[-1] > 2.0), df) is True

# level, not edge; predicate was already true one bar earlier and still is
def test_became_true_false_when_already_true():
    df = _make_df([3.0, 3.0, 3.0])
    assert became_true(lambda d: bool(d["close"].iloc[-1] >= 3.0), df) is False

# no signal; predicate false on the latest bar regardless of the bar before
def test_became_true_false_when_predicate_false_on_latest():
    df = _make_df([3.0, 3.0, 1.0])
    assert became_true(lambda d: bool(d["close"].iloc[-1] >= 3.0), df) is False

# return type; must be bool
def test_became_true_returns_bool():
    df = _make_df([1.0, 2.0, 3.0])
    result = became_true(lambda d: bool(d["close"].iloc[-1] > 2.0), df)
    assert isinstance(result, bool)

# warm-up boundary; df has exactly `period` rows so predicate(df) succeeds, but
# df.iloc[:-1] is one bar short of the same indicator's minimum -- the
# ValueError from require_columns propagates rather than being swallowed
def test_became_true_propagates_error_at_warmup_boundary():
    df = _make_df([100.0] * 14)
    with pytest.raises(ValueError):
        became_true(lambda d: is_atr_above_threshold(d, threshold=0.0, period=14), df)
