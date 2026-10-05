"""
================================================================================
UNIT TESTS FOR domain/bars.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_bars.py -v

To run a specific test function:
    pytest tests/test_bars.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pandas as pd
from lighthouse.domain.bars import closed_bars

################### TESTS ##########################

# multi-row df: drops only the last (current/forming) row
def test_closed_bars_drops_last_row():
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )

    result = closed_bars(df)

    assert len(result) == 2
    assert list(result["close"]) == [1.0, 2.0]

# single-row df: the only row is the forming bar, so no closed bars exist
def test_closed_bars_single_row_returns_empty():
    df = pd.DataFrame(
        {"close": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
    )

    result = closed_bars(df)

    assert result.empty

# empty df: stays empty, no error
def test_closed_bars_empty_df_returns_empty():
    df = pd.DataFrame({"close": []})

    result = closed_bars(df)

    assert result.empty

# does not mutate the input df
def test_closed_bars_does_not_mutate_input():
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )

    closed_bars(df)

    assert len(df) == 3
