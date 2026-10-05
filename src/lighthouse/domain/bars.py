"""
================================================================================
BARS MODULE
================================================================================

Per ADR 0001 (docs/adr/0001-last-row-is-current-period.md), the last row of any
OHLCV DataFrame handed to a bot is always the current period — complete or not
— in both live and backtest. Bots that need only settled data call
closed_bars(df) to drop it.

Functions:
    closed_bars() : drop the last (current/forming) row of an OHLCV DataFrame
================================================================================
"""

from __future__ import annotations

################# IMPORTS ##################
import pandas as pd

################# FUNCTIONS ##################

def closed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the last row of an OHLCV DataFrame, which always represents the
    current (possibly still-forming) period.

    Args:
        df: OHLCV DataFrame whose last row is the current period.

    Returns:
        A copy of df excluding its last row. Empty (0 or 1 row) input returns
        an empty DataFrame.
    """
    return df.iloc[:-1].copy()
