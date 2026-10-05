"""
================================================================================
SIGNALS - EXAMPLE BOT STRATEGY
================================================================================

This module contains breakout signal detection functions.

Each function takes an OHLCV DataFrame and parameters, and returns a single
boolean indicating whether a breakout condition is met on the last row of df.
The caller picks the frame — pass closed_bars(df) (lighthouse.domain.bars) for
closed-bar semantics, or df itself to evaluate the forming bar (see docs/adr/0001).

The DataFrame is expected to have:
- Columns: 'open', 'high', 'low', 'close', 'volume'
- Datetime index

A breakout is confirmed when the last row's close price exceeds the
swing high (long) or falls below the swing low (short) by at least
atr_buffer_mult * ATR, providing a volatility-scaled confirmation buffer.

Functions:
    is_long_breakout()  : True if the last row signals a long breakout
    is_short_breakout() : True if the last row signals a short breakout
================================================================================
"""

################# IMPORTS ##################

import pandas as pd
from lighthouse.domain import indicators

################# SIGNALS ##################

# ── Breakout signals ────────────────────────────────────────────────────────
def is_long_breakout(df: pd.DataFrame, window: int, atr_period: int, atr_buffer_mult: float) -> bool:
    """
    Return True if the last row of df signals a long breakout.

    A long breakout is confirmed when the last row's close exceeds
    the swing high of the lookback window by at least atr_buffer_mult * ATR.
    This buffer prevents triggering on a marginal poke above the range.

    Args:
        df:              OHLCV DataFrame with a datetime index; last row is evaluated.
        window:          Number of bars to look back for the swing high.
        atr_period:      ATR calculation period.
        atr_buffer_mult: Multiplier applied to ATR to form the confirmation buffer.

    Returns:
        True if close[-1] > swing_high + (atr_buffer_mult * ATR).
    """

    # Get the last row's close price
    last_close = df["close"].iloc[-1]

    # Calculate the swing high over the lookback window, excluding the last
    # row to avoid including its own high
    swing_high = indicators.swing_high(df.iloc[:-1], window)

    # Calculate current ATR
    atr = indicators.atr_current(df, atr_period)

    # Check if the last row's close exceeds the swing high plus the ATR buffer
    return bool(last_close > swing_high + (atr_buffer_mult * atr))


def is_short_breakout(df: pd.DataFrame, window: int, atr_period: int, atr_buffer_mult: float) -> bool:
    """
    Return True if the last row of df signals a short breakout.

    A short breakout is confirmed when the last row's close falls
    below the swing low of the lookback window by at least atr_buffer_mult * ATR.
    This buffer prevents triggering on a marginal poke below the range.

    Args:
        df:              OHLCV DataFrame with a datetime index; last row is evaluated.
        window:          Number of bars to look back for the swing low.
        atr_period:      ATR calculation period.
        atr_buffer_mult: Multiplier applied to ATR to form the confirmation buffer.

    Returns:
        True if close[-1] < swing_low - (atr_buffer_mult * ATR).
    """
    # Get the last row's close price
    last_close = df["close"].iloc[-1]

    # Calculate the swing low over the lookback window, excluding the last
    # row to avoid including its own low
    swing_low = indicators.swing_low(df.iloc[:-1], window)

    # Calculate current ATR
    atr = indicators.atr_current(df, atr_period)

    # Check if the last row's close falls below the swing low minus the ATR buffer
    return bool(last_close < swing_low - (atr_buffer_mult * atr))
