"""
================================================================================
INDICATORS MODULE 
================================================================================

This module contains functions related to calculating numerical indicators.

Functions:
    swing_high() : highest high over a lookback window
    swing_low()  : lowest low over a lookback window
    atr()        : average true range (ATR)
    atr_current(): most recent ATR value as a scalar
    atr_pct()    : ATR expressed as a percentage of current close
    bollinger_bands() : Bollinger Bands (middle, upper, lower)
    bollinger_bandwidth() : normalized width of the Bollinger Bands
    bollinger_pct_b() : Bollinger %B; position of price within the bands
    rolling_volume_avg() : simple rolling average of volume
    rolling_volume_avg_current() : most recent rolling volume average as a scalar
    volume_ratio_current() : current bar's volume as a multiple of the rolling average
    sma() : simple moving average (SMA)
    ema() : exponential moving average (EMA)
    rsi() : relative strength index (RSI)

================================================================================
"""

################# IMPORTS ##################
import pandas as pd
import numpy as np
from lighthouse.utils.validation import require_columns

################# INDICATORS ##################

# ── Support/resistance ─────────────────────────────────────────────────────────
def swing_high(df: pd.DataFrame, window: int) -> float:
    """
    Highest high over a lookback window.

    Args:
        df       : DataFrame with a 'high' column
        window   : Number of bars to look back 

    Returns:
        The highest high value over the lookback window.

    Raises:
        ValueError: If the DataFrame does not contain a 'high' column or if it has fewer rows than the specified window.
    """
    require_columns(df, ["high"], window)
    return float(df["high"].iloc[-window:].max())

def swing_low(df: pd.DataFrame, window: int) -> float:
    """
    Lowest low over a lookback window.

    Args:
        df       : DataFrame with a 'low' column
        window   : Number of bars to look back 

    Returns:
        The lowest low value over the lookback window.

    Raises:
        ValueError: If the DataFrame does not contain a 'low' column or if it has fewer rows than the specified window.
    """
    require_columns(df, ["low"], window)
    return float(df["low"].iloc[-window:].min())

# ── Volatility ─────────────────────────────────────────────────────────
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range (ATR).

    Measures market volatility using the full high/low/close range.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR        = Wilder's smoothed moving average of True Range over 'period' bars.

    Args:
        df     : DataFrame with 'high', 'low', 'close' columns.
        period : Smoothing period.

    Returns:
        A pandas series of ATR values aligned to df's index. First 'period' values are NaN.
    
    Raises:
        ValueError: If the DataFrame does not contain 'high', 'low', or 'close' columns or has fewer than `period` rows.
    """
    require_columns(df, ["high", "low", "close"], period)

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    
    alpha = 1.0 / period # Wilder's smoothing 

    # pandas.ewm uses a different seeding than original Wilder's, converges after enough bars.
    # e.g. fetching 200 bars, ATR values stabilize after approximately 2*period bars.
    # Traders are mostly interested in the end of the series, which are the latest ATR values and the most stable.
    return tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def atr_current(df: pd.DataFrame, period: int) -> float:
    """
    Most recent ATR value as a scalar.

    Convenience wrapper around atr() for strategies that only need
    the latest reading.

    Args:
        df     : DataFrame with 'high', 'low', 'close' columns.
        period : Smoothing period.

    Returns:
        The most recent ATR value.
    """
    return float(atr(df, period).iloc[-1])


def atr_pct(df: pd.DataFrame, period: int) -> float:
    """
    ATR expressed as a percentage of the current closing price.

    Args:
        df     : DataFrame with 'high', 'low', 'close' columns.
        period : Smoothing period for ATR.

    Returns:
        The current ATR as a percentage of the current closing price.
    """
    return (atr_current(df, period) / df["close"].iloc[-1]) * 100


def bollinger_bands(df: pd.DataFrame, period: int, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands (middle, upper, lower).

    Args:
        df      : DataFrame with a 'close' column.
        period  : Rolling window for SMA and standard deviation.
        num_std : Number of standard deviations for the bands (default 2.0).

    Returns:
        A tuple of (middle, upper, lower) as pd.Series aligned to df's index. First `period` values are NaN.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'close' column or has fewer than `period` rows.
    """
    require_columns(df, ["close"], period)

    middle = df["close"].rolling(window=period).mean()
    std    = df["close"].rolling(window=period).std(ddof=0)
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return middle, upper, lower


def bollinger_bandwidth(df: pd.DataFrame, period: int, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger bandwidth; normalized width of the Bollinger Bands.

    Measures how wide the bands are relative to the middle band (SMA).

    Args:
        df      : DataFrame with a 'close' column.
        period  : Rolling window passed to bollinger_bands.
        num_std : Number of standard deviations for the bands (default = 2.0).

    Returns:
        A pandas series of bandwidth values aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'close' column or has fewer than `period` rows.
    """
    middle, upper, lower = bollinger_bands(df, period, num_std)
    return (upper - lower) / middle


def bollinger_pct_b(df: pd.DataFrame, period: int, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger %B — position of price within the Bollinger Bands.

    Expresses where the current close sits relative to the bands on a
    0-1 scale. Values outside [0, 1] mean price has broken outside
    the bands.

    1 = close is at upper band
    0 = close is at lower band

    Args:
        df      : DataFrame with a 'close' column.
        period  : Rolling window passed to bollinger_bands.
        num_std : Number of standard deviations for the bands (default = 2.0).

    Returns:
        A pandas series of %B values aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'close' column or has fewer than `period` rows.
    """
    middle, upper, lower = bollinger_bands(df, period, num_std)
    return (df["close"] - lower) / (upper - lower)


# ── Volume ─────────────────────────────────────────────────────────
def rolling_volume_avg(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Simple rolling average of volume.

    Args:
        df     : DataFrame with a 'volume' column.
        period : Rolling window size.

    Returns:
        A pandas series containing the rolling means of volume aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'volume' column or has fewer than `period` rows.
    """
    require_columns(df, ["volume"], period)
    return df["volume"].rolling(window=period).mean()


def rolling_volume_avg_current(df: pd.DataFrame, period: int) -> float:
    """
    Most recent rolling volume average as a scalar.

    Args:
        df     : DataFrame with a 'volume' column.
        period : Rolling window size.

    Returns:
        The most recent rolling volume average as a scalar.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'volume' column.
    """
    return float(rolling_volume_avg(df, period).iloc[-1])


def volume_ratio_current(df: pd.DataFrame, period: int) -> float:
    """
    Current bar's volume expressed as a multiple of the rolling average.

    Args:
        df     : DataFrame with a 'volume' column.
        period : Rolling window for the average.

    Returns:
        The ratio of current volume to rolling average.
    
    Raises:
        ValueError: If the DataFrame does not contain a 'volume' column.
    """
    avg = rolling_volume_avg_current(df, period)
    if avg == 0:
        return float("nan")
    return float(df["volume"].iloc[-1] / avg)


# ── Moving averages ─────────────────────────────────────────────────────────
def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """
    Computes a simple moving average.
    By default used for close price, but can be used for any column.

    Args:
        df      : DataFrame containing `column`.
        period  : Rolling window size.
        column  : Column to compute SMA on (default 'close').

    Returns:
        A pandas series of SMA values aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain the specified column or has fewer than `period` rows.
    """
    require_columns(df, [column], period)
    return df[column].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """
    Exponential Moving Average.

    By default used for close price, but can be used for any column.

    Uses the standard smoothing factor: alpha = 2 / (period + 1).

    Args:
        df      : DataFrame containing `column`.
        period  : Span for EMA smoothing.
        column  : Column to compute EMA on (default 'close').

    Returns:
        A pandas series of EMA values aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain the specified column or has fewer than `period` rows.
    """
    require_columns(df, [column], period)
    return df[column].ewm(span=period, adjust=False).mean()


# ── Momentum ─────────────────────────────────────────────────────────
def rsi(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """
    Relative Strength Index (RSI).

    Measures momentum by comparing average gains to average losses
    over a rolling window. Uses Wilder's smoothing (EWMA with alpha=1/period).

    Args:
        df      : DataFrame containing `column`.
        period  : Lookback period.
        column  : Column to compute RSI on (default 'close').

    Returns:
        A pandas series of RSI values aligned to df's index.
    
    Raises:
        ValueError: If the DataFrame does not contain the specified column or has fewer than `period` rows.
    """
    require_columns(df, [column], period)
    delta = df[column].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


