"""
================================================================================
SIGNALS - GENERAL FILTERS MODULE 
================================================================================

This module contains general-purpose boolean signal filters.

Each function takes an OHLCV DataFrame, parameters and does a single boolean check.

The DataFrame is expected to have:
- Columns: 'timestamp','open', 'high', 'low', 'close', 'volume'
- Datetime index 

================================================================================
"""

################# IMPORTS ##################
from datetime import time
from zoneinfo import ZoneInfo
import pandas as pd
from lighthouse.domain import indicators
 
################# FILTERS ##################

# ── Volume ─────────────────────────────────────────────────────────
def is_volume_above_average(df: pd.DataFrame, period: int, multiplier: float = 1.0) -> bool:
    """
    Return True if the last row of df has volume exceeding its rolling average.
 
    Args:
        df:         OHLCV DataFrame with a datetime index.
        period:     Lookback period for the rolling volume average.
        multiplier: Scale factor applied to the average before comparison.
                    (e.g. 1.5 means volume must be at least 1.5x the average).
 
    Returns:
        True if the last row's volume >= (rolling_average * multiplier).
    """
    avg = float(indicators.rolling_volume_avg(df, period).iloc[-1])
    current_volume = float(df["volume"].iloc[-1])
    return current_volume >= avg * multiplier


# ── Volatility ─────────────────────────────────────────────────────────
def is_atr_above_threshold(df: pd.DataFrame, threshold: float, period: int) -> bool:
    """
    Return True if the current ATR exceeds a minimum volatility threshold.
 
    Args:
        df:        OHLCV DataFrame with a datetime index.
        threshold: Minimum ATR value required to pass the filter.
        period:    ATR calculation period.
 
    Returns:
        True if ATR > threshold.
    """
    current_atr = indicators.atr_current(df, period)
    return current_atr > threshold
 
 
def is_atr_below_threshold(df: pd.DataFrame, threshold: float, period: int) -> bool:
    """
    Return True if the current ATR is below a maximum volatility threshold.
 
    Args:
        df:        OHLCV DataFrame with a datetime index.
        threshold: Maximum ATR value allowed to pass the filter.
        period:    ATR calculation period.
 
    Returns:
        True if ATR < threshold.
    """
    current_atr = indicators.atr_current(df, period)
    return current_atr < threshold


# ── Momentum ─────────────────────────────────────────────────────────
def is_rsi_above(df: pd.DataFrame, period: int, level: float = 50.0) -> bool:
    """
    Return True if the RSI reading on the last row of df is above level.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        level:  RSI threshold (0-100). Defaults to 50 (bullish momentum).
        period: RSI calculation period.
 
    Returns:
        True if the last row's RSI > level.
    """
    rsi_series = indicators.rsi(df, period)
    return float(rsi_series.iloc[-1]) > level
 
 
def is_rsi_below(df: pd.DataFrame, period: int, level: float = 50.0) -> bool:
    """
    Return True if the RSI reading on the last row of df is below level.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        level:  RSI threshold (0-100). Defaults to 50 (bearish momentum).
        period: RSI calculation period.
 
    Returns:
        True if the last row's RSI < level.
    """
    rsi_series = indicators.rsi(df, period)
    return float(rsi_series.iloc[-1]) < level
 
 
def is_rsi_in_range(df: pd.DataFrame, period: int, lower: float = 30.0, upper: float = 70.0) -> bool:
    """
    Return True if RSI is within [lower, upper] — useful for ranging filters.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        lower:  Lower RSI bound (inclusive).
        upper:  Upper RSI bound (inclusive).
        period: RSI calculation period.
 
    Returns:
        True if lower <= current RSI <= upper.
    """
    rsi_series = indicators.rsi(df, period)
    value = float(rsi_series.iloc[-1])
    return lower <= value <= upper
 
 
# ── Trend ─────────────────────────────────────────────────────────
def is_price_above_ema(df: pd.DataFrame, period: int) -> bool:
    """
    Return True if close price is above the EMA.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        period: EMA period.
    
    Returns:
        True if current close price > current EMA.
    """
    ema_series = indicators.ema(df, period)
    return float(df["close"].iloc[-1]) > float(ema_series.iloc[-1])
 

def is_price_below_ema(df: pd.DataFrame, period: int) -> bool:
    """
    Return True if close price is below the EMA.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        period: EMA period.
    
    Returns:
        True if current close price < current EMA.
    """
    ema_series = indicators.ema(df, period)
    return float(df["close"].iloc[-1]) < float(ema_series.iloc[-1])

 
def is_price_above_sma(df: pd.DataFrame, period: int) -> bool:
    """
    Return True if close price is above the SMA.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        period: SMA period.

    Returns:
        True if current close price > current SMA.
    """
    sma_series = indicators.sma(df, period)
    return float(df["close"].iloc[-1]) > float(sma_series.iloc[-1])
 
 
def is_price_below_sma(df: pd.DataFrame, period: int) -> bool:
    """
    Return True if close price is below the SMA.
 
    Args:
        df:     OHLCV DataFrame with a datetime index.
        period: SMA period.
    
    Returns:
        True if current close price < current SMA.
    """
    sma_series = indicators.sma(df, period)
    return float(df["close"].iloc[-1]) < float(sma_series.iloc[-1])
 

def is_sma_above_sma(df: pd.DataFrame, fast_period: int = 50, slow_period: int = 200) -> bool:
    """
    Return True if the fast SMA is above the slow SMA (golden cross condition).
 
    Args:
        df:           OHLCV DataFrame with a datetime index.
        fast_period:  Period for the faster SMA.
        slow_period:  Period for the slower SMA.
 
    Returns:
        True if current SMA(fast) > current SMA(slow).
    """
    fast = indicators.sma(df, fast_period)
    slow = indicators.sma(df, slow_period)
    return float(fast.iloc[-1]) > float(slow.iloc[-1])


def is_ema_above_ema(df: pd.DataFrame, fast_period: int = 50, slow_period: int = 200) -> bool:
    """
    Return True if the fast EMA is above the slow EMA (golden cross condition).
 
    Args:
        df:           OHLCV DataFrame with a datetime index.
        fast_period:  Period for the faster EMA.
        slow_period:  Period for the slower EMA.

    Returns:
        True if current EMA(fast) > current EMA(slow).
    """
    fast = indicators.ema(df, fast_period)
    slow = indicators.ema(df, slow_period)
    return float(fast.iloc[-1]) > float(slow.iloc[-1])


# ── Bollinger bands ─────────────────────────────────────────────────────────
def is_price_above_upper_band(df: pd.DataFrame, period: int, std_dev: float = 2.0) -> bool:
    """
    Return True if close price is above the upper Bollinger Band.
 
    Args:
        df:      OHLCV DataFrame with a datetime index.
        period:  Bollinger Band period.
        std_dev: Number of standard deviations for the bands.

    Returns:
        True if current close price > current upper_band value.
    """
    middle, upper, lower = indicators.bollinger_bands(df, period, std_dev)
    return float(df["close"].iloc[-1]) > float(upper.iloc[-1])
 
 
def is_price_below_lower_band(df: pd.DataFrame, period: int, std_dev: float = 2.0) -> bool:
    """
    Return True if close price is below the lower Bollinger Band.
 
    Args:
        df:      OHLCV DataFrame with a datetime index.
        period:  Bollinger Band period.
        std_dev: Number of standard deviations for the bands.

    Returns:
        True if current close price < current lower_band value.
    """
    middle, upper, lower = indicators.bollinger_bands(df, period, std_dev)
    return float(df["close"].iloc[-1]) < float(lower.iloc[-1])
 
 
def is_price_inside_bands(df: pd.DataFrame, period: int, std_dev: float = 2.0) -> bool:
    """
    Return True if close price is within the Bollinger Bands.
 
    Args:
        df:      OHLCV DataFrame with a datetime index.
        period:  Bollinger Band period.
        std_dev: Number of standard deviations for the bands.

    Returns:
        True if current close price is within the Bollinger Bands.
    """
    middle, upper, lower = indicators.bollinger_bands(df, period, std_dev)
    close = float(df["close"].iloc[-1])
    return float(lower.iloc[-1]) <= close <= float(upper.iloc[-1])
 
 
# ── Swing structure ─────────────────────────────────────────────────────────
def is_price_above_swing_high(df: pd.DataFrame, window: int) -> bool:
    """
    Return True if close price has broken above the recent swing high.

    Swing high computation excludes the current bar to avoid lookahead bias in reference level.
 
    Args:
        df:       OHLCV DataFrame with a datetime index.
        window:   Number of bars to look back when identifying the swing high.

    Returns:
        True if current close price > swing_high(df, window).
    """
    high = indicators.swing_high(df.iloc[:-1], window)
    return float(df["close"].iloc[-1]) > high
 
 
def is_price_below_swing_low(df: pd.DataFrame, window: int) -> bool:
    """
    Return True if close price has broken below the recent swing low.

    Swing low computation excludes the current bar to avoid lookahead bias in reference level.
 
    Args:
        df:       OHLCV DataFrame with a datetime index.
        window:   Number of bars to look back when identifying the swing low.

    Returns:
        True if current close price < swing_low(df, window).
    """
    low = indicators.swing_low(df.iloc[:-1], window)
    return float(df["close"].iloc[-1]) < low
 
 
# ── Trading sessions ─────────────────────────────────────────────────────────

# Session windows defined in each market's local time, paired with its IANA timezone.
# Format: (session_open, session_close, iana_timezone)
SESSIONS: dict[str, tuple[time, time, str]] = {
    "london":               (time(8, 0),  time(17, 0), "Europe/London"),       # LSE hours
    "ny":                   (time(9, 30), time(16, 0), "America/New_York"),    # NYSE hours
    "tokyo":                (time(9, 0),  time(15, 30),"Asia/Tokyo"),          # TSE hours (no DST)
    "sydney":               (time(10, 0), time(16, 0), "Australia/Sydney"),    # ASX hours
    "london_ny_overlap":    (time(14, 0), time(17, 0), "Europe/London"),       # 14:00–17:00 London local
    "tokyo_london_overlap": (time(8, 0),  time(9, 0),  "Europe/London"),       # 08:00–09:00 London local
}
 
 
def is_within_session(df: pd.DataFrame, session_open: time, session_close: time, tz: str = "UTC") -> bool:
    """
    Return True if the last row of df falls within a custom trading session.
 
    Args:
        df:            OHLCV DataFrame with a tz-aware or tz-naive datetime index.
        session_open:  Session start time (expressed in the timezone given by `tz`).
        session_close: Session end time (expressed in the timezone given by `tz`).
        tz:            IANA timezone name (e.g. "Europe/London", "America/New_York").
                       The bar timestamp is converted to this timezone before comparison.

    Returns:
        True if the last row's timestamp is within [session_open, session_close).
    """
    last_ts = df.index[-1]
 
    # Localise tz-naive indices; convert tz-aware indices to the target tz.
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(ZoneInfo(tz)) # zoneinfo handles DST automatically
    else:
        last_ts = last_ts.astimezone(ZoneInfo(tz)) # zoneinfo handles DST automatically
 
    bar_time = last_ts.time()
 
    # Handle sessions that wrap past midnight.
    if session_open <= session_close:
        return session_open <= bar_time < session_close
    else:
        return bar_time >= session_open or bar_time < session_close
 
 
def is_in_named_session(df: pd.DataFrame, session: str) -> bool:
    """
    Convenience wrapper for predefined sessions in SESSIONS.

    Return True if the last row of df falls within a named trading session.
 
    Available sessions: "london", "ny", "tokyo", "sydney",
                        "london_ny_overlap", "tokyo_london_overlap".
 
    Args:
        df:      OHLCV DataFrame with a tz-aware or tz-naive datetime index.
        session: One of the session keys in SESSIONS.
 
    Returns:
        True if the last row is within the named session window.
    """
    if session not in SESSIONS:
        raise ValueError(
            f"Unknown session '{session}'. "
            f"Choose from: {list(SESSIONS.keys())}"
        )
    open_, close_, tz = SESSIONS[session]
    return is_within_session(df, open_, close_, tz=tz)