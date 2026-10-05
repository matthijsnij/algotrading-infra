"""
================================================================================
BASE DATA SOURCE
================================================================================

Abstract base class for all OHLCV data sources used to load historical data
into BacktestExchange.

To implement a custom data source, subclass BaseDataSource and implement load().

Functions:
    validate_dataframe(): Validate that a DataFrame conforms to the requirements for BacktestExchange.
    validate_bar_continuity(): Validate that gaps between consecutive bars are structurally legal for a TradingCalendar.
================================================================================
"""

from __future__ import annotations

######### IMPORTS #########
from abc import ABC, abstractmethod
import pandas as pd

from lighthouse.domain.calendars import TradingCalendar
from lighthouse.domain.timeframe import Timeframe
from lighthouse.utils.logging import get_logger

logger = get_logger(__name__)

######### VALIDATION #########
def validate_dataframe(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "",
    calendar: TradingCalendar | None = None,
) -> None:
    """
    Validate that a DataFrame conforms to OHLCV standard for BacktestExchange.

    Raises ValueError if any validation check fails.

    Args:
        df        : DataFrame to validate
        symbol    : symbol name (for error messages only)
        timeframe : expected timeframe; verifies bar interval matches
        calendar  : optional TradingCalendar; when supplied, also runs
                    validate_bar_continuity() to reject structurally illegal gaps.
                    Omitted (default None) leaves behaviour unchanged for existing callers.

    Checks:
        - Not empty
        - Has DatetimeIndex in UTC
        - Has exactly [open, high, low, close, volume] columns (lowercase)
        - No NaN in OHLCV columns
        - OHLCV logic: high >= low, close in [low, high]
        - Volume >= 0
        - No duplicate timestamps
        - Sorted ascending by timestamp
        - Bar interval matches expected timeframe
        - Gap continuity against `calendar`, if supplied
    """
    context = f"symbol={symbol}" if symbol else "DataFrame"

    # Check 1: Not empty
    if df.empty:
        raise ValueError(f"[{context}] DataFrame is empty")

    # Check 2: DatetimeIndex exists and is UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"[{context}] Index must be DatetimeIndex, got {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise ValueError(
            f"[{context}] DatetimeIndex must be timezone-aware (UTC), got naive index"
        )
    if str(df.index.tz) != "UTC":
        raise ValueError(
            f"[{context}] DatetimeIndex timezone must be UTC, got {df.index.tz}"
        )

    # Check 3: Correct columns
    required_cols = {"open", "high", "low", "close", "volume"}
    actual_cols = set(df.columns)
    if actual_cols != required_cols:
        missing = required_cols - actual_cols
        extra = actual_cols - required_cols
        msg = f"[{context}] Column mismatch."
        if missing:
            msg += f" Missing: {missing}."
        if extra:
            msg += f" Extra: {extra}."
        raise ValueError(msg)

    # Check 4: No NaN in OHLCV
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"[{context}] Column '{col}' has {nan_count} NaN values"
            )

    # Check 5: OHLCV logic
    invalid_high_low = (df["high"] < df["low"]).any()
    if invalid_high_low:
        raise ValueError(
            f"[{context}] Found rows where high < low (data logic error)"
        )

    invalid_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).any()
    if invalid_close:
        raise ValueError(
            f"[{context}] Found rows where close not in [low, high] (data logic error)"
        )

    # Check 6: Volume >= 0
    negative_volume = (df["volume"] < 0).any()
    if negative_volume:
        raise ValueError(
            f"[{context}] Found negative volume values (volume must be >= 0)"
        )

    # Check 7: No duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        raise ValueError(
            f"[{context}] Found {dup_count} duplicate timestamps"
        )

    # Check 8: Sorted ascending
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"[{context}] Index not sorted ascending (required for backtest)"
        )

    # Check 9: Bar interval matches expected timeframe
    try:
        expected_seconds = Timeframe.parse(timeframe).seconds
    except ValueError as exc:
        raise ValueError(f"[{context}] Unknown timeframe '{timeframe}': {exc}") from exc

    # Calculate median bar interval from timestamps
    if len(df) > 1:
        time_deltas = df.index.to_series().diff().dt.total_seconds()
        # Skip the first NaT value
        time_deltas = time_deltas.dropna()

        if len(time_deltas) > 0:
            median_interval = time_deltas.median()
            # Allow small tolerance (±2% to account for DST or tick precision)
            tolerance = expected_seconds * 0.02
            if abs(median_interval - expected_seconds) > tolerance:
                raise ValueError(
                    f"[{context}] Bar interval mismatch. Expected {timeframe} "
                    f"(~{expected_seconds}s), but found median interval of {median_interval:.0f}s. "
                    f"Check that the correct CSV file is loaded."
                )

    # Check 10: Gap continuity against the trading calendar, if supplied
    if calendar is not None:
        validate_bar_continuity(df, calendar, timeframe)


def validate_bar_continuity(df: pd.DataFrame, calendar: TradingCalendar, base_timeframe: str) -> None:
    """
    Validate that every gap between consecutive bars is structurally legal for `calendar`.

    Raises ValueError if any validation check fails.

    Args:
        df             : DataFrame with a UTC DatetimeIndex, sorted, no duplicates expected
        calendar       : TradingCalendar governing which gaps are legal (see domain/calendars.py)
        base_timeframe : timeframe string of df, used only in error messages

    Checks:
        - No duplicate timestamps
        - Sorted ascending by timestamp
        - Reference interval is the modal (most common) consecutive delta, so it stays
          correct even when the data contains gaps.
        - Every consecutive pair either matches the modal interval exactly or is a gap
          the calendar structurally allows (calendar.allows_gap()); otherwise ValueError,
          naming both offending timestamps and the observed delta.
        - Every accepted gap larger than the modal interval is logged as a warning
          (timestamps + length), so wholly-missing trading days stay visible even
          though they pass validation.
    """
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        raise ValueError(f"[{base_timeframe}] Found {dup_count} duplicate timestamps")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{base_timeframe}] Index not sorted ascending (required for backtest)")

    if len(df) < 2:
        return

    deltas = df.index.to_series().diff().dropna()
    modal_delta = deltas.mode().iloc[0]

    for prev_ts, next_ts, delta in zip(df.index[:-1], df.index[1:], deltas):
        if delta == modal_delta:
            continue
        if calendar.allows_gap(prev_ts, next_ts, modal_delta):
            logger.warning(
                "[%s] Accepted gap between %s and %s (delta=%s, expected %s) under calendar '%s'.",
                base_timeframe, prev_ts, next_ts, delta, modal_delta, calendar.name,
            )
            continue
        raise ValueError(
            f"[{base_timeframe}] Illegal gap between {prev_ts} and {next_ts} "
            f"(delta={delta}, expected {modal_delta}) — calendar '{calendar.name}' does not "
            f"structurally allow this gap. If this gap is expected, use a different calendar."
        )


######### CLASS #########
class BaseDataSource(ABC):
    """
    Abstract interface for loading historical OHLCV data into BacktestExchange.

    Subclasses must implement _load_impl(), not load(). The load() method is
    concrete and enforces validation automatically.

    Methods:
        load()      : concrete method that validates the returned DataFrame and defers to _load_impl()
        _load_impl(): abstract method that subclasses must implement to load data
    """

    def __init_subclass__(cls, **kwargs):
        """
        Enforce that subclasses implement _load_impl(), not load().

        Raises TypeError if a subclass tries to override load().
        """
        super().__init_subclass__(**kwargs)
        
        if "load" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must implement _load_impl(), not load(). "
                f"The load() method is the template that enforces validation."
            )

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for the given symbol and timeframe.

        This method is concrete and enforces validation. Subclasses must
        implement _load_impl() instead.

        Args:
            symbol    : trading symbol 
            timeframe : timeframe string 
            start_date: optional start date/time string 
            end_date  : optional end date/time string 

        Returns:
            pandas DataFrame with DatetimeIndex (UTC) and columns:
            [open, high, low, close, volume].
        """
        df = self._load_impl(symbol, timeframe, start_date, end_date)
        validate_dataframe(df, symbol, timeframe)
        return df

    @abstractmethod
    def _load_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data. Subclasses must implement this method.

        The returned DataFrame will be validated by load(). Ensure it conforms to:
            - DatetimeIndex in UTC
            - Columns: open, high, low, close, volume (float)
            - Sorted ascending by timestamp
            - No NaN in OHLCV columns

        Args:
            symbol    : trading symbol 
            timeframe : timeframe string 
            start_date: optional start date/time string 
            end_date  : optional end date/time string 

        Returns:
            pandas DataFrame with OHLCV data.
        """
        ...
