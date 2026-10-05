"""
================================================================================
DATA WINDOW 
================================================================================

This file contains the DataWindow class, which provides bar-by-bar OHLCV data access with optional timeframe aggregation.

All methods enforce strict lookahead prevention: only bars up to and including
the current cursor are ever returned.

Helper functions:
    _tf_to_seconds()     : convert timeframe string to seconds
    _tf_to_pandas_freq() : convert timeframe string to pandas frequency string
================================================================================
"""

############ IMPORTS #############
import pandas as pd
from typing import Any
from lighthouse.domain.timeframe import Timeframe
from .state import BacktestState

########## CONSTANTS ##########

# Shared aggregation spec for resampling base bars into a coarser timeframe.
_AGG_SPEC: dict[str, tuple[str, str]] = {
    "open":   ("open",   "first"),
    "high":   ("high",   "max"),
    "low":    ("low",    "min"),
    "close":  ("close",  "last"),
    "volume": ("volume", "sum"),
}

########## HELPER FUNCTIONS ##########
def _tf_to_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to its equivalent duration in seconds.

    Args:
        timeframe : timeframe string (e.g., "1m", "5m", "1h", "1d")
    
    Returns:
        Duration in seconds as an integer.

    Raises:
        ValueError: `timeframe` does not parse as a Timeframe.
    """
    try:
        return Timeframe.parse(timeframe).seconds
    except ValueError as exc:
        raise ValueError(f"BacktestExchange: unsupported timeframe '{timeframe}'. {exc}") from exc

def _tf_to_pandas_freq(timeframe: str) -> str:
    """
    Convert a timeframe string to its equivalent pandas frequency string.

    Args:
        timeframe : timeframe string (e.g., "1m", "5m", "1h", "1d")
    
    Returns:
        Pandas frequency string (e.g., "1min", "5min", "1h", "1D").

    Raises:
        ValueError: `timeframe` does not parse as a Timeframe.
    """
    try:
        return Timeframe.parse(timeframe).pandas_freq
    except ValueError as exc:
        raise ValueError(f"BacktestExchange: unsupported timeframe '{timeframe}'. {exc}") from exc

########### CLASS ##########
class DataWindow:
    """
    Provides bar-by-bar OHLCV data access with optional timeframe aggregation.

    All data access is gated by the current cursor position so the simulation
    never exposes future bars to the strategy bot.

    Methods:
        get_window()         : return last limit bars for a given timeframe, ending with the current (possibly forming) period
        is_signal_bar_close(): check if the current base bar is the last bar of its signal period
        fetch_ohlcv()        : return OHLCV data in list-of-lists format
        fetch_bid_ask()      : derive bid/ask from current bar's close and configured spread
        
    """

    def __init__(self, state: BacktestState) -> None:
        """
        Constructor. Initialize the DataWindow with shared simulation state.

        Args:
            state : BacktestState instance (shared by all engines)        
        """
        self._s = state

    def get_window(self, limit: int, timeframe: str) -> pd.DataFrame:
        """
        Return the last 'limit' bars for the requested timeframe, ending with the
        current period (per ADR 0001: always the last row, complete or not).

        When timeframe == base_timeframe, returns raw base bars directly.
        When timeframe is coarser, resamples base bars up to and including the
        still-forming last bucket.

        Args:
            limit     : number of bars to return
            timeframe : target timeframe string

        Returns:
            DataFrame with [open, high, low, close, volume] and a datetime index.
        """
        s = self._s

        # Pass-through: no aggregation needed
        if timeframe == s.base_timeframe:
            start  = max(0, s.cursor - limit + 1)
            window = s.df.iloc[start : s.cursor + 1].copy()
            return window.set_index(s.ts_col)

        # Validate: requested timeframe must be a whole multiple of the base timeframe.
        # A plain seconds comparison would wrongly accept e.g. a 90m signal timeframe
        # on a 1h base (5400s >= 3600s passes) even though 90m bars can't be built
        # from clean 1h base bars.
        base_tf   = Timeframe.parse(s.base_timeframe)
        signal_tf = Timeframe.parse(timeframe)
        if not signal_tf.is_multiple_of(base_tf):
            raise ValueError(
                f"BacktestExchange.get_window(): requested timeframe '{timeframe}' "
                f"({signal_tf.seconds}s) is not a whole multiple of base_timeframe "
                f"'{s.base_timeframe}' ({base_tf.seconds}s). Cannot create signal-timeframe "
                f"candles that don't align with the base timeframe's bars."
            )

        # Aggregate all base bars up to cursor into signal-timeframe candles
        freq      = _tf_to_pandas_freq(timeframe)
        base_bars = s.df.iloc[: s.cursor + 1].copy()
        base_bars = base_bars.set_index(s.ts_col)
        base_bars.index = pd.to_datetime(base_bars.index, utc=True)

        # Session-aware aggregation: a signal candle must never merge bars from two
        # different trading sessions (e.g. Friday close + Monday open). Grouping by a
        # single constant session id (24/7) reduces to a plain resample, so this is a
        # strict superset of the old behaviour, not a separate code path.
        session_ids = base_bars.index.map(s.calendar.session_id)
        if session_ids.nunique() <= 1:
            agg = base_bars.resample(
                freq, closed="left", label="left"
            ).agg(**_AGG_SPEC).dropna(subset=["open"])
        else:
            parts = [
                group.resample(freq, closed="left", label="left")
                     .agg(**_AGG_SPEC).dropna(subset=["open"])
                for _, group in base_bars.groupby(session_ids)
            ]
            agg = pd.concat(parts).sort_index()

        # Per ADR 0001, the last bucket is always included, complete or not — it
        # represents the current period, matching live's in-progress candle.
        return agg.iloc[-limit:]

    def is_signal_bar_close(self, timeframe: str) -> bool:
        """
        Return True if the current base bar is the last bar of its signal period.

        Compares the current bar's timestamp against the actual next row in the data
        (not an arithmetic guess), so a period boundary is detected correctly even when
        the next bar is separated by a calendar gap (e.g. overnight or weekend). Falls
        back to arithmetic only at the end of the data, where no next row exists, to
        avoid leaking a partial candle.

        Args:
            timeframe : signal timeframe to check against

        Returns:
            True if this is the last base bar in the current signal period.
        """
        s = self._s
        if timeframe == s.base_timeframe:
            return True

        curr_ts = s.get_bar_timestamp(s.cursor)
        if curr_ts is None:
            raise RuntimeError(
                f"BacktestExchange: get_bar_timestamp returned None at cursor={s.cursor}. "
                f"Timestamp column '{s.ts_col}' may be missing or corrupt."
            )

        curr_ts = pd.Timestamp(curr_ts)
        freq    = _tf_to_pandas_freq(timeframe)

        if s.cursor + 1 < len(s.df):
            next_ts = pd.Timestamp(s.get_bar_timestamp(s.cursor + 1))
            return curr_ts.floor(freq) != next_ts.floor(freq)

        # End of data: no next row to check against, fall back to arithmetic so no
        # partial candle is ever reported as closed.
        base_delta = pd.Timedelta(seconds=_tf_to_seconds(s.base_timeframe))
        return curr_ts.floor(freq) != (curr_ts + base_delta).floor(freq)

    def fetch_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int) -> list:
        """
        Return OHLCV data in the raw list-of-lists format used by BaseExchange.

        Delegates to get_window() so all aggregation and lookahead protection apply.

        Args:
            symbol    : trading symbol (unused; present for interface compatibility)
            timeframe : target timeframe
            limit     : number of bars to return

        Returns:
            List of [timestamp_ms, open, high, low, close, volume].
        """
        df     = self.get_window(limit, str(timeframe))
        result = []
        for ts, row in df.iterrows():
            result.append([
                int(pd.Timestamp(ts).timestamp() * 1000),
                row["open"], row["high"], row["low"], row["close"], row["volume"],
            ])
        return result

    def fetch_bid_ask(self, symbol: str) -> dict[str, Any]:
        """
        Derive bid/ask from the current bar's close price and the configured spread.

        Args:
            symbol : trading symbol (unused; present for interface compatibility)

        Returns:
            Dict with keys {"symbol", "bid", "ask"}.
        """
        s     = self._s
        close = float(s.current_bar["close"])
        return {
            "symbol": s.symbol,
            "bid":    close * (1.0 - s.spread),
            "ask":    close * (1.0 + s.spread),
        }
