"""
================================================================================
FUNDING MODEL - HISTORICAL FUNDING RATES FROM EXCHANGE
================================================================================

The HistoricalFundingModel class replays real historical funding rates during a backtest.
Historical funding rates are loaded from a CSV file (as written by BaseFundingFetcher.save()).
================================================================================
"""

######## IMPORTS ##################

from pathlib import Path
import numpy as np
import pandas as pd
from .base import BaseFundingModel

######## CLASS ####################

class FundingModel_Historical(BaseFundingModel):
    """
    Funding rate model that replays real historical funding rates.

    On construction, a pd.Series of (timestamp → rate) is forward-filled and
    aligned to the OHLCV bar DatetimeIndex so that compute_rate() is a simple
    array lookup. Bars before the first available funding rate fall back to
    base_funding_rate.

    Attributes:
        _rates : numpy float array of pre-aligned rates, shape (n_bars,).
                 Coverage is guaranteed by the constructor's range check.

    Methods:
        from_csv()     : classmethod constructor, load rates from a CSV file
        compute_rate() : return the historical rate for the given bar
    """

    def __init__(self, rates: pd.Series, bar_timestamps: pd.DatetimeIndex) -> None:
        """
        Align funding rates to bar timestamps and store as a pre-computed array.

        Args:
            rates          : pd.Series with a timezone-aware (UTC) DatetimeIndex.
                             Values are unscaled funding rates.
            bar_timestamps : timezone-aware (UTC) DatetimeIndex of the OHLCV bars
                             used in the backtest.

        Raises:
            ValueError : if either index is timezone-naive, or if funding data does
                         not fully cover the backtest period (start and end times).
        """

        # Validate that both indices are timezone-aware (UTC)
        if rates.index.tz is None:
            raise ValueError(
                "FundingModel_Historical: 'rates' index must be timezone-aware (UTC). "
                "Use pd.to_datetime(..., utc=True) before constructing."
            )
        if bar_timestamps.tz is None:
            raise ValueError(
                "FundingModel_Historical: 'bar_timestamps' must be timezone-aware (UTC). "
                "Ensure the OHLCV DatetimeIndex has UTC timezone set."
            )

        # Validate that funding data fully covers the backtest period
        if rates.index.min() > bar_timestamps.min():
            raise ValueError(
                f"FundingModel_Historical: funding data starts at {rates.index.min()} "
                f"but backtest starts at {bar_timestamps.min()}. "
                f"Fetch funding data earlier or start backtest later."
            )
        if rates.index.max() < bar_timestamps.max():
            raise ValueError(
                f"FundingModel_Historical: funding data ends at {rates.index.max()} "
                f"but backtest ends at {bar_timestamps.max()}. "
                f"Fetch more funding data or end backtest earlier."
            )

        # Combined index of all bar timestamps and all funding rate timestamps
        combined = bar_timestamps.union(rates.index)

        # Forward-fill funding rates onto the combined index
        aligned  = rates.reindex(combined).ffill()

        # Re-index to keep only the bar timestamps
        aligned  = aligned.reindex(bar_timestamps)

        # Convert to a numpy array
        self._rates: np.ndarray = aligned.to_numpy(dtype=float)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        bar_timestamps: pd.DatetimeIndex,
    ) -> "FundingModel_Historical":
        """
        Construct a FundingModel_Historical from a funding rate CSV file.

        Expected CSV format (as written by BaseFundingFetcher.save()):
            timestamp,funding_rate
            2024-01-01 00:00:00+00:00,0.0001
            2024-01-01 08:00:00+00:00,0.00012
            ...

        Args:
            path           : path to the funding rate CSV file
            bar_timestamps : timezone-aware (UTC) DatetimeIndex of the OHLCV bars

        Returns:
            A FundingModel_Historical instance
        """
        if not path.exists():
            raise FileNotFoundError(
                f"FundingModel_Historical: funding rate CSV not found at '{path}'."
            )

        df = pd.read_csv(path, index_col="timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

        if "funding_rate" not in df.columns:
            raise ValueError(
                f"FundingModel_Historical: CSV at '{path}' must contain a "
                "'funding_rate' column."
            )

        rates = df["funding_rate"].astype(float)
        return cls(rates, bar_timestamps)

    def compute_rate(self, bar_idx: int, close_price: float, base_funding_rate: float) -> float:
        """
        Return the historical funding rate for the current bar.

        Under normal construction the coverage check in __init__ ensures all bars
        have a valid rate. The NaN fallback to base_funding_rate is a last-resort
        guard against data corruption and should never be reached in practice.

        Args:
            bar_idx           : current bar index (cursor position)
            close_price       : current bar's close price (unused)
            base_funding_rate : fallback rate from InstrumentSpec

        Returns:
            Signed funding rate for this bar.
        """
        rate = self._rates[bar_idx]
        if np.isnan(rate):
            return base_funding_rate
        return float(rate)
