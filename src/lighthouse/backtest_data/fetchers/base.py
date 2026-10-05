"""
================================================================================
FUNDING FETCHER - ABSTRACT BASE CLASS
================================================================================

Abstract base class for all exchange-specific funding rate fetchers.

Fetchers are offline data-acquisition utilities. They pull historical funding
rates from an exchange and save them to a standardized CSV. They have no role
during backtest execution as that is handled by FundingModel_Historical.

================================================================================
"""

######## IMPORTS ##################

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from lighthouse.utils.paths import backtest_data_dir

######## CONSTANTS ################

_FUNDING_DATA_DIR = backtest_data_dir() / "funding"

######## CLASS ####################

class BaseFundingFetcher(ABC):
    """
    Abstract interface for fetching historical funding rates from an exchange.

    Subclasses implement fetch() for a specific exchange REST API. The shared
    save() method writes the result to the standardized CSV format consumed by
    FundingModel_Historical.

    Methods:
        fetch() : pull historical funding rates from the exchange (abstract)
        save()  : write a funding rate DataFrame to a CSV file
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates for a symbol over a date range.

        Args:
            symbol     : market symbol 
            start_date : inclusive start, as ISO date string (UTC)
            end_date   : inclusive end, as ISO date string (UTC)

        Returns:
            A pd.DataFrame with:
                - UTC DatetimeIndex named 'timestamp' (one row per settlement)
                - Single float column 'funding_rate' (unscaled, e.g. 0.0001 = 0.01%)
        """
        ...

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        """
        Save a funding rate DataFrame to backtest_data/funding/<symbol>.csv.

        Args:
            df     : DataFrame as returned by fetch()
            symbol : market symbol, used as the CSV filename stem

        Returns:
            The absolute Path object of the written CSV file.
        """
        _FUNDING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _FUNDING_DATA_DIR / f"{symbol}.csv"
        df.to_csv(path, index_label="timestamp")
        return path
