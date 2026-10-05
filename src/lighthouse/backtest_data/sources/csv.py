"""
================================================================================
CSV DATA SOURCE
================================================================================

Loads historical OHLCV data from local CSV files into BacktestExchange.

Expected file location:
    <data_dir>/<SYMBOL>/<timeframe>.csv

Expected CSV format:
    timestamp,open,high,low,close,volume
    2024-01-01 00:00:00,42000.0,42100.0,41900.0,42050.0,1250.5
    ...
================================================================================
"""

from __future__ import annotations

####### IMPORTS #########
import pandas as pd
from pathlib import Path
from .base import BaseDataSource

########## CONSTANTS #########
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

######## CLASS #########
class CSVDataSource(BaseDataSource):
    """
    Load OHLCV data from CSV files stored under a local data directory.

    File path convention: <data_dir>/<SYMBOL>/<timeframe>.csv

    Methods:
        _load_impl(): load OHLCV data from CSV file and return as DataFrame
    """

    def __init__(self, data_dir: str | Path) -> None:
        """
        Initialize with the root data directory.

        Args:
            data_dir : path to the root backtest data folder
        """
        self._data_dir = Path(data_dir)

    def _load_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
    
        path = self._data_dir / symbol / f"{timeframe}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"CSVDataSource: no data file found at '{path}'. "
                f"Expected: <data_dir>/<symbol>/<timeframe>.csv"
            )

        df = pd.read_csv(path, parse_dates=["timestamp"]) 
        df = df.set_index("timestamp") # set timestamp as index
        df.index = pd.to_datetime(df.index, utc=True) # ensure index is in UTC
        df = df[["open", "high", "low", "close", "volume"]].astype(float) # ensure OHLCV columns are float

        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

        return df
