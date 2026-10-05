"""
================================================================================
UNIT TESTS FOR exchanges/backtest/data_sources/

Covers:
    base.py      : validate_dataframe()
    csv.py       : CSVDataSource

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_data_sources.py -v

To run a specific test function:
    pytest tests/backtest/test_data_sources.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from pathlib import Path
from lighthouse.backtest_data.sources.base import validate_dataframe, validate_bar_continuity
from lighthouse.backtest_data.sources.csv import CSVDataSource
from lighthouse.domain.calendars import CALENDAR_24_7, CALENDAR_WEEKDAY

################### HELPERS ##########################

def make_valid_df(n_bars: int = 5, timeframe_freq: str = "1h") -> pd.DataFrame:
    """Return a minimal valid OHLCV DataFrame with UTC DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=timeframe_freq, tz="UTC")
    return pd.DataFrame({
        "open":   [100.0] * n_bars,
        "high":   [101.0] * n_bars,
        "low":    [ 99.0] * n_bars,
        "close":  [100.0] * n_bars,
        "volume": [1000.0] * n_bars,
    }, index=idx)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to CSV in the format CSVDataSource expects."""
    path.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out.index.name = "timestamp"
    df_out.reset_index().to_csv(path / "1h.csv", index=False)

################### TESTS ##########################

# ── validate_dataframe() ────────────────────────────────────────────────────────

# raises on empty DataFrame
def test_validate_raises_empty_df():
    df = make_valid_df().iloc[0:0]  # empty but correct structure
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(df, timeframe="1h")


# raises when index is not a DatetimeIndex
def test_validate_raises_non_datetime_index():
    df = make_valid_df().reset_index(drop=True)  # integer index
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_dataframe(df, timeframe="1h")


# raises when DatetimeIndex is timezone-naive
def test_validate_raises_naive_datetime_index():
    df = make_valid_df()
    df.index = df.index.tz_localize(None)  # strip UTC
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_dataframe(df, timeframe="1h")


# raises when a required OHLCV column is missing
def test_validate_raises_missing_column():
    df = make_valid_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing"):
        validate_dataframe(df, timeframe="1h")


# raises when any OHLCV column contains NaN
def test_validate_raises_nan_in_ohlcv():
    df = make_valid_df()
    df.loc[df.index[2], "close"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_dataframe(df, timeframe="1h")


# raises when high < low on any row
def test_validate_raises_high_less_than_low():
    df = make_valid_df()
    df.loc[df.index[0], "high"] = 98.0  # below low of 99.0
    with pytest.raises(ValueError, match="high < low"):
        validate_dataframe(df, timeframe="1h")


# raises when close is outside [low, high] on any row
def test_validate_raises_close_outside_range():
    df = make_valid_df()
    df.loc[df.index[0], "close"] = 105.0  # above high of 101.0
    with pytest.raises(ValueError, match="close not in"):
        validate_dataframe(df, timeframe="1h")


# raises when volume is negative
def test_validate_raises_negative_volume():
    df = make_valid_df()
    df.loc[df.index[0], "volume"] = -1.0
    with pytest.raises(ValueError, match="negative volume"):
        validate_dataframe(df, timeframe="1h")


# raises on duplicate timestamps
def test_validate_raises_duplicate_timestamps():
    df = make_valid_df()
    df = pd.concat([df, df.iloc[[0]]])  # append first row again
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataframe(df, timeframe="1h")


# raises when index is not sorted ascending
def test_validate_raises_not_sorted():
    df = make_valid_df()
    df = df.iloc[::-1]  # reverse order
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_dataframe(df, timeframe="1h")


# raises when bar interval doesn't match the requested timeframe
def test_validate_raises_bar_interval_mismatch():
    df = make_valid_df(timeframe_freq="1h")  # 1h bars
    with pytest.raises(ValueError, match="Bar interval mismatch"):
        validate_dataframe(df, timeframe="1d")  # claimed to be daily


# valid DataFrame passes without raising
def test_validate_passes_valid_df():
    df = make_valid_df(timeframe_freq="1h")
    validate_dataframe(df, timeframe="1h")  # should not raise

# ── validate_bar_continuity() ────────────────────────────────────────────────

# 24/7 calendar accepts a gapless df (delta always equals the modal interval)
def test_validate_bar_continuity_24_7_accepts_gapless():
    df = make_valid_df(n_bars=5, timeframe_freq="1h")
    validate_bar_continuity(df, CALENDAR_24_7, "1h")  # should not raise


# 24/7 calendar rejects any gap, even one spanning a day boundary
def test_validate_bar_continuity_24_7_rejects_gap():
    idx = pd.DatetimeIndex(["2024-01-01 23:00", "2024-01-02 02:00", "2024-01-02 03:00"], tz="UTC")
    df = make_valid_df(n_bars=3, timeframe_freq="1h")
    df.index = idx
    with pytest.raises(ValueError, match="Illegal gap"):
        validate_bar_continuity(df, CALENDAR_24_7, "1h")


# weekday calendar accepts a Mon-Fri gappy daily df (weekend gaps are structurally legal)
def test_validate_bar_continuity_weekday_accepts_gappy_df():
    idx = pd.date_range("2024-01-08 21:00", periods=10, freq="B", tz="UTC")  # two trading weeks, timestamped at session close
    df = make_valid_df(n_bars=10, timeframe_freq="1D")
    df.index = idx
    validate_bar_continuity(df, CALENDAR_WEEKDAY, "1d")  # should not raise


# weekday calendar still rejects a hole inside a single trading day
def test_validate_bar_continuity_weekday_rejects_in_session_hole():
    idx = pd.DatetimeIndex(["2024-01-08 14:30", "2024-01-08 15:30", "2024-01-08 17:30"], tz="UTC")
    df = make_valid_df(n_bars=3, timeframe_freq="1h")
    df.index = idx
    with pytest.raises(ValueError, match="Illegal gap"):
        validate_bar_continuity(df, CALENDAR_WEEKDAY, "1h")


# reference interval is the modal delta, not the first delta: a df whose first
# gap is the atypical one (a weekend) still validates against the 1h modal interval
def test_validate_bar_continuity_uses_modal_delta_not_first_delta():
    idx = pd.DatetimeIndex(
        ["2024-01-05 20:00", "2024-01-08 14:30"]  # Friday -> Monday gap comes first
        + [f"2024-01-08 {h:02d}:30" for h in range(15, 20)],  # then five gapless 1h bars
        tz="UTC",
    )
    df = make_valid_df(n_bars=len(idx), timeframe_freq="1h")
    df.index = idx
    validate_bar_continuity(df, CALENDAR_WEEKDAY, "1h")  # should not raise


# duplicate timestamps raise regardless of calendar
def test_validate_bar_continuity_raises_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC")
    df = make_valid_df(n_bars=3, timeframe_freq="1h")
    df.index = idx
    with pytest.raises(ValueError, match="duplicate"):
        validate_bar_continuity(df, CALENDAR_24_7, "1h")


# out-of-order timestamps raise regardless of calendar
def test_validate_bar_continuity_raises_out_of_order():
    idx = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 02:00"], tz="UTC")
    df = make_valid_df(n_bars=3, timeframe_freq="1h")
    df.index = idx
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_bar_continuity(df, CALENDAR_24_7, "1h")


# an accepted gap larger than the modal interval is logged as a warning
def test_validate_bar_continuity_logs_warning_on_accepted_gap(caplog):
    idx = pd.date_range("2024-01-08 21:00", periods=10, freq="B", tz="UTC")
    df = make_valid_df(n_bars=10, timeframe_freq="1D")
    df.index = idx
    with caplog.at_level("WARNING", logger="lighthouse.backtest_data.sources.base"):
        validate_bar_continuity(df, CALENDAR_WEEKDAY, "1d")
    assert "Accepted gap" in caplog.text


# ── CSVDataSource ────────────────────────────────────────────────────────

# raises FileNotFoundError when CSV file does not exist
def test_csv_raises_file_not_found(tmp_path):
    source = CSVDataSource(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        source.load("BTC/USDT", "1h")


# loads a valid CSV and returns a DataFrame with correct shape and UTC DatetimeIndex
def test_csv_loads_valid_file(tmp_path):
    df = make_valid_df(n_bars=5, timeframe_freq="1h")
    write_csv(tmp_path / "BTC/USDT", df)

    source = CSVDataSource(data_dir=tmp_path)
    result = source.load("BTC/USDT", "1h")

    assert len(result)                == 5
    assert isinstance(result.index,    pd.DatetimeIndex)
    assert str(result.index.tz)       == "UTC"
    assert list(result.columns)       == ["open", "high", "low", "close", "volume"]


# start_date and end_date filter rows correctly
def test_csv_date_filtering(tmp_path):
    df = make_valid_df(n_bars=10, timeframe_freq="1h")
    write_csv(tmp_path / "BTC/USDT", df)

    source = CSVDataSource(data_dir=tmp_path)
    result = source.load(
        "BTC/USDT", "1h",
        start_date = "2024-01-01 02:00",
        end_date   = "2024-01-01 05:00",
    )

    assert result.index[0]  == pd.Timestamp("2024-01-01 02:00", tz="UTC")
    assert result.index[-1] == pd.Timestamp("2024-01-01 05:00", tz="UTC")
    assert len(result)      == 4  # bars at 02:00, 03:00, 04:00, 05:00
