"""
================================================================================
UNIT TESTS FOR exchanges/funding_fetchers/

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_funding_fetchers.py -v

To run a specific test function:
    pytest tests/test_funding_fetchers.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import lighthouse.backtest_data.fetchers.base as _base_module
from lighthouse.backtest_data.fetchers.base import BaseFundingFetcher
from lighthouse.backtest_data.fetchers.phemex import PhemexFundingFetcher

################### HELPERS ##########################

class _ConcreteFetcher(BaseFundingFetcher):
    """Minimal concrete implementation for testing the abstract base."""

    def fetch(self, symbol, start_date, end_date) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            ["2024-01-01 00:00:00+00:00", "2024-01-01 08:00:00+00:00"],
            name="timestamp",
        )
        return pd.DataFrame({"funding_rate": [0.0001, 0.0002]}, index=idx)


################### TESTS ##########################

# ── BaseFundingFetcher.save() ─────────────────────────────────────────────────

# save() writes a CSV file to the fixed backtest_data/funding/ directory
def test_base_save_creates_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(_base_module, "_FUNDING_DATA_DIR", tmp_path)
    fetcher = _ConcreteFetcher()
    df = fetcher.fetch("BTCUSD", "2024-01-01", "2024-01-02")

    path = fetcher.save(df, "BTCUSD")

    assert path.exists()
    assert path.name == "BTCUSD.csv"


# saved CSV contains 'timestamp' and 'funding_rate' columns and correct values
def test_base_save_csv_content(tmp_path, monkeypatch):
    monkeypatch.setattr(_base_module, "_FUNDING_DATA_DIR", tmp_path)
    fetcher = _ConcreteFetcher()
    df = fetcher.fetch("BTCUSD", "2024-01-01", "2024-01-02")
    path = fetcher.save(df, "BTCUSD")

    result = pd.read_csv(path, parse_dates=["timestamp"])
    result = result.set_index("timestamp")
    result.index = pd.to_datetime(result.index, utc=True)

    assert list(result.columns) == ["funding_rate"]
    assert result["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert result["funding_rate"].iloc[1] == pytest.approx(0.0002)


# save() creates the output directory if it does not already exist
def test_base_save_creates_missing_directory(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b" / "c"
    monkeypatch.setattr(_base_module, "_FUNDING_DATA_DIR", nested)
    fetcher = _ConcreteFetcher()
    df = fetcher.fetch("BTCUSD", "2024-01-01", "2024-01-02")

    fetcher.save(df, "BTCUSD")

    assert (nested / "BTCUSD.csv").exists()


# ── PhemexFundingFetcher._get() ───────────────────────────────────────────────

# _get() raises RuntimeError when the response body contains a non-zero code
def test_phemex_get_raises_on_api_error_code():
    fetcher = PhemexFundingFetcher()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"code": 10001, "msg": "invalid symbol", "data": None}

    with patch.object(fetcher._session, "get", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="10001"):
            fetcher._get({})


# _get() returns the parsed body when code == 0
def test_phemex_get_returns_body_on_success():
    fetcher = PhemexFundingFetcher()

    payload = {"code": 0, "msg": "", "data": [{"fundingTime": 1704067200000, "fundingRate": 0.0001}]}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = payload

    with patch.object(fetcher._session, "get", return_value=mock_resp):
        body = fetcher._get({})

    assert body == payload
