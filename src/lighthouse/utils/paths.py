"""
================================================================================
RUNTIME PATHS
================================================================================
Centralized path resolution for the lighthouse package.

Functions:
    logs_dir() : directory for log files (trades and events)
    backtest_data_dir() : directory for OHLCV/funding CSVs
    state_dir() : directory for live state persistence
================================================================================
"""

from pathlib import Path

def logs_dir() -> Path:
    """
    Return the root directory for all log files.

    This is a single directory for all logs, not one per bot or per session.
    """
    return Path(__file__).resolve().parents[3] / "logs"

def backtest_data_dir() -> Path:
    """
    Return the root directory for all backtest data.
    """
    return Path(__file__).resolve().parents[3] / "backtest_data"

def state_dir() -> Path:
    """
    Return the root directory for all state files.
    """
    return Path(__file__).resolve().parents[3] / "state"