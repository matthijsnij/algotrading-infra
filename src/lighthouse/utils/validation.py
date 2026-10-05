"""
================================================================================
VALIDATION UTILITIES
================================================================================
Lightweight, reusable DataFrame validation helpers shared across layers.

Not to be confused with backtest_data.sources.base.validate_dataframe(), which
enforces the full OHLCV schema (UTC index, exact columns, no NaNs, etc.) for
raw data sources. This module only checks that specific columns are present.

Functions:
    require_columns() : check a DataFrame has required columns and enough rows
================================================================================
"""

################# IMPORTS ##################
import pandas as pd

################# VALIDATION ##################
def require_columns(df: pd.DataFrame, required_columns: list[str], min_rows: int = 1) -> None:
    """
    Validate that a DataFrame contains required columns and sufficient rows.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        min_rows: Minimum number of rows required (default 1)

    Raises:
        ValueError: If any required column is missing or DataFrame has insufficient rows
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {', '.join(missing)}")
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows; need at least {min_rows}.")
