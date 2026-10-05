"""
================================================================================
UTILITY FUNCTIONS
================================================================================
"""

######## IMPORTS ##################

from datetime import datetime, timezone

######## FUNCTIONS ##################

def to_utc_datetime(value: str | datetime) -> datetime:
    """
    Parse a date string or datetime to a UTC-aware datetime.
    
    Args:
        value : ISO date string or datetime object

    Returns:
        A UTC-aware datetime object.
    """
    if isinstance(value, str):
        dt = datetime.fromisoformat(value)
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def to_epoch_ms(date_str: str) -> int:
    """
    Convert an ISO date string to UTC epoch milliseconds.
    
    Args:
        date_str : ISO date string

    Returns:
        UTC epoch milliseconds as an integer.
    """
    return int(to_utc_datetime(date_str).timestamp() * 1000)