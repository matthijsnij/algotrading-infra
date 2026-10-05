"""
================================================================================
FUNDING FETCHER - PHEMEX
================================================================================

PhemexFundingFetcher retrieves historical funding rate settlements from the
Phemex public REST API and returns them in the standardized DataFrame format
expected by BaseFundingFetcher and consumed by FundingModel_Historical.
================================================================================
"""

######## IMPORTS ##################

import time
import requests
import pandas as pd
from lighthouse.utils.time import to_epoch_ms
from .base import BaseFundingFetcher

######## CONSTANTS ################

_BASE_URL    = "https://api.phemex.com"
_ENDPOINT    = "/api-data/public/data/funding-rate-history"
_PAGE_SIZE   = 100
_MAX_RETRIES = 3

######## CLASS ####################

class PhemexFundingFetcher(BaseFundingFetcher):
    """
    Fetches historical funding rates from the Phemex public REST API.

    Paginates automatically and retries transient failures (429, 5xx) with
    exponential backoff.

    Methods:
        fetch() : fetch all settlements in a date range
    """

    def __init__(self, timeout: int = 10) -> None:
        """
        Constructor. Initialize the PhemexFundingFetcher with an optional timeout.

        Args:
            timeout : HTTP request timeout in seconds (default: 10)
        """
        self._session = requests.Session()
        self._timeout = timeout

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all funding rate settlements for symbol in [start_date, end_date].

        Args:
            symbol     : Phemex funding rate index symbol
            start_date : inclusive start as ISO date string
            end_date   : inclusive end as ISO date string

        Returns:
            A pd.DataFrame with UTC DatetimeIndex ('timestamp') and float column
            'funding_rate'.

        Raises:
            RuntimeError       : if the API returns a non-zero error code
            requests.HTTPError : on a non-2xx HTTP status after retries
        """
        # Convert ISO date strings to UTC epoch milliseconds, expected by API
        cursor_ms = to_epoch_ms(start_date)
        end_ms    = to_epoch_ms(end_date)

        # Fetch all pages of funding rate settlements, accumulating them in a list
        rows: list[dict] = []
        while True:
            # GET request 
            body = self._get({"symbol": symbol, "start": cursor_ms, "end": end_ms, "limit": _PAGE_SIZE})
            data = body["data"]
            page: list[dict] = data["rows"] if isinstance(data, dict) else data
            rows.extend(page) # add page to the result list

            # Stop if this is the last page 
            if len(page) < _PAGE_SIZE:
                break

            # Update cursor to the next page (exclusive) for the next iteration
            cursor_ms = page[-1]["fundingTime"] + 1

        # Return empty if no rows were fetched
        if not rows:
            return pd.DataFrame(
                columns=["funding_rate"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )

        # Convert the list of dicts to a DataFrame with the expected format
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df.index.name = "timestamp"
        df = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
        df["funding_rate"] = df["funding_rate"].astype(float)
        return df.sort_index().drop_duplicates()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get(self, params: dict) -> dict:
        """
        GET request with exponential backoff on 429 and 5xx.

        Args:
            params : query parameters for the GET request

        Raises:
            RuntimeError       : on API-level error code
            requests.HTTPError : on non-2xx status after retries exhausted
        """
        # Construct the full URL 
        url = _BASE_URL + _ENDPOINT

        # Retry loop
        for attempt in range(_MAX_RETRIES):
            # Make the GET request with the provided params and timeout
            resp = self._session.get(url, params=params, timeout=self._timeout)

            # Retry on 429 or 5xx with exponential backoff, up to _MAX_RETRIES
            # 429 = rate limit exceeded, 5xx = server error
            if resp.status_code == 429 or resp.status_code >= 500:
                # Raise on last attempt, otherwise sleep and retry
                if attempt == _MAX_RETRIES - 1:
                    resp.raise_for_status()
                time.sleep(2 ** attempt) # Exponential backoff
                continue

            resp.raise_for_status() # passes silently on 2xx (success), raises on others
            body: dict = resp.json() # parse JSON response body to dict

            # Raise RuntimeError if the API-level code is non-zero, indicating an error
            if body.get("code", 0) != 0:
                raise RuntimeError(f"Phemex API error {body['code']}: {body.get('msg', 'unknown')}")
            
            return body


