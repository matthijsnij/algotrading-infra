"""
================================================================================
DATA MODULE 
================================================================================

This module contains functions related to fetching market and account data.

Functions:
    get_ohlcv_df(): fetches OHLCV data and returns a clean DataFrame
    get_bid_ask(): fetches the current best bid and ask prices for a symbol
    get_balance(): fetches the current account balance
    get_spot_equity(): calculates total portfolio equity in a quote currency (quote balance + mark-to-market of held assets)

================================================================================
"""

################# IMPORTS ##################
from typing import Any

import pandas as pd

from lighthouse.domain.timeframe import Timeframe
from lighthouse.exchanges.base import BaseExchange
from lighthouse.utils.logging import get_logger

logger = get_logger(__name__)

################# FUNCTIONS ##################

# ── OHLCV market data ─────────────────────────────────────────────────────────
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
def get_ohlcv_df(exchange: BaseExchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Convert a raw exchange OHLCV list to a clean pandas DataFrame. Wraps around BaseExchange.fetch_ohlcv().
    
    Args:
        exchange:  any BaseExchange instance
        symbol:    market symbol
        timeframe: timeframe string e.g. "15m", "1h"
        limit:     number of closed candles to return

    Returns:
        A DataFrame with columns: ["open", "high", "low", "close", "volume"],
        a datetime index in UTC, and limit+1 rows: limit closed bars plus the
        current (possibly still-forming) bar as the last row (see ADR 0001).
    
    Raises:
    """
    try:
        exchange_ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=Timeframe.parse(timeframe), limit=limit + 1)
    except Exception as exc:
        raise ValueError(
            f"Could not build OHLCV DataFrame for {symbol}: {exc}"
        ) from exc

    if not isinstance(exchange_ohlcv, list):
        raise ValueError(
            f"Expected fetch_ohlcv to return a list, got {type(exchange_ohlcv).__name__} for {symbol}"
        )

    # Create DataFrame, parse timestamps, and set index
    df = pd.DataFrame(exchange_ohlcv, columns=OHLCV_COLUMNS)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True) # from milliseconds to datetime in UTC
    except Exception as exc:
        raise ValueError(f"Failed to parse OHLCV timestamps for {symbol}: {exc}") from exc
    df.set_index("timestamp", inplace=True)
    df.index.name = "timestamp (UTC)" 

    return df

# ── Bid, ask ─────────────────────────────────────────────────────────
def get_bid_ask(exchange: BaseExchange, symbol: str) -> dict[str, Any]:
    """
    Fetch the current best bid and ask prices for a symbol.

    Args:
        exchange: any BaseExchange instance
        symbol:   trading symbol

    Returns:
        A dict with keys: {"symbol": str, "bid": float, "ask": float, "spread": float}
    """
    try:
        bid_ask = exchange.fetch_bid_ask(symbol)
    except Exception as exc:
        raise ValueError(
            f"Could not fetch bid/ask for {symbol}: {exc}"
        ) from exc
    
    # Guard against None values
    raw_bid = bid_ask.get("bid")
    raw_ask = bid_ask.get("ask")
    if raw_bid is None or raw_ask is None:
        logger.warning(
            "Orderbook for %s returned None for bid or ask — market may be closed or illiquid",
            symbol,
        )
        raise ValueError(
            f"Orderbook for {symbol} returned None for bid or ask — market may be closed or illiquid"
        )
    
    # Cast to float and calculate spread
    bid = float(raw_bid)
    ask = float(raw_ask)
    spread = ask - bid

    # Return dict with spread
    return {
        "symbol": symbol, 
        "bid": bid, 
        "ask": ask,
        "spread": spread
    }

# ── Balance ─────────────────────────────────────────────────────────
def get_balance(exchange: BaseExchange) -> dict[str, dict[str, float]]:
    """
    Fetch the current account balance.

    Args:
        exchange: any BaseExchange instance

    Returns:
        A dict with structure: {"currency": {"free": float, "used": float, "total": float}}
    """
    try:
        balance = exchange.fetch_balance()
    except Exception as exc:
        raise ValueError(
            f"Could not fetch balance: {exc}"
        ) from exc
    return balance


def get_spot_equity(exchange: BaseExchange, symbols: list[str], quote_currency: str) -> float:
    """
    Calculate total portfolio equity in quote_currency terms.

    Sums the quote currency cash balance with the mark-to-market value of all
    base asset holdings derived from the given symbols. Uses the mid price
    (average of bid and ask) to value each held asset.

    This gives an accurate real-time equity figure that stays flat on trade
    entry and only moves with actual p&l.

    Helper for ccxt-style live adapters only: derives base assets by
    splitting each symbol on "/" (e.g. "BTC/USDT:USDT" -> "BTC"), which
    requires ccxt-format symbols. Shared code should call
    `exchange.get_equity(symbols, quote_currency)` instead — that
    abstraction lets each exchange (backtest, non-ccxt live adapters, etc.)
    compute equity in whatever way fits its own account model.

    Args:
        exchange:       any BaseExchange instance
        symbols:        list of trading pair strings, e.g. ["BTC/USDT", "ETH/USDT"]
                        also accepts futures-style pairs like "BTC/USDT:USDT"
        quote_currency: the currency to express equity in, e.g. "USDT"

    Returns:
        Total portfolio equity as a float in quote_currency terms.
    """
    balance = get_balance(exchange)

    # Start with the cash balance in the quote currency
    equity = balance.get(quote_currency, {}).get("total", 0.0)

    # Derive unique base assets from symbols (e.g. "BTC/USDT:USDT" -> "BTC")
    base_assets = set()
    for symbol in symbols:
        base = symbol.split("/")[0]
        base_assets.add(base)

    # Mark-to-market each held base asset
    failed_assets = [] # to store any assets we fail to price
    for base in base_assets:
        held = balance.get(base, {}).get("total", 0.0)
        if held <= 0.0:
            continue
        # Fetch mid price for this asset against the quote currency
        pair = f"{base}/{quote_currency}"
        try:
            bid_ask = get_bid_ask(exchange, pair)
            mid_price = (bid_ask["bid"] + bid_ask["ask"]) / 2.0
        except Exception as exc:
            logger.error(
                "get_spot_equity: could not price %s, excluding from equity | error=%s",
                pair, exc,
            )
            failed_assets.append(base)
            continue
        equity += held * mid_price
        logger.debug(
            "get_spot_equity: %s held=%.6f mid=%.4f value=%.4f %s",
            base, held, mid_price, held * mid_price, quote_currency,
        )

    if failed_assets:
        logger.warning(
            "get_spot_equity: equity may be understated — "
            "could not price %d asset(s): %s",
            len(failed_assets), failed_assets,
        )

    logger.debug("get_spot_equity: total equity=%.4f %s", equity, quote_currency)
    return equity

