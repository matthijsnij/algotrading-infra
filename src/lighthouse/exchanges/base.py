"""
================================================================================
ABSTRACT BASE EXCHANGE CLASS
================================================================================

This module contains the BaseExchange class, which defines the interface that all exchanges must implement.

================================================================================
"""

from __future__ import annotations

########################### IMPORTS ###################
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from lighthouse.domain.exchange_types import NormalizedPosition, NormalizedOrder
from lighthouse.domain.timeframe import Timeframe

########################### BASE EXCHANGE CLASS ####################
class BaseExchange(ABC):
    """
    Abstract base class that every exchange must implement.

    Attributes:
        supports_testnet : ClassVar, True if the concrete exchange has a sandbox/testnet
                           environment. Default False; override on subclasses that support it.

    Methods:
        __init__()                    : validate testnet capability and store testnet flag
        fetch_ohlcv()                 : fetch historical OHLCV data
        fetch_bid_ask()               : fetch current best bid and ask prices
        fetch_balance()               : fetch available account balance
        fetch_open_positions()        : fetch open positions for a specific symbol
        get_equity()                   : fetch total account equity in quote_currency terms
        place_buy_market_order()      : place a buy market order
        place_sell_market_order()     : place a sell market order
        place_buy_limit_order()       : place a buy limit order
        place_sell_limit_order()      : place a sell limit order
        place_buy_stop_order()        : place a buy stop order
        place_sell_stop_order()       : place a sell stop order
        place_buy_stop_limit_order()  : place a buy stop-limit order
        place_sell_stop_limit_order() : place a sell stop-limit order
        cancel_order()                : cancel a single open order by ID
        cancel_all_orders()           : cancel all open orders for a symbol
        fetch_open_orders()           : fetch open orders for a specific symbol
        fetch_order()                 : fetch a single order by ID
        fetch_funding_payments()      : fetch signed sum of funding payments for a symbol in a time window
    """

    supports_testnet: ClassVar[bool] = False

    def __init__(self, *, testnet: bool = False) -> None:
        if testnet and not self.supports_testnet:
            raise NotImplementedError(
                f"{type(self).__name__} does not support a testnet/sandbox environment. "
                "Set testnet: false in the config, or pick a different exchange."
            )
        self._testnet = testnet

    @property
    def testnet(self) -> bool:
        """Whether this exchange instance is connected to a testnet/sandbox environment."""
        return self._testnet

    # ── Market data ─────────────────────────────────────────────────────────
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int) -> list:
        """
        Fetch historical candles.

        Per ADR 0001 (docs/adr/0001-last-row-is-current-period.md), the last
        candle returned is always the current period, complete or not — 'limit'
        is a plain row count, not a count of closed candles. (get_ohlcv_df() in
        execution/data.py is what turns 'limit' into "limit closed bars" for
        bot-facing callers, by requesting limit+1 here.)

        Args:
            symbol (str)         : market symbol
            timeframe (Timeframe): candle timeframe
            limit (int)          : number of candles to return

        Returns:
            A list of candles, each candle being:
            [timestamp, open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def fetch_bid_ask(self, symbol: str) -> dict[str, Any]:
        """
        Fetch the current best bid and ask prices.

        Args:
            symbol (str) : market symbol

        Returns:
            A dict with at minimum:
            {"symbol": str, "bid": float, "ask": float}
        """
        pass

    # ── Account ───────────────────────────────────────────────────────────────
    @abstractmethod
    def fetch_balance(self) -> dict[str, dict[str, float]]:
        """
        Fetch available account balance.

        Returns:
            A dict of dicts keyed by currency string, e.g.:
            {
                "USDT": { "free": float, "used": float, "total": float },
                "BTC":  { "free": float, "used": float, "total": float },
                ...
            }
        """
        pass

    # ── Positions ────────────────────────────────────────────────────────────────
    @abstractmethod
    def fetch_open_positions(self, symbol: str) -> list[NormalizedPosition]:
        """
        Fetch open positions for a specific symbol.

        Args:
            symbol (str) : market symbol

        Returns:
            A list of NormalizedPosition instances (see domain/exchange_types.py).
            Sides must be normalized to the Side enum at this boundary.
        """
        pass

    @abstractmethod
    def get_equity(self, symbols: list[str], quote_currency: str) -> float:
        """
        Fetch total account equity in quote_currency terms.

        Args:
            symbols:        trading symbols this equity figure should cover
                            (used to mark-to-market held/positioned assets)
            quote_currency: currency to express equity in, e.g. "USDT"

        Returns:
            Total equity as a float in quote_currency terms.
        """
        pass

    # ── Orders ────────────────────────────────────────────────────────────────
    @abstractmethod
    def place_buy_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        """
        Place a buy market order.

        Args:
            symbol (str) : market symbol
            size (float) : quantity to buy (in base currency)

        Returns:
            A NormalizedOrder instance (see domain/exchange_types.py).
        """
        pass

    @abstractmethod
    def place_sell_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        """
        Place a sell market order.

        Args:
            symbol (str) : market symbol
            size (float) : quantity to sell (in base currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_buy_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        """
        Place a buy limit order.

        Args:
            symbol (str)        : market symbol
            size (float)        : quantity to buy (in base currency)
            limit_price (float) : limit price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_sell_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        """
        Place a sell limit order.

        Args:
            symbol (str)        : market symbol
            size (float)        : quantity to sell (in base currency)
            limit_price (float) : limit price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_buy_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        """
        Place a buy stop order.

        Args:
            symbol (str)       : market symbol
            size (float)       : quantity to buy (in base currency)
            stop_price (float) : trigger price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_sell_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        """
        Place a sell stop order.

        Args:
            symbol (str)       : market symbol
            size (float)       : quantity to sell (in base currency)
            stop_price (float) : trigger price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_buy_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        """
        Place a buy stop-limit order.

        Args:
            symbol (str)        : market symbol
            size (float)        : quantity to buy (in base currency)
            stop_price (float)  : trigger price (in quote currency)
            limit_price (float) : limit price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    @abstractmethod
    def place_sell_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        """
        Place a sell stop-limit order.

        Args:
            symbol (str)        : market symbol
            size (float)        : quantity to sell (in base currency)
            stop_price (float)  : trigger price (in quote currency)
            limit_price (float) : limit price (in quote currency)

        Returns:
            A NormalizedOrder instance. See place_buy_market_order() for the contract.
        """
        pass

    # ── Order management ─────────────────────────────────────────────────────

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> None:
        """
        Cancel a single open order by ID.

        Args:
            order_id (str) : exchange-assigned order identifier
            symbol (str)   : market symbol the order was placed on
        """
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol (str) : market symbol
        """
        pass

    @abstractmethod
    def fetch_open_orders(self, symbol: str) -> list[NormalizedOrder]:
        """
        Fetch open orders for a specific symbol. Used for checking pending orders.

        Args:
            symbol (str) : market symbol

        Returns:
            A list of NormalizedOrder instances (see domain/exchange_types.py). Note
            that filled, fill_price, timestamp, and fee are intentionally absent/None,
            open orders have not filled yet. Use fetch_order() for completed order data.
        """
        pass

    @abstractmethod
    def fetch_order(self, order_id: str, symbol: str) -> NormalizedOrder:
        """
        Fetch a single order by ID. Used for getting information on fills like actual fill price, execution time, fees.

        Args:
            order_id (str) : exchange-assigned order identifier
            symbol (str)   : market symbol the order was placed on

        Returns:
            A NormalizedOrder instance (see domain/exchange_types.py), with filled,
            fill_price, timestamp, and fee populated where available.
        """
        pass

    def fetch_funding_payments(self, symbol: str, start: datetime, end: datetime) -> float | None:
        """
        Fetch the signed sum of account funding payments for a symbol over a time window.

        Not abstract: exchanges/instruments without a funding concept (spot, dated
        futures) need no override and simply inherit this default.

        Args:
            symbol (str)     : market symbol
            start (datetime) : UTC start of the window (inclusive)
            end (datetime)   : UTC end of the window (inclusive)

        Returns:
            Signed sum of funding payments in settlement_currency, where positive means
            paid and negative means received. None if the exchange/instrument has no
            funding concept or the data could not be retrieved.
        """
        return None