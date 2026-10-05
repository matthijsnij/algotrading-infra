"""
================================================================================
PHEMEX EXCHANGE CLASS
================================================================================

This file contains the PhemexExchange class, which implements the BaseExchange interface for the Phemex exchange using the ccxt library.

================================================================================
"""

from __future__ import annotations

########################### IMPORTS ###################
import ccxt
import threading
from datetime import datetime
from typing import Any, ClassVar
from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.enums import normalize_side
from lighthouse.domain.exchange_types import NormalizedPosition, NormalizedOrder, Fee
from lighthouse.domain.timeframe import Timeframe
from lighthouse.utils.logging import get_logger

############################# CONSTANTS ############################
logger = get_logger(__name__)

############################# CLASS ############################
class PhemexExchange(BaseExchange):

    """
    Implementation of the Phemex exchange using the ccxt library.
    
    Attributes:
        client:                 ccxt.phemex instance.
        _lock:                  threading.Lock to ensure thread safety of API calls.
    
    Methods:
        All BaseExchange methods
        fetch_funding_payments() : fetch signed sum of funding payments via ccxt fetchFundingHistory
    """

    # Class variable indicating whether the exchange supports testnet
    supports_testnet: ClassVar[bool] = True

    # Market types Phemex supports via ccxt's "defaultType" option
    SUPPORTED_MARKET_TYPES: ClassVar[set[str]] = {"spot", "swap"}

    def __init__(self, credentials: dict[str, str], market_type: str, testnet: bool = False) -> None:

        if market_type not in self.SUPPORTED_MARKET_TYPES:
            raise ValueError(
                f"Phemex does not support market_type '{market_type}'. "
                f"Supported: {sorted(self.SUPPORTED_MARKET_TYPES)}."
            )

        # BaseExchange constructor initializes the testnet attribute
        super().__init__(testnet=testnet)
        self.market_type = market_type

        # Initialize the ccxt Phemex client with provided credentials and options
        self.client = ccxt.phemex({
            "apiKey": credentials["api_key"],
            "secret": credentials["api_secret"],
            "enableRateLimit": True,        # enable built-in rate limiter
            "options": {
                "defaultType": market_type 
            },
        })

        # Route to sandbox mode if testnet is enabled
        if self.testnet:
            self.client.set_sandbox_mode(True)
        self._lock = threading.Lock()   # create a lock for thread safety

    # ── Market data ───────────────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int) -> list:
        with self._lock:
            ohlcv = self.client.fetchOHLCV(symbol=symbol, timeframe=str(timeframe), limit=limit)
        return ohlcv

    def fetch_bid_ask(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            orderbook = self.client.fetchOrderBook(symbol=symbol)
        bid = orderbook["bids"][0][0] if len(orderbook["bids"]) > 0 else None
        ask = orderbook["asks"][0][0] if len(orderbook["asks"]) > 0 else None
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
        }

    # ── Account ───────────────────────────────────────────────────────────────
    def fetch_balance(self) -> dict[str, dict[str, float]]:
        with self._lock:
            raw = self.client.fetchBalance()
        # Keep only currency entries (ccxt mixes in metadata keys like "info", "total")
        return {
            currency: {
                "free": float(raw[currency]["free"] or 0.0),
                "used": float(raw[currency]["used"] or 0.0),
                "total": float(raw[currency]["total"]),
            }
            for currency in raw.get("total", {})
            if raw[currency].get("total") is not None
        }

    # ── Positions ─────────────────────────────────────────────────────────────
    def fetch_open_positions(self, symbol: str) -> list[NormalizedPosition]:
        with self._lock:
            raw = self.client.fetchPositions([symbol])
        return [
            NormalizedPosition(
                symbol=p["symbol"],
                side=normalize_side(p["side"]),
                size=float(p["contracts"]),  # size is represented by 'contracts' in Phemex
            )
            for p in raw
            if p.get("contracts") is not None and p["contracts"] > 0
        ]

    def get_equity(self, symbols: list[str], quote_currency: str) -> float:
        balance = self.fetch_balance()
        equity = balance.get(quote_currency, {}).get("total", 0.0)

        if self.market_type == "spot":
            # Mark-to-market wallet holdings. Mirrors execution/data.py::get_spot_equity,
            # duplicated here rather than imported to avoid an exchanges -> execution import.
            base_assets = {symbol.split("/")[0] for symbol in symbols}
            for base in base_assets:
                held = balance.get(base, {}).get("total", 0.0)
                if held <= 0.0:
                    continue
                try:
                    bid_ask = self.fetch_bid_ask(f"{base}/{quote_currency}")
                    bid, ask = bid_ask.get("bid"), bid_ask.get("ask")
                    if bid is None or ask is None:
                        raise ValueError(f"no bid/ask for {base}/{quote_currency}")
                    equity += held * (bid + ask) / 2.0
                except Exception:
                    logger.warning("get_equity: could not price %s, excluding from equity", base, exc_info=True)
        else:
            # swap: wallet balance plus unrealized PnL summed across open positions
            with self._lock:
                raw = self.client.fetchPositions(symbols)
            equity += sum(float(p["unrealizedPnl"]) for p in raw if p.get("unrealizedPnl") is not None)

        return equity

    # ── Orders ────────────────────────────────────────────────────────────────
    def place_buy_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="market", side="buy", amount=size)
        if raw.get("id") is None:
            raise ValueError(f"place_buy_market_order: exchange returned no order id for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="buy",
            type="market",
            size=float(raw["amount"]),
        )

    def place_sell_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="market", side="sell", amount=size)
        if raw.get("id") is None:
            raise ValueError(f"place_sell_market_order: exchange returned no order id for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="sell",
            type="market",
            size=float(raw["amount"]),
        )

    def place_buy_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="limit", side="buy", amount=size, price=limit_price)
        if raw.get("id") is None:
            raise ValueError(f"place_buy_limit_order: exchange returned no order id for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="buy",
            type="limit",
            size=float(raw["amount"]),
            price=float(raw["price"]) if raw["price"] else None,
        )

    def place_sell_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="limit", side="sell", amount=size, price=limit_price)
        if raw.get("id") is None:
            raise ValueError(f"place_sell_limit_order: exchange returned no order id for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="sell",
            type="limit",
            size=float(raw["amount"]),
            price=float(raw["price"]) if raw["price"] else None,
        )

    def place_buy_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="market", side="buy", amount=size, price=None, params={"triggerPrice": stop_price})
        if raw.get("id") is None:
            raise ValueError(f"place_buy_stop_order: exchange returned no order id for {symbol}")
        _trigger = raw.get("triggerPrice")
        if _trigger is None:
            raise ValueError(f"place_buy_stop_order: exchange response missing 'triggerPrice' for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="buy",
            type="stop",
            size=float(raw["amount"]),
            stop_price=float(_trigger),
        )

    def place_sell_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="market", side="sell", amount=size, price=None, params={"triggerPrice": stop_price})
        if raw.get("id") is None:
            raise ValueError(f"place_sell_stop_order: exchange returned no order id for {symbol}")
        _trigger = raw.get("triggerPrice")
        if _trigger is None:
            raise ValueError(f"place_sell_stop_order: exchange response missing 'triggerPrice' for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="sell",
            type="stop",
            size=float(raw["amount"]),
            stop_price=float(_trigger),
        )

    def place_buy_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="limit", side="buy", amount=size, price=limit_price, params={"triggerPrice": stop_price})
        if raw.get("id") is None:
            raise ValueError(f"place_buy_stop_limit_order: exchange returned no order id for {symbol}")
        _trigger = raw.get("triggerPrice")
        if _trigger is None:
            raise ValueError(f"place_buy_stop_limit_order: exchange response missing 'triggerPrice' for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="buy",
            type="stop_limit",
            size=float(raw["amount"]),
            price=float(raw["price"]) if raw["price"] else None,
            stop_price=float(_trigger),
        )

    def place_sell_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        with self._lock:
            raw = self.client.createOrder(symbol=symbol, type="limit", side="sell", amount=size, price=limit_price, params={"triggerPrice": stop_price})
        if raw.get("id") is None:
            raise ValueError(f"place_sell_stop_limit_order: exchange returned no order id for {symbol}")
        _trigger = raw.get("triggerPrice")
        if _trigger is None:
            raise ValueError(f"place_sell_stop_limit_order: exchange response missing 'triggerPrice' for {symbol}")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side="sell",
            type="stop_limit",
            size=float(raw["amount"]),
            price=float(raw["price"]) if raw["price"] else None,
            stop_price=float(_trigger),
        )

    # ── Order management ─────────────────────────────────────────────────────
    def cancel_order(self, order_id: str, symbol: str) -> None:
        with self._lock:
            self.client.cancelOrder(order_id, symbol)

    def cancel_all_orders(self, symbol: str) -> None:
        with self._lock:
            self.client.cancelAllOrders(symbol)

    def fetch_open_orders(self, symbol: str) -> list[NormalizedOrder]:
        with self._lock:
            raw = self.client.fetchOpenOrders(symbol)
        return [
            NormalizedOrder(
                id=o["id"],
                status=o["status"],
                symbol=o["symbol"],
                side=o["side"],
                type=o["type"],
                size=float(o["amount"]),
                price=float(o["price"]) if o["price"] else None,
            )
            for o in raw
        ]

    def fetch_order(self, order_id: str, symbol: str) -> NormalizedOrder:
        with self._lock:
            raw = self.client.fetchOrder(order_id, symbol)
        fee = raw.get("fee")
        return NormalizedOrder(
            id=raw["id"],
            status=raw["status"],
            symbol=raw["symbol"],
            side=raw["side"],
            type=raw["type"],
            size=float(raw["amount"]),
            filled=float(raw["filled"]) if raw.get("filled") is not None else 0.0,
            fill_price=float(raw["average"]) if raw.get("average") is not None else None,
            timestamp=raw.get("timestamp"),  # ms since epoch UTC, or None
            fee=Fee(cost=float(fee["cost"]), currency=fee["currency"]) if fee and fee.get("cost") is not None else None,
        )

    def fetch_funding_payments(self, symbol: str, start: datetime, end: datetime) -> float | None:
        # No funding history endpoint on this client -> instrument has no funding concept
        if not self.client.has.get("fetchFundingHistory"):
            return None

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        try:
            with self._lock:
                history = self.client.fetchFundingHistory(symbol, since=since_ms)
        except Exception:
            logger.warning("Failed to fetch funding history for %s", symbol, exc_info=True)
            return None

        # ccxt's fetchFundingHistory does not document a fixed sign convention for
        # 'amount' (it's exchange-specific, not unified) — as of writing this was NOT
        # verified against a real Phemex account. Assumed here: 'amount' already
        # matches our convention (positive = paid, negative = received), i.e. no sign
        # flip. VERIFY against a live Phemex funding history response before relying
        # on this in production; negate below if it turns out to be reversed.
        total = sum(
            float(entry["amount"])
            for entry in history
            if entry.get("timestamp") is not None and entry["timestamp"] <= end_ms
        )
        return total