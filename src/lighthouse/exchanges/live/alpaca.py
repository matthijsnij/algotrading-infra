"""
================================================================================
ALPACA EXCHANGE CLASS
================================================================================

This file contains the AlpacaExchange class, which implements the BaseExchange interface for the Alpaca brokerage using the alpaca-py SDK. Stock tickers (e.g. "AAPL") are passed through natively with no symbol-mapping layer.

================================================================================
"""

from __future__ import annotations

########################### IMPORTS ###################
import threading
from typing import Any, ClassVar

from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)
from lighthouse.domain.enums import normalize_side
from lighthouse.domain.exchange_types import NormalizedOrder, NormalizedPosition
from lighthouse.domain.timeframe import Timeframe, TimeframeUnit
from lighthouse.exchanges.base import BaseExchange

############################# CONSTANTS ############################

# Maps a Timeframe's unit to alpaca-py's TimeFrameUnit
_TIMEFRAME_UNITS = {
    TimeframeUnit.MINUTE: TimeFrameUnit.Minute,
    TimeframeUnit.HOUR: TimeFrameUnit.Hour,
    TimeframeUnit.DAY: TimeFrameUnit.Day,
    TimeframeUnit.WEEK: TimeFrameUnit.Week,
}


############################# HELPERS ############################
def _to_alpaca_timeframe(timeframe: Timeframe) -> TimeFrame:
    return TimeFrame(timeframe.amount, _TIMEFRAME_UNITS[timeframe.unit])


def _whole_shares(size: float) -> float:
    """Limit/stop orders require whole shares; fractional shares are market-order only."""
    return float(max(1, round(size)))


def _base_order_kwargs(raw: AlpacaOrder) -> dict[str, Any]:
    return {
        "id": str(raw.id),
        "status": raw.status.value,
        "symbol": raw.symbol,
        "side": raw.side.value,
        "type": raw.type.value,
        "size": float(raw.qty),
    }


############################# CLASS ############################
class AlpacaExchange(BaseExchange):

    """
    Implementation of the Alpaca brokerage using the alpaca-py SDK.

    Attributes:
        client:                 alpaca-py TradingClient instance.
        data_client:            alpaca-py StockHistoricalDataClient instance.
        _lock:                  threading.Lock to ensure thread safety of API calls.

    Methods:
        All BaseExchange methods.
        fetch_funding_payments() inherited unchanged from BaseExchange: stocks have no funding concept.
    """

    supports_testnet: ClassVar[bool] = True

    # Alpaca only trades US equities in this codebase; no swap/margin market types
    SUPPORTED_MARKET_TYPES: ClassVar[set[str]] = {"stock"}

    def __init__(self, credentials: dict[str, str], market_type: str, testnet: bool = False) -> None:

        if market_type not in self.SUPPORTED_MARKET_TYPES:
            raise ValueError(
                f"Alpaca does not support market_type '{market_type}'. "
                f"Supported: {sorted(self.SUPPORTED_MARKET_TYPES)}."
            )

        # BaseExchange constructor initializes the testnet attribute
        super().__init__(testnet=testnet)
        self.market_type = market_type

        self.client = TradingClient(
            api_key=credentials["api_key"],
            secret_key=credentials["api_secret"],
            paper=self.testnet,
        )
        # Market data feed is the same (IEX on the free tier) for paper and live accounts
        self.data_client = StockHistoricalDataClient(
            api_key=credentials["api_key"],
            secret_key=credentials["api_secret"],
        )
        self._lock = threading.Lock()   # create a lock for thread safety

    # ── Market data ───────────────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int) -> list:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_to_alpaca_timeframe(timeframe),
            limit=limit,
            feed=DataFeed.IEX,
            sort=Sort.DESC,
        )
        with self._lock:
            bars = self.data_client.get_stock_bars(request)
        candles = bars.data.get(symbol, [])
        return [
            [int(bar.timestamp.timestamp() * 1000), bar.open, bar.high, bar.low, bar.close, bar.volume]
            for bar in reversed(candles)  # requested desc (latest-first) for the limit to apply to recent bars
        ]

    def fetch_bid_ask(self, symbol: str) -> dict[str, Any]:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        with self._lock:
            quotes = self.data_client.get_stock_latest_quote(request)
        quote = quotes[symbol]
        return {
            "symbol": symbol,
            "bid": quote.bid_price,
            "ask": quote.ask_price,
        }

    # ── Account ───────────────────────────────────────────────────────────────
    def fetch_balance(self) -> dict[str, dict[str, float]]:
        with self._lock:
            account = self.client.get_account()
        cash = float(account.cash)
        equity = float(account.equity)
        return {
            "USD": {
                "free": cash,
                "used": max(equity - cash, 0.0),
                "total": equity,
            }
        }

    # ── Positions ─────────────────────────────────────────────────────────────
    def fetch_open_positions(self, symbol: str) -> list[NormalizedPosition]:
        with self._lock:
            try:
                position = self.client.get_open_position(symbol)
            except APIError:
                return []  # Alpaca raises when there is no open position for the symbol
        return [
            NormalizedPosition(
                symbol=position.symbol,
                side=normalize_side(position.side.value),
                size=abs(float(position.qty)),
            )
        ]

    def get_equity(self, symbols: list[str], quote_currency: str) -> float:
        # Alpaca account equity is already all-in (cash + positions marked to market),
        # unlike ccxt spot accounts (see PhemexExchange.get_equity) -- no calculation needed
        with self._lock:
            account = self.client.get_account()
        return float(account.equity)

    # ── Orders ────────────────────────────────────────────────────────────────
    def place_buy_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        request = MarketOrderRequest(symbol=symbol, qty=size, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(**_base_order_kwargs(raw))

    def place_sell_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        request = MarketOrderRequest(symbol=symbol, qty=size, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(**_base_order_kwargs(raw))

    def place_buy_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        request = LimitOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, limit_price=limit_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            price=float(raw.limit_price) if raw.limit_price is not None else None,
        )

    def place_sell_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        request = LimitOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY, limit_price=limit_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            price=float(raw.limit_price) if raw.limit_price is not None else None,
        )

    def place_buy_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        request = StopOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, stop_price=stop_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
        )

    def place_sell_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        request = StopOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY, stop_price=stop_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
        )

    def place_buy_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        request = StopLimitOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, stop_price=stop_price, limit_price=limit_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            price=float(raw.limit_price) if raw.limit_price is not None else None,
            stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
        )

    def place_sell_stop_limit_order(self, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder:
        request = StopLimitOrderRequest(
            symbol=symbol, qty=_whole_shares(size), side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY, stop_price=stop_price, limit_price=limit_price,
        )
        with self._lock:
            raw = self.client.submit_order(request)
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            price=float(raw.limit_price) if raw.limit_price is not None else None,
            stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
        )

    # ── Order management ─────────────────────────────────────────────────────
    def cancel_order(self, order_id: str, symbol: str) -> None:
        # symbol is unused: Alpaca cancels by order id alone, kept for BaseExchange interface parity
        with self._lock:
            self.client.cancel_order_by_id(order_id)

    def cancel_all_orders(self, symbol: str) -> None:
        # Alpaca has no per-symbol bulk-cancel endpoint; cancel each open order individually
        for order in self.fetch_open_orders(symbol):
            self.cancel_order(order.id, symbol)

    def fetch_open_orders(self, symbol: str) -> list[NormalizedOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        with self._lock:
            raw_orders = self.client.get_orders(request)
        return [
            NormalizedOrder(
                **_base_order_kwargs(raw),
                price=float(raw.limit_price) if raw.limit_price is not None else None,
                stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
            )
            for raw in raw_orders
        ]

    def fetch_order(self, order_id: str, symbol: str) -> NormalizedOrder:
        with self._lock:
            raw = self.client.get_order_by_id(order_id)
        fill_time = raw.filled_at or raw.submitted_at
        return NormalizedOrder(
            **_base_order_kwargs(raw),
            price=float(raw.limit_price) if raw.limit_price is not None else None,
            stop_price=float(raw.stop_price) if raw.stop_price is not None else None,
            filled=float(raw.filled_qty) if raw.filled_qty is not None else 0.0,
            fill_price=float(raw.filled_avg_price) if raw.filled_avg_price is not None else None,
            timestamp=int(fill_time.timestamp() * 1000) if fill_time is not None else None,
            # fee is left None: Alpaca equities trading is commission-free
        )
