"""
================================================================================
ORDER BOOK
================================================================================

This file contains the OrderBook class, which manages the resting order book and queued market orders for the BacktestExchange simulation engine.

It handles registration, evaluation, cancellation, and routing to FillEngine.
All order type constants are defined here and re-exported for use in exchange.py.
================================================================================
"""

from __future__ import annotations

######## IMPORTS ##########
import uuid
from typing import Any
from .state import BacktestState
from .fill_engine import FillEngine

########## CONSTANTS ##########
TYPE_MARKET     = "market"
TYPE_LIMIT      = "limit"
TYPE_STOP       = "stop"
TYPE_STOP_LIMIT = "stop_limit"

########### CLASS ##########
class OrderBook:
    """
    Manages the resting order book and queued market orders.

    Responsibilities:
      - Register new resting stop/limit orders with is_entry flag
      - Queue market orders for latency-delayed execution
      - Process queued market orders on each bar (step 1 of advance())
      - Evaluate resting orders each bar (step 2 of advance()); routes fills
        to FillEngine
      - Cancel individual or all orders
      - Expose order lists via fetch_open_orders()

    Methods:
        queue_market_order() : create a market order and queue it for latency-delayed execution
        fill_queued_market_orders(): fill any queued market orders whose latency has elapsed
        register_order()     : register a new resting stop or limit/stop-limit order
        check_pending_orders(): evaluate all resting orders against the current bar OHLCV
        cancel_order()       : remove a resting or queued order by ID
        cancel_all_orders()  : remove all resting and queued orders
        fetch_open_orders()  : return all open orders with internal fields stripped
    """

    def __init__(self, state: BacktestState, fill_engine: FillEngine) -> None:
        """
        Constructor. Initialize the OrderBook with shared simulation state and fill engine.

        Args:
            state       : BacktestState instance (shared by all engines)
            fill_engine : FillEngine instance for routing fills
        """
        self._s     = state
        self._fills = fill_engine

    # ── Market order routing ──────────────────────────────────────────────────

    def queue_market_order(self, symbol: str, side: str, size: float) -> dict[str, Any]:
        """
        Create a market order and queue it for latency-delayed execution.

        All market orders are queued for execution at the open of cursor + latency_bars.
        This prevents look-ahead bias: the order is placed during analysis of bar N,
        and fills at bar N+latency_bars' open (never at bar N's price).

        Args:
            symbol : trading symbol
            side   : "buy" or "sell"
            size   : order size

        Returns:
            Normalized order dict.
        """
        s = self._s

        # can_short guard
        if side == "sell" and not s.instrument.can_short:
            if s.position is None or s.position["side"] != "long":
                raise ValueError(
                    "BacktestExchange: instrument does not support short selling "
                    "(can_short=False). A market sell order requires an open long position."
                )

        # Create order dict
        order_id = str(uuid.uuid4())
        order = {
            "id":          order_id,
            "status":      "open",
            "symbol":      symbol,
            "side":        side,
            "type":        TYPE_MARKET,
            "size":        size,
            "price":       None,
            "stop_price":  None,
            "fill_at_bar": s.cursor + s.latency_bars,
        }

        # Queue for latency-delayed execution at bar open
        s.queued_market_orders.append(order)
        return order

    def fill_queued_market_orders(self) -> None:
        """
        Fill any queued market orders whose latency has elapsed.

        Fills at the current bar's OPEN ± spread ± slippage (adverse direction).
        Called as step 1 of advance().
        """
        s             = self._s
        still_pending = []
        for order in s.queued_market_orders:
            if order["fill_at_bar"] <= s.cursor:
                open_price = float(s.current_bar["open"])
                if order["side"] == "buy":
                    fill_price = open_price * (1.0 + s.spread + s.slippage)
                else:
                    fill_price = open_price * (1.0 - s.spread - s.slippage)
                self._fills.execute_market_fill(order, fill_price)
            else:
                still_pending.append(order)
        s.queued_market_orders = still_pending

    # ── Resting order registration ────────────────────────────────────────────

    def register_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Register a new resting stop or limit/stop-limit order.

        Sets is_entry=True if placed while flat (entry intent),
        is_entry=False if placed with an open position (exit intent).
        This flag is used by FillEngine to prevent orphaned exit orders
        from opening new positions.

        Args:
            symbol     : market symbol
            side       : "buy" or "sell"
            order_type : TYPE_LIMIT, TYPE_STOP, or TYPE_STOP_LIMIT
            size       : order size
            price      : limit price (limit and stop-limit orders)
            stop_price : trigger price (stop and stop-limit orders)

        Returns:
            Normalized order dict (including internal is_entry field).
        """
        s = self._s
        if order_type not in (TYPE_LIMIT, TYPE_STOP, TYPE_STOP_LIMIT):
            raise ValueError(
                f"BacktestExchange: unsupported order_type '{order_type}'. "
                f"Supported: {TYPE_LIMIT}, {TYPE_STOP}, {TYPE_STOP_LIMIT}"
            )

        order_id = str(uuid.uuid4())
        order = {
            "id":         order_id,
            "status":     "open",
            "symbol":     symbol,
            "side":       side,
            "type":       order_type,
            "size":       size,
            "price":      price,
            "stop_price": stop_price,
            "is_entry":   s.position is None,  # True = entry intent; False = exit intent
        }
        s.open_orders[order_id] = order
        return order

    # ── Resting order evaluation ──────────────────────────────────────────────

    def check_pending_orders(self) -> None:
        """
        Evaluate all resting orders against the current bar OHLCV.

        Processing order:
          1. Stop/stop-limit orders (SL before TP, conservative worst case)
          2. Limit orders

        Gap protection for stops:
          - Buy stop  fills at max(stop_price, bar_open) * (1 + slippage)
          - Sell stop fills at min(stop_price, bar_open) * (1 - slippage)

        Partial fill guard for limits (when partial_fill_fraction < 1.0):
          - Only fills when bar_volume * partial_fill_fraction >= order_size

        Triggered orders are passed to FillEngine.fill_pending_order().
        Called as step 2 of advance().
        """
        s = self._s
        if not s.open_orders:
            return

        bar        = s.current_bar
        high       = float(bar["high"])
        low        = float(bar["low"])
        open_price = float(bar["open"])
        volume     = float(bar["volume"])

        # Stops first, then limits
        orders = sorted(
            s.open_orders.values(),
            key=lambda o: 0 if o["type"] in (TYPE_STOP, TYPE_STOP_LIMIT) else 1,
        )

        for order in orders:
            if order["id"] not in s.open_orders:
                # Already filled earlier this bar (e.g., SL closed position)
                continue

            side       = order["side"]
            order_type = order["type"]

            # ── Stop and stop-limit orders ────────────────────────────────────
            if order_type in (TYPE_STOP, TYPE_STOP_LIMIT):
                stop_price = order["stop_price"]
                if side == "buy" and high >= stop_price:
                    fill_price  = max(stop_price, open_price)
                    fill_price *= (1.0 + s.slippage)
                    self._fills.fill_pending_order(order, fill_price, is_maker=False)
                elif side == "sell" and low <= stop_price:
                    fill_price  = min(stop_price, open_price)
                    fill_price *= (1.0 - s.slippage)
                    self._fills.fill_pending_order(order, fill_price, is_maker=False)

            # ── Limit orders ──────────────────────────────────────────────────
            elif order_type == TYPE_LIMIT:
                limit_price = order["price"]

                # Partial fill guard
                if s.partial_fill_fraction < 1.0:
                    if volume * s.partial_fill_fraction < order["size"]:
                        continue  # Defer — insufficient volume this bar

                if s.limit_fill_policy == "through":
                    # Conservative: price must trade THROUGH limit by one tick
                    tick = s.instrument.tick_size
                    if side == "buy" and low < limit_price - tick:
                        self._fills.fill_pending_order(order, limit_price, is_maker=True)
                    elif side == "sell" and high > limit_price + tick:
                        self._fills.fill_pending_order(order, limit_price, is_maker=True)
                else:
                    # "touch" — legacy optimistic: fills when price touches the limit
                    if side == "buy" and low <= limit_price:
                        self._fills.fill_pending_order(order, limit_price, is_maker=True)
                    elif side == "sell" and high >= limit_price:
                        self._fills.fill_pending_order(order, limit_price, is_maker=True)

    # ── Cancellation and fetch ────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> None:
        """Remove a resting or queued order by ID."""
        s = self._s
        s.open_orders.pop(order_id, None)
        s.queued_market_orders = [o for o in s.queued_market_orders if o["id"] != order_id]

    def cancel_all_orders(self) -> None:
        """Remove all resting and queued orders."""
        s = self._s
        s.open_orders.clear()
        s.queued_market_orders.clear()

    def fetch_open_orders(self) -> list[dict[str, Any]]:
        """
        Return all open orders with internal fields stripped.

        Strips:
          - is_entry  from resting orders
          - fill_at_bar from queued market orders

        Returns:
            Combined list of resting + queued order dicts.
        """
        s = self._s
        resting = [
            {k: v for k, v in o.items() if k != "is_entry"}
            for o in s.open_orders.values()
        ]
        queued = [
            {k: v for k, v in o.items() if k != "fill_at_bar"}
            for o in s.queued_market_orders
        ]
        return resting + queued
