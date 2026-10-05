"""
================================================================================
FILL ENGINE
================================================================================

This file contains the FillEngine class, which executes all trade fills and maintains position and balance accounting for the BacktestExchange simulation engine.

Imports calc_liquidation_price from mechanics.py as a pure function to avoid
a circular dependency (LiquidationEngine → FillEngine.close_position).
================================================================================
"""

######### IMPORTS #########
import uuid
from typing import Any
from .state import BacktestState
from .mechanics import calc_liquidation_price

########## CLASS #########
class FillEngine:
    """
    Executes all trade fills and maintains position and balance accounting.

    Methods:
        fill_pending_order() : route a triggered resting order to entry or exit
        execute_market_fill():   open a position via market or stop order (taker fee)
        execute_limit_fill() : open a position via limit order (maker fee)
        close_position()     : close the current position, record trade log entry
    """

    def __init__(self, state: BacktestState) -> None:
        """
        Constructor. Initialize the FillEngine with shared simulation state.
        
        Args:
            state : BacktestState instance (shared by all engines)
        """
        self._s = state

    # ── Pending order routing ─────────────────────────────────────────────────

    def fill_pending_order(self, order: dict, fill_price: float, is_maker: bool) -> None:
        """
        Route a triggered resting stop or limit order to either entry or exit.

        Routes based on position state and the order's is_entry flag:
          - Flat + is_entry=True  : open position (limit → execute_limit_fill,
                                    stop → execute_market_fill)
          - Flat + is_entry=False : orphaned exit (position closed earlier this
                                    bar by another order), discarded silently
          - Position open + is_entry=False : exit order → close_position
          - Position open + is_entry=True  : error (overlapping entry logic)

        Args:
            order      : the triggered resting order dict
            fill_price : execution price for the fill
            is_maker   : True for limit fills, False for stop fills
        """
        s = self._s
        is_entry = order["is_entry"]  # Explicit access; fails loudly if missing
        
        if s.position is None:
            # Position is flat
            if is_entry:
                # Entry order: open position based on fill type
                if is_maker:
                    self.execute_limit_fill(order, fill_price)
                else:
                    self.execute_market_fill(order, fill_price)
            # else: orphaned exit order (position closed earlier this bar), discard silently
        else:
            # Position is open
            if is_entry:
                raise ValueError(
                    f"BacktestExchange: entry order {order['id']} is filling with position already open. "
                    f"Indicates overlapping entry logic in the bot."
                )
            else:
                # Exit order: close position
                self.close_position(fill_price, order_id=order["id"], is_maker=is_maker)

        s.open_orders.pop(order["id"], None)

    # ── Entry fills ───────────────────────────────────────────────────────────

    def execute_market_fill(self, order: dict[str, Any], fill_price: float) -> None:
        """
        Open a position via a market or stop order fill.

        If an opposing position is open, it is closed first at the same price
        (taker fee).  The new position then opens with taker fee.

        Args:
            order      : order dict being filled
            fill_price : execution price
        """
        s      = self._s
        symbol = order["symbol"]
        side   = order["side"]
        size   = order["size"]

        # Close any opposing position first (reversal)
        if s.position is not None:
            pos_side    = s.position["side"]
            is_opposing = (
                (pos_side == "long"  and side == "sell") or
                (pos_side == "short" and side == "buy")
            )
            if is_opposing:
                self.close_position(fill_price, order_id=order["id"], is_maker=False, close_reason="reversal")

        # Balance update
        notional = size * fill_price
        fee      = notional * s.taker_fee
        margin   = notional / s.instrument.leverage

        # Check margin requirement for leveraged instruments
        required = margin + fee
        if s.balance["free"] < required:
            s.orders_rejected += 1
            return  # Insufficient margin; order remains queued

        s.balance["free"]  -= margin + fee
        s.balance["used"]  += margin
        s.balance["total"] -= fee
        s.total_fees_paid  += fee

        # Open new position
        position_side = "long" if side == "buy" else "short"
        s.position = {
            "symbol":       symbol,
            "side":         position_side,
            "size":         size,
            "entry_price":  fill_price,
            "mark_price":   fill_price,
            "leverage":     s.instrument.leverage,
        }

        # Compute liquidation price if the instrument supports it
        if s.instrument.has_liquidation:
            s.liquidation_price = calc_liquidation_price(
                fill_price, position_side,
                s.instrument.leverage, s.instrument.maintenance_margin,
            )
        else:
            s.liquidation_price = None

        # Open trade record (accumulates entry metadata for trade log on close)
        s.open_trade = {
            "symbol":       symbol,
            "side":         position_side,
            "entry_price":  fill_price,
            "size":         size,
            "entry_bar":    s.cursor,
            "entry_ts":     s.get_bar_timestamp(s.cursor),
            "entry_fee":    fee,
            "funding_paid": 0.0,
        }

        # Update order dict to reflect fill
        order["status"] = "closed"
        order["price"]  = fill_price

    def execute_limit_fill(self, order: dict[str, Any], fill_price: float) -> None:
        """
        Open a position via a limit order fill.

        Identical to execute_market_fill() except that the new position is opened
        with maker_fee instead of taker_fee.  Any opposing position is still closed
        aggressively (taker fee) since reversals are always market-speed.

        Args:
            order      : order dict being filled
            fill_price : execution price (the limit price itself)
        """
        s      = self._s
        symbol = order["symbol"]
        side   = order["side"]
        size   = order["size"]

        # Close any opposing position first (taker fee — reversal is aggressive)
        if s.position is not None:
            pos_side    = s.position["side"]
            is_opposing = (
                (pos_side == "long"  and side == "sell") or
                (pos_side == "short" and side == "buy")
            )
            if is_opposing:
                self.close_position(fill_price, order_id=order["id"], is_maker=False, close_reason="reversal")

        # Balance update with maker fee
        notional = size * fill_price
        fee      = notional * s.maker_fee
        margin   = notional / s.instrument.leverage

        # Check margin requirement for leveraged instruments
        required = margin + fee
        if s.balance["free"] < required:
            s.orders_rejected += 1
            return  # Insufficient margin; order remains queued

        s.balance["free"]  -= margin + fee
        s.balance["used"]  += margin
        s.balance["total"] -= fee
        s.total_fees_paid  += fee

        # Open new position
        position_side = "long" if side == "buy" else "short"
        s.position = {
            "symbol":       symbol,
            "side":         position_side,
            "size":         size,
            "entry_price":  fill_price,
            "mark_price":   fill_price,
            "leverage":     s.instrument.leverage,
        }

        # Compute liquidation price if the instrument supports it
        if s.instrument.has_liquidation:
            s.liquidation_price = calc_liquidation_price(
                fill_price, position_side,
                s.instrument.leverage, s.instrument.maintenance_margin,
            )
        else:
            s.liquidation_price = None

        # Open trade record (accumulates entry metadata for trade log on close)
        s.open_trade = {
            "symbol":       symbol,
            "side":         position_side,
            "entry_price":  fill_price,
            "size":         size,
            "entry_bar":    s.cursor,
            "entry_ts":     s.get_bar_timestamp(s.cursor),
            "entry_fee":    fee,
            "funding_paid": 0.0,
        }

        # Update order dict to reflect fill
        order["status"] = "closed"
        order["price"]  = fill_price

    # ── Position close ────────────────────────────────────────────────────────

    def close_position(
        self,
        fill_price: float,
        order_id: str,
        is_maker: bool,
        close_reason: str = "exit_order",
    ) -> None:
        """
        Close the current open position at fill_price.

        Realizes p&l, applies the correct fee, releases margin, and appends a
        completed trade record to the trade log.

        Args:
            fill_price   : execution price for the closing fill
            order_id     : ID of the order that triggered the close
            is_maker     : True → apply maker_fee; False → apply taker_fee
            close_reason : reason for closure: "exit_order", "reversal", "liquidation", or "finalize"
        """
        s = self._s
        if s.position is None:
            return

        entry_price   = s.position["entry_price"]
        size          = s.position["size"]
        position_side = s.position["side"]

        # Compute realized p&l
        if position_side == "long":
            pnl = (fill_price - entry_price) * size
        else:
            pnl = (entry_price - fill_price) * size

        # Balance update
        notional = size * fill_price
        fee_rate = s.maker_fee if is_maker else s.taker_fee
        fee      = notional * fee_rate
        margin   = (size * entry_price) / s.instrument.leverage

        s.balance["used"]  -= margin
        s.balance["free"]  += margin + pnl - fee
        s.balance["total"] += pnl - fee
        s.total_fees_paid  += fee

        # Finalize trade record
        if s.open_trade is not None:
            funding_paid   = s.open_trade.get("funding_paid", 0.0)
            total_fees     = s.open_trade["entry_fee"] + fee
            net_pnl        = pnl - total_fees - funding_paid
            initial_margin = size * entry_price / s.instrument.leverage
            s.trade_log.append({
                "symbol":       s.open_trade["symbol"],
                "side":         s.open_trade["side"],
                "entry_price":  s.open_trade["entry_price"],
                "exit_price":   fill_price,
                "size":         size,
                "entry_bar":    s.open_trade["entry_bar"],
                "exit_bar":     s.cursor,
                "entry_ts":     s.open_trade["entry_ts"],
                "exit_ts":      s.get_bar_timestamp(s.cursor),
                "pnl":          round(pnl, 6),
                "entry_fee":    round(s.open_trade["entry_fee"], 6),
                "exit_fee":     round(fee, 6),
                "funding_paid": round(funding_paid, 6),
                "total_fees":   round(total_fees, 6),
                "net_pnl":      round(net_pnl, 6),
                "return_pct":   round(net_pnl / initial_margin * 100, 4) if initial_margin > 0 else 0.0,
                "close_reason": close_reason,
            })
            s.open_trade = None

        s.position          = None
        s.liquidation_price = None
        s.filled_order_ids.add(order_id)


