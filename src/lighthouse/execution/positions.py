"""
================================================================================
POSITION MODULE
================================================================================

This module contains stateless position query functions.

Functions:
    get_open_positions(): fetches all open positions for a symbol
    can_trade(): checks if opening a new trade is permitted (no existing position in that direction)
    get_position(): returns the open NormalizedPosition for a symbol and side, or None if flat
    is_position_closed(): checks if a tracked SL/TP order has been fully executed
================================================================================
"""

from __future__ import annotations

################# IMPORTS ##################

from lighthouse.utils.logging import get_logger
from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition

class OrderCancelledError(Exception):
    """
    Raised when a tracked SL/TP order is no longer on the exchange
    but the position is still open, indicating the order was cancelled
    rather than filled.
    """

logger = get_logger(__name__)

################# FUNCTIONS ##################

# ── Fetching all open positions ─────────────────────────────────────────────────────
def get_open_positions(exchange: BaseExchange, symbol: str) -> list[NormalizedPosition]:
    """
    Fetch all open positions for a symbol from the exchange. 

    Args:
        exchange: any BaseExchange instance
        symbol:   trading symbol

    Returns:
        A list of NormalizedPosition instances.
    """
    try:
        return exchange.fetch_open_positions(symbol)
    except Exception as exc:
        raise ValueError(
            f"Could not fetch open positions for {symbol}: {exc}"
        ) from exc

def _get_position(exchange: BaseExchange, symbol: str, side: Side) -> NormalizedPosition | None:
    """
    Return the open position matching symbol and side, or None if not found.

    Args:
        exchange: any BaseExchange instance
        symbol:   trading symbol
        side:     Side.LONG or Side.SHORT

    Returns:
        A NormalizedPosition or None if no matching position is found.
    """
    try:
        positions = get_open_positions(exchange, symbol)
        for pos in positions:
            if pos.symbol == symbol and pos.side == side:
                return pos
        return None
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Error fetching open positions for %s %s: %s", side.value, symbol, exc)
        return None

# ── Other functions ─────────────────────────────────────────────────────
def _is_position_open(exchange: BaseExchange, symbol: str, side: Side) -> bool:
    """
    Check whether a position is currently open for symbol and side.

    Args:
        exchange: any BaseExchange instance
        symbol:   trading symbol
        side:     Side.LONG or Side.SHORT

    Returns:
        True if a matching open position exists, False otherwise.
    """
    return _get_position(exchange, symbol, side) is not None


# ── Trade permission ─────────────────────────────────────────────────────────
def can_trade(exchange: BaseExchange, symbol: str, side: Side) -> bool:
    """
    Check whether opening a new trade is permitted.

    Blocks trading if a position in the given direction is already open on
    the symbol.

    Args:  
        exchange: any BaseExchange instance
        symbol:   trading symbol
        side:     intended trade direction; Side.LONG or Side.SHORT

    Returns: 
        True if trading is allowed, False otherwise.
    """
    try:
        if _is_position_open(exchange, symbol, side):
            logger.info(
                "Trade blocked — position already open | symbol=%s side=%s",
                symbol, side.value,
            )
            return False
        logger.debug("Trade permitted | symbol=%s side=%s", symbol, side.value)
        return True
    except Exception as exc:
        logger.error(
            "can_trade check failed, blocking trade as safe default | symbol=%s side=%s error=%s",
            symbol, side.value, exc,
        )
        return False


def get_position(exchange: BaseExchange, symbol: str, side: Side) -> NormalizedPosition | None:
    """
    Return the open position for symbol and side, or None if flat.

    Args:
        exchange: any BaseExchange instance
        symbol:   trading symbol
        side:     Side.LONG or Side.SHORT

    Returns:
        A NormalizedPosition or None if no matching position is open.
    """
    return _get_position(exchange, symbol, side)


def is_position_closed(exchange: BaseExchange, order_id: str, symbol: str, side: Side) -> bool:
    """
    Check whether a tracked SL or TP order has been fully executed.

    An order is considered closed when two conditions are both true:
    1. The order ID is no longer present in the exchange's open orders
       (it was executed or cancelled).
    2. No open position exists for the symbol and side (confirming the
       position was actually closed, not just that the order was cancelled).

    Args:
        exchange:  active exchange instance implementing ``BaseExchange``
        order_id:  exchange-assigned ID of the SL or TP order to check
        symbol:    trading symbol
        side:      Side.LONG or Side.SHORT

    Returns:
        True if the order is gone and the position is confirmed closed.
        False if the order is still open, or if the check itself fails.
    """
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        order_still_open = any(o.id == order_id for o in open_orders)
        if order_still_open:
            return False
        position_still_open = _is_position_open(exchange, symbol, side)
        closed = not position_still_open
        if not closed:
            raise OrderCancelledError(
                f"Order gone but position still open — order likely cancelled | "
                f"order_id={order_id} symbol={symbol} side={side.value}"
            )
        return closed
    except OrderCancelledError:
        raise
    except Exception as exc:
        logger.error(
            "is_position_closed check failed | order_id=%s symbol=%s error=%s",
            order_id, symbol, exc,
        )
        return False