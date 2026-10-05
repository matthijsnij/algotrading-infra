"""
================================================================================
ORDERS MODULE 
================================================================================

This module contains stateless order management functions.

Wraps BaseExchange order methods with consistent logging and error handling. 

Functions:
    place_market_buy()       : place a market buy order
    place_market_sell()      : place a market sell order
    place_limit_buy()        : place a limit buy order
    place_limit_sell()       : place a limit sell order
    place_stop_buy()         : place a stop market buy order
    place_stop_sell()        : place a stop market sell order
    place_stop_limit_buy()   : place a stop-limit buy order
    place_stop_limit_sell()  : place a stop-limit sell order
    cancel_order()           : cancel an open order by ID
    cancel_all_orders()      : cancel all open orders for a symbol
    get_open_orders()        : fetch all open orders for a symbol
    fetch_order()            : fetch a single closed or open order by ID
    parse_order_result()     : extract fill_price, fee, and fill_time from an order dict
    emergency_close()        : close an open position with a market order in the opposite direction
================================================================================
"""

from __future__ import annotations

############## IMPORTS ###############

from datetime import datetime, timezone

from lighthouse.utils.logging import get_logger
from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedOrder, Fee

logger = get_logger(__name__)

############## FUNCTIONS ###############

# ── Market orders ─────────────────────────────────────────────────────────
def place_market_buy(exchange: BaseExchange, symbol: str, size: float) -> NormalizedOrder | None:
    """
    Place a market buy order

    Args:
        exchange: active exchange instance 
        symbol:   trading symbol
        size:     order quantity

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info("Placing market BUY | symbol=%s size=%s", symbol, size)
    try:
        result = exchange.place_buy_market_order(symbol, size)
        logger.info("Market BUY submitted | id=%s symbol=%s size=%s", result.id, symbol, size)
        return result
    except Exception as exc:
        logger.error("Market BUY failed | symbol=%s size=%s error=%s", symbol, size, exc)
        return None


def place_market_sell(exchange: BaseExchange, symbol: str, size: float) -> NormalizedOrder | None:
    """
    Place a market sell order

    Args:
        exchange: active exchange instance 
        symbol:   trading symbol 
        size:     order quantity

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info("Placing market SELL | symbol=%s size=%s", symbol, size)
    try:
        result = exchange.place_sell_market_order(symbol, size)
        logger.info("Market SELL submitted | id=%s symbol=%s size=%s", result.id, symbol, size)
        return result
    except Exception as exc:
        logger.error("Market SELL failed | symbol=%s size=%s error=%s", symbol, size, exc)
        return None


# ── Limit orders ─────────────────────────────────────────────────────────
def place_limit_buy(exchange: BaseExchange, symbol: str, size: float, limit_price: float) -> NormalizedOrder | None:
    """
    Place a limit buy order

    Args:
        exchange:    active exchange instance 
        symbol:      trading symbol
        size:        order quantity
        limit_price: limit price

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing limit BUY | symbol=%s size=%s limit_price=%s",
        symbol, size, limit_price,
    )
    try:
        result = exchange.place_buy_limit_order(symbol, size, limit_price)
        logger.info(
            "Limit BUY submitted | id=%s symbol=%s size=%s limit_price=%s",
            result.id, symbol, size, limit_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Limit BUY failed | symbol=%s size=%s limit_price=%s error=%s",
            symbol, size, limit_price, exc,
        )
        return None


def place_limit_sell(exchange: BaseExchange, symbol: str, size: float, limit_price: float) -> NormalizedOrder | None:
    """
    Place a limit sell order

    Args:
        exchange:    active exchange instance 
        symbol:      trading symbol
        size:        order quantity
        limit_price: limit price

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing limit SELL | symbol=%s size=%s limit_price=%s",
        symbol, size, limit_price,
    )
    try:
        result = exchange.place_sell_limit_order(symbol, size, limit_price)
        logger.info(
            "Limit SELL submitted | id=%s symbol=%s size=%s limit_price=%s",
            result.id, symbol, size, limit_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Limit SELL failed | symbol=%s size=%s limit_price=%s error=%s",
            symbol, size, limit_price, exc,
        )
        return None


# ── Stop orders ─────────────────────────────────────────────────────────
def place_stop_buy(exchange: BaseExchange, symbol: str, size: float, stop_price: float) -> NormalizedOrder | None:
    """
    Place a stop market buy order (triggers a market buy when stop_price is hit).

    Args:
        exchange:   active exchange instance 
        symbol:     trading symbol
        size:       order quantity
        stop_price: price that triggers the order

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing stop BUY | symbol=%s size=%s stop_price=%s",
        symbol, size, stop_price,
    )
    try:
        result = exchange.place_buy_stop_order(symbol, size, stop_price)
        logger.info(
            "Stop BUY submitted | id=%s symbol=%s size=%s stop_price=%s",
            result.id, symbol, size, stop_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Stop BUY failed | symbol=%s size=%s stop_price=%s error=%s",
            symbol, size, stop_price, exc,
        )
        return None


def place_stop_sell(exchange: BaseExchange, symbol: str, size: float, stop_price: float) -> NormalizedOrder | None:
    """
    Place a stop market sell order (triggers a market sell when stop_price is hit)

    Args:
        exchange:   active exchange instance
        symbol:     trading symbol
        size:       order quantity
        stop_price: price that triggers the order

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing stop SELL | symbol=%s size=%s stop_price=%s",
        symbol, size, stop_price,
    )
    try:
        result = exchange.place_sell_stop_order(symbol, size, stop_price)
        logger.info(
            "Stop SELL submitted | id=%s symbol=%s size=%s stop_price=%s",
            result.id, symbol, size, stop_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Stop SELL failed | symbol=%s size=%s stop_price=%s error=%s",
            symbol, size, stop_price, exc,
        )
        return None


# ── Stop limit orders ─────────────────────────────────────────────────────────
def place_stop_limit_buy(exchange: BaseExchange, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder | None:
    """
    Place a stop-limit buy order

    When stop_price is hit a limit buy at limit_price is submitted

    Args:
        exchange:    active exchange instance 
        symbol:      trading symbol
        size:        order quantity
        stop_price:  price that activates the limit order
        limit_price: limit price to pay once activated

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing stop-limit BUY | symbol=%s size=%s stop_price=%s limit_price=%s",
        symbol, size, stop_price, limit_price,
    )
    try:
        result = exchange.place_buy_stop_limit_order(symbol, size, stop_price, limit_price)
        logger.info(
            "Stop-limit BUY submitted | id=%s symbol=%s size=%s stop_price=%s limit_price=%s",
            result.id, symbol, size, stop_price, limit_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Stop-limit BUY failed | symbol=%s size=%s stop_price=%s limit_price=%s error=%s",
            symbol, size, stop_price, limit_price, exc,
        )
        return None


def place_stop_limit_sell(exchange: BaseExchange, symbol: str, size: float, stop_price: float, limit_price: float) -> NormalizedOrder | None:
    """
    Place a stop-limit sell order

    When stop_price is hit a limit sell at limit_price is submitted

    Args:
        exchange:    active exchange instance 
        symbol:      trading symbol
        size:        order quantity
        stop_price:  price that activates the limit order
        limit_price: limit price to accept once activated

    Returns:
        NormalizedOrder result, or None on failure
    """
    logger.info(
        "Placing stop-limit SELL | symbol=%s size=%s stop_price=%s limit_price=%s",
        symbol, size, stop_price, limit_price,
    )
    try:
        result = exchange.place_sell_stop_limit_order(symbol, size, stop_price, limit_price)
        logger.info(
            "Stop-limit SELL submitted | id=%s symbol=%s size=%s stop_price=%s limit_price=%s",
            result.id, symbol, size, stop_price, limit_price,
        )
        return result
    except Exception as exc:
        logger.error(
            "Stop-limit SELL failed | symbol=%s size=%s stop_price=%s limit_price=%s error=%s",
            symbol, size, stop_price, limit_price, exc,
        )
        return None


# ── Order management ─────────────────────────────────────────────────────────
def cancel_order(exchange: BaseExchange, order_id: str, symbol: str) -> bool:
    """
    Cancel a single open order by ID.

    Args:
        exchange:  active exchange instance 
        order_id:  exchange-assigned order ID
        symbol:    trading symbol

    Returns:
        True if the cancel succeeded, False otherwise.
    """
    logger.info("Cancelling order | symbol=%s order_id=%s", symbol, order_id)
    try:
        exchange.cancel_order(order_id, symbol)
        logger.info("Order cancelled | id=%s symbol=%s", order_id, symbol)
        return True
    except Exception as exc:
        logger.error(
            "Order cancellation failed | symbol=%s order_id=%s error=%s",
            symbol, order_id, exc,
        )
        return False


def cancel_all_orders(exchange: BaseExchange, symbol: str) -> bool:
    """
    Cancel all open orders for a given symbol.

    Args:
        exchange: active exchange instance 
        symbol:   trading symbol

    Returns:
        True if the cancel succeeded, False otherwise.
    """
    logger.info("Cancelling all orders | symbol=%s", symbol)
    try:
        exchange.cancel_all_orders(symbol)
        logger.info("All orders cancelled | symbol=%s", symbol)
        return True
    except Exception as exc:
        logger.error(
            "Cancel all orders failed | symbol=%s error=%s",
            symbol, exc,
        )
        return False
    
# ── Fetch open orders ─────────────────────────────────────────────────────────
def get_open_orders(exchange: BaseExchange, symbol: str) -> list[NormalizedOrder]:
    """
    Fetch all open orders for a symbol.

    Args:
        exchange: active exchange instance 
        symbol:   trading symbol

    Returns:
        A list of NormalizedOrder instances.
    """
    logger.debug("Fetching open orders | symbol=%s", symbol)
    try:
        orders = exchange.fetch_open_orders(symbol)
        logger.debug("Open orders fetched | symbol=%s count=%d", symbol, len(orders))
        return orders
    except Exception as exc:
        logger.error("Fetch open orders failed | symbol=%s error=%s", symbol, exc)
        raise

def fetch_order(exchange: BaseExchange, order_id: str, symbol: str) -> NormalizedOrder | None:
    """
    Fetch a single closed or open order by ID.

    Args:
        exchange:  active exchange instance
        order_id:  exchange-assigned order identifier
        symbol:    trade symbol the order was placed on

    Returns:
        The NormalizedOrder on success, or None on failure.
    """
    try:
        result = exchange.fetch_order(order_id, symbol)
        logger.debug("Fetched order | id=%s symbol=%s status=%s fill_price=%s fee=%s",
                     order_id, symbol, result.status, result.fill_price, result.fee)
        return result
    except Exception as exc:
        logger.error("fetch_order failed | id=%s symbol=%s error=%s", order_id, symbol, exc)
        return None


# ── Order result parsing ─────────────────────────────────────────────────────
def parse_order_result(order_data: NormalizedOrder | None) -> tuple[float | None, Fee | None, datetime | None]:
    """
    Extract fill_price, fee, and fill_time from a NormalizedOrder.

    Args:
        order_data: NormalizedOrder returned by fetch_order, or None if the fetch failed.

    Returns:
        (fill_price, fee, fill_time); each field is None if order_data is None
        or the corresponding attribute is unset on the order.
    """
    if order_data is None:
        return None, None, None
    fill_price = order_data.fill_price
    fee        = order_data.fee
    _ts        = order_data.timestamp
    fill_time  = datetime.fromtimestamp(_ts / 1000, tz=timezone.utc) if _ts else None
    return fill_price, fee, fill_time


# ── Emergency close ─────────────────────────────────────────────────────────
def emergency_close(exchange: BaseExchange, symbol: str, side: Side, size: float) -> NormalizedOrder | None:
    """
    Place a market order in the opposite direction of an open position, closing it immediately.

    This function serves as an emergency backup for situations where security is breached, like SL/TP
    orders failing silently, being rejected, or being lost.

    Args:
        exchange: active exchange instance
        symbol:   trading symbol
        side:     Side.LONG or Side.SHORT, the direction of the position to close
        size:     quantity to close

    Returns:
        The NormalizedOrder from the closing market order, or None on failure.
    """
    logger.warning(
        "Emergency close triggered | symbol=%s side=%s size=%s",
        symbol, side.value, size,
    )
    if side == Side.LONG:
        return place_market_sell(exchange, symbol, size)
    elif side == Side.SHORT:
        return place_market_buy(exchange, symbol, size)
    else:
        logger.error(
            "emergency_close: cannot close a FLAT position | symbol=%s", symbol
        )
        return None

