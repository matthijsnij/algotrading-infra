"""
================================================================================
EXCHANGE VALUE TYPES
================================================================================

Typed, immutable value classes for the data every exchange returns across the
BaseExchange boundary. These replace the raw dicts previously documented only
in the BaseExchange method docstrings: exchanges construct them, execution/
and bots/ consume them by attribute access.

Conventions:
- Sides are normalized at the exchange boundary — NormalizedPosition.side is a
  Side enum, so consumers never call normalize_side() themselves.
- Instances are frozen: they are snapshots of exchange state, never mutated.

Classes:
    Fee                : cost + currency of a filled order's fee
    NormalizedPosition : an open position (symbol, side, size)
    NormalizedOrder    : an order as reported by the exchange
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import dataclasses

from lighthouse.domain.enums import Side

############ DATACLASSES ############

@dataclasses.dataclass(frozen=True)
class Fee:
    """
    Fee charged on a filled order.

    Attributes:
        cost     : fee amount, denominated in `currency`
        currency : currency the fee was charged in, e.g. "USDT"
    """

    cost:     float
    currency: str


@dataclasses.dataclass(frozen=True)
class NormalizedPosition:
    """
    An open position as reported by an exchange.

    Attributes:
        symbol : market symbol, e.g. "BTC/USDT:USDT"
        side   : Side.LONG or Side.SHORT (normalized at the exchange boundary)
        size   : position size in base currency, always positive
    """

    symbol: str
    side:   Side
    size:   float


@dataclasses.dataclass(frozen=True)
class NormalizedOrder:
    """
    An order as reported by an exchange (just placed, open, or fetched by ID).

    Attributes:
        id         : exchange-assigned order ID
        status     : order status, e.g. "open", "closed", "cancelled"
        symbol     : market symbol, e.g. "BTC/USDT:USDT"
        side       : "buy" or "sell"
        type       : order type, e.g. "market", "limit", "stop", "stop-limit"
        size       : requested quantity in base currency
        price      : limit price, if any
        stop_price : stop trigger price, if any
        filled     : quantity filled so far, if known
        fill_price : average fill price, if filled
        timestamp  : fill/creation time in ms since epoch (UTC), if known
        fee        : fee charged, if known
    """

    id:         str
    status:     str
    symbol:     str
    side:       str
    type:       str
    size:       float
    price:      float | None = None
    stop_price: float | None = None
    filled:     float | None = None
    fill_price: float | None = None
    timestamp:  int | None   = None
    fee:        Fee | None   = None
