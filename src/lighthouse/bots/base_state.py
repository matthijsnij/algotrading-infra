"""
================================================================================
BASE STATE DATACLASS
================================================================================

This file contains the BaseState class, which defines the interface that all state classes must implement.

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from lighthouse.domain.enums import Side  
from lighthouse.domain.exchange_types import Fee

############ CLASSES ##############

class BotMode(Enum):
    """
    Lifecycle phase of the bot.
    """
    IDLE          = "idle"           # No open position, actively evaluating entry conditions on every tick
    IN_POSITION   = "in_position"    # A position is live
    WAITING_FILL  = "waiting_fill"   # An entry order has been placed and is pending fill
    HALTED        = "halted"         # Bot has stopped due to a risk breach or repeated errors


@dataclass
class BaseState:
    """
    Universal state container shared by every strategy bot.

    Attributes:
        mode:               BotMode
        side:               Side
        lifetime_tick_count: number of ticks since this state was created — a Warm Start
                            carries this over, only a Cold Start resets it
        last_closed_bar:    timestamp of the last fully closed candle; a data-feed fact
                            maintained solely by BaseBot.tick(), not touched by reset()
        last_exit_bar:      last_closed_bar as of the last transition out of IN_POSITION
                            (normal close or halt); a cross-trade fact used by the Same-Bar
                            Re-Entry Guard (see CONTEXT.md, ADR 0003), so it survives reset()
        halt_reason:        reason for halting the bot, if applicable
        entry_limit_order_id:     exchange ID of a pending limit entry order (WAITING_FILL only)
        entry_time:         UTC fill time of the current open position
        entry_price:        fill price of the current open position
        entry_fee:          fee paid on entry; None if unavailable
        sl_price:           planned stop-loss price of the current open position; None if the
                            strategy does not use SL orders
        tp_price:           planned take-profit price of the current open position; None if the
                            strategy does not use TP orders
        lifetime_trade_count: number of trades closed over the whole life of this state
        halt_restartable:    whether the watchdog may auto-restart this bot after a halt
        entry_failure_count:      consecutive failed entry attempts since the last successful entry
        entry_retry_after_tick:   lifetime_tick_count that must be reached before the next entry
                                  attempt is allowed (Entry Failure Backoff, see CONTEXT.md); 0 means
                                  no backoff active
    
    Methods:
        reset() : reset all per-trade fields to their default values
    """

    mode:            BotMode           = field(default=BotMode.IDLE) # BotMode
    side:            Side              = field(default=Side.FLAT)   # Side
    lifetime_tick_count:      int               = field(default=0)           # number of ticks since this state was created
    last_closed_bar: datetime | None = field(default=None)       # timestamp of the last fully closed candle; maintained by BaseBot.tick(), not reset()
    last_exit_bar:   datetime | None = field(default=None)       # last_closed_bar at the last exit from IN_POSITION; drives the Same-Bar Re-Entry Guard, not reset()
    halt_reason:     str | None      = field(default=None)       # reason for halting the bot, if applicable
    position_size:   float             = field(default=0.0)         # current open position size in base currency
    entry_limit_order_id:  str | None     = field(default=None)        # exchange ID of a pending limit entry order (WAITING_FILL only)
    sl_order_id:     str | None     = field(default=None)        # exchange ID of the active stop-loss order
    tp_order_id:     str | None     = field(default=None)        # exchange ID of the active take-profit order
    entry_time:      datetime | None = field(default=None)       # UTC fill time of the current open position
    entry_price:     float | None   = field(default=None)        # fill price of the current open position
    entry_fee:       Fee | None     = field(default=None)        # fee paid on entry; None if unavailable
    sl_price:        float | None   = field(default=None)        # planned stop-loss price of the current open position
    tp_price:        float | None   = field(default=None)        # planned take-profit price of the current open position
    lifetime_trade_count:     int               = field(default=0)           # number of trades closed over the whole life of this state
    halt_restartable: bool             = field(default=True)        # whether the watchdog may auto-restart this bot after a halt
    entry_failure_count:    int        = field(default=0)           # consecutive failed entry attempts since the last successful entry
    entry_retry_after_tick: int        = field(default=0)           # lifetime_tick_count required before the next entry attempt; 0 = no backoff active

    def reset(self) -> None:
        """
        Return all per-trade fields to their default values.

        last_closed_bar and last_exit_bar are intentionally NOT reset here: both are
        cross-trade facts (a data-feed fact and the Same-Bar Re-Entry Guard's stamp,
        respectively), not per-trade facts.
        """
        self.mode            = BotMode.IDLE
        self.side            = Side.FLAT
        self.halt_reason     = None
        self.position_size   = 0.0
        self.entry_limit_order_id  = None
        self.sl_order_id     = None
        self.tp_order_id     = None
        self.entry_time      = None
        self.entry_price     = None
        self.entry_fee       = None
        self.sl_price        = None
        self.tp_price        = None
