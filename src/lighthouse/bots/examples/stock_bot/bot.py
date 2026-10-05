"""
================================================================================
STOCK BOT
================================================================================

Holds the StockBot class, an Example Bot (CONTEXT.md) illustrating the
BaseBot/BaseState interfaces with a long-only moving-average crossover
strategy on a holiday-aware, non-24/7 stock instrument.

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.base_state import BotMode
from lighthouse.bots.examples.stock_bot.state import StockBotState
from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.enums import Side
from lighthouse.domain.bars import closed_bars
from lighthouse.domain.signals.general_filters import is_sma_above_sma
from lighthouse.domain.risk import calc_pct_stop_loss, calc_size_fixedfractional
from lighthouse.execution.orders import place_market_buy, place_market_sell, fetch_order, parse_order_result
from lighthouse.execution.positions import can_trade

############ CLASS ##############

class StockBot(BaseBot):
    """
    Example stock strategy bot. An Example Bot (CONTEXT.md): a minimal, complete
    reference implementation of the bot lifecycle, not a trading strategy
    recommendation.

    Symbol: AAPL
    Calendar: nyse (holiday-aware)
    Strategy type: long-only moving-average crossover, evaluated on closed bars

    Attributes:
        all attributes inherited from BaseBot (see bots/base_bot.py)
        state: StockBotState instance

    Config keys:
        symbol                  : trading symbol
        timeframe               : OHLCV timeframe
        limit                   : number of candles to fetch per tick
        scan_interval           : seconds between ticks when IDLE (scanning for entry)
        position_interval       : seconds between ticks when IN_POSITION (monitoring exit)
        fast_period             : fast SMA period
        slow_period             : slow SMA period
        risk_pct                : fraction of account balance to risk per trade
        sizing_stop_pct         : synthetic stop-loss distance (fraction of entry price) fed into
                                   domain/risk.py's fixed-fractional sizing helper only — this
                                   strategy places no real SL/TP protective orders
        max_consecutive_errors  : halt after this many consecutive tick errors
        max_entry_failures      : halt (restartable=False) after this many consecutive failed entry attempts

    Methods:
        on_scan()     : enter long when the fast SMA crosses above the slow SMA
        on_position() : exit when the fast SMA is no longer above the slow SMA
    """

    def __init__(self, exchange: BaseExchange, config: dict, notifier: Any | None = None, portfolio: Any | None = None) -> None:
        """
        Initialize the StockBot.

        Calls super().__init__() and sets up StockBotState.

        Args:
            exchange  : BaseExchange instance
            config    : strategy config (see class docstring for required keys)
            notifier  : optional Notifier for alerting; forwarded to BaseBot.__init__()
            portfolio : optional runtime.portfolio.Portfolio; forwarded to BaseBot.__init__()
        """
        super().__init__(exchange, config, notifier=notifier, portfolio=portfolio)
        self.state = StockBotState()

    # ── Entry logic ───────────────────────────────────────────────────────────
    def on_scan(self, df: pd.DataFrame) -> None:
        """
        Enter long when the fast SMA crosses above the slow SMA.

        Steps:
        1. Signal check — is_sma_above_sma(fast, slow) on the last closed bar
                            (closed_bars(df), per docs/adr/0001); no signal: return
        2. can_trade()   — confirm no existing long position on the exchange; if False, halts
                            immediately (restartable=True) instead of recording a failure —
                            state and exchange disagree, so reconcile() on restart is the fix
        3. Size          — a synthetic stop-loss distance (sizing_stop_pct) feeds
                            domain/risk.py's fixed-fractional sizing helper; no real SL order
                            is placed. Zero size records an entry failure and returns
        4. Enter         — place market order, record entry_price on state; a failed order
                            records an entry failure and returns
        5. State update  — set mode = IN_POSITION, side = Side.LONG

        A level check (fast above slow) rather than an edge trigger (domain/signals/triggers.py's
        became_true()), deliberately: the Entry Failure Backoff (docs/adr/0002) is scan-keyed, not
        bar-keyed, and relies on the gate staying open across bar boundaries so record_entry_failure()
        keeps retrying — and eventually halts restartable=False on exhaustion — as long as the trend
        holds. Edge-triggering would close the gate the moment a new bar closes without a fresh cross,
        silently abandoning retries instead of following ADR 0002's backoff-then-halt path. The accepted
        tradeoff: a bot started (or restarted) mid-uptrend enters immediately rather than waiting for a
        fresh cross, since evaluating on closed bars (docs/adr/0001) gives no way to tell "already above"
        from "just crossed" without the history became_true() would need — and ADR 0002 already rejects
        tracking the bar clock for retry purposes for the same reason.

        Args:
            df: OHLCV DataFrame
        """
        closed = closed_bars(df)
        if not is_sma_above_sma(closed, self.config["fast_period"], self.config["slow_period"]):
            return  # no signal, exit

        self.logger.info("Crossover signal detected | symbol=%s", self.config["symbol"])

        # Check can_trade
        if not can_trade(self.exchange, self.config["symbol"], Side.LONG):
            # Exempt from Entry Failure Backoff: state and exchange disagree, so this halts
            # immediately rather than waiting — reconcile() on restart resolves it (see docs/adr/0002).
            self.halt(
                "can_trade() returned False for long — bot state and exchange disagree.",
                restartable=True,
            )
            return

        # Estimate entry price from current ask
        entry_price = self.bid_ask["ask"]

        # Synthetic stop-loss distance for sizing only — this strategy places no real SL order
        sizing_sl_price = calc_pct_stop_loss(entry_price, Side.LONG, self.config["sizing_stop_pct"])

        # Calculate position size
        account_equity = self.get_sizing_equity()
        position_size = calc_size_fixedfractional(account_equity, self.config["risk_pct"], entry_price, sizing_sl_price)
        if position_size == 0:
            self.logger.warning("Position size is 0, skipping entry")
            self.record_entry_failure("position_size == 0")
            return

        # Place market order
        order = place_market_buy(self.exchange, self.config["symbol"], position_size)
        if order is None:
            self.logger.error("Market order failed, aborting entry")
            self.record_entry_failure("market order placement failed")
            return

        # Fetch full order details for actual fill price, entry fee, and fill timestamp
        entry_order = fetch_order(self.exchange, order.id, self.config["symbol"])
        actual_entry_price, entry_fee_data, actual_entry_time = parse_order_result(entry_order)
        actual_entry_price = actual_entry_price or entry_price  # fall back to estimated ask price
        actual_entry_time  = actual_entry_time or datetime.now(timezone.utc)
        filled             = (entry_order.filled or 0.0) if entry_order is not None else 0.0
        position_size      = filled if filled > 0 else position_size  # fall back to requested size if filled is missing

        # Update state
        self.state.entry_time    = actual_entry_time
        self.state.entry_price   = actual_entry_price
        self.state.entry_fee     = entry_fee_data
        self.state.position_size = position_size
        self.state.mode          = BotMode.IN_POSITION
        self.state.side          = Side.LONG
        self.logger.info(
            "Position opened | symbol=%s entry=%.4f size=%s",
            self.config["symbol"], actual_entry_price, position_size,
        )

    # ── Exit logic ────────────────────────────────────────────────────────────
    def on_position(self, df: pd.DataFrame) -> None:
        """
        Exit when the fast SMA is no longer above the slow SMA (crossed back below).

        A level check rather than an edge trigger: domain/signals/ booleans are
        level-triggered by default (CONTEXT.md, "Signal"), so if the exit order placement
        fails below, the same condition still holds on the next tick and on_position() retries
        instead of missing the exit.

        Args:
            df: OHLCV DataFrame
        """
        closed = closed_bars(df)
        if is_sma_above_sma(closed, self.config["fast_period"], self.config["slow_period"]):
            return  # fast still above slow, stay in position

        order = place_market_sell(self.exchange, self.config["symbol"], self.state.position_size)
        if order is None:
            self.logger.error("Exit market order failed — position remains open, will retry next tick.")
            return

        # Fetch full order details for actual fill price, exit fee, and fill timestamp
        exit_order = fetch_order(self.exchange, order.id, self.config["symbol"])
        exit_price, exit_fee_data, exit_time = parse_order_result(exit_order)
        # Caller resolves exit_price: fall back to the current bid if the fill price
        # couldn't be parsed from the order response.
        exit_price = exit_price if exit_price is not None else (self.bid_ask["bid"] if self.bid_ask else None)
        self.close_trade("cross_down", exit_price=exit_price, exit_fee=exit_fee_data, exit_time=exit_time)
