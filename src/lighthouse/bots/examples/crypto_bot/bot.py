"""
================================================================================
CRYPTO BOT
================================================================================

Holds the CryptoBot class, an example crypto breakout strategy illustrating the
BaseBot/BaseState interfaces.

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

import pandas as pd
from datetime import datetime, timezone
from typing import Any
from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.base_state import BotMode
from lighthouse.bots.examples.crypto_bot.state import CryptoBotState
from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.enums import Side
from lighthouse.domain import indicators
from lighthouse.domain.bars import closed_bars
from lighthouse.domain.signals.example import is_long_breakout, is_short_breakout
from lighthouse.domain.signals.general_filters import is_volume_above_average, is_atr_above_threshold
from lighthouse.domain.risk import calc_size_fixedfractional
from lighthouse.execution.orders import place_market_buy, place_market_sell, place_stop_sell, place_stop_buy, place_limit_sell, place_limit_buy, cancel_order, cancel_all_orders, emergency_close, fetch_order, parse_order_result
from lighthouse.execution.positions import can_trade, is_position_closed, OrderCancelledError

############ CLASS ##############

class CryptoBot(BaseBot):
    """
    Example crypto strategy bot. An Example Bot (CONTEXT.md): a minimal, complete
    reference implementation of the bot lifecycle, not a trading strategy
    recommendation.

    Symbol: BTC/USDT 
    Timeframe: 1h
    Strategy type: Range breakout using swing high/low

    Attributes:
        all attributes inherited from BaseBot (see bots/base_bot.py) 
        state:              CryptoBotState instance 
        _direction:         resolved direction string ("long", "short", or "both")

    Config keys:
        symbol                  : trading symbol
        timeframe               : OHLCV timeframe
        limit                   : number of candles to fetch per tick
        scan_interval           : seconds between ticks when IDLE (scanning for entry)
        position_interval       : seconds between ticks when IN_POSITION (monitoring exit)
        direction               : "long", "short", or "both"
        breakout_window         : bars to look back for swing high/low
        atr_period              : ATR calculation period
        atr_buffer_mult         : ATR multiplier for breakout confirmation buffer
        sl_atr_mult             : ATR multiplier for stop loss distance
        rr_ratio                : reward/risk ratio for take profit
        volume_period           : lookback period for rolling volume average
        volume_mult             : volume must be >= volume_mult * rolling average
        atr_threshold           : minimum ATR value in price units to allow entry
        risk_pct                : fraction of account balance to risk per trade
        max_consecutive_errors  : halt after this many consecutive tick errors
        max_entry_failures      : halt (restartable=False) after this many consecutive failed entry attempts
    
    Methods:
        on_scan()     : evaluate entry conditions and open a position if all pass; calls
                        self.record_entry_failure() on every bail-out that leaves the bot
                        flat, per BaseBot's Entry Failure Backoff contract (docs/adr/0002)
        on_position() : check whether the SL or TP order has been filled and handle accordingly,
                        calling self.close_trade() (BaseBot) to finalize the trade
    """

    def __init__(self, exchange: BaseExchange, config: dict, notifier: Any | None = None, portfolio: Any | None = None) -> None:
        """
        Initialize the CryptoBot.

        Calls super().__init__(), sets up CryptoBotState, and resolves
        the configured direction to a string.

        Args:
            exchange  : BaseExchange instance
            config    : strategy config (see class docstring for required keys)
            notifier  : optional Notifier for alerting; forwarded to BaseBot.__init__()
            portfolio : optional runtime.portfolio.Portfolio; forwarded to BaseBot.__init__()
        """
        super().__init__(exchange, config, notifier=notifier, portfolio=portfolio)

        # initialize strategy state
        self.state = CryptoBotState()

        # resolve allowed trading direction to string
        direction = config.get("direction", "").lower()
        if direction not in {"long", "short", "both"}:
            raise ValueError(f"Invalid allowed direction '{direction}' - must be 'long', 'short', or 'both'")
        self._direction = direction

    # ── Entry logic ───────────────────────────────────────────────────────────
    def on_scan(self, df: pd.DataFrame) -> None:
        """
        Evaluate entry conditions and open a position if all pass.

        Steps:
        1. ATR filter    — skip if ATR is below atr_threshold (evaluated on the forming bar)
        2. Volume filter — skip if volume is below volume_mult * rolling average (forming bar)
        3. Signal check  — determine side for this tick, evaluated on the last closed bar
                             (closed_bars(df), per docs/adr/0001):
                             if direction is "long" or "both": check is_long_breakout → side = Side.LONG
                             elif direction is "short" or "both": check is_short_breakout → side = Side.SHORT
                             if no signal fires: return
                             (long is checked first; on "both", a long signal takes priority)
        4. can_trade()   — confirm no existing position on the exchange; if False, halts
                            immediately (restartable=True) instead of recording a failure —
                            state and exchange disagree, so reconcile() on restart is the fix
        5. Size          — fetch balance and calculate position size; zero size records an
                            entry failure via self.record_entry_failure() and returns
        6. Enter         — place market order, record entry_price on state; a failed order
                            records an entry failure and returns
        7. SL/TP         — compute sl_price and tp_price, place orders, store IDs; a failed
                            placement emergency-closes the position, records an entry
                            failure, and returns
        8. State update  — set mode = IN_POSITION, side = detected side (Side.LONG or Side.SHORT)

        Note that first checks are to avoid unnecessary API calls. Post-exit
        re-entry on an unchanged signal is handled by BaseBot's Same-Bar
        Re-Entry Guard (see CONTEXT.md, ADR 0003), not by this bot.

        df : OHLCV DataFrame
        """

        # Check ATR filter
        if not is_atr_above_threshold(df, self.config["atr_threshold"], self.config["atr_period"]):
            return
        
        # Check volume filter
        if not is_volume_above_average(df, self.config["volume_period"], self.config["volume_mult"]):
            return
        
        # Check breakout signals
        if self._direction in {"long", "both"} and is_long_breakout(closed_bars(df), self.config["breakout_window"], self.config["atr_period"], self.config["atr_buffer_mult"]):
            side = Side.LONG
        elif self._direction in {"short", "both"} and is_short_breakout(closed_bars(df), self.config["breakout_window"], self.config["atr_period"], self.config["atr_buffer_mult"]):
            side = Side.SHORT
        else:
            return  # no signal, exit

        self.logger.info("Signal detected | symbol=%s side=%s", self.config["symbol"], side.value)

        # Check can_trade
        if not can_trade(self.exchange, self.config["symbol"], side):
            # Exempt from Entry Failure Backoff: state and exchange disagree, so this halts
            # immediately rather than waiting — reconcile() on restart resolves it (see docs/adr/0002).
            self.halt(
                f"can_trade() returned False for {side.value} — bot state and exchange disagree.",
                restartable=True,
            )
            return

        # Estimate entry price from current bid/ask
        entry_price = self.bid_ask["ask"] if side == Side.LONG else self.bid_ask["bid"]

        # Compute SL/TP prices
        atr = indicators.atr_current(df, self.config["atr_period"])
        sl_distance = self.config["sl_atr_mult"] * atr
        if side == Side.LONG:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + (self.config["rr_ratio"] * sl_distance)
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - (self.config["rr_ratio"] * sl_distance)

        # Calculate position size
        account_equity = self.get_sizing_equity()
        position_size = calc_size_fixedfractional(account_equity, self.config["risk_pct"], entry_price, sl_price)
        if position_size == 0:
            self.logger.warning("Position size is 0, skipping entry")
            self.record_entry_failure("position_size == 0")
            return

        # Place market order
        if side == Side.LONG:
            order = place_market_buy(self.exchange, self.config["symbol"], position_size)
        else:
            order = place_market_sell(self.exchange, self.config["symbol"], position_size)
        if order is None:
            self.logger.error("Market order failed, aborting entry")
            self.record_entry_failure("market order placement failed")
            return

        # Fetch full order details for actual fill price, entry fee, and fill timestamp
        entry_order = fetch_order(self.exchange, order.id, self.config["symbol"])
        actual_entry_price, entry_fee_data, actual_entry_time = parse_order_result(entry_order)
        actual_entry_price = actual_entry_price or entry_price  # fall back to estimated bid/ask price
        actual_entry_time  = actual_entry_time or datetime.now(timezone.utc)
        filled             = (entry_order.filled or 0.0) if entry_order is not None else 0.0
        position_size      = filled if filled > 0 else position_size  # fall back to requested size if filled is missing

        # Place SL and TP orders
        if side == Side.LONG:
            sl_order = place_stop_sell(self.exchange, self.config["symbol"], position_size, sl_price)
            tp_order = place_limit_sell(self.exchange, self.config["symbol"], position_size, tp_price)
        else:
            sl_order = place_stop_buy(self.exchange, self.config["symbol"], position_size, sl_price)
            tp_order = place_limit_buy(self.exchange, self.config["symbol"], position_size, tp_price)

        # Emergency close if either SL or TP order placement failed, to avoid unprotected position
        if sl_order is None or tp_order is None:
            self.logger.error(
                "SL or TP order placement failed — triggering emergency close | sl=%s tp=%s",
                sl_order, tp_order,
            )
            # Cancel whichever order did succeed before closing
            if sl_order is not None:
                cancel_order(self.exchange, sl_order.id, self.config["symbol"])
            if tp_order is not None:
                cancel_order(self.exchange, tp_order.id, self.config["symbol"])
            emergency_close(self.exchange, self.config["symbol"], side, position_size)
            self.record_entry_failure("SL/TP order placement failed — emergency closed")
            return

        # Update state
        self.state.entry_time      = actual_entry_time
        self.state.entry_price     = actual_entry_price
        self.state.entry_fee       = entry_fee_data
        self.state.sl_order_id     = sl_order.id
        self.state.tp_order_id     = tp_order.id
        self.state.sl_price        = sl_price
        self.state.tp_price        = tp_price
        self.state.position_size   = position_size
        self.state.mode            = BotMode.IN_POSITION
        self.state.side            = side
        self.logger.info(
            "Position opened | symbol=%s side=%s entry=%.4f sl=%.4f tp=%.4f size=%s",
            self.config["symbol"], side.value, entry_price, sl_price, tp_price, position_size,
        )


    # ── Exit logic ────────────────────────────────────────────────────────────
    def on_position(self, df: pd.DataFrame) -> None:
        """
        Check whether the SL or TP order has been filled and handle accordingly.

        Steps:
        1. SL check — call is_position_closed(sl_order_id)
                      if True: cancel TP order, record loss, call self.close_trade("sl")
        2. TP check — call is_position_closed(tp_order_id)
                      if True: cancel SL order, record win, call self.close_trade("tp")

        df : OHLCV DataFrame (not used directly, kept for interface consistency)
        """

        # Log unrealized p&l
        if self.state.entry_price is not None and self.bid_ask:
            mid = (self.bid_ask["bid"] + self.bid_ask["ask"]) / 2
            direction_sign = 1 if self.state.side == Side.LONG else -1
            unrealized_pnl = (mid - self.state.entry_price) * self.state.position_size * direction_sign
            unrealized_pnl_pct = (mid - self.state.entry_price) / self.state.entry_price * direction_sign * 100
            self.logger.debug("Current unrealized p&l | symbol=%s p&l=%.4f (%.2f%%)", self.config["symbol"], unrealized_pnl, unrealized_pnl_pct)

        # Safety guard, if IN_POSITION but either SL or TP id is missing
        if self.state.sl_order_id is None or self.state.tp_order_id is None:
            self.logger.error(
                "IN_POSITION but no SL/TP order IDs — triggering emergency close | symbol=%s side=%s",
                self.config["symbol"], self.state.side.value,
            )
            # Cancel whichever order ID is still present before closing
            surviving_order_id = self.state.sl_order_id or self.state.tp_order_id
            if surviving_order_id:
                if not cancel_order(self.exchange, surviving_order_id, self.config["symbol"]):
                    self.logger.warning("Cancel failed for surviving order — attempting cancel_all_orders | symbol=%s", self.config["symbol"])
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.halt("Failed to cancel surviving order before emergency close (missing SL/TP) — manual intervention required.")
                        return
            order = emergency_close(self.exchange, self.config["symbol"], self.state.side, self.state.position_size)
            exit_price = order.fill_price if order is not None else None
            _ts        = order.timestamp  if order is not None else None
            exit_time  = datetime.fromtimestamp(_ts / 1000, tz=timezone.utc) if _ts else None
            self.close_trade("emergency_missing_orders", exit_price=exit_price, exit_time=exit_time)
            return

        # Check SL hit
        try:
            sl_closed = self.state.sl_order_id and is_position_closed(self.exchange, self.state.sl_order_id, self.config["symbol"], self.state.side)
        except OrderCancelledError as exc:
            self.logger.error("SL order cancelled but position still open — triggering emergency close | %s", exc)

            # Cancel TP order
            if self.state.tp_order_id:
                if not cancel_order(self.exchange, self.state.tp_order_id, self.config["symbol"]):
                    self.logger.warning("TP cancel failed during SL-cancelled path — attempting cancel_all_orders | symbol=%s", self.config["symbol"])
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.halt("Failed to cancel orphaned TP order during SL-cancelled emergency — manual intervention required.")
                        return

            # Trigger emergency close
            order = emergency_close(self.exchange, self.config["symbol"], self.state.side, self.state.position_size)
            exit_price = order.fill_price if order is not None else None
            _ts        = order.timestamp  if order is not None else None
            exit_time  = datetime.fromtimestamp(_ts / 1000, tz=timezone.utc) if _ts else None
            self.close_trade("emergency_sl_cancelled", exit_price=exit_price, exit_time=exit_time)
            return
        if sl_closed:
            # Fetch actual SL fill price, exit fee, and fill timestamp
            if self.state.tp_order_id:
                if not cancel_order(self.exchange, self.state.tp_order_id, self.config["symbol"]):
                    self.logger.warning("TP cancel failed after SL hit — attempting cancel_all_orders | symbol=%s", self.config["symbol"])
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.halt("Failed to cancel orphaned TP order after SL hit — manual intervention required.")
                        return
            sl_order_data                  = fetch_order(self.exchange, self.state.sl_order_id, self.config["symbol"])
            exit_price, exit_fee_data, exit_time = parse_order_result(sl_order_data)
            # Caller resolves exit_price: fall back to the planned SL price if the fill
            # price couldn't be parsed from the order response.
            exit_price = exit_price if exit_price is not None else self.state.sl_price
            self.close_trade("sl", exit_price=exit_price, exit_fee=exit_fee_data, exit_time=exit_time)
            return

        # Check TP hit
        try:
            tp_closed = self.state.tp_order_id and is_position_closed(self.exchange, self.state.tp_order_id, self.config["symbol"], self.state.side)
        except OrderCancelledError as exc:
            self.logger.error("TP order cancelled but position still open — triggering emergency close | %s", exc)

            # Cancel SL order
            if self.state.sl_order_id:
                if not cancel_order(self.exchange, self.state.sl_order_id, self.config["symbol"]):
                    self.logger.warning("SL cancel failed during TP-cancelled path — attempting cancel_all_orders | symbol=%s", self.config["symbol"])
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.halt("Failed to cancel orphaned SL order during TP-cancelled emergency — manual intervention required.")
                        return

            # Trigger emergency close
            order = emergency_close(self.exchange, self.config["symbol"], self.state.side, self.state.position_size)
            exit_price = order.fill_price if order is not None else None
            _ts        = order.timestamp  if order is not None else None
            exit_time  = datetime.fromtimestamp(_ts / 1000, tz=timezone.utc) if _ts else None
            self.close_trade("emergency_tp_cancelled", exit_price=exit_price, exit_time=exit_time)
            return
        if tp_closed:
            # Fetch actual TP fill price, exit fee, and fill timestamp
            if self.state.sl_order_id:
                if not cancel_order(self.exchange, self.state.sl_order_id, self.config["symbol"]):
                    self.logger.warning("SL cancel failed after TP hit — attempting cancel_all_orders | symbol=%s", self.config["symbol"])
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.halt("Failed to cancel orphaned SL order after TP hit — manual intervention required.")
                        return
            tp_order_data                  = fetch_order(self.exchange, self.state.tp_order_id, self.config["symbol"])
            exit_price, exit_fee_data, exit_time = parse_order_result(tp_order_data)
            # Caller resolves exit_price: fall back to the planned TP price if the fill
            # price couldn't be parsed from the order response.
            exit_price = exit_price if exit_price is not None else self.state.tp_price
            self.close_trade("tp", exit_price=exit_price, exit_fee=exit_fee_data, exit_time=exit_time)
            return

