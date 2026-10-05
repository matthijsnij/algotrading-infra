"""
================================================================================
ABSTRACT BASE BOT CLASS
================================================================================

Abstract base class for all strategy bots.

Owns orchestration only: the tick loop, data fetching, error handling, and
lifecycle management. Contains zero strategy-specific logic.

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

import threading
import uuid
import pandas as pd
from abc import ABC
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

from lighthouse.exchanges.base import BaseExchange
from lighthouse.execution.data import get_ohlcv_df, get_bid_ask
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import NormalizedPosition, NormalizedOrder, Fee
from lighthouse.domain.calendars import create_calendar, TradingCalendar
from lighthouse.domain.timeframe import Timeframe
from lighthouse.domain.bars import closed_bars
from lighthouse.execution.orders import get_open_orders, emergency_close, cancel_all_orders, cancel_order
from lighthouse.execution.positions import get_open_positions
from lighthouse.domain.trade_log import TradeLogger, TradeRecord
from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.notifications.notifier import Notifier
from lighthouse.utils.logging import get_logger, add_bot_file_handler

############ HELPERS ##############

def _last_closed_bar_time(timeframe: str, calendar: TradingCalendar) -> datetime:
    """
    Compute the open timestamp of the last fully closed candle
    without making an API call.

    Floors the current UTC time to the nearest timeframe boundary, then subtracts one
    interval to get the previous (closed) bar. For calendars that are not always open,
    walks back further until landing on a timestamp inside a trading session (e.g. the
    last bar before a weekend). 24/7 -> unchanged result (loop never runs).
    """
    interval = Timeframe.parse(timeframe).seconds
    now      = datetime.now(timezone.utc)
    floored  = datetime.fromtimestamp((now.timestamp() // interval) * interval, tz=timezone.utc) # open time of current bar
    candidate = floored - timedelta(seconds=interval) # open time of last closed bar

    max_steps = max(1, int(7 * 86400 / interval)) # at most one week back
    for _ in range(max_steps):
        if calendar.is_open(pd.Timestamp(candidate)):
            return candidate
        candidate -= timedelta(seconds=interval)

    raise RuntimeError(
        f"_last_closed_bar_time: no in-session bar found within a week before {floored} "
        f"for calendar '{calendar.name}'."
    )


def _find_order_by_id(orders: list[NormalizedOrder], order_id: str | None) -> NormalizedOrder | None:
    """
    Look up a persisted order ID among fetched open orders. Returns None if order_id is None
    (nothing persisted) or if no open order matches it (persisted order no longer open).
    """
    if order_id is None:
        return None
    return next((o for o in orders if o.id == order_id), None)


############ CLASS ##############

class BaseBot(ABC):
    """
    Abstract base class for all strategy bots.

    Attributes:
        exchange:            BaseExchange instance
        config:              bot configuration dictionary (see below)
        logger:              Logger instance named after the bot class
        notifier:            Notifier instance for alerting; a disabled no-op Notifier([]) when none is supplied
        portfolio:           Optional runtime.portfolio.Portfolio for system-wide equity sizing;
                             None in backtests (one bot + one exchange, so exchange-local
                             equity already equals system equity) and wired in live.py.
                             Typed Any, not Portfolio, so bots never import the runtime layer.
        trade_logger:        TradeLogger instance for writing trade records to CSV
        state:               Optional BaseState instance for tracking bot state across ticks
        bid_ask:             Optional dict with 'bid' and 'ask' prices, updated on every tick when IDLE or IN_POSITION
        _stop_event:         threading.Event used to signal the tick loop to exit
        _consecutive_errors: integer counter for consecutive tick errors
        _db_path:            Optional path to the SQLite state store; None disables persistence
                              (e.g. backtests). Set post-construction by attach_persistence(),
                              not a constructor arg.
        _bot_key:            Optional "<name>:<symbol>" key this bot is persisted under; set
                              alongside _db_path.
        _persist_state_fn:   Optional callable (db_path, bot_key, state) -> None used by
                              persist_state(); set alongside _db_path by attach_persistence()
                              so Bots never imports runtime.state_store directly.
        _state_lock:         threading.Lock serializing halt()'s mutate+persist against tick()'s
                              own persist — halt() can be called from a different thread (command
                              listener, drawdown monitor) while the tick loop's own thread is
                              mid-tick; without this, a slightly-stale tick() write can land after
                              (and silently clobber) halt()'s HALTED write.

    config must include the following keys:
        symbol:                 market symbol
        timeframe:              primary OHLCV timeframe, drives the tick interval and the main OHLCV fetch; bots needing additional timeframes fetch them directly inside on_tick()
        limit:                  number of candles to fetch per tick
        scan_interval:          seconds between ticks when IDLE (scanning for entry)
        position_interval:      seconds between ticks when IN_POSITION (monitoring exit)
        max_consecutive_errors: halt after this many consecutive tick errors; defaults to 5
        max_entry_failures:     halt (restartable=False) after this many consecutive failed Entry
                                Attempts; defaults to 3. See record_entry_failure().
        calendar:               optional TradingCalendar name (domain/calendars.py); defaults to "24/7"
        equity_scope:           "exchange" (default) or "system" — which equity get_sizing_equity() uses
        capital_fraction:       fraction of that equity get_sizing_equity() returns; defaults to 1.0

    Methods:
        run()                   : start the tick loop (intended to be run in a thread)
        halt()                  : stop the tick loop and perform cleanup
        on_halt()               : called by halt() for exchange cleanup, can be overridden
        attach_persistence()    : wire up SQLite state persistence; called by
                                  runtime.supervisor.prepare_bot(), not by bots themselves
        reconcile()             : synchronously check for orphaned positions/orders before starting
                                  threads; called by runtime.supervisor.prepare_bot()
        on_reconcile()          : called by reconcile() when an orphaned position turns out to be protected (CONTEXT.md, docs/adr/0004)
        tick()                  : execute a single tick, fetch data, call on_tick(), handle errors
        on_tick()               : concrete mode dispatcher — routes to on_scan() (IDLE), on_position()
                                  (IN_POSITION), or on_pending_fill() (WAITING_FILL); a no-op when
                                  HALTED. Still overridable by subclasses that want the raw hook
                                  instead of per-mode dispatch (e.g. StockBot).
        on_scan()               : called by on_tick() while IDLE; default no-op
        on_position()           : called by on_tick() while IN_POSITION; default no-op
        on_pending_fill()       : called by on_tick() while WAITING_FILL; default no-op
        record_entry_failure()  : bot-author obligation — call from every on_scan()-style
                                  bail-out path that leaves the bot flat, to drive the Entry
                                  Failure Backoff (see CONTEXT.md, docs/adr/0002). tick() only
                                  suppresses on_tick() during the backoff; it cannot detect the
                                  failure itself, since BaseBot has no chokepoint on order placement.
        close_trade()           : finalize a closed trade — generic P&L/fee accounting, TradeRecord
                                  write, and state.reset(). Caller resolves reason and exit_price.
        get_sizing_equity()     : equity to size positions against, per config equity_scope/capital_fraction
        persist_state()          : save current state to SQLite if persistence is enabled (no-op otherwise);
                                  called by runtime.supervisor.prepare_bot() and internally by tick()/halt()
        build_summary()         : generic snapshot dict for the periodic summary alert (runtime/summary.py)
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: dict,
        notifier: Notifier | None = None,
        portfolio: Any | None = None,
    ) -> None:
        """
        Initialize the bot.

        Args:
            exchange:  BaseExchange instance 
            config:    bot configuration dictionary (see class docstring)
            notifier:  optional Notifier for alerting; defaults to a disabled no-op
                      Notifier([]) when None, so every notify_*() call site below is
                      always safe to call without an `if notifier:` guard.
            portfolio: optional runtime.portfolio.Portfolio for "system" equity_scope sizing;
                      None (default) falls back to exchange-local equity in get_sizing_equity().
        """
        self.exchange = exchange
        self.config = config
        self.calendar = create_calendar(config.get("calendar", "24/7"))
        self.logger = get_logger(self.__class__.__name__)
        add_bot_file_handler(self.__class__.__name__)
        self.notifier = notifier if notifier is not None else Notifier([])
        self.portfolio = portfolio
        self.trade_logger = TradeLogger(self.__class__.__name__, on_log=self.notifier.notify_trade_closed)
        self.state: BaseState | None = None  
        self.bid_ask: dict | None = None 
        self._stop_event = threading.Event()
        self._consecutive_errors: int = 0
        # Set post-construction by attach_persistence(); None means persistence is
        # disabled (e.g. backtests never call attach_persistence()).
        self._db_path: Path | None = None
        self._bot_key: str | None = None
        # Injected alongside _db_path/_bot_key so Bots never imports runtime.state_store
        # directly (Runtime orchestrates Bots, not the other way around).
        self._persist_state_fn: (Callable[[Path, str, "BaseState"], None]) | None = None
        self._state_lock = threading.Lock()

    def attach_persistence(
        self,
        db_path: Path,
        bot_key: str,
        persist_state_fn: Callable[[Path, str, "BaseState"], None],
    ) -> None:
        """
        Wire up SQLite state persistence. Called by runtime.supervisor.prepare_bot()
        post-construction, once, before reconcile(); bots never call this themselves.
        """
        self._db_path = db_path
        self._bot_key = bot_key
        self._persist_state_fn = persist_state_fn

    def reconcile(self) -> None:
            """
            Called synchronously from live.py for each bot before any threads start.
    
            Queries the exchange for open positions on this bot's symbol. Three outcomes:
              - No position: proceed normally.
              - Position with a persisted state.sl_order_id that is still open on the exchange:
                call on_reconcile() to restore state and resume IN_POSITION. SL/TP are identified
                by persisted order ID only — order-type strings are adapter-specific and free-form
                (see docs/adr/0004) and are never inspected here.
              - Position with no persisted sl_order_id, or one that is no longer open: unprotected
                — halt (restartable=False). If persisted state is IN_POSITION (this bot has a full
                record of the trade), on_halt() owns the emergency close (and, in subclasses, the
                trade record); otherwise reconcile() closes it itself, since on_halt() is a no-op
                when the mode isn't IN_POSITION. Exactly one owner closes — never both.

            A failed position or order fetch halts (restartable) with a "state unknown" reason
            rather than falling through to a decision — an unknown exchange state is never treated
            as "nothing there".

            When persisted state was loaded before this call (see runtime.supervisor.prepare_bot()),
            the "no position" branch additionally reconciles the persisted mode against reality:
              - Persisted IN_POSITION but exchange shows no position: the position closed while the
                bot was down. state.reset() back to IDLE and log CRITICAL (no trade record — that
                close is unrecoverable from here, see state_store.py module docstring).
              - Persisted WAITING_FILL: if the pending entry order is still open on the exchange, it
                is a legitimate order (not orphaned) and is left alone; only OTHER open orders are
                cancelled. If that order is no longer open (and still no position), treated the same
                as the stale IN_POSITION case above.
            """
            symbol = self.config["symbol"]
            self.logger.info("Reconciliation check | symbol=%s", symbol)
    
            # Fetch open positions for this bot's symbol
            try:
                positions = get_open_positions(self.exchange, symbol)
            except Exception as exc:
                self.logger.error("Reconciliation: failed to fetch positions — halting, state unknown | error=%s", exc)
                self.halt("Reconciliation: could not fetch positions — state unknown.")
                return
    
            # No open position found.
            # Reconcile persisted state (Warm Start: this branch handles persisted
            # IN_POSITION and WAITING_FILL below) and cancel orphaned orders if any.
            if not positions:
                self.logger.info("Reconciliation: no open positions found | symbol=%s", symbol)
    
                # Fetch open orders once up front; reused by both the WAITING_FILL check
                # below and the orphaned-order cleanup at the end of this branch. A failed
                # fetch means "unknown", not "no orders" — halt rather than fall through.
                try:
                    open_orders = get_open_orders(self.exchange, symbol)
                except Exception as exc:
                    self.logger.error("Reconciliation: failed to fetch open orders — halting, state unknown | symbol=%s error=%s", symbol, exc)
                    self.halt("Reconciliation: could not fetch open orders — state unknown.")
                    return
    
                # Reconcile stale persisted mode against reality before touching orders.
                preserve_order_id: str | None = None
                if self.state is not None and self.state.mode == BotMode.IN_POSITION:
                    self.logger.critical(
                        "Reconciliation: persisted state was IN_POSITION but exchange shows no "
                        "position — it closed while the bot was down. Resetting to IDLE; the trade "
                        "record for that close is lost | symbol=%s",
                        symbol,
                    )
                    # Notify the user that the bot is resetting to IDLE and that the trade record is lost
                    self.notifier.notify_reconcile_issue(
                        self.__class__.__name__, symbol,
                        "persisted IN_POSITION but exchange shows no position — reset to IDLE, trade record lost",
                        "critical",
                    )
                    self.state.reset()
                # WAITING_FILL: if the pending entry order is still open, it's legitimate and left alone; only orphaned orders are cancelled. If that order is no longer open (and still no position), treated the same as the stale IN_POSITION case above
                elif self.state is not None and self.state.mode == BotMode.WAITING_FILL:
                    pending_id = self.state.entry_limit_order_id
                    if pending_id is not None and any(o.id == pending_id for o in open_orders):
                        self.logger.info(
                            "Reconciliation: resuming WAITING_FILL, pending entry order still open | "
                            "symbol=%s order_id=%s",
                            symbol, pending_id,
                        )
                        preserve_order_id = pending_id
                    else:
                        self.logger.critical(
                            "Reconciliation: persisted state was WAITING_FILL but the pending entry "
                            "order is no longer open and no position exists — resetting to IDLE | "
                            "symbol=%s order_id=%s",
                            symbol, pending_id,
                        )
                        self.notifier.notify_reconcile_issue(
                            self.__class__.__name__, symbol,
                            f"persisted WAITING_FILL order {pending_id} no longer open and no position — reset to IDLE",
                            "critical",
                        )
                        self.state.reset()
    
                # Cancel any orphaned orders left from a previous session (e.g. SL after TP filled),
                # but never cancel a still-legitimately-pending own entry order (preserve_order_id).
                try:
                    orphaned = [o for o in open_orders if o.id != preserve_order_id]
                    if orphaned:
                        self.logger.warning(
                            "Reconciliation: %d orphaned order(s) found with no position — cancelling | symbol=%s",
                            len(orphaned), symbol,
                        )
                        if preserve_order_id is None:
                            cancel_all_orders(self.exchange, symbol)
                        else:
                            for o in orphaned:
                                cancel_order(self.exchange, o.id, symbol)
                except Exception as exc:
                    self.logger.error("Reconciliation: failed to cancel orphaned orders | error=%s", exc)
                return
    
            # Get details of the open position
            position = positions[0] # there should only be one position per symbol
            size = position.size
            side = position.side
    
            self.logger.critical(
                "Reconciliation: orphaned position detected | symbol=%s side=%s size=%s",
                symbol, side.value, size,
            )
            self.notifier.notify_reconcile_issue(
                self.__class__.__name__, symbol,
                f"orphaned position detected side={side.value} size={size}",
                "critical",
            )
    
            if size <= 0:
                self.logger.error(
                    "Reconciliation: unreadable position (side=%s size=%s) — halting",
                    side.value, size,
                )
                self.halt("Reconciliation: unreadable orphaned position.")
                return
    
# Fetch open orders to check for SL protection. A failed fetch means "unknown",
            # not "no SL" — halt rather than falling through to an emergency close decision.
            try:
                open_orders = get_open_orders(self.exchange, symbol)
            except Exception as exc:
                self.logger.error("Reconciliation: failed to fetch open orders — halting, state unknown | symbol=%s error=%s", symbol, exc)
                self.halt("Reconciliation: could not fetch open orders — state unknown.")
                return

            # SL/TP are identified by persisted order ID, never by order-type string (docs/adr/0004):
            # the exchange says what exists, persisted state says which of it is this bot's SL/TP.
            sl_order_id = self.state.sl_order_id if self.state is not None else None
            tp_order_id = self.state.tp_order_id if self.state is not None else None
            sl_order = _find_order_by_id(open_orders, sl_order_id)
            tp_order = _find_order_by_id(open_orders, tp_order_id)

            # No persisted SL order ID, or the persisted SL order is no longer open — unaccounted
            # for. Treated as unprotected with no inspection of order types at all.
            if sl_order is None:
                self.logger.critical(
                    "Reconciliation: unprotected position (no persisted SL order still open) detected | "
                    "symbol=%s side=%s size=%s persisted_sl_order_id=%s",
                    symbol, side.value, size, sl_order_id,
                )
                # Persisted IN_POSITION: this bot has a full record of this trade, so on_halt()
                # owns the close (and, in subclasses, the trade record) — closing here too
                # would fire a second emergency close against an already-flat position.
                if self.state is not None and self.state.mode == BotMode.IN_POSITION:
                    self.halt(
                        "Reconciliation: unprotected orphaned position — emergency close executed.",
                        restartable=False,
                    )
                    return

                # No persisted position for this trade (fresh/IDLE state) — on_halt() will be
                # a no-op (mode != IN_POSITION), so reconcile() must perform the close itself.
                close_order = emergency_close(self.exchange, symbol, side, size)
                self.notifier.notify_emergency_close(
                    self.__class__.__name__, symbol, side.value,
                    "reconciliation: unprotected orphaned position", close_order is not None,
                )
                self.halt(
                    "Reconciliation: unprotected orphaned position — emergency close executed.",
                    restartable=False,
                )
                return
    
            self.logger.warning(
                "Reconciliation: orphaned position with SL intact — resuming IN_POSITION | "
                "symbol=%s side=%s size=%s sl_id=%s",
                symbol, side.value, size, sl_order.id,
            )

            # SL order found
            self.on_reconcile(position, sl_order, tp_order)


    def on_reconcile(self, position: NormalizedPosition, sl_order: NormalizedOrder, tp_order: NormalizedOrder | None) -> None:
        """
        Called by reconcile() when an orphaned position with SL protection is found.

        Default implementation restores side, mode, position_size, sl_order_id, and tp_order_id
        from exchange data. Override in subclasses to also restore additional strategy-specific
        state fields needed by exit logic (e.g. sl_price, tp_price for trailing stop strategies).

        position:  NormalizedPosition from the exchange
        sl_order:  the identified NormalizedOrder for the stop order
        tp_order:  the identified NormalizedOrder for the limit order, or None if not found
        """
        if self.state is None:
            return
        self.state.side          = position.side
        self.state.mode          = BotMode.IN_POSITION
        self.state.position_size = position.size
        self.state.sl_order_id   = sl_order.id
        self.state.tp_order_id   = tp_order.id if tp_order is not None else None

    def run(self) -> None:
        """
        Start the tick loop.

        Designed to be run inside a threading.Thread so multiple bots can execute concurrently.
        Fires tick() immediately on start, then waits for the appropriate interval before each
        subsequent tick: scan_interval when IDLE, position_interval when IN_POSITION.

        The loop exits cleanly when halt() is called or when
        max_consecutive_errors is exceeded.
        """
        self._stop_event.clear() # ensure the stop event is cleared before starting

        # Guard: ensure state was initialized by subclass
        if self.state is None:
            raise RuntimeError(
                f"{self.__class__.__name__} failed to initialize self.state in __init__(). "
                "Subclass __init__() must create and assign self.state = YourStateClass()."
            )

        if self.state.mode == BotMode.HALTED:
            self.logger.error("Bot.run() called on a halted bot — aborting.")
            return
        scan_interval:     float = float(self.config["scan_interval"]) 
        position_interval: float = float(self.config["position_interval"])
        self.logger.info(
            "Bot started | symbol=%s | timeframe=%s | scan_interval=%ss | position_interval=%ss",
            self.config.get("symbol"),
            self.config.get("timeframe"),
            scan_interval,
            position_interval,
        )

        # Tick loop
        while not self._stop_event.is_set():
            self.tick()
            self._stop_event.wait(timeout=self._sleep_seconds())

        self.logger.info("Bot stopped.")


    def record_entry_failure(self, reason: str) -> None:
        """
        Record a failed Entry Attempt and apply the Entry Failure Backoff (see CONTEXT.md).

        Bot authors must call this from every on_scan()-style bail-out path that acts
        on a signal and leaves the bot flat (mode still IDLE) — market order failure,
        zero position size, protective-order placement failure. Do not call it for
        can_trade() == False; that halts immediately instead (state and exchange disagree,
        so waiting only delays the alert — see docs/adr/0002).

        Increments state.entry_failure_count and sets state.entry_retry_after_tick so
        tick() suppresses on_tick() for a growing number of scans (1, then 2, then 4).
        Once max_entry_failures (default 3) consecutive failures have accumulated without
        an intervening successful entry, halts with restartable=False — a restart would
        only repeat the same failing attempt.

        Args:
            reason: why the entry attempt failed; logged and passed to halt() if the cap is hit.
        """
        if self.state is None:
            return
        self.state.entry_failure_count += 1
        backoff_scans = 2 ** (self.state.entry_failure_count - 1) # 1, 2, 4, ...
        self.state.entry_retry_after_tick = self.state.lifetime_tick_count + backoff_scans
        max_failures: int = int(self.config.get("max_entry_failures", 3))
        self.logger.warning(
            "Entry attempt failed (%d/%d) | reason=%s | retry after tick %d",
            self.state.entry_failure_count, max_failures, reason, self.state.entry_retry_after_tick,
        )
        if self.state.entry_failure_count >= max_failures:
            self.halt(
                f"Exceeded {max_failures} consecutive entry failures — last: {reason}",
                restartable=False,
            )


    def tick(self) -> None:
            """
            Execute a single tick.
    
            Gated on calendar.is_open(): when the market is closed, both fetches and
            on_tick() are skipped entirely and _consecutive_errors is left untouched
            (stops the overnight halt cascade for non-24/7 calendars; 24/7 is always open).
    
            OHLCV data (primary timeframe) and bid/ask are only fetched when the bot is IDLE, where they are needed for entry evaluation.
    
            Staleness backstop: for non-24/7 calendars, if the newest closed bar has not
            advanced since the previous tick (a holiday with no holiday list still passes
            is_open()), on_tick() is skipped for this tick only.
    
            Entry Failure Backoff: while IDLE, on_tick() is also skipped for as long as
            state.lifetime_tick_count has not yet reached state.entry_retry_after_tick, set by
            record_entry_failure() after a failed entry attempt (see CONTEXT.md,
            docs/adr/0002). A tick that detects a successful entry (mode transitions to
            IN_POSITION) resets the backoff.

            Same-Bar Re-Entry Guard: while IDLE, on_tick() is also skipped for as long as
            state.last_closed_bar has not advanced past state.last_exit_bar — the newest
            closed bar as of the last exit from IN_POSITION (see CONTEXT.md, ADR 0003).
            A tick that detects a transition out of IN_POSITION (normal close or halt())
            stamps last_exit_bar from the last_closed_bar computed earlier this tick. That
            same transition is also compared against state.lifetime_trade_count: if it did
            not move, close_trade() was likely never called, and a WARNING is logged
            (BaseBot has no chokepoint on order placement, so this cannot be enforced, only
            detected).

            Exceptions raised during data fetching or inside on_tick() are
            caught, logged, and counted. If the count reaches
            max_consecutive_errors (default 5), halt() is called
            automatically. A successful tick resets the counter to zero.
            """
            try:
                symbol    = self.config["symbol"]
                timeframe = self.config["timeframe"]
                limit     = int(self.config["limit"])
    
                if not self.calendar.is_open(pd.Timestamp(datetime.now(timezone.utc))):
                    self.logger.debug("Market closed (calendar=%s) — skipping tick.", self.calendar.name)
                    return
    
                if self.state is not None:
                    self.state.lifetime_tick_count += 1
    
                if self.state is not None and self.state.mode == BotMode.IDLE:
                    df           = get_ohlcv_df(self.exchange, symbol, timeframe, limit)
                    self.bid_ask = get_bid_ask(self.exchange, symbol)
                else:
                    df = None
                    self.bid_ask = get_bid_ask(self.exchange, symbol)
    
                skip_on_tick = False
                if self.state is not None:
                    previous_closed_bar = self.state.last_closed_bar
                    closed = closed_bars(df) if df is not None else None
                    self.state.last_closed_bar = (
                        pd.Timestamp(closed.index[-1]).to_pydatetime()
                        if closed is not None and not closed.empty
                        else _last_closed_bar_time(timeframe, self.calendar)
                    )
                    skip_on_tick = (
                        # Reconsidered for issue #2: kept 24/7-exempt. This backstop exists for
                        # calendar-close staleness (e.g. a holiday with no holiday list), not to
                        # gate re-evaluation when scan_interval < timeframe; 24/7 bots are expected
                        # to re-run on_tick() every scan even without a new closed bar (e.g. to
                        # re-check bid/ask while IN_POSITION).
                        self.calendar.name != "24/7"
                        and previous_closed_bar is not None
                        and self.state.last_closed_bar <= previous_closed_bar
                    )
    
                # Entry Failure Backoff (see CONTEXT.md, docs/adr/0002): suppress on_tick()
                # entirely while IDLE and still waiting out a growing number of scans after
                # a failed entry attempt, recorded via record_entry_failure().
                backoff_active = (
                    self.state is not None
                    and self.state.mode == BotMode.IDLE
                    and self.state.lifetime_tick_count <= self.state.entry_retry_after_tick
                )
    
                # Same-Bar Re-Entry Guard (see CONTEXT.md, ADR 0003): suppress on_tick()
                # entirely while IDLE and the newest closed bar is still the one that was
                # newest at the last exit from IN_POSITION (stamped below, and in halt()).
                reentry_guard_active = (
                    self.state is not None
                    and self.state.mode == BotMode.IDLE
                    and self.state.last_exit_bar is not None
                    and self.state.last_closed_bar is not None
                    and self.state.last_closed_bar <= self.state.last_exit_bar
                )

                if skip_on_tick:
                    self.logger.debug("No new closed bar (calendar=%s) — skipping on_tick().", self.calendar.name)
                elif backoff_active:
                    self.logger.debug(
                        "Entry failure backoff active (retry after tick %d, currently %d) — skipping on_tick().",
                        self.state.entry_retry_after_tick, self.state.lifetime_tick_count,
                    )
                elif reentry_guard_active:
                    self.logger.debug(
                        "Same-Bar Re-Entry Guard active (last_exit_bar=%s, last_closed_bar=%s) — skipping on_tick().",
                        self.state.last_exit_bar, self.state.last_closed_bar,
                    )
                else:
                    mode_before        = self.state.mode if self.state is not None else None
                    trade_count_before = self.state.lifetime_trade_count if self.state is not None else None
                    self.on_tick(df)
                    if self.state is not None and mode_before is not None and mode_before != self.state.mode:
                        if mode_before != BotMode.IN_POSITION and self.state.mode == BotMode.IN_POSITION:
                            self.state.entry_failure_count      = 0
                            self.state.entry_retry_after_tick   = 0
                            self.notifier.notify_position_opened(
                                self.__class__.__name__, symbol,
                                self.state.side.value if self.state.side else "unknown",
                                self.state.entry_price, self.state.position_size,
                            )
                        if mode_before == BotMode.IN_POSITION and self.state.mode != BotMode.IN_POSITION:
                            self.state.last_exit_bar = self.state.last_closed_bar
                            if self.state.lifetime_trade_count == trade_count_before:
                                self.logger.warning(
                                    "Mode left IN_POSITION but lifetime_trade_count did not change — "
                                    "close_trade() may not have been called | symbol=%s new_mode=%s",
                                    symbol, self.state.mode.value,
                                )
                self._consecutive_errors = 0

            except Exception as exc:
                self._consecutive_errors += 1
                max_errors: int = int(self.config.get("max_consecutive_errors", 5))
                self.logger.exception(
                    "Tick error (%d/%d): %s",
                    self._consecutive_errors,
                    max_errors,
                    exc,
                ) # log as ERROR with stack trace
                if self._consecutive_errors >= max_errors:
                    self.halt(
                        f"Exceeded {max_errors} consecutive tick errors — last: {exc}"
                    )
    
            finally:
                # Persist on every tick outcome (success, caught error, or market-closed
                # early return) so counters/mode never lag behind by more than one tick.
                # Locked against halt() (see _state_lock) so an external halt() call
                # racing with this tick can't have its HALTED write clobbered below.
                with self._state_lock:
                    self.persist_state()


    def on_tick(self, df: pd.DataFrame) -> None:
        """
        Strategy entry point called on every tick.

        Concrete mode dispatcher: routes to on_scan() (IDLE), on_position() (IN_POSITION),
        or on_pending_fill() (WAITING_FILL); does nothing when HALTED or when state is None.

        Still overridable — a bot that wants the raw hook instead of per-mode dispatch
        overrides this method directly (e.g. StockBot's no-op skeleton).

        df: OHLCV DataFrame
        """
        if self.state is None or self.state.mode == BotMode.HALTED:
            return
        if self.state.mode == BotMode.IDLE:
            self.on_scan(df)
        elif self.state.mode == BotMode.IN_POSITION:
            self.on_position(df)
        elif self.state.mode == BotMode.WAITING_FILL:
            self.on_pending_fill(df)


    def on_scan(self, df: pd.DataFrame) -> None:
        """
        Called by on_tick() while IDLE (scanning for entry). Default no-op.

        Args:
            df: OHLCV DataFrame
        """
        pass


    def on_position(self, df: pd.DataFrame) -> None:
        """
        Called by on_tick() while IN_POSITION (monitoring exit). Default no-op.

        Args:
            df: OHLCV DataFrame
        """
        pass


    def on_pending_fill(self, df: pd.DataFrame) -> None:
        """
        Called by on_tick() while WAITING_FILL (awaiting a resting entry order fill). Default no-op.

        Args:
            df: OHLCV DataFrame
        """
        pass


    def close_trade(self, reason: str, exit_price: float | None, exit_fee: Fee | None = None, exit_time: datetime | None = None) -> None:
        """
        Finalize a closed trade: generic P&L/fee accounting, TradeRecord write, and state.reset().

        Bot-author-facing contract surface, alongside record_entry_failure() and
        get_sizing_equity() — call this from every exchange-side close (SL/TP fill,
        emergency close) once the closing order has been resolved.

        Args:
            reason:     free-form exit reason, passed directly into TradeRecord.exit_reason
                       (e.g. "sl", "tp", "emergency_halt"). No enum — vocabulary is bot-specific.
            exit_price: actual average fill price of the closing order. The caller resolves
                       this (including any fallback to a planned SL/TP price); None means the
                       closing order failed entirely, and yields outcome="unknown".
            exit_fee:   fee charged on the closing order; None if unavailable.
            exit_time:  exchange-side fill timestamp; falls back to datetime.now() when None
                       (e.g. order response missing timestamp field).
        """
        self.state.lifetime_trade_count += 1
        exit_time = exit_time or datetime.now(timezone.utc)

        try:
            direction_sign = 1 if self.state.side == Side.LONG else -1
            if exit_price is not None and self.state.entry_price is not None:
                gross_pnl = (exit_price - self.state.entry_price) * self.state.position_size * direction_sign
            else:
                gross_pnl = 0.0  # cannot compute without both prices

            # Compute fees and net P&L
            entry_fee_cost = self.state.entry_fee.cost if self.state.entry_fee else 0.0
            exit_fee_cost  = exit_fee.cost             if exit_fee           else 0.0
            funding_paid   = None
            if exit_price is not None and self.state.entry_time is not None:
                funding_paid = self.exchange.fetch_funding_payments(
                    self.config["symbol"], self.state.entry_time, exit_time,
                )
            net_pnl        = gross_pnl - entry_fee_cost - exit_fee_cost - (funding_paid or 0.0)
            notional       = self.state.entry_price * self.state.position_size if self.state.entry_price else 0.0
            net_pnl_pct    = net_pnl / notional if notional else 0.0
            duration       = int((exit_time - self.state.entry_time).total_seconds()) if self.state.entry_time else 0
            settlement_ccy = self.config.get("settlement_currency", self.config["quote_currency"])

            # Derive outcome purely from realized P&L, independent of reason
            if exit_price is None or self.state.entry_price is None:
                trade_outcome = "unknown"
            elif net_pnl > 0:
                trade_outcome = "win"
            elif net_pnl < 0:
                trade_outcome = "loss"
            else:
                trade_outcome = "breakeven"

            record = TradeRecord(
                trade_id            = str(uuid.uuid4()),
                bot_name            = self.__class__.__name__,
                exchange            = type(self.exchange).__name__,
                symbol              = self.config["symbol"],
                settlement_currency = settlement_ccy,
                side                = self.state.side.value,
                entry_time          = self.state.entry_time,
                entry_price         = self.state.entry_price,
                position_size       = self.state.position_size,
                entry_fee           = self.state.entry_fee.cost if self.state.entry_fee else None,
                planned_sl_price    = self.state.sl_price,
                planned_tp_price    = self.state.tp_price,
                exit_time           = exit_time,
                exit_price          = exit_price,
                exit_reason         = reason,
                exit_fee            = exit_fee.cost if exit_fee else None,
                gross_pnl           = gross_pnl,
                funding_paid        = funding_paid,
                net_pnl             = net_pnl,
                net_pnl_pct         = net_pnl_pct,
                duration_seconds    = duration,
                outcome             = trade_outcome,
            )
            self.logger.info(
                "Trade closed | reason=%s entry=%.4f exit=%s size=%s pnl=%.4f",
                reason, self.state.entry_price, exit_price, self.state.position_size, net_pnl,
            )
            self.trade_logger.log(record)
        except Exception as exc:
            self.logger.error("Failed to write trade record: %s", exc)

        self.state.reset()


    def get_sizing_equity(self) -> float:
        """
        Return the equity this bot should size positions against.

        Reads config keys "equity_scope" ("exchange" default, or "system") and
        "capital_fraction" (default 1.0, multiplied into the result so multiple
        bots sharing one portfolio don't each size against the full pool).

        "system" scope requires self.portfolio to be wired (live.py only) and
        falls back to exchange-local equity when it isn't (e.g. backtests, where
        one bot + one exchange makes exchange-local equity == system equity by
        definition — see memories/repo/portfolio_system_equity_plan.md).
        """
        scope = self.config.get("equity_scope", "exchange")
        capital_fraction = self.config.get("capital_fraction", 1.0)

        if scope == "system" and self.portfolio is not None:
            equity = self.portfolio.total_equity()
        else:
            equity = self.exchange.get_equity([self.config["symbol"]], self.config["quote_currency"])

        return equity * capital_fraction


    def persist_state(self) -> None:
        """
        Save the bot's current state to SQLite, if persistence is enabled.

        No-op when _db_path is None (persistence disabled, e.g. backtests) or when
        _persist_state_fn hasn't been injected. Never raises — persistence is a
        safety net and must not crash the bot; failures are logged as a warning.
        """
        if self._db_path is None or self.state is None or self._persist_state_fn is None:
            return
        try:
            self._persist_state_fn(self._db_path, self._bot_key, self.state)
        except Exception as exc:
            self.logger.warning("Failed to persist state | error=%s", exc)


    def _sleep_seconds(self) -> float:
        """
        Compute how long run() should sleep before the next tick.

        IN_POSITION: always position_interval (unchanged).
        Otherwise, when the market is open: scan_interval (unchanged).
        When the market is closed: min(scan_interval, seconds until calendar.next_open()),
        so the loop never sleeps past the next open. 24/7 -> always open -> unchanged.
        """
        scan_interval:     float = float(self.config["scan_interval"])
        position_interval: float = float(self.config["position_interval"])

        if self.state is not None and self.state.mode == BotMode.IN_POSITION:
            return position_interval

        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        if self.calendar.is_open(now_utc):
            return scan_interval

        seconds_until_open = (pd.Timestamp(self.calendar.next_open(now_utc)) - now_utc).total_seconds()
        return min(scan_interval, max(0.0, seconds_until_open))


    def halt(self, reason: str, level: str = "error", restartable: bool = True) -> None:
        """
        Halt the bot.

        Records halt_reason on state, logs the event, calls on_halt() for exchange cleanup,
        then sets state.mode to BotMode.HALTED and signals the tick loop to exit.

        Stamps state.last_exit_bar from state.last_closed_bar when called while
        IN_POSITION — halt() is a transition out of IN_POSITION that bypasses tick()'s
        own stamping (see CONTEXT.md, ADR 0003), and it can be called from outside the
        tick loop (command listener, drawdown monitor) or from on_halt()'s emergency
        close, where instant re-entry after a watchdog restart is worst.

        Args:
            reason:      Why the bot was halted.
            level:       Log level for the halt message (default "error"). Use "info" for deliberate manual stops.
            restartable: Whether the watchdog may auto-restart this bot (default True). Pass
                         False for deliberate/system halts (manual stop, circuit breaker) that
                         should not be fought by the watchdog.
        """
        mode_before = self.state.mode if self.state is not None else None
        if self.state is not None:
            self.state.halt_reason = reason
        log = getattr(self.logger, level, self.logger.error) # log level method
        log("HALTED — %s", reason)
        self.notifier.notify_halt(self.__class__.__name__, reason, level, restartable)
        try:
            self.on_halt()
        except Exception as exc:
            self.logger.exception("on_halt() raised an exception — stop event still set: %s", exc)
        finally:
            if self.state is not None:
                if mode_before == BotMode.IN_POSITION:
                    self.state.last_exit_bar = self.state.last_closed_bar
                self.state.mode = BotMode.HALTED
                self.state.halt_restartable = restartable
            # halt() may be called from outside the tick loop (command listener,
            # drawdown monitor) — tick()'s own save may never run again after this.
            # Locked together with the mutation above so a concurrently-running
            # tick() in the bot's own thread can't persist a stale snapshot after
            # this write and silently clobber the HALTED status.
            with self._state_lock:
                self.persist_state()
            self._stop_event.set()


    def on_halt(self) -> NormalizedOrder | None:
        """
        Called by halt() before the tick loop exits.

        Default implementation handles two cases:
          - WAITING_FILL: cancels the pending limit entry order and returns None.
          - IN_POSITION:  triggers an emergency market close, cancels orphaned SL/TP orders,
                          writes the "emergency_halt" trade record via close_trade(), and
                          returns the close NormalizedOrder.
        If not in either of those modes, returns None immediately, nothing to clean up.
        If the emergency close fails, leaves the SL intact on the exchange as the last line of
        defence and no trade record is written (position is still open).

        Subclasses should call super().on_halt() first, then handle any remaining
        strategy-specific cleanup.

        Returns:
            The emergency close NormalizedOrder if an emergency close was executed, or None otherwise.
        """
        if self.state is None:
            return None

        # Cancel pending entry order if bot is waiting for fill
        if self.state.mode == BotMode.WAITING_FILL:
            if self.state.entry_limit_order_id is not None:
                self.logger.warning(
                    "Halt cleanup: cancelling pending entry order | symbol=%s order_id=%s",
                    self.config["symbol"], self.state.entry_limit_order_id,
                )
                if not cancel_order(self.exchange, self.state.entry_limit_order_id, self.config["symbol"]):
                    self.logger.warning(
                        "cancel_order failed for pending entry — attempting cancel_all_orders | symbol=%s order_id=%s",
                        self.config["symbol"], self.state.entry_limit_order_id,
                    )
                    if not cancel_all_orders(self.exchange, self.config["symbol"]):
                        self.logger.critical(
                            "Failed to cancel pending entry order on halt — order may fill unmanaged, manual intervention required | symbol=%s order_id=%s",
                            self.config["symbol"], self.state.entry_limit_order_id,
                        )
            return None

        if self.state.mode != BotMode.IN_POSITION:
            return None

        self.logger.warning(
            "Halt cleanup triggered | symbol=%s side=%s",
            self.config["symbol"], self.state.side.value if self.state.side else "unknown",
        )

        # Emergency close first, if it fails, leave SL intact as last line of defence
        order = None
        if self.state.side is not None and self.state.position_size is not None:
            order = emergency_close(self.exchange, self.config["symbol"], self.state.side, self.state.position_size)

        side_label = self.state.side.value if self.state.side else "unknown"
        if order is None:
            self.logger.error(
                "Emergency close failed on halt — leaving SL intact | symbol=%s",
                self.config["symbol"],
            )
            self.notifier.notify_emergency_close(
                self.__class__.__name__, self.config["symbol"], side_label,
                self.state.halt_reason or "halt", False,
            )
            return None

        # Close succeeded, cancel SL/TP (now orphaned)
        if self.state.sl_order_id is not None:
            cancel_order(self.exchange, self.state.sl_order_id, self.config["symbol"])
        if self.state.tp_order_id is not None:
            cancel_order(self.exchange, self.state.tp_order_id, self.config["symbol"])

        self.notifier.notify_emergency_close(
            self.__class__.__name__, self.config["symbol"], side_label,
            self.state.halt_reason or "halt", True,
        )

        exit_price = order.fill_price
        _ts        = order.timestamp
        exit_time  = datetime.fromtimestamp(_ts / 1000, tz=timezone.utc) if _ts else None
        self.close_trade("emergency_halt", exit_price=exit_price, exit_time=exit_time)

        return order


    def build_summary(self) -> dict[str, Any]:
        """
        Build a generic snapshot for the periodic summary alert (runtime/summary.py).

        Works for any bot subclass with zero strategy-specific code: reads only
        BaseState fields plus self.bid_ask. Unrealized P&L is computed from
        self.bid_ask's mid price using the same formula CryptoBot already uses
        inline for its debug log (bot.py); None when not IN_POSITION or bid_ask/
        entry_price is unavailable.

        Returns:
            dict with keys: symbol, mode, side, entry_price, position_size,
                          unrealized_pnl, lifetime_trade_count.
        """
        unrealized_pnl: float | None = None
        if (
            self.state is not None
            and self.state.mode == BotMode.IN_POSITION
            and self.state.entry_price is not None
            and self.bid_ask is not None
        ):
            mid = (self.bid_ask["bid"] + self.bid_ask["ask"]) / 2
            direction_sign = 1 if self.state.side == Side.LONG else -1
            unrealized_pnl = (mid - self.state.entry_price) * self.state.position_size * direction_sign

        return {
            "symbol":         self.config.get("symbol"),
            "mode":           self.state.mode.value if self.state is not None else None,
            "side":           self.state.side.value if self.state is not None and self.state.side else None,
            "entry_price":    self.state.entry_price if self.state is not None else None,
            "position_size":  self.state.position_size if self.state is not None else None,
            "unrealized_pnl": unrealized_pnl,
            "lifetime_trade_count": self.state.lifetime_trade_count if self.state is not None else 0,
        }

