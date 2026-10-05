"""
================================================================================
UNIT TESTS FOR BASE_BOT.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_bots/test_base_bot.py -v

To run a specific test function:
    pytest tests/test_bots/test_base_bot.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import Fee, NormalizedPosition

################### HELPERS ##########################

CONFIG = {
    "symbol":                 "BTC/USDT",
    "quote_currency":         "USDT",
    "timeframe":              "1h",
    "limit":                  100,
    "scan_interval":          60,
    "position_interval":      10,
    "max_consecutive_errors": 5,
}


class ConcreteBot(BaseBot):
    """Minimal concrete subclass of BaseBot for testing."""
    def on_tick(self, df: pd.DataFrame) -> None:
        pass


def make_bot() -> ConcreteBot:
    """Return a ConcreteBot with a BaseState attached."""
    bot = ConcreteBot(exchange=MagicMock(), config=CONFIG)
    bot.state = BaseState()
    return bot


################### TESTS ##########################

# ── halt() ────────────────────────────────────────────────────────

# halt_reason is written to state before on_halt() is called
def test_halt_sets_halt_reason_before_on_halt():
    bot = make_bot()
    captured = {}

    def spy():
        captured["halt_reason"] = bot.state.halt_reason

    with patch.object(bot, "on_halt", side_effect=spy):
        bot.halt("risk breach")

    assert captured["halt_reason"] == "risk breach"

# mode is set to HALTED in the finally block even if on_halt() raises
def test_halt_sets_mode_halted_even_if_on_halt_raises():
    bot = make_bot()

    with patch.object(bot, "on_halt", side_effect=RuntimeError("exchange down")):
        bot.halt("some reason")

    assert bot.state.mode == BotMode.HALTED

# _stop_event is set after halt completes
def test_halt_sets_stop_event():
    bot = make_bot()

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("some reason")

    assert bot._stop_event.is_set()

# level="info" routes to logger.info; the default routes to logger.error
def test_halt_default_level_logs_error():
    bot = make_bot()

    with patch.object(bot, "on_halt", return_value=None), \
         patch.object(bot.logger, "error") as mock_error:
        bot.halt("some reason")

    assert mock_error.called
    logged = str(mock_error.call_args_list)
    assert "HALTED" in logged

# level="info" routes to logger.info instead of logger.error
def test_halt_info_level_logs_info():
    bot = make_bot()

    with patch.object(bot, "on_halt", return_value=None), \
         patch.object(bot.logger, "info") as mock_info:
        bot.halt("planned stop", level="info")

    assert mock_info.called
    logged = str(mock_info.call_args_list)
    assert "HALTED" in logged

# restartable defaults to True when the kwarg is omitted
def test_halt_default_restartable_is_true():
    bot = make_bot()

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("some reason")

    assert bot.state.halt_restartable is True

# restartable=False is recorded on state (deliberate/system halts)
def test_halt_explicit_restartable_false():
    bot = make_bot()

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("manual stop", restartable=False)

    assert bot.state.halt_restartable is False


# halt() while IN_POSITION stamps last_exit_bar from last_closed_bar (Same-Bar
# Re-Entry Guard, issue #6) — covers exit paths that bypass tick(), e.g. an
# emergency close inside on_halt() or a halt() called from another thread
def test_halt_in_position_stamps_last_exit_bar():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION
    bot.state.last_closed_bar = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("some reason")

    assert bot.state.last_exit_bar == bot.state.last_closed_bar

# halt() while not IN_POSITION (e.g. already IDLE or WAITING_FILL) does not stamp
def test_halt_not_in_position_does_not_stamp_last_exit_bar():
    bot = make_bot()
    bot.state.mode = BotMode.IDLE
    bot.state.last_closed_bar = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("some reason")

    assert bot.state.last_exit_bar is None


# ── record_entry_failure() ────────────────────────────────────────────────────

# first failure: counter becomes 1, backoff is 1 scan, cap (default 3) not reached
def test_record_entry_failure_first_failure_sets_backoff_of_one_scan():
    bot = make_bot()
    bot.state.lifetime_tick_count = 10

    with patch.object(bot, "halt") as mock_halt:
        bot.record_entry_failure("market order failed")

    assert bot.state.entry_failure_count == 1
    assert bot.state.entry_retry_after_tick == 11
    mock_halt.assert_not_called()

# second consecutive failure: counter becomes 2, backoff grows to 2 scans
def test_record_entry_failure_second_failure_sets_backoff_of_two_scans():
    bot = make_bot()
    bot.state.lifetime_tick_count = 10
    bot.state.entry_failure_count = 1

    with patch.object(bot, "halt") as mock_halt:
        bot.record_entry_failure("market order failed")

    assert bot.state.entry_failure_count == 2
    assert bot.state.entry_retry_after_tick == 12
    mock_halt.assert_not_called()

# third consecutive failure with default max_entry_failures=3: halts, non-restartable
def test_record_entry_failure_hits_default_cap_halts_non_restartable():
    bot = make_bot()
    bot.state.entry_failure_count = 2

    with patch.object(bot, "halt") as mock_halt:
        bot.record_entry_failure("sl/tp placement failed")

    assert bot.state.entry_failure_count == 3
    mock_halt.assert_called_once()
    assert mock_halt.call_args.kwargs["restartable"] is False

# max_entry_failures is configurable — cap of 1 halts on the very first failure
def test_record_entry_failure_respects_configured_max():
    bot = make_bot()
    bot.config["max_entry_failures"] = 1

    with patch.object(bot, "halt") as mock_halt:
        bot.record_entry_failure("position_size == 0")

    assert bot.state.entry_failure_count == 1
    mock_halt.assert_called_once()
    assert mock_halt.call_args.kwargs["restartable"] is False

# below the cap, halt() is never invoked
def test_record_entry_failure_below_cap_does_not_halt():
    bot = make_bot()
    bot.config["max_entry_failures"] = 5
    bot.state.entry_failure_count = 3

    with patch.object(bot, "halt") as mock_halt:
        bot.record_entry_failure("market order failed")

    mock_halt.assert_not_called()


# ── get_sizing_equity() ──────────────────────────────────────────────────────

# default equity_scope ("exchange"): uses exchange.get_equity(), ignores portfolio
def test_get_sizing_equity_default_scope_uses_exchange_equity():
    bot = make_bot()
    bot.portfolio = MagicMock()
    bot.exchange.get_equity.return_value = 1000.0

    result = bot.get_sizing_equity()

    bot.exchange.get_equity.assert_called_once_with(["BTC/USDT"], "USDT")
    bot.portfolio.total_equity.assert_not_called()
    assert result == 1000.0

# equity_scope="system" with a portfolio wired: uses portfolio.total_equity()
def test_get_sizing_equity_system_scope_uses_portfolio():
    bot = ConcreteBot(exchange=MagicMock(), config={**CONFIG, "equity_scope": "system"})
    bot.state = BaseState()
    bot.portfolio = MagicMock()
    bot.portfolio.total_equity.return_value = 5000.0

    result = bot.get_sizing_equity()

    bot.exchange.get_equity.assert_not_called()
    assert result == 5000.0

# equity_scope="system" with no portfolio wired (e.g. backtests): falls back to exchange equity
def test_get_sizing_equity_system_scope_without_portfolio_falls_back():
    bot = ConcreteBot(exchange=MagicMock(), config={**CONFIG, "equity_scope": "system"})
    bot.state = BaseState()
    bot.exchange.get_equity.return_value = 2000.0

    result = bot.get_sizing_equity()

    bot.exchange.get_equity.assert_called_once_with(["BTC/USDT"], "USDT")
    assert result == 2000.0

# capital_fraction multiplies the scoped equity
def test_get_sizing_equity_applies_capital_fraction():
    bot = ConcreteBot(exchange=MagicMock(), config={**CONFIG, "capital_fraction": 0.25})
    bot.state = BaseState()
    bot.exchange.get_equity.return_value = 1000.0

    result = bot.get_sizing_equity()

    assert result == 250.0


# ── close_trade() ──────────────────────────────────────────────────────────

def make_in_position_bot() -> ConcreteBot:
    """Return a ConcreteBot in an open long position, ready to close_trade()."""
    bot = make_bot()
    bot.state.mode        = BotMode.IN_POSITION
    bot.state.side        = Side.LONG
    bot.state.entry_price = 100.0
    bot.state.entry_time  = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bot.state.position_size = 1.0
    bot.exchange.fetch_funding_payments.return_value = 0.0
    return bot

# winning long trade: gross/net P&L computed, outcome "win", lifetime_trade_count incremented, state reset
def test_close_trade_win_computes_pnl_and_resets_state():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("tp", exit_price=110.0, exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

    mock_log.assert_called_once()
    record = mock_log.call_args.args[0]
    assert record.gross_pnl == 10.0
    assert record.net_pnl == 10.0
    assert record.outcome == "win"
    assert record.exit_reason == "tp"
    assert bot.state.lifetime_trade_count == 1
    assert bot.state.mode == BotMode.IDLE  # state.reset() was called
    assert bot.state.entry_price is None

# losing trade: net_pnl negative, outcome "loss"
def test_close_trade_loss_outcome():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("sl", exit_price=90.0, exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

    record = mock_log.call_args.args[0]
    assert record.gross_pnl == -10.0
    assert record.net_pnl == -10.0
    assert record.outcome == "loss"

# exact breakeven (no fees): net_pnl == 0, outcome "breakeven"
def test_close_trade_breakeven_outcome():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("sl", exit_price=100.0, exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

    record = mock_log.call_args.args[0]
    assert record.net_pnl == 0.0
    assert record.outcome == "breakeven"

# exit_price is None (closing order failed entirely): outcome "unknown", gross_pnl 0.0
def test_close_trade_none_exit_price_yields_unknown_outcome():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("emergency_missing_orders", exit_price=None)

    record = mock_log.call_args.args[0]
    assert record.gross_pnl == 0.0
    assert record.outcome == "unknown"
    assert record.exit_price is None

# entry_fee/exit_fee are subtracted from gross_pnl to compute net_pnl
def test_close_trade_subtracts_entry_and_exit_fees():
    bot = make_in_position_bot()
    bot.state.entry_fee = Fee(cost=1.0, currency="USDT")
    exit_fee = Fee(cost=0.5, currency="USDT")

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("tp", exit_price=110.0, exit_fee=exit_fee, exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

    record = mock_log.call_args.args[0]
    assert record.gross_pnl == 10.0
    assert record.net_pnl == 8.5  # 10.0 - 1.0 - 0.5

# exit_time=None falls back to datetime.now(timezone.utc)
def test_close_trade_none_exit_time_falls_back_to_now():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log") as mock_log:
        bot.close_trade("tp", exit_price=110.0, exit_time=None)

    record = mock_log.call_args.args[0]
    assert record.exit_time is not None
    assert isinstance(record.exit_time, datetime)

# a TradeLogger failure is caught and logged, but the structured info line still fires
# and state.reset() still runs
def test_close_trade_log_failure_still_resets_state():
    bot = make_in_position_bot()

    with patch.object(bot.trade_logger, "log", side_effect=Exception("disk full")), \
         patch.object(bot.logger, "error") as mock_error, \
         patch.object(bot.logger, "info") as mock_info:
        bot.close_trade("tp", exit_price=110.0, exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

    mock_error.assert_called_once()
    mock_info.assert_called_once()
    assert bot.state.mode == BotMode.IDLE
    assert bot.state.lifetime_trade_count == 1


# ── on_tick() dispatch ───────────────────────────────────────────────

class DispatchBot(BaseBot):
    """Concrete BaseBot subclass that overrides the per-mode hooks (not on_tick itself) to record calls."""
    def on_scan(self, df: pd.DataFrame) -> None:
        self.scan_called = True
    def on_position(self, df: pd.DataFrame) -> None:
        self.position_called = True
    def on_pending_fill(self, df: pd.DataFrame) -> None:
        self.pending_fill_called = True


def make_dispatch_bot() -> DispatchBot:
    """Return a DispatchBot with a BaseState attached and call flags cleared."""
    bot = DispatchBot(exchange=MagicMock(), config=CONFIG)
    bot.state = BaseState()
    bot.scan_called = False
    bot.position_called = False
    bot.pending_fill_called = False
    return bot

# IDLE dispatches to on_scan() only
def test_on_tick_dispatches_to_on_scan_when_idle():
    bot = make_dispatch_bot()
    bot.state.mode = BotMode.IDLE

    bot.on_tick(pd.DataFrame())

    assert bot.scan_called is True
    assert bot.position_called is False
    assert bot.pending_fill_called is False

# IN_POSITION dispatches to on_position() only
def test_on_tick_dispatches_to_on_position_when_in_position():
    bot = make_dispatch_bot()
    bot.state.mode = BotMode.IN_POSITION

    bot.on_tick(pd.DataFrame())

    assert bot.scan_called is False
    assert bot.position_called is True
    assert bot.pending_fill_called is False

# WAITING_FILL dispatches to on_pending_fill() only
def test_on_tick_dispatches_to_on_pending_fill_when_waiting_fill():
    bot = make_dispatch_bot()
    bot.state.mode = BotMode.WAITING_FILL

    bot.on_tick(pd.DataFrame())

    assert bot.scan_called is False
    assert bot.position_called is False
    assert bot.pending_fill_called is True

# HALTED dispatches to nothing
def test_on_tick_dispatches_to_nothing_when_halted():
    bot = make_dispatch_bot()
    bot.state.mode = BotMode.HALTED

    bot.on_tick(pd.DataFrame())

    assert bot.scan_called is False
    assert bot.position_called is False
    assert bot.pending_fill_called is False


# ── on_halt() ────────────────────────────────────────────────────────

# not IN_POSITION, returns None immediately, no exchange calls made
def test_on_halt_not_in_position_returns_none():
    bot = make_bot()
    bot.state.mode = BotMode.IDLE

    with patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_not_called()
    mock_cancel.assert_not_called()


# IN_POSITION, close succeeds, cancels both SL and TP, returns order dict
def test_on_halt_in_position_close_succeeds_cancels_sl_and_tp(make_order):
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = MagicMock()          # Side.LONG equivalent
    bot.state.sl_order_id   = "sl-123"
    bot.state.tp_order_id   = "tp-456"

    fake_order = make_order(fill_price=50000.0, timestamp=1700000000000)

    with patch("lighthouse.bots.base_bot.emergency_close", return_value=fake_order) as mock_close, \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch.object(bot, "close_trade") as mock_close_trade:
        result = bot.on_halt()

    assert result == fake_order
    mock_close.assert_called_once()
    assert mock_cancel.call_count == 2
    cancelled_ids = {call.args[1] for call in mock_cancel.call_args_list}
    assert cancelled_ids == {"sl-123", "tp-456"}
    mock_close_trade.assert_called_once()
    call_kwargs = mock_close_trade.call_args
    assert call_kwargs.args[0] == "emergency_halt"
    assert call_kwargs.kwargs["exit_price"] == 50000.0
    assert call_kwargs.kwargs["exit_time"] is not None


# IN_POSITION, close succeeds, no TP, only SL is cancelled
def test_on_halt_in_position_close_succeeds_no_tp(make_order):
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = MagicMock()
    bot.state.sl_order_id   = "sl-123"
    bot.state.tp_order_id   = None

    fake_order = make_order(fill_price=50000.0, timestamp=1700000000000)

    with patch("lighthouse.bots.base_bot.emergency_close", return_value=fake_order), \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch.object(bot, "close_trade") as mock_close_trade:
        result = bot.on_halt()

    assert result == fake_order
    assert mock_cancel.call_count == 1
    assert mock_cancel.call_args.args[1] == "sl-123"
    mock_close_trade.assert_called_once()
    assert mock_close_trade.call_args.args[0] == "emergency_halt"


# IN_POSITION, close succeeds but fill timestamp is missing, exit_time is None
def test_on_halt_in_position_close_succeeds_no_timestamp(make_order):
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = MagicMock()
    bot.state.sl_order_id   = "sl-123"
    bot.state.tp_order_id   = "tp-456"

    fake_order = make_order(fill_price=50000.0)  # no timestamp

    with patch("lighthouse.bots.base_bot.emergency_close", return_value=fake_order), \
         patch("lighthouse.bots.base_bot.cancel_order"), \
         patch.object(bot, "close_trade") as mock_close_trade:
        bot.on_halt()

    mock_close_trade.assert_called_once()
    assert mock_close_trade.call_args.kwargs["exit_time"] is None


# IN_POSITION, close fails, SL/TP not cancelled, returns None, close_trade not called
def test_on_halt_in_position_close_fails_leaves_sl_intact():
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = MagicMock()
    bot.state.sl_order_id   = "sl-123"
    bot.state.tp_order_id   = "tp-456"

    with patch("lighthouse.bots.base_bot.emergency_close", return_value=None) as mock_close, \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch.object(bot, "close_trade") as mock_close_trade:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_called_once()
    mock_cancel.assert_not_called()
    mock_close_trade.assert_not_called()


# WAITING_FILL with entry_limit_order_id set, cancel is called, returns None
def test_on_halt_waiting_fill_with_order_id_cancels_and_returns_none():
    bot = make_bot()
    bot.state.mode                  = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id  = "entry-789"

    with patch("lighthouse.bots.base_bot.cancel_order", return_value=True) as mock_cancel, \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_not_called()
    mock_cancel.assert_called_once()
    assert mock_cancel.call_args.args[1] == "entry-789"


# WAITING_FILL with no entry_limit_order_id, cancel is not called, returns None
def test_on_halt_waiting_fill_no_order_id_returns_none():
    bot = make_bot()
    bot.state.mode                  = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id  = None

    with patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_not_called()
    mock_cancel.assert_not_called()


# WAITING_FILL, cancel_order fails, cancel_all_orders succeeds, returns None
def test_on_halt_waiting_fill_cancel_order_fails_falls_back_to_cancel_all():
    bot = make_bot()
    bot.state.mode                  = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id  = "entry-789"

    with patch("lighthouse.bots.base_bot.cancel_order", return_value=False) as mock_cancel, \
         patch("lighthouse.bots.base_bot.cancel_all_orders", return_value=True) as mock_cancel_all, \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_not_called()
    mock_cancel.assert_called_once()
    mock_cancel_all.assert_called_once_with(bot.exchange, CONFIG["symbol"])


# WAITING_FILL, cancel_order fails, cancel_all_orders also fails, returns None
def test_on_halt_waiting_fill_both_cancels_fail_returns_none():
    bot = make_bot()
    bot.state.mode                  = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id  = "entry-789"

    with patch("lighthouse.bots.base_bot.cancel_order", return_value=False), \
         patch("lighthouse.bots.base_bot.cancel_all_orders", return_value=False) as mock_cancel_all, \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close:
        result = bot.on_halt()

    assert result is None
    mock_close.assert_not_called()
    mock_cancel_all.assert_called_once_with(bot.exchange, CONFIG["symbol"])


# ── reconcile() ──────────────────────────────────────────────────────────────

# no open positions, no orphaned orders then nothing cancelled, bot stays IDLE
def test_reconcile_no_position_no_orders():
    bot = make_bot()

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]) as mock_orders, \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all:
        bot.reconcile()

    assert bot.state.mode == BotMode.IDLE
    mock_cancel_all.assert_not_called()


# no open positions but orphaned orders found, cancel_all_orders called
def test_reconcile_no_position_orphaned_orders_cancelled(make_order):
    bot = make_bot()

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[make_order(id="stale-sl")]), \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all:
        bot.reconcile()

    mock_cancel_all.assert_called_once_with(bot.exchange, CONFIG["symbol"])


# persisted sl_order_id present and still open -> on_reconcile() called with that exact order.
# type is deliberately "stop_limit" (not "stop") to lock in that type strings are never inspected.
def test_reconcile_position_with_sl_and_tp_calls_on_reconcile(make_order):
    bot = make_bot()
    bot.state.sl_order_id = "sl-1"
    bot.state.tp_order_id = "tp-1"

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    sl_order = make_order(id="sl-1", type="stop_limit")
    tp_order = make_order(id="tp-1", type="stop_limit")

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[sl_order, tp_order]), \
         patch.object(bot, "on_reconcile") as mock_reconcile:
        bot.reconcile()

    mock_reconcile.assert_called_once_with(position, sl_order, tp_order)


# persisted sl_order_id present and open, no persisted tp_order_id -> on_reconcile receives tp_order=None
def test_reconcile_position_with_sl_no_tp(make_order):
    bot = make_bot()
    bot.state.sl_order_id = "sl-1"

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    sl_order = make_order(id="sl-1", type="stop_limit")

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[sl_order]), \
         patch.object(bot, "on_reconcile") as mock_reconcile:
        bot.reconcile()

    mock_reconcile.assert_called_once_with(position, sl_order, None)


# position found with no persisted sl_order_id -> close + halt(restartable=False); no order-type
# inspection influences the outcome, even though a stop-type order is open on the exchange
def test_reconcile_position_no_sl_emergency_close_and_halt(make_order):
    bot = make_bot()

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    stray_stop_order = make_order(id="not-mine", type="stop")

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[stray_stop_order]), \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_close.assert_called_once()
    mock_halt.assert_called_once()
    assert mock_halt.call_args.kwargs["restartable"] is False


# persisted sl_order_id present but absent from open orders -> close + halt(restartable=False)
def test_reconcile_persisted_sl_order_id_not_in_open_orders_closes_and_halts():
    bot = make_bot()
    bot.state.sl_order_id = "sl-gone"

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_close.assert_called_once()
    mock_halt.assert_called_once()
    assert mock_halt.call_args.kwargs["restartable"] is False


# position found with size 0 (unreadable), halt called immediately, no close attempt
def test_reconcile_unreadable_position_halts():
    bot = make_bot()

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.0)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_halt.assert_called_once()
    mock_close.assert_not_called()


# get_open_positions raises → bot halts, state is unknown
def test_reconcile_fetch_positions_raises_halts():
    bot = make_bot()

    with patch("lighthouse.bots.base_bot.get_open_positions", side_effect=RuntimeError("network error")), \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_halt.assert_called_once()


# get_open_orders raises in the no-position branch → halts, state unknown;
# neither state.reset() nor any order cancellation is attempted (issue #8, bug 1)
def test_reconcile_no_position_order_fetch_raises_halts():
    bot = make_bot()
    bot.state.mode = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id = "entry-1"

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", side_effect=RuntimeError("network error")), \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all, \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_halt.assert_called_once()
    assert bot.state.mode == BotMode.WAITING_FILL  # state.reset() not called
    mock_cancel_all.assert_not_called()
    mock_cancel.assert_not_called()


# get_open_orders raises in the position-found branch → halts, state unknown;
# emergency_close is never attempted on an unverified "no SL" reading (issue #8, bug 1)
def test_reconcile_position_order_fetch_raises_halts():
    bot = make_bot()
    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", side_effect=RuntimeError("network error")), \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch.object(bot, "halt") as mock_halt:
        bot.reconcile()

    mock_halt.assert_called_once()
    mock_close.assert_not_called()


# persisted IN_POSITION + unprotected position found: reconcile() does not close itself;
# it halts (non-restartable) and on_halt() performs the single emergency close (issue #8, bug 2)
def test_reconcile_persisted_in_position_no_sl_closes_via_on_halt(make_order):
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = Side.LONG
    bot.state.sl_order_id   = "sl-123"  # not present among the open orders returned below

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    fake_order = make_order(fill_price=50000.0, timestamp=1700000000000)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.emergency_close", return_value=fake_order) as mock_close, \
         patch("lighthouse.bots.base_bot.cancel_order"), \
         patch.object(bot, "close_trade") as mock_close_trade, \
         patch.object(bot, "on_halt", wraps=bot.on_halt) as mock_on_halt:
        bot.reconcile()

    mock_close.assert_called_once()  # exactly one close, and it happened inside on_halt()
    mock_on_halt.assert_called_once()
    mock_close_trade.assert_called_once()
    assert mock_close_trade.call_args.args[0] == "emergency_halt"
    assert bot.state.mode == BotMode.HALTED
    assert bot.state.halt_restartable is False


# no persisted position (fresh/IDLE) + unprotected position found on the exchange:
# reconcile() performs the close itself; on_halt() has nothing to do (mode stays IDLE going in)
def test_reconcile_idle_no_sl_closes_from_reconcile_not_on_halt():
    bot = make_bot()
    assert bot.state.mode == BotMode.IDLE

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.emergency_close") as mock_close, \
         patch.object(bot, "on_halt", wraps=bot.on_halt) as mock_on_halt:
        bot.reconcile()

    mock_close.assert_called_once()  # closed once, by reconcile()
    mock_on_halt.assert_called_once()  # on_halt() still runs (via halt()) but is a no-op: mode != IN_POSITION


# ── on_reconcile() ────────────────────────────────────────────────────────────

# all five state fields are restored correctly from exchange data
def test_on_reconcile_restores_all_fields(make_order):
    bot = make_bot()

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.05)
    sl_order = make_order(id="sl-99")
    tp_order = make_order(id="tp-88")

    bot.on_reconcile(position, sl_order, tp_order)

    assert bot.state.mode          == BotMode.IN_POSITION
    assert bot.state.side          == Side.LONG
    assert bot.state.position_size == 0.05
    assert bot.state.sl_order_id   == "sl-99"
    assert bot.state.tp_order_id   == "tp-88"


# short side is carried through correctly
def test_on_reconcile_short_side(make_order):
    bot = make_bot()

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.SHORT, size=0.02)
    sl_order = make_order(id="sl-1")

    bot.on_reconcile(position, sl_order, None)

    assert bot.state.side == Side.SHORT


# tp_order=None → tp_order_id stays None, no error
def test_on_reconcile_no_tp_order(make_order):
    bot = make_bot()

    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    sl_order = make_order(id="sl-1")

    bot.on_reconcile(position, sl_order, None)

    assert bot.state.tp_order_id is None
    assert bot.state.sl_order_id == "sl-1"


# ── tick() ─────────────────────────────────────────────────────────────────────

# closed market: both fetches and on_tick() are skipped, _consecutive_errors untouched
def test_tick_closed_market_skips_fetch_and_on_tick():
    bot = make_bot()
    bot.calendar = MagicMock()
    bot.calendar.name = "weekday"
    bot.calendar.is_open.return_value = False
    bot._consecutive_errors = 3

    with patch("lighthouse.bots.base_bot.get_ohlcv_df") as mock_ohlcv, \
         patch("lighthouse.bots.base_bot.get_bid_ask") as mock_bid_ask, \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_ohlcv.assert_not_called()
    mock_bid_ask.assert_not_called()
    mock_on_tick.assert_not_called()
    assert bot._consecutive_errors == 3


# open market: fetches and on_tick() run as before, consecutive errors reset
def test_tick_open_market_calls_fetch_and_on_tick():
    bot = make_bot()  # default calendar "24/7", state IDLE
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df) as mock_ohlcv, \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_ohlcv.assert_called_once()
    mock_on_tick.assert_called_once()
    assert bot._consecutive_errors == 0


# the default "24/7" calendar is always open, regardless of wall-clock time
def test_tick_24_7_calendar_never_gated():
    bot = make_bot()
    assert bot.calendar.name == "24/7"
    assert bot.calendar.is_open(pd.Timestamp.now(tz="UTC")) is True


# staleness backstop: a non-24/7 calendar skips on_tick() when the newest bar
# has not advanced since the previous tick (holiday with no holiday list)
def test_tick_stale_bar_skipped_for_non_24_7():
    bot = make_bot()
    bot.calendar = MagicMock()
    bot.calendar.name = "weekday"
    bot.calendar.is_open.return_value = True

    ts = pd.Timestamp("2024-01-08 20:30", tz="UTC")
    bot.state.last_closed_bar = ts.to_pydatetime()
    # last row is the current/forming bar (ADR 0001); ts is the newest closed bar
    forming_ts = ts + pd.Timedelta(hours=1)
    df = pd.DataFrame({"open": [1.0, 1.0]}, index=pd.DatetimeIndex([ts, forming_ts]))

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_not_called()
    assert bot._consecutive_errors == 0


# staleness backstop is inactive for the 24/7 calendar: on_tick() still runs
# even when the fetched data's last bar equals the previously stored one
def test_tick_stale_bar_backstop_inactive_for_24_7():
    bot = make_bot()  # default calendar "24/7"
    ts = pd.Timestamp("2024-01-08 20:30", tz="UTC")
    bot.state.last_closed_bar = ts.to_pydatetime()
    forming_ts = ts + pd.Timedelta(hours=1)
    df = pd.DataFrame({"open": [1.0, 1.0]}, index=pd.DatetimeIndex([ts, forming_ts]))

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# staleness backstop survives a trade close: last_closed_bar is a data-feed
# fact, not a per-trade field, so state.reset() (called on every trade close)
# must not clear it — otherwise the backstop loses a tick (issue #11)
def test_tick_stale_bar_skipped_for_non_24_7_after_trade_close():
    bot = make_bot()
    bot.calendar = MagicMock()
    bot.calendar.name = "weekday"
    bot.calendar.is_open.return_value = True

    ts = pd.Timestamp("2024-01-08 20:30", tz="UTC")
    bot.state.last_closed_bar = ts.to_pydatetime()
    bot.state.reset()  # simulates a trade close (e.g. BaseBot's close_trade())

    forming_ts = ts + pd.Timedelta(hours=1)
    df = pd.DataFrame({"open": [1.0, 1.0]}, index=pd.DatetimeIndex([ts, forming_ts]))

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_not_called()

# state.last_closed_bar is taken from the last CLOSED bar, not the fetched
# df's last row, which is always the current/forming period (ADR 0001)
def test_tick_last_closed_bar_excludes_forming_row():
    bot = make_bot()
    closed_ts  = pd.Timestamp("2024-01-08 20:00", tz="UTC")
    forming_ts = pd.Timestamp("2024-01-08 21:00", tz="UTC")
    df = pd.DataFrame({"open": [1.0, 1.0]}, index=pd.DatetimeIndex([closed_ts, forming_ts]))

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick"):
        bot.tick()

    assert bot.state.last_closed_bar == closed_ts.to_pydatetime()

# entry failure backoff: IDLE and entry_retry_after_tick not yet reached skips on_tick()
def test_tick_entry_failure_backoff_active_skips_on_tick():
    bot = make_bot()  # default calendar "24/7", state IDLE
    bot.state.entry_retry_after_tick = 1  # lifetime_tick_count becomes 1 on this tick — still gated

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=None), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_not_called()
    assert bot._consecutive_errors == 0

# entry failure backoff: once lifetime_tick_count exceeds entry_retry_after_tick, on_tick() resumes
def test_tick_entry_failure_backoff_expired_calls_on_tick():
    bot = make_bot()  # default calendar "24/7", state IDLE
    bot.state.lifetime_tick_count = 5
    bot.state.entry_retry_after_tick = 5  # next lifetime_tick_count (6) clears the gate

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=None), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# entry failure backoff only gates IDLE — an IN_POSITION bot is unaffected
def test_tick_entry_failure_backoff_ignored_when_not_idle():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION
    bot.state.entry_retry_after_tick = 999  # would gate IDLE, irrelevant here

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# ── Same-Bar Re-Entry Guard (issue #6, ADR 0003) ────────────────────────────

# a transition out of IN_POSITION during on_tick() (a normal trade close) stamps
# last_exit_bar from the last_closed_bar value computed earlier this tick
def test_tick_exit_transition_stamps_last_exit_bar():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION

    def close_trade(_df):
        bot.state.mode = BotMode.IDLE

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick", side_effect=close_trade):
        bot.tick()

    assert bot.state.last_exit_bar == bot.state.last_closed_bar
    assert bot.state.last_exit_bar is not None

# staying IN_POSITION across ticks does not stamp last_exit_bar
def test_tick_no_exit_stamp_when_staying_in_position():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick"):
        bot.tick()

    assert bot.state.last_exit_bar is None

# a transition out of IN_POSITION whose lifetime_trade_count did NOT change (close_trade()
# was never called) logs a WARNING to flag the missing trade record
def test_tick_exit_without_lifetime_trade_count_change_logs_warning():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION

    def leave_position_without_close(_df):
        bot.state.mode = BotMode.IDLE  # lifetime_trade_count left untouched

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick", side_effect=leave_position_without_close), \
         patch.object(bot.logger, "warning") as mock_warning:
        bot.tick()

    assert mock_warning.called

# a transition out of IN_POSITION whose lifetime_trade_count DID change (close_trade() was
# called as expected) does not log the missed-close WARNING
def test_tick_exit_with_lifetime_trade_count_change_no_warning():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION

    def leave_position_via_close(_df):
        bot.state.mode = BotMode.IDLE
        bot.state.lifetime_trade_count += 1

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick", side_effect=leave_position_via_close), \
         patch.object(bot.logger, "warning") as mock_warning:
        bot.tick()

    mock_warning.assert_not_called()

# guard blocks entry while the newest closed bar is still the one that was
# newest at the last exit
def test_tick_reentry_guard_blocks_entry_when_bar_unchanged():
    bot = make_bot()  # default calendar "24/7", state IDLE
    ts = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)
    bot.state.last_exit_bar = ts

    with patch("lighthouse.bots.base_bot._last_closed_bar_time", return_value=ts), \
         patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=None), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_not_called()

# guard lifts once a new bar has closed since the exit
def test_tick_reentry_guard_lifts_after_new_bar_closes():
    bot = make_bot()
    old_ts = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)
    new_ts = old_ts + timedelta(hours=1)
    bot.state.last_exit_bar = old_ts

    with patch("lighthouse.bots.base_bot._last_closed_bar_time", return_value=new_ts), \
         patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=None), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# guard only gates IDLE (entries) — an IN_POSITION bot is unaffected
def test_tick_reentry_guard_ignored_when_not_idle():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION
    ts = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)
    bot.state.last_exit_bar = ts

    with patch("lighthouse.bots.base_bot._last_closed_bar_time", return_value=ts), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# before the first trade, last_exit_bar is None — guard does nothing
def test_tick_reentry_guard_inactive_before_first_trade():
    bot = make_bot()
    assert bot.state.last_exit_bar is None
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick") as mock_on_tick:
        bot.tick()

    mock_on_tick.assert_called_once()

# ── run() ─────────────────────────────────────────────────────────────────────

# already HALTED bot, run() returns immediately without calling tick()
def test_run_halted_bot_aborts_immediately():
    bot = make_bot()
    bot.state.mode = BotMode.HALTED

    with patch.object(bot, "tick") as mock_tick:
        bot.run()

    mock_tick.assert_not_called()


# normal bot, tick() is called at least once before the loop exits
def test_run_calls_tick_at_least_once():
    bot = make_bot()

    def stop_after_first_tick():
        bot._stop_event.set()

    with patch.object(bot, "tick", side_effect=stop_after_first_tick) as mock_tick:
        bot.run()

    assert mock_tick.call_count >= 1


# ── persist_state() ─────────────────────────────────────────────────────────

# _db_path is None (default, matches every existing make_bot() usage): no-op, no exception
def test_persist_state_noop_when_db_path_none():
    bot = make_bot()
    assert bot._db_path is None

    mock_save = MagicMock()
    bot._persist_state_fn = mock_save
    bot.persist_state()

    mock_save.assert_not_called()

# _db_path/_bot_key set: the injected _persist_state_fn is called with the bot's db_path, key, and state
def test_persist_state_calls_save_state_when_enabled():
    bot = make_bot()
    bot._db_path = "fake/path.db"
    bot._bot_key = "concretebot:BTC/USDT"

    mock_save = MagicMock()
    bot._persist_state_fn = mock_save
    bot.persist_state()

    mock_save.assert_called_once_with("fake/path.db", "concretebot:BTC/USDT", bot.state)

# _persist_state_fn raising is swallowed (logged), never propagates
def test_persist_state_swallows_save_state_exception():
    bot = make_bot()
    bot._db_path = "fake/path.db"
    bot._bot_key = "concretebot:BTC/USDT"

    bot._persist_state_fn = MagicMock(side_effect=RuntimeError("disk full"))
    bot.persist_state()  # must not raise


# tick() persists once per tick when enabled
def test_tick_persists_state_when_enabled():
    bot = make_bot()
    bot._db_path = "fake/path.db"
    bot._bot_key = "concretebot:BTC/USDT"
    mock_save = MagicMock()
    bot._persist_state_fn = mock_save
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick"):
        bot.tick()

    mock_save.assert_called_once_with("fake/path.db", "concretebot:BTC/USDT", bot.state)

# halt() persists state too (may run outside the tick loop)
def test_halt_persists_state_when_enabled():
    bot = make_bot()
    bot._db_path = "fake/path.db"
    bot._bot_key = "concretebot:BTC/USDT"
    mock_save = MagicMock()
    bot._persist_state_fn = mock_save

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("some reason")

    mock_save.assert_called_once_with("fake/path.db", "concretebot:BTC/USDT", bot.state)


# ── reconcile(): persisted-state-aware "no position" branches ──────────────

# persisted IN_POSITION but exchange shows no position: reset to IDLE, critical logged
def test_reconcile_stale_in_position_resets_to_idle():
    bot = make_bot()
    bot.state.mode = BotMode.IN_POSITION
    bot.state.position_size = 0.02

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all, \
         patch.object(bot.logger, "critical") as mock_critical:
        bot.reconcile()

    assert bot.state.mode == BotMode.IDLE
    assert bot.state.position_size == 0.0
    mock_critical.assert_called_once()
    mock_cancel_all.assert_not_called()  # no orphaned orders found

# WAITING_FILL with the pending entry order still open: preserved, not cancelled,
# but OTHER orphaned orders are still cancelled
def test_reconcile_waiting_fill_pending_order_preserved(make_order):
    bot = make_bot()
    bot.state.mode = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id = "entry-1"

    pending_order = make_order(id="entry-1", type="limit")
    orphaned_order = make_order(id="stale-sl", type="stop")

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[pending_order, orphaned_order]), \
         patch("lighthouse.bots.base_bot.cancel_order") as mock_cancel, \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all:
        bot.reconcile()

    assert bot.state.mode == BotMode.WAITING_FILL  # untouched, not reset
    mock_cancel_all.assert_not_called()
    mock_cancel.assert_called_once_with(bot.exchange, "stale-sl", CONFIG["symbol"])

# WAITING_FILL but the pending entry order is gone and there's no position:
# resets to IDLE, critical logged, same as the stale IN_POSITION case
def test_reconcile_waiting_fill_order_gone_resets_to_idle():
    bot = make_bot()
    bot.state.mode = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id = "entry-1"

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.cancel_all_orders") as mock_cancel_all, \
         patch.object(bot.logger, "critical") as mock_critical:
        bot.reconcile()

    assert bot.state.mode == BotMode.IDLE
    mock_critical.assert_called_once()
    mock_cancel_all.assert_not_called()  # no orphaned orders found


# ── notifier wiring ───────────────────────────────────────────────────────────

# default notifier is a disabled no-op Notifier([]) when none is supplied
def test_default_notifier_is_disabled_noop():
    bot = make_bot()
    assert bot.notifier._channels == []

# halt() calls notify_halt() with the bot name, reason, level, and restartable flag
def test_halt_calls_notify_halt():
    bot = make_bot()
    bot.notifier = MagicMock()

    with patch.object(bot, "on_halt", return_value=None):
        bot.halt("risk breach", level="warning", restartable=False)

    bot.notifier.notify_halt.assert_called_once_with("ConcreteBot", "risk breach", "warning", False)

# on_halt(): IN_POSITION emergency close succeeds -> notify_emergency_close(success=True)
def test_on_halt_emergency_close_success_notifies(make_order):
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = Side.LONG
    bot.state.halt_reason   = "manual stop"

    fake_order = make_order(fill_price=50000.0, timestamp=1700000000000)
    with patch("lighthouse.bots.base_bot.emergency_close", return_value=fake_order), \
         patch("lighthouse.bots.base_bot.cancel_order"), \
         patch.object(bot, "close_trade"):
        bot.on_halt()

    bot.notifier.notify_emergency_close.assert_called_once_with(
        "ConcreteBot", CONFIG["symbol"], "long", "manual stop", True,
    )

# on_halt(): IN_POSITION emergency close fails -> notify_emergency_close(success=False)
def test_on_halt_emergency_close_failure_notifies():
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.position_size = 0.01
    bot.state.side          = Side.SHORT
    bot.state.halt_reason   = None

    with patch("lighthouse.bots.base_bot.emergency_close", return_value=None):
        bot.on_halt()

    bot.notifier.notify_emergency_close.assert_called_once_with(
        "ConcreteBot", CONFIG["symbol"], "short", "halt", False,
    )

# reconcile(): stale IN_POSITION reset fires notify_reconcile_issue
def test_reconcile_stale_in_position_notifies_reconcile_issue():
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.mode = BotMode.IN_POSITION

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]):
        bot.reconcile()

    bot.notifier.notify_reconcile_issue.assert_called_once()
    assert bot.notifier.notify_reconcile_issue.call_args.args[0] == "ConcreteBot"
    assert bot.notifier.notify_reconcile_issue.call_args.args[3] == "critical"

# reconcile(): stale WAITING_FILL reset (order gone) fires notify_reconcile_issue
def test_reconcile_stale_waiting_fill_notifies_reconcile_issue():
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.mode = BotMode.WAITING_FILL
    bot.state.entry_limit_order_id = "entry-1"

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]):
        bot.reconcile()

    bot.notifier.notify_reconcile_issue.assert_called_once()

# reconcile(): orphaned position detected fires notify_reconcile_issue
def test_reconcile_orphaned_position_notifies_reconcile_issue(make_order):
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.sl_order_id = "sl-1"
    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)
    sl_order = make_order(id="sl-1", type="stop_limit")

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[sl_order]), \
         patch.object(bot, "on_reconcile"):
        bot.reconcile()

    bot.notifier.notify_reconcile_issue.assert_called_once()

# reconcile(): unprotected position (no SL) fires notify_emergency_close
def test_reconcile_unprotected_position_notifies_emergency_close(make_order):
    bot = make_bot()
    bot.notifier = MagicMock()
    position = NormalizedPosition(symbol=CONFIG["symbol"], side=Side.LONG, size=0.01)

    with patch("lighthouse.bots.base_bot.get_open_positions", return_value=[position]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.emergency_close", return_value=make_order(fill_price=100.0)):
        bot.reconcile()

    bot.notifier.notify_emergency_close.assert_called_once_with(
        "ConcreteBot", CONFIG["symbol"], "long", "reconciliation: unprotected orphaned position", True,
    )

# tick(): IDLE -> IN_POSITION transition inside on_tick() fires notify_position_opened
def test_tick_position_opened_notifies_on_mode_transition():
    bot = make_bot()  # default calendar "24/7", state IDLE
    bot.notifier = MagicMock()
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )

    def open_position(_df):
        bot.state.mode          = BotMode.IN_POSITION
        bot.state.side          = Side.LONG
        bot.state.entry_price   = 100.0
        bot.state.position_size = 0.5

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick", side_effect=open_position):
        bot.tick()

    bot.notifier.notify_position_opened.assert_called_once_with(
        "ConcreteBot", CONFIG["symbol"], "long", 100.0, 0.5,
    )

# tick(): IDLE -> IN_POSITION transition resets the entry failure backoff
def test_tick_successful_entry_resets_entry_failure_backoff():
    bot = make_bot()  # default calendar "24/7", state IDLE
    bot.state.entry_failure_count = 2
    bot.state.entry_retry_after_tick = 0  # backoff window already expired, attempt now succeeds
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )

    def open_position(_df):
        bot.state.mode = BotMode.IN_POSITION

    with patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick", side_effect=open_position):
        bot.tick()

    assert bot.state.entry_failure_count == 0
    assert bot.state.entry_retry_after_tick == 0

# tick(): staying IN_POSITION across ticks does not re-fire notify_position_opened
def test_tick_no_notify_when_already_in_position():
    bot = make_bot()
    bot.notifier = MagicMock()
    bot.state.mode = BotMode.IN_POSITION

    with patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch.object(bot, "on_tick"):
        bot.tick()

    bot.notifier.notify_position_opened.assert_not_called()


# ── build_summary() ───────────────────────────────────────────────────────────

# IDLE bot: unrealized_pnl is None, other fields reflect state
def test_build_summary_idle_returns_none_pnl():
    bot = make_bot()

    summary = bot.build_summary()

    assert summary["symbol"] == CONFIG["symbol"]
    assert summary["mode"] == BotMode.IDLE.value
    assert summary["unrealized_pnl"] is None
    assert summary["lifetime_trade_count"] == 0

# IN_POSITION long: unrealized_pnl computed from bid/ask mid price
def test_build_summary_in_position_long_computes_pnl():
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.side          = Side.LONG
    bot.state.entry_price   = 100.0
    bot.state.position_size = 2.0
    bot.bid_ask              = {"bid": 104.0, "ask": 106.0}  # mid = 105.0

    summary = bot.build_summary()

    assert summary["unrealized_pnl"] == pytest.approx((105.0 - 100.0) * 2.0)
    assert summary["side"] == "long"

# IN_POSITION short: direction sign flips the P&L calculation
def test_build_summary_in_position_short_computes_pnl():
    bot = make_bot()
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.side          = Side.SHORT
    bot.state.entry_price   = 100.0
    bot.state.position_size = 2.0
    bot.bid_ask              = {"bid": 94.0, "ask": 96.0}  # mid = 95.0

    summary = bot.build_summary()

    assert summary["unrealized_pnl"] == pytest.approx((95.0 - 100.0) * 2.0 * -1)

