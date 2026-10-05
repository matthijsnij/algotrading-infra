"""
================================================================================
UNIT TESTS FOR EXAMPLES/CRYPTO_BOT/BOT.PY

The on_scan() entry-failure bail-out paths, plus direct calls to on_scan() and
on_position(), are tested here; on_halt() and close_trade() now live in
BaseBot and are tested in test_bots/test_base_bot.py. Other paths are covered
by backtests.

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_bots/examples/test_crypto_bot.py -v

To run a specific test function:
    pytest tests/test_bots/examples/test_crypto_bot.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from lighthouse.bots.examples.crypto_bot.bot import CryptoBot
from lighthouse.bots.base_state import BotMode
from lighthouse.domain.enums import Side

################### HELPERS ##########################

CONFIG = {
    "symbol":                 "BTC/USDT",
    "quote_currency":         "USDT",
    "timeframe":              "1h",
    "limit":                  100,
    "scan_interval":          60,
    "position_interval":      10,
    "max_consecutive_errors": 5,
    "direction":              "both",
    "breakout_window":        20,
    "atr_period":             14,
    "atr_buffer_mult":        0.2,
    "sl_atr_mult":            1.5,
    "rr_ratio":               2.0,
    "volume_period":          20,
    "volume_mult":            1.2,
    "atr_threshold":          50.0,
    "risk_pct":               0.01,
}


def make_bot() -> CryptoBot:
    """Return a CryptoBot with state set to IN_POSITION."""
    bot = CryptoBot(exchange=MagicMock(), config=CONFIG)
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.side          = Side.LONG
    bot.state.position_size = 0.01
    bot.state.sl_order_id   = "sl-123"
    bot.state.tp_order_id   = "tp-456"
    return bot


# ── on_scan() bail-outs (issue #3 — Entry Failure Backoff) ──────────────────

def make_idle_bot() -> CryptoBot:
    """Return a CryptoBot with state IDLE and bid/ask primed, ready for on_tick()."""
    bot = CryptoBot(exchange=MagicMock(), config=CONFIG)
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    return bot


DF = pd.DataFrame(
    {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0]},
    index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
)


# can_trade() == False halts immediately (restartable=True) instead of backing off
def test_on_scan_can_trade_false_halts_restartable_no_backoff():
    bot = make_idle_bot()

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=False), \
         patch.object(bot, "record_entry_failure") as mock_record, \
         patch.object(bot, "halt") as mock_halt:
        bot.on_tick(DF)

    mock_record.assert_not_called()
    mock_halt.assert_called_once()
    assert mock_halt.call_args.kwargs["restartable"] is True

# position_size == 0 records an entry failure and aborts entry
def test_on_scan_zero_position_size_records_failure():
    bot = make_idle_bot()

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.indicators.atr_current", return_value=10.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.calc_size_fixedfractional", return_value=0.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_market_buy") as mock_order, \
         patch.object(bot, "record_entry_failure") as mock_record:
        bot.on_tick(DF)

    mock_record.assert_called_once()
    mock_order.assert_not_called()
    assert bot.state.mode == BotMode.IDLE

# market order placement failure records an entry failure and aborts entry
def test_on_scan_market_order_failed_records_failure():
    bot = make_idle_bot()

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.indicators.atr_current", return_value=10.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.calc_size_fixedfractional", return_value=0.01), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_market_buy", return_value=None), \
         patch.object(bot, "record_entry_failure") as mock_record:
        bot.on_tick(DF)

    mock_record.assert_called_once()
    assert bot.state.mode == BotMode.IDLE

# SL/TP placement failure triggers emergency close AND records an entry failure
def test_on_scan_sl_tp_placement_failed_emergency_closes_and_records_failure(make_order):
    bot = make_idle_bot()
    entry_order = make_order(fill_price=101.0, timestamp=1700000000000, filled=0.01)

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.indicators.atr_current", return_value=10.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.calc_size_fixedfractional", return_value=0.01), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_market_buy", return_value=entry_order), \
         patch("lighthouse.bots.examples.crypto_bot.bot.fetch_order", return_value=entry_order), \
         patch("lighthouse.bots.examples.crypto_bot.bot.parse_order_result", return_value=(101.0, None, None)), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_stop_sell", return_value=None), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_limit_sell", return_value=None), \
         patch("lighthouse.bots.examples.crypto_bot.bot.emergency_close") as mock_emergency_close, \
         patch.object(bot, "record_entry_failure") as mock_record:
        bot.on_tick(DF)

    mock_emergency_close.assert_called_once()
    mock_record.assert_called_once()
    assert bot.state.mode == BotMode.IDLE

# on_scan() called directly (not via on_tick): full successful entry transitions IDLE -> IN_POSITION
def test_on_scan_direct_call_successful_entry_transitions_to_in_position(make_order):
    bot = make_idle_bot()
    entry_order = make_order(id="entry-1", fill_price=101.0, timestamp=1700000000000, filled=0.01)
    sl_order    = make_order(id="sl-1")
    tp_order    = make_order(id="tp-1")

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.indicators.atr_current", return_value=10.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.calc_size_fixedfractional", return_value=0.01), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_market_buy", return_value=entry_order), \
         patch("lighthouse.bots.examples.crypto_bot.bot.fetch_order", return_value=entry_order), \
         patch("lighthouse.bots.examples.crypto_bot.bot.parse_order_result", return_value=(101.0, None, None)), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_stop_sell", return_value=sl_order), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_limit_sell", return_value=tp_order):
        bot.on_scan(DF)

    assert bot.state.mode == BotMode.IN_POSITION
    assert bot.state.side == Side.LONG
    assert bot.state.sl_order_id == "sl-1"
    assert bot.state.tp_order_id == "tp-1"
    assert bot.state.entry_price == 101.0


# ── on_position() direct calls ──────────────────────────────────────────────

def make_in_position_bot() -> CryptoBot:
    """Return a CryptoBot IN_POSITION with entry_price/bid_ask primed, ready for on_position()."""
    bot = make_bot()
    bot.state.entry_price = 100.0
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    return bot


# on_position() called directly: SL fill cancels TP and closes the trade with reason "sl"
def test_on_position_direct_call_sl_hit_closes_trade_as_sl(make_order):
    bot = make_in_position_bot()
    sl_fill = make_order(id="sl-123", fill_price=95.0, timestamp=1700000000000)

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_position_closed", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.cancel_order", return_value=True) as mock_cancel, \
         patch("lighthouse.bots.examples.crypto_bot.bot.fetch_order", return_value=sl_fill), \
         patch("lighthouse.bots.examples.crypto_bot.bot.parse_order_result", return_value=(95.0, None, None)), \
         patch.object(bot, "close_trade") as mock_close:
        bot.on_position(DF)

    mock_cancel.assert_called_once_with(bot.exchange, "tp-456", "BTC/USDT")
    mock_close.assert_called_once()
    assert mock_close.call_args.args[0] == "sl"
    assert mock_close.call_args.kwargs["exit_price"] == 95.0


# on_position() called directly: TP fill (SL not closed) cancels SL and closes the trade with reason "tp"
def test_on_position_direct_call_tp_hit_closes_trade_as_tp(make_order):
    bot = make_in_position_bot()
    tp_fill = make_order(id="tp-456", fill_price=110.0, timestamp=1700000000000)

    with patch("lighthouse.bots.examples.crypto_bot.bot.is_position_closed", side_effect=[False, True]), \
         patch("lighthouse.bots.examples.crypto_bot.bot.cancel_order", return_value=True) as mock_cancel, \
         patch("lighthouse.bots.examples.crypto_bot.bot.fetch_order", return_value=tp_fill), \
         patch("lighthouse.bots.examples.crypto_bot.bot.parse_order_result", return_value=(110.0, None, None)), \
         patch.object(bot, "close_trade") as mock_close:
        bot.on_position(DF)

    mock_cancel.assert_called_once_with(bot.exchange, "sl-123", "BTC/USDT")
    mock_close.assert_called_once()
    assert mock_close.call_args.args[0] == "tp"
    assert mock_close.call_args.kwargs["exit_price"] == 110.0
