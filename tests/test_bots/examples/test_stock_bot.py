"""
================================================================================
UNIT TESTS FOR EXAMPLES/STOCK_BOT/BOT.PY

Covers the moving-average crossover strategy's three core behaviors: entry on
a cross-up, exit on a cross-down, and no action while no cross has occurred.
Other on_scan() bail-out paths (zero size, failed order) mirror CryptoBot's
and are covered by test_entry_failure_contract.py.

To run all tests in this file:
    pytest tests/test_bots/examples/test_stock_bot.py -v
================================================================================
"""

################### IMPORTS ##########################

import pandas as pd
from unittest.mock import MagicMock, patch

from lighthouse.bots.examples.stock_bot.bot import StockBot
from lighthouse.bots.base_state import BotMode
from lighthouse.domain.enums import Side

################### HELPERS ##########################

CONFIG = {
    "symbol":                 "AAPL",
    "quote_currency":         "USD",
    "calendar":               "nyse",
    "timeframe":              "1d",
    "limit":                  200,
    "scan_interval":          86400,
    "position_interval":      3600,
    "max_consecutive_errors": 5,
    "max_entry_failures":     3,
    "fast_period":            3,
    "slow_period":            5,
    "risk_pct":               0.01,
    "sizing_stop_pct":        0.05,
}


def make_closes_df(closes: list[float]) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of close prices; last row is the forming bar."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open":   closes,
            "high":   [c + 0.5 for c in closes],
            "low":    [c - 0.5 for c in closes],
            "close":  closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def make_idle_bot() -> StockBot:
    """Return a StockBot with state IDLE and bid/ask primed, ready for on_scan()."""
    bot = StockBot(exchange=MagicMock(), config=CONFIG)
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    return bot


def make_in_position_bot() -> StockBot:
    """Return a StockBot IN_POSITION with entry_price/bid_ask primed, ready for on_position()."""
    bot = StockBot(exchange=MagicMock(), config=CONFIG)
    bot.state.mode          = BotMode.IN_POSITION
    bot.state.side          = Side.LONG
    bot.state.entry_price   = 100.0
    bot.state.position_size = 10.0
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    return bot


# On the last closed bar, fast SMA(3) > slow SMA(5): a clear cross-up (rising closes).
CROSS_UP_DF = make_closes_df([100, 101, 102, 103, 104, 105, 106])

# On the last closed bar, fast SMA(3) < slow SMA(5): a clear cross-down (falling closes).
CROSS_DOWN_DF = make_closes_df([106, 105, 104, 103, 102, 101, 100])

# Flat closes: fast SMA(3) == slow SMA(5), no cross either way.
FLAT_DF = make_closes_df([100, 100, 100, 100, 100, 100, 100])

# Long uptrend lead-in: fast SMA(3) has already been above slow SMA(5) for several bars
# before the last closed bar, so there is no fresh cross on it (see bot.py.on_scan's
# docstring and docs/adr/0002: the signal check is a deliberate level check, not an
# edge trigger).
ALREADY_TRENDING_UP_DF = make_closes_df([90, 95, 100, 102, 103, 104, 105, 106, 107, 108])


################### on_scan() TESTS ##########################

# entry on cross-up: fast SMA crossing above slow SMA opens a long position
def test_on_scan_cross_up_enters_long(make_order):
    bot = make_idle_bot()
    entry_order = make_order(id="entry-1", fill_price=101.0, timestamp=1700000000000, filled=10.0)

    with patch("lighthouse.bots.examples.stock_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.stock_bot.bot.calc_size_fixedfractional", return_value=10.0), \
         patch("lighthouse.bots.examples.stock_bot.bot.place_market_buy", return_value=entry_order), \
         patch("lighthouse.bots.examples.stock_bot.bot.fetch_order", return_value=entry_order), \
         patch("lighthouse.bots.examples.stock_bot.bot.parse_order_result", return_value=(101.0, None, None)):
        bot.on_scan(CROSS_UP_DF)

    assert bot.state.mode == BotMode.IN_POSITION
    assert bot.state.side == Side.LONG
    assert bot.state.entry_price == 101.0
    assert bot.state.position_size == 10.0


# no action without a cross: flat SMAs place no order and leave the bot IDLE
def test_on_scan_no_cross_does_nothing():
    bot = make_idle_bot()

    with patch("lighthouse.bots.examples.stock_bot.bot.can_trade") as mock_can_trade, \
         patch("lighthouse.bots.examples.stock_bot.bot.place_market_buy") as mock_order:
        bot.on_scan(FLAT_DF)

    mock_can_trade.assert_not_called()
    mock_order.assert_not_called()
    assert bot.state.mode == BotMode.IDLE


# intended level-check behavior: a bot (re)started mid-uptrend enters immediately even
# though the last closed bar is not a fresh cross — see on_scan's docstring/docs/adr/0002
def test_on_scan_already_trending_enters_without_fresh_cross(make_order):
    bot = make_idle_bot()
    entry_order = make_order(id="entry-1", fill_price=101.0, timestamp=1700000000000, filled=10.0)

    with patch("lighthouse.bots.examples.stock_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.stock_bot.bot.calc_size_fixedfractional", return_value=10.0), \
         patch("lighthouse.bots.examples.stock_bot.bot.place_market_buy", return_value=entry_order), \
         patch("lighthouse.bots.examples.stock_bot.bot.fetch_order", return_value=entry_order), \
         patch("lighthouse.bots.examples.stock_bot.bot.parse_order_result", return_value=(101.0, None, None)):
        bot.on_scan(ALREADY_TRENDING_UP_DF)

    assert bot.state.mode == BotMode.IN_POSITION


################### on_position() TESTS ##########################

# exit on cross-down: fast SMA crossing below slow SMA closes the long position
def test_on_position_cross_down_exits_long(make_order):
    bot = make_in_position_bot()
    exit_order = make_order(id="exit-1", fill_price=99.0, timestamp=1700000000000)

    with patch("lighthouse.bots.examples.stock_bot.bot.place_market_sell", return_value=exit_order) as mock_sell, \
         patch("lighthouse.bots.examples.stock_bot.bot.fetch_order", return_value=exit_order), \
         patch("lighthouse.bots.examples.stock_bot.bot.parse_order_result", return_value=(99.0, None, None)), \
         patch.object(bot, "close_trade") as mock_close:
        bot.on_position(CROSS_DOWN_DF)

    mock_sell.assert_called_once_with(bot.exchange, "AAPL", 10.0)
    mock_close.assert_called_once()
    assert mock_close.call_args.args[0] == "cross_down"
    assert mock_close.call_args.kwargs["exit_price"] == 99.0


# no action without a cross: fast SMA still above slow SMA keeps the position open
def test_on_position_no_cross_stays_in_position():
    bot = make_in_position_bot()

    with patch("lighthouse.bots.examples.stock_bot.bot.place_market_sell") as mock_sell, \
         patch.object(bot, "close_trade") as mock_close:
        bot.on_position(CROSS_UP_DF)

    mock_sell.assert_not_called()
    mock_close.assert_not_called()
    assert bot.state.mode == BotMode.IN_POSITION
