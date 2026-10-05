"""
================================================================================
CONTRACT TEST: RECORD_ENTRY_FAILURE() WIRING ACROSS THE BOT REGISTRY

Iterates bots/registry.py and, for every bot that has entry-attempt logic,
forces a bail-out against a stub exchange and asserts state.entry_failure_count
moved. Catches an author who wires a new on_scan()-style bail-out path
without calling self.record_entry_failure() (see docs/adr/0002, CONTEXT.md
"Entry Failure Backoff").

To run all tests in this file:
    pytest tests/test_bots/examples/test_entry_failure_contract.py -v
================================================================================
"""

################### IMPORTS ##########################

from typing import Callable
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.registry import _BOT_REGISTRY
from lighthouse.bots.examples.crypto_bot.config import crypto_config
from lighthouse.bots.examples.stock_bot.config import stock_config

################### HELPERS ##########################

DF = pd.DataFrame(
    {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0]},
    index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
)


def _crypto_bot_forced_entry_failure(bot_cls: type) -> BaseBot:
    """
    Build a CryptoBot, drive it through on_tick() with a forced position_size == 0
    bail-out, and return it for the caller to inspect state on.
    """
    bot = bot_cls(exchange=MagicMock(), config=crypto_config)
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    with patch("lighthouse.bots.examples.crypto_bot.bot.is_atr_above_threshold", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_volume_above_average", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.is_long_breakout", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.crypto_bot.bot.indicators.atr_current", return_value=10.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.calc_size_fixedfractional", return_value=0.0), \
         patch("lighthouse.bots.examples.crypto_bot.bot.place_market_buy") as mock_order:
        bot.on_tick(DF)
    assert not mock_order.called  # sanity: never reached order placement
    return bot


def _stock_bot_forced_entry_failure(bot_cls: type) -> BaseBot:
    """
    Build a StockBot, drive it through on_tick() with a forced position_size == 0
    bail-out, and return it for the caller to inspect state on.
    """
    bot = bot_cls(exchange=MagicMock(), config=stock_config)
    bot.bid_ask = {"bid": 99.0, "ask": 101.0}
    with patch("lighthouse.bots.examples.stock_bot.bot.is_sma_above_sma", return_value=True), \
         patch("lighthouse.bots.examples.stock_bot.bot.can_trade", return_value=True), \
         patch("lighthouse.bots.examples.stock_bot.bot.calc_size_fixedfractional", return_value=0.0), \
         patch("lighthouse.bots.examples.stock_bot.bot.place_market_buy") as mock_order:
        bot.on_tick(DF)
    assert not mock_order.called  # sanity: never reached order placement
    return bot


# Bot name -> factory that forces one of its on_scan()-style bail-outs and
# returns the bot for the contract test to inspect. Add an entry here for every
# bot with entry-attempt logic; new bots without one fail loudly below instead
# of silently skipping coverage.
_ENTRY_FAILURE_SCENARIOS: dict[str, Callable[[type], BaseBot]] = {
    "crypto_bot": _crypto_bot_forced_entry_failure,
    "stock_bot":  _stock_bot_forced_entry_failure,
}

# Bots registered in bots/registry.py that have no entry-attempt logic yet
# (on_tick() is a no-op skeleton) and so have nothing to force a failure in.
_NO_ENTRY_LOGIC_YET: set[str] = set()

################### TESTS ##########################

# every bot in bots/registry.py is either covered by a forced-failure scenario or
# explicitly exempted for having no entry-attempt logic yet — nothing falls through
def test_every_registered_bot_is_covered_or_exempt():
    uncovered = set(_BOT_REGISTRY) - set(_ENTRY_FAILURE_SCENARIOS) - _NO_ENTRY_LOGIC_YET
    assert not uncovered, (
        f"Bot(s) {sorted(uncovered)} are registered in bots/registry.py but have no "
        "entry-failure scenario in _ENTRY_FAILURE_SCENARIOS and are not listed in "
        "_NO_ENTRY_LOGIC_YET. Add a scenario that forces one of its on_scan()-style "
        "bail-outs, or add it to _NO_ENTRY_LOGIC_YET if it has no entry-attempt logic."
    )

# forcing an on_scan()-style bail-out increments state.entry_failure_count by exactly 1
@pytest.mark.parametrize("bot_name", sorted(_ENTRY_FAILURE_SCENARIOS))
def test_forced_entry_bailout_records_entry_failure(bot_name):
    bot_cls = _BOT_REGISTRY[bot_name]
    scenario = _ENTRY_FAILURE_SCENARIOS[bot_name]

    bot = scenario(bot_cls)

    assert bot.state.entry_failure_count == 1
