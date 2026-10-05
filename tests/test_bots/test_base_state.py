"""
================================================================================
UNIT TESTS FOR BASE_STATE.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_bots/test_base_state.py -v

To run a specific test function:
    pytest tests/test_bots/test_base_state.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

from datetime import datetime, timezone

from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.domain.enums import Side

################### TESTS ##########################

# ── reset() ────────────────────────────────────────────────────────

# last_closed_bar is a data-feed fact maintained by BaseBot.tick(), not a
# per-trade fact, so reset() (called on every trade close) must not clear it
def test_reset_preserves_last_closed_bar():
    state = BaseState()
    ts = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)
    state.last_closed_bar = ts

    state.reset()

    assert state.last_closed_bar == ts


# last_exit_bar drives the Same-Bar Re-Entry Guard across trades, so reset()
# (called on every trade close) must not clear it either
def test_reset_preserves_last_exit_bar():
    state = BaseState()
    ts = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)
    state.last_exit_bar = ts

    state.reset()

    assert state.last_exit_bar == ts


# reset() still clears the per-trade fields it owns
def test_reset_clears_per_trade_fields():
    state = BaseState()
    state.mode          = BotMode.IN_POSITION
    state.side          = Side.LONG
    state.halt_reason   = "test"
    state.position_size = 1.0
    state.entry_price   = 100.0

    state.reset()

    assert state.mode == BotMode.IDLE
    assert state.side == Side.FLAT
    assert state.halt_reason is None
    assert state.position_size == 0.0
    assert state.entry_price is None

# sl_price/tp_price are per-trade planned exit levels, so reset() must clear them too
def test_reset_clears_sl_tp_price():
    state = BaseState()
    state.sl_price = 95.0
    state.tp_price = 110.0

    state.reset()

    assert state.sl_price is None
    assert state.tp_price is None
