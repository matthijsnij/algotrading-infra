"""
================================================================================
UNIT TESTS FOR SUMMARY.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_summary.py -v

To run a specific test function:
    pytest tests/test_summary.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import threading
import time
from unittest.mock import MagicMock

from lighthouse.runtime.summary import periodic_summary

################### HELPERS ##########################

INTERVAL = 1  # seconds; periodic_summary only accepts int interval_seconds
TIMEOUT  = 2.0


def _run_summary(bot, notifier, interval_seconds=INTERVAL) -> tuple:
    """Run periodic_summary() in a background thread and return (shutdown_event, thread)."""
    shutdown_event = threading.Event()
    thread = threading.Thread(
        target=periodic_summary,
        args=(bot, notifier, interval_seconds, shutdown_event),
        daemon=True,
    )
    thread.start()
    return shutdown_event, thread

################### TESTS ##########################

# after one interval, build_summary() + send_summary() are called with the expected kwargs
def test_periodic_summary_sends_after_interval():
    bot = MagicMock()
    bot.__class__.__name__ = "TrivialBot"
    bot.build_summary.return_value = {
        "symbol": "BTC/USDT:USDT", "mode": "idle", "side": None,
        "entry_price": None, "position_size": 0.0, "unrealized_pnl": None, "lifetime_trade_count": 0,
    }
    notifier = MagicMock()

    shutdown_event, thread = _run_summary(bot, notifier, interval_seconds=0)
    time.sleep(0.05)  # let the loop run at least once (sleep(0) is effectively a no-op)
    shutdown_event.set()
    thread.join(timeout=TIMEOUT)

    bot.build_summary.assert_called()
    notifier.send_summary.assert_called()
    kwargs = notifier.send_summary.call_args.kwargs
    assert kwargs["bot_name"] == "TrivialBot"
    assert kwargs["symbol"] == "BTC/USDT:USDT"
    assert kwargs["mode"] == "idle"


# shutdown_event already set before periodic_summary starts: exits without calling build_summary()
def test_periodic_summary_exits_if_shutdown_already_set():
    bot = MagicMock()
    notifier = MagicMock()
    shutdown_event = threading.Event()
    shutdown_event.set()

    thread = threading.Thread(
        target=periodic_summary,
        args=(bot, notifier, 60, shutdown_event),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=TIMEOUT)

    bot.build_summary.assert_not_called()
    notifier.send_summary.assert_not_called()


# an exception building/sending the summary is logged and swallowed — loop keeps running
def test_periodic_summary_swallows_exceptions():
    bot = MagicMock()
    bot.build_summary.side_effect = RuntimeError("boom")
    notifier = MagicMock()

    shutdown_event, thread = _run_summary(bot, notifier, interval_seconds=0)
    time.sleep(0.05)
    shutdown_event.set()
    thread.join(timeout=TIMEOUT)

    assert not thread.is_alive()  # loop exited cleanly despite the exception
