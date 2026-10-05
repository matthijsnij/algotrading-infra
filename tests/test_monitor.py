"""
================================================================================
UNIT TESTS FOR MONITOR.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_monitor.py -v

To run a specific test function:
    pytest tests/test_monitor.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import threading
import time
import pytest
from unittest.mock import MagicMock
from lighthouse.runtime.monitor import drawdown_monitor

################### HELPERS ##########################

INTERVAL   = 0.02   # short check interval so tests run fast
TIMEOUT    = 2.0    # max time to wait for shutdown_event to be set

def _make_bots(n: int = 1) -> list:
    """Return n mock bot instances."""
    return [MagicMock() for _ in range(n)]

def _run_monitor(equity_values: list, starting_equity: float, max_drawdown_pct: float = 0.10, consecutive_limit: int = 2, notifier=None) -> tuple:
    """
    Run drawdown_monitor in a background thread and return (shutdown_event, bots).

    equity_values: list of floats returned by equity_fn on successive calls.
                   After the list is exhausted, equity_fn raises StopIteration
                   (caught by the monitor as an error, loop continues until shutdown).
    """
    bots = _make_bots()
    shutdown_event = threading.Event()
    equity_iter = iter(equity_values)

    def equity_fn():
        return next(equity_iter)

    thread = threading.Thread(
        target=drawdown_monitor,
        args=(equity_fn, bots, shutdown_event, starting_equity),
        kwargs={
            "max_drawdown_pct": max_drawdown_pct, "check_interval": INTERVAL,
            "consecutive_limit": consecutive_limit, "notifier": notifier,
        },
        daemon=True,
    )
    thread.start()
    return shutdown_event, bots, thread

################### TESTS ##########################

# ── drawdown_monitor ────────────────────────────────────────────────────────

# equity stays above threshold; shutdown_event not set, bots not halted
def test_drawdown_monitor_no_breach():
    # starting_equity=1000, max_drawdown=10%; equity stays at 950 (5% drawdown —> within limit)
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[950.0, 950.0, 950.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    time.sleep(INTERVAL * 5) # sleep to allow monitor loop to run a few times
    assert not shutdown_event.is_set()
    bots[0].halt.assert_not_called()

# single breach below consecutive_limit; no halt triggered
def test_drawdown_monitor_single_breach_no_halt():
    # one check at 850 (15% drawdown —> exceeds 10% limit), then equity recovers
    # consecutive_limit=2, so single breach should not halt
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 950.0, 950.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    time.sleep(INTERVAL * 5) # sleep to allow monitor loop to run a few times
    assert not shutdown_event.is_set()
    bots[0].halt.assert_not_called()

# two consecutive breaches; all bots halted and shutdown_event set
def test_drawdown_monitor_two_consecutive_breaches_halts():
    # two consecutive checks at 850 (15% drawdown —> exceeds 10% limit)
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 850.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT) # wait for shutdown_event to be set
    assert triggered, "shutdown_event was not set within timeout"
    bots[0].halt.assert_called_once()

# circuit-breaker halt is marked non-restartable so the watchdog doesn't fight it
def test_drawdown_monitor_halt_passes_restartable_false():
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 850.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered, "shutdown_event was not set within timeout"
    assert bots[0].halt.call_args.kwargs["restartable"] is False

# breach then recovery resets counter — two non-consecutive breaches do not halt
def test_drawdown_monitor_breach_then_recovery_resets_counter():
    # breach → recovery → breach: counter resets on recovery, so second breach is count=1, no halt
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 950.0, 850.0, 950.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    time.sleep(INTERVAL * 6) # sleep to allow monitor loop to run enough times
    assert not shutdown_event.is_set()
    bots[0].halt.assert_not_called() # test for one bot

# HWM updates; drawdown calculated against new HWM, not starting equity
def test_drawdown_monitor_hwm_updates():
    # equity rises to 1200 (new HWM), then drops to 1100 (8.3% from HWM — within 10%)
    # if calculated against starting_equity=1000, it would look like a gain and not trip
    # if HWM is tracked correctly, 1100/1200 is within limit → no halt
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[1200.0, 1100.0, 1100.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    time.sleep(INTERVAL * 5) # sleep to allow monitor loop to run enough times
    assert not shutdown_event.is_set()
    bots[0].halt.assert_not_called()

# HWM updates; large drop from new HWM correctly triggers circuit breaker
def test_drawdown_monitor_hwm_drop_triggers_halt():
    # equity rises to 1200 (new HWM), then drops to 1050 (12.5% from HWM — exceeds 10%)
    # two consecutive checks at 1050 → should halt
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[1200.0, 1050.0, 1050.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered, "shutdown_event was not set within timeout"
    bots[0].halt.assert_called_once()

# multiple bots; all are halted when circuit breaker fires
def test_drawdown_monitor_all_bots_halted():
    bots = _make_bots(n=3)
    shutdown_event = threading.Event()
    equity_values = iter([850.0, 850.0])

    thread = threading.Thread(
        target=drawdown_monitor,
        args=(lambda: next(equity_values), bots, shutdown_event, 1000.0),
        kwargs={"max_drawdown_pct": 0.10, "check_interval": INTERVAL, "consecutive_limit": 2},
        daemon=True,
    )
    thread.start()
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered
    for bot in bots:
        bot.halt.assert_called_once()

# shutdown_event already set before monitor starts; exits without calling equity_fn
def test_drawdown_monitor_exits_if_shutdown_already_set():
    shutdown_event = threading.Event()
    shutdown_event.set()
    equity_fn = MagicMock(return_value=1000.0)
    bots = _make_bots()

    thread = threading.Thread(
        target=drawdown_monitor,
        args=(equity_fn, bots, shutdown_event, 1000.0),
        kwargs={"check_interval": INTERVAL},
        daemon=True,
    )
    thread.start()
    thread.join(timeout=TIMEOUT)
    equity_fn.assert_not_called()


# ── notifier wiring ───────────────────────────────────────────────────────────

# first breach of a new streak -> notify_drawdown_breach() called once with raw fractions
def test_drawdown_monitor_notifies_on_first_breach():
    notifier = MagicMock()
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 950.0, 950.0],  # single breach, then recovery — below consecutive_limit
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
        notifier=notifier,
    )
    time.sleep(INTERVAL * 5)
    assert not shutdown_event.is_set()
    notifier.notify_drawdown_breach.assert_called_once()
    args = notifier.notify_drawdown_breach.call_args[0]
    assert args[0] == pytest.approx(0.15)
    assert args[1] == pytest.approx(0.10)
    notifier.notify_circuit_breaker.assert_not_called()

# second consecutive breach does not re-notify notify_drawdown_breach — only the first does
def test_drawdown_monitor_does_not_renotify_breach_on_second_consecutive():
    notifier = MagicMock()
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 850.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
        notifier=notifier,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered
    notifier.notify_drawdown_breach.assert_called_once()

# circuit breaker trip -> notify_circuit_breaker() called with raw fractions
def test_drawdown_monitor_notifies_circuit_breaker_on_trip():
    notifier = MagicMock()
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 850.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
        notifier=notifier,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered
    notifier.notify_circuit_breaker.assert_called_once()
    args = notifier.notify_circuit_breaker.call_args[0]
    assert args[0] == pytest.approx(0.15)
    assert args[1] == pytest.approx(0.10)

# notifier=None (default): no crash on breach or circuit-breaker paths
def test_drawdown_monitor_no_notifier_is_safe():
    shutdown_event, bots, thread = _run_monitor(
        equity_values=[850.0, 850.0],
        starting_equity=1000.0,
        max_drawdown_pct=0.10,
        consecutive_limit=2,
    )
    triggered = shutdown_event.wait(timeout=TIMEOUT)
    assert triggered
    bots[0].halt.assert_called_once()
