"""
================================================================================
MONITOR MODULE
================================================================================

System-wide drawdown monitor.

Functions:
    drawdown_monitor() : background thread that monitors system-wide drawdown
                         and halts all bots if the limit is exceeded.
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import threading
import time
from contextlib import nullcontext
from typing import Any, Callable
from lighthouse.utils.logging import get_logger

############ LOGGER ############

logger = get_logger(__name__)

############ DRAWDOWN MONITOR ############

def drawdown_monitor(
    equity_fn: Callable[[], float],
    bots: list,
    shutdown_event: threading.Event,
    starting_equity: float,
    max_drawdown_pct: float = 0.10,
    check_interval: int     = 60,
    consecutive_limit: int  = 2,
    lock: threading.Lock | None = None,
    notifier: Any | None = None,
) -> None:
    """
    Background thread that monitors system-wide drawdown using a rolling
    high-water mark (HWM) on portfolio equity. Halts all bots and triggers
    shutdown if drawdown exceeds max_drawdown_pct for consecutive_limit
    consecutive checks.

    Args:
        equity_fn:        zero-argument callable that returns current portfolio
                        equity as a float. Defined by the caller — the monitor
                        has no knowledge of exchanges, currencies, or asset types.
        bots:             list of all running bot instances
        shutdown_event:   threading.Event shared with the main thread
        starting_equity:  initial equity used as the first HWM value
        max_drawdown_pct: drawdown fraction that triggers the circuit breaker
        check_interval:   seconds to sleep between equity checks
        consecutive_limit: number of consecutive breaching checks before halting
        lock:             optional lock shared with the command listener and watchdog,
                          held during the halt-all sweep so it can't interleave with a
                          concurrent restart. No locking if None.
        notifier:         optional Notifier for alerting; notified on the first breach
                          of a new consecutive streak (0->1 transition) and on the
                          circuit-breaker trip.
    """
    hwm = starting_equity
    consecutive_breaches = 0
    logger.info(
        "Drawdown monitor started | hwm=%.4f max_drawdown=%.1f%%",
        hwm, max_drawdown_pct * 100,
    )

    while not shutdown_event.is_set():
        time.sleep(check_interval)

        if shutdown_event.is_set():
            break

        try:
            equity = equity_fn()
        except Exception as exc:
            logger.error("Drawdown monitor: failed to fetch equity | error=%s", exc)
            continue

        # Update rolling high-water mark
        if equity > hwm:
            hwm = equity
            logger.debug("Drawdown monitor: new HWM=%.4f", hwm)

        # Calculate current drawdown
        drawdown = (hwm - equity) / hwm if hwm > 0 else 0.0

        if drawdown >= max_drawdown_pct:
            consecutive_breaches += 1
            logger.warning(
                "Drawdown monitor: drawdown %.2f%% exceeds limit %.2f%% "
                "(%d/%d consecutive breaches) | hwm=%.4f equity=%.4f",
                drawdown * 100, max_drawdown_pct * 100,
                consecutive_breaches, consecutive_limit,
                hwm, equity,
            )
            if consecutive_breaches == 1 and notifier is not None:
                notifier.notify_drawdown_breach(drawdown, max_drawdown_pct)
            if consecutive_breaches >= consecutive_limit:
                logger.critical(
                    "Drawdown monitor: circuit breaker triggered — "
                    "drawdown %.2f%% exceeded limit for %d consecutive checks. "
                    "Halting all bots.",
                    drawdown * 100, consecutive_limit,
                )
                if notifier is not None:
                    notifier.notify_circuit_breaker(drawdown, max_drawdown_pct)
                with lock or nullcontext():
                    for bot in bots:
                        bot.halt("System drawdown limit exceeded.", level="critical", restartable=False)
                shutdown_event.set()
                break
        else:
            if consecutive_breaches > 0:
                logger.info(
                    "Drawdown monitor: drawdown %.2f%% back within limit — "
                    "resetting consecutive breach counter.",
                    drawdown * 100,
                )
            consecutive_breaches = 0
            logger.debug(
                "Drawdown monitor: drawdown=%.2f%% hwm=%.4f equity=%.4f",
                drawdown * 100, hwm, equity,
            )

    logger.info("Drawdown monitor stopped.")
