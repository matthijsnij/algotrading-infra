"""
================================================================================
SUMMARY MODULE
================================================================================

Periodic bot-status summary notifier.

Functions:
    periodic_summary() : background thread that periodically sends a bot's
                         status summary through the Notifier.
================================================================================
"""

############ IMPORTS ############

import threading
import time
from typing import Any
from lighthouse.utils.logging import get_logger

############ LOGGER ############

logger = get_logger(__name__)

############ PERIODIC SUMMARY ############

def periodic_summary(
    bot: Any,
    notifier: Any,
    interval_seconds: int,
    shutdown_event: threading.Event,
) -> None:
    """
    Background thread that periodically builds and sends a status summary for one bot.

    Runs until shutdown_event is set, sleeping interval_seconds between each summary.
    A failure to build or send a summary is logged and the loop continues — a
    transient error here must never take down a bot thread or the main process.

    Args:
        bot:              the bot instance to summarize (calls bot.build_summary())
        notifier:         Notifier instance to dispatch the summary through
                         (calls notifier.send_summary(bot_name=..., **summary))
        interval_seconds: seconds to sleep between summaries
        shutdown_event:   threading.Event shared with the main thread
    """
    bot_name = bot.__class__.__name__
    logger.info("Periodic summary started for %s | interval=%ds", bot_name, interval_seconds)

    while not shutdown_event.is_set():
        time.sleep(interval_seconds)

        if shutdown_event.is_set():
            break

        try:
            summary = bot.build_summary()
            notifier.send_summary(bot_name=bot_name, **summary)
        except Exception as exc:
            logger.error("Periodic summary: failed to build/send summary for %s | error=%s", bot_name, exc)

    logger.info("Periodic summary stopped for %s.", bot_name)
