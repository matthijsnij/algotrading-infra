"""
================================================================================
UNIT TESTS FOR NOTIFICATIONS/NOTIFIER.PY

To run all tests in this file:
    pytest tests/test_notifications/test_notifier.py -v
================================================================================
"""

################### IMPORTS ##########################

import threading
import time
from unittest.mock import patch

from lighthouse.notifications.base import BaseNotificationChannel
from lighthouse.notifications.notifier import ALERT_CATEGORIES, Notifier

################### HELPERS ##########################

class FakeChannel(BaseNotificationChannel):
    """In-process channel double — records (message, level) tuples, thread-safe."""
    def __init__(self, on_send=None):
        self.messages = []
        self._lock = threading.Lock()
        self._on_send = on_send

    def send(self, message: str, level: str) -> None:
        if self._on_send is not None:
            self._on_send()
        with self._lock:
            self.messages.append((message, level))


class RaisingChannel(BaseNotificationChannel):
    """Channel double that always raises — used to prove the worker survives it."""
    def send(self, message: str, level: str) -> None:
        raise RuntimeError("channel is down")

################### TESTS ##########################

# ── disabled (no channels) ──────────────────────────────────────────────────

# Notifier([]) never starts a worker thread
def test_disabled_notifier_starts_no_thread():
    notifier = Notifier([])
    assert notifier._thread is None

# every notify_* call is a silent no-op when disabled
def test_disabled_notifier_notify_calls_are_noop():
    notifier = Notifier([])
    notifier.notify_startup("phemex", ["ExampleBot"])
    notifier.notify_halt("ExampleBot", "reason", "error", True)
    notifier.close()  # must not raise/hang

# ── async dispatch ───────────────────────────────────────────────────────────

# a notify_* call is delivered to the channel asynchronously; close() flushes it
def test_notify_dispatches_to_channel():
    channel = FakeChannel()
    notifier = Notifier([channel])

    notifier.notify_startup("phemex", ["ExampleBot"])
    notifier.close()

    assert len(channel.messages) == 1
    message, level = channel.messages[0]
    assert "phemex" in message
    assert "ExampleBot" in message
    assert level == "info"

# all configured channels receive every message
def test_notify_dispatches_to_all_channels():
    channel_a = FakeChannel()
    channel_b = FakeChannel()
    notifier = Notifier([channel_a, channel_b])

    notifier.notify_shutdown("manual stop")
    notifier.close()

    assert len(channel_a.messages) == 1
    assert len(channel_b.messages) == 1

# ── per-category filtering ───────────────────────────────────────────────────

# a category left out of enabled_categories never reaches the channel
def test_disabled_category_not_enqueued():
    channel = FakeChannel()
    enabled = set(ALERT_CATEGORIES) - {"startup"}
    notifier = Notifier([channel], enabled_categories=enabled)

    notifier.notify_startup("phemex", ["ExampleBot"])
    notifier.notify_shutdown("manual stop")
    notifier.close()

    assert len(channel.messages) == 1
    assert "manual stop" in channel.messages[0][0]

# notify_restart_exhausted() includes both the bot name and its state-store identifier
def test_notify_restart_exhausted_includes_state_key():
    channel = FakeChannel()
    notifier = Notifier([channel])

    notifier.notify_restart_exhausted("CryptoBot", "crypto_bot:BTC/USDT:USDT", 3)
    notifier.close()

    assert len(channel.messages) == 1
    message, level = channel.messages[0]
    assert "CryptoBot" in message
    assert "crypto_bot:BTC/USDT:USDT" in message
    assert level == "critical"

# ── close() semantics ────────────────────────────────────────────────────────

# close() joins the worker thread — it is no longer alive afterwards
def test_close_joins_worker_thread():
    channel = FakeChannel()
    notifier = Notifier([channel])
    notifier.notify_startup("phemex", ["ExampleBot"])

    notifier.close(timeout=2.0)

    assert not notifier._thread.is_alive()

# close() on an already-disabled notifier is a no-op (no thread to join)
def test_close_disabled_notifier_is_noop():
    notifier = Notifier([])
    notifier.close()  # must not raise

# ── worker robustness ────────────────────────────────────────────────────────

# a channel raising an exception does not crash the worker thread or subsequent sends
def test_channel_exception_does_not_crash_worker():
    good_channel = FakeChannel()
    notifier = Notifier([RaisingChannel(), good_channel])

    notifier.notify_startup("phemex", ["ExampleBot"])
    notifier.notify_shutdown("manual stop")
    notifier.close()

    assert len(good_channel.messages) == 2

# ── queue-full drop ──────────────────────────────────────────────────────────

# when the queue is full, the extra message is dropped and a warning is logged
def test_queue_full_drops_message_and_logs_warning():
    release = threading.Event()
    blocked_channel = FakeChannel(on_send=lambda: release.wait(timeout=2.0))
    notifier = Notifier([blocked_channel], queue_maxsize=1)

    with patch("lighthouse.notifications.notifier.logger") as mock_logger:
        notifier.notify_startup("phemex", ["ExampleBot"])  # picked up immediately, worker blocks in send()
        time.sleep(0.1)  # let the worker pull this item off the queue
        notifier.notify_shutdown("r1")   # fills the queue (maxsize=1)
        notifier.notify_halt("bot", "reason", "error", True)  # queue full -> dropped

        release.set()
        notifier.close()

    assert mock_logger.warning.called
