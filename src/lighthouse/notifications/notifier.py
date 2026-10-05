"""
================================================================================
NOTIFIER
================================================================================
Async, non-blocking dispatcher for all alerting events. Owns a bounded queue
and a single daemon worker thread; every notify_*/send_summary() call just
formats a message and enqueues it — callers (bot tick thread, monitor thread,
supervisor thread) never block on real network I/O.

Per-category toggles are checked before enqueueing, so a disabled category
never touches the queue. When constructed with an empty channel list (alerting
disabled), every notify_* method is a no-op — call sites never need to guard
with `if notifier is not None`.

Constants:
    ALERT_CATEGORIES : canonical set of category keys, one per event below —
                      mirrored by config/models.py::AlertingConfig.categories.

Classes:
    Notifier : queue + worker thread + one notify_* method per event category.
================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

import queue
import threading
from typing import TYPE_CHECKING

from lighthouse.notifications.base import BaseNotificationChannel
from lighthouse.utils.logging import get_logger

if TYPE_CHECKING:
    from lighthouse.domain.trade_log import TradeRecord

############ CONSTANTS ##############

ALERT_CATEGORIES: tuple = (
    "startup",
    "shutdown",
    "halt",
    "emergency_close",
    "reconcile_issue",
    "restart_success",
    "restart_failed_attempt",
    "restart_exhausted",
    "drawdown_breach",
    "circuit_breaker",
    "position_opened",
    "trade_closed",
    "summary",
)

_SENTINEL = object()  # enqueued by close() to unblock the worker's queue.get()

logger = get_logger(__name__)

############ CLASS ##############

class Notifier:
    """
    Formats and dispatches alert messages to one or more notification channels.

    Attributes:
        _channels:  list of BaseNotificationChannel instances; empty -> disabled no-op.
        _enabled:   set of category keys allowed through to the queue.
        _queue:     bounded queue.Queue of (message, level) tuples.
        _thread:    daemon worker thread; None when disabled (no channels).
    """

    def __init__(
        self,
        channels: list[BaseNotificationChannel],
        enabled_categories: set | None = None,
        queue_maxsize: int = 100,
    ) -> None:
        """
        Args:
            channels:            configured notification channels; empty list disables alerting entirely.
            enabled_categories:  set of category keys to allow through; defaults to all of ALERT_CATEGORIES.
            queue_maxsize:       bound on the pending-message queue (spam safety net).
        """
        self._channels = channels
        self._enabled = enabled_categories if enabled_categories is not None else set(ALERT_CATEGORIES)
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._thread: threading.Thread | None = None
        if self._channels:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ── internals ─────────────────────────────────────────────────────────────

    def _enqueue(self, category: str, message: str, level: str) -> None:
        """Format-and-forget: enqueue (message, level) if enabled, else no-op."""
        if not self._channels or category not in self._enabled:
            return
        try:
            self._queue.put_nowait((message, level))
        except queue.Full:
            logger.warning("Notifier: queue full — dropping message | category=%s", category)

    def _worker(self) -> None:
        """Background loop: pull messages off the queue and dispatch to every channel."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            message, level = item
            for channel in self._channels:
                try:
                    channel.send(message, level)
                except Exception as exc:
                    logger.warning("Notifier: channel send failed | error=%s", exc)

    def close(self, timeout: float = 5.0) -> None:
        """Flush and stop the worker thread; no-op if alerting is disabled."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            logger.warning("Notifier: queue full while closing — worker may not exit promptly.")
        self._thread.join(timeout=timeout)

    # ── notify_* methods ──────────────────────────────────────────────────────

    def notify_startup(self, exchange_name: str, bot_names: list) -> None:
        message = f"Live trading started | exchange={exchange_name} bots={', '.join(bot_names)}"
        self._enqueue("startup", message, "info")

    def notify_shutdown(self, reason: str) -> None:
        message = f"Live trading shut down | reason={reason}"
        self._enqueue("shutdown", message, "info")

    def notify_halt(self, bot_name: str, reason: str, level: str, restartable: bool) -> None:
        message = f"{bot_name} HALTED | reason={reason} restartable={restartable}"
        self._enqueue("halt", message, level)

    def notify_emergency_close(self, bot_name: str, symbol: str, side: str, reason: str, success: bool) -> None:
        outcome = "succeeded" if success else "FAILED"
        message = f"{bot_name} emergency close {outcome} | symbol={symbol} side={side} reason={reason}"
        self._enqueue("emergency_close", message, "info" if success else "critical")

    def notify_reconcile_issue(self, bot_name: str, symbol: str, detail: str, level: str) -> None:
        message = f"{bot_name} reconciliation issue | symbol={symbol} detail={detail}"
        self._enqueue("reconcile_issue", message, level)

    def notify_restart_success(self, bot_name: str, attempt: int, max_attempts: int) -> None:
        message = f"{bot_name} restarted | attempt={attempt}/{max_attempts}"
        self._enqueue("restart_success", message, "info")

    def notify_restart_failed_attempt(self, bot_name: str, attempt: int, max_attempts: int, detail: str) -> None:
        message = f"{bot_name} restart attempt failed | attempt={attempt}/{max_attempts} detail={detail}"
        self._enqueue("restart_failed_attempt", message, "warning")

    def notify_restart_exhausted(self, bot_name: str, bot_key: str, max_attempts: int) -> None:
        message = (
            f"{bot_name} exceeded max restart attempts ({max_attempts}) — manual intervention "
            f"required | state_key={bot_key}"
        )
        self._enqueue("restart_exhausted", message, "critical")

    def notify_drawdown_breach(self, pct: float, limit_pct: float) -> None:
        message = f"Drawdown breach | drawdown={pct * 100:.2f}% limit={limit_pct * 100:.2f}%"
        self._enqueue("drawdown_breach", message, "warning")

    def notify_circuit_breaker(self, pct: float, limit_pct: float) -> None:
        message = f"Circuit breaker tripped — all bots halted | drawdown={pct * 100:.2f}% limit={limit_pct * 100:.2f}%"
        self._enqueue("circuit_breaker", message, "critical")

    def notify_position_opened(self, bot_name: str, symbol: str, side: str, entry_price: float, size: float) -> None:
        message = f"{bot_name} position opened | symbol={symbol} side={side} entry={entry_price} size={size}"
        self._enqueue("position_opened", message, "info")

    def notify_trade_closed(self, record: "TradeRecord") -> None:
        message = (
            f"{record.bot_name} trade closed | symbol={record.symbol} side={record.side} "
            f"outcome={record.outcome} net_pnl={record.net_pnl:.4f} ({record.net_pnl_pct * 100:.2f}%) "
            f"reason={record.exit_reason}"
        )
        self._enqueue("trade_closed", message, "info")

    def send_summary(
        self,
        bot_name: str,
        symbol: str,
        mode: str,
        side: str,
        entry_price: float | None,
        position_size: float,
        unrealized_pnl: float | None,
        lifetime_trade_count: int,
    ) -> None:
        message = (
            f"{bot_name} summary | symbol={symbol} mode={mode} side={side} entry={entry_price} "
            f"size={position_size} unrealized_pnl={unrealized_pnl} total_trades={lifetime_trade_count}"
        )
        self._enqueue("summary", message, "info")
