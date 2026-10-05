"""
================================================================================
TELEGRAM NOTIFICATION CHANNEL
================================================================================
Sends alert messages to a Telegram chat via the Bot API's sendMessage HTTPS
endpoint. Plain `requests` POST.

Classes:
    TelegramChannel : BaseNotificationChannel implementation for Telegram
================================================================================
"""

############ IMPORTS ##############

import time

import requests

from lighthouse.notifications.base import BaseNotificationChannel
from lighthouse.utils.logging import get_logger

############ CONSTANTS ##############

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_RETRY_DELAYS = (1.0, 3.0)  # seconds; up to 2 retries after the initial attempt
_TIMEOUT_SECONDS = 5

logger = get_logger(__name__)

############ CLASS ##############

class TelegramChannel(BaseNotificationChannel):
    """
    Sends messages to a single Telegram chat via bot token + chat ID.

    Retries up to len(_RETRY_DELAYS) times with a short fixed delay on failure,
    honoring Telegram's Retry-After header on an HTTP 429. Never raises — all
    failures are logged as a warning and swallowed, since a notification failure
    must not be allowed to crash the caller.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._url = _API_URL.format(token=bot_token)
        self._chat_id = chat_id

    def send(self, message: str, level: str) -> None:
        """
        POST message to the configured Telegram chat. Never raises.

        Args:
            message: text to send.
            level:   severity label, prefixed onto the message for quick triage in-chat.
        """
        payload = {"chat_id": self._chat_id, "text": f"[{level.upper()}] {message}"}

        attempts = len(_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                response = requests.post(self._url, json=payload, timeout=_TIMEOUT_SECONDS)
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    if attempt < attempts - 1:
                        time.sleep(retry_after)
                        continue
                    logger.warning("Telegram send failed after retries: rate-limited (429).")
                    return
                response.raise_for_status()
                return
            except Exception as exc:
                if attempt < attempts - 1:
                    time.sleep(_RETRY_DELAYS[attempt])
                    continue
                logger.warning("Telegram send failed after %d attempt(s): %s", attempts, exc)
                return

    @staticmethod
    def _parse_retry_after(response: "requests.Response") -> float:
        """Parse the Retry-After header (seconds); falls back to the first retry delay."""
        try:
            return float(response.headers.get("Retry-After", _RETRY_DELAYS[0]))
        except (TypeError, ValueError):
            return _RETRY_DELAYS[0]
