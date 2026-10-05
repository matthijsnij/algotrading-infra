"""
================================================================================
UNIT TESTS FOR NOTIFICATIONS/FACTORY.PY

To run all tests in this file:
    pytest tests/test_notifications/test_factory.py -v
================================================================================
"""

################### IMPORTS ##########################

import pytest

from lighthouse.notifications.factory import create_notifier
from lighthouse.notifications.telegram import TelegramChannel

################### TESTS ##########################

# enabled=False returns a disabled Notifier([]) regardless of other settings
def test_disabled_settings_returns_noop_notifier():
    notifier = create_notifier({"enabled": False, "channel": "telegram", "categories": {}})
    assert notifier._thread is None
    assert notifier._channels == []

# enabled=True with telegram channel builds a TelegramChannel from supplied credentials
def test_enabled_telegram_builds_channel():
    notifier = create_notifier(
        {
            "enabled": True,
            "channel": "telegram",
            "categories": {"startup": True, "shutdown": False},
        },
        credentials={"bot_token": "tok", "chat_id": "chat"},
    )

    assert len(notifier._channels) == 1
    assert isinstance(notifier._channels[0], TelegramChannel)
    assert notifier._enabled == {"startup"}

# enabled=True with telegram channel but no credentials supplied raises RuntimeError
def test_enabled_telegram_missing_credentials_raises():
    with pytest.raises(RuntimeError):
        create_notifier({"enabled": True, "channel": "telegram", "categories": {}})

# unsupported channel name raises ValueError
def test_unsupported_channel_raises():
    with pytest.raises(ValueError, match="Unsupported alerting channel"):
        create_notifier(
            {"enabled": True, "channel": "discord", "categories": {}},
            credentials={"bot_token": "tok", "chat_id": "chat"},
        )
