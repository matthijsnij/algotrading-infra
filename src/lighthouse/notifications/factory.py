"""
================================================================================
NOTIFIER FACTORY
================================================================================
Build a configured Notifier from the "alerting_settings" dict produced by
config/convert.py::config_to_live_context(). Currently only the Telegram
channel is implemented.

Credentials are resolved by the caller (runtime/live.py, which already depends
on the config layer) and passed in already-resolved, so this module has no
import-time dependency on lighthouse.config — avoiding a
notifications -> config -> bots -> notifications import cycle.

Functions:
    create_notifier(settings: dict, credentials: dict | None = None) -> Notifier
================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

from lighthouse.notifications.notifier import Notifier
from lighthouse.notifications.telegram import TelegramChannel

############ FACTORY ##############

def create_notifier(settings: dict, credentials: dict | None = None) -> Notifier:
    """
    Build a Notifier from an alerting settings dict.

    Args:
        settings: {"enabled": bool, "channel": str, "categories": dict[str, bool]}
        credentials: pre-resolved credentials for the configured channel (e.g.
                    {"bot_token": ..., "chat_id": ...} for telegram), or None when
                    alerting is disabled. Resolving credentials is the caller's
                    responsibility (see runtime/live.py::main()).

    Returns:
        A disabled no-op Notifier([]) if settings["enabled"] is falsy, otherwise a
        Notifier wired to the configured channel with only the enabled categories
        allowed through.

    Raises:
        ValueError: settings["channel"] is not a supported channel name.
        RuntimeError: settings["enabled"] is truthy but no credentials were supplied
                     for the configured channel.
    """
    if not settings.get("enabled", False):
        return Notifier([])

    channel_name = settings.get("channel", "telegram") # default to telegram if missing
    categories = settings.get("categories", {})
    enabled_categories = {key for key, value in categories.items() if value}

    if channel_name == "telegram":
        if not credentials:
            raise RuntimeError(
                "Telegram alerting is enabled but no credentials were supplied to create_notifier()."
            )
        channels = [TelegramChannel(credentials["bot_token"], credentials["chat_id"])]
    else:
        raise ValueError(f"Unsupported alerting channel: '{channel_name}'")

    return Notifier(channels, enabled_categories=enabled_categories)
