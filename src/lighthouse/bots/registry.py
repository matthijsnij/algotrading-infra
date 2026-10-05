"""
================================================================================
BOT REGISTRY
================================================================================

This module contains the bot registry and functionality to resolve a bot
class by name (as configured in a YAML config's `bots[].name` field).

Functions:
    resolve_bot(): resolve a bot class by name from the registry.
================================================================================
"""

from typing import Type

from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.examples.crypto_bot.bot import CryptoBot
from lighthouse.bots.examples.stock_bot.bot import StockBot

# ── Registry ──────────────────────────────────────────────────────────────────
_BOT_REGISTRY = {
    "crypto_bot": CryptoBot,
    "stock_bot": StockBot,
}


# ── Factory ───────────────────────────────────────────────────────────────────
def resolve_bot(name: str) -> Type[BaseBot]:
    """
    Resolve a bot class by its registered name.

    Args:
        name: bot name as it appears in `bots[].name` in a config YAML

    Returns:
        The BaseBot subclass registered under `name`.
    """
    if name not in _BOT_REGISTRY:
        available = ", ".join(sorted(_BOT_REGISTRY))
        raise ValueError(f"Bot '{name}' not registered. Available bots: {available}")
    return _BOT_REGISTRY[name]
