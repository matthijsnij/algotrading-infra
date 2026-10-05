"""
================================================================================
CRYPTO BOT STATE
================================================================================

State container for the CryptoBot.

Explicit (currently empty) subclass of BaseState — every bot gets a named
place for its own state, even when there's nothing strategy-specific in it
yet. sl_price/tp_price live in BaseState (shared by every strategy that uses
SL/TP orders, and needed by BaseBot.close_trade()).

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

from dataclasses import dataclass

from lighthouse.bots.base_state import BaseState

############ CLASS ##############

@dataclass
class CryptoBotState(BaseState):
    """Strategy-specific state for the CryptoBot (currently identical to BaseState)."""

