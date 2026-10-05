"""
================================================================================
STOCK BOT STATE
================================================================================

State container for the StockBot — an Example Bot (CONTEXT.md).

Thin subclass of BaseState — the moving-average crossover strategy (see
bot.py) needs no additional fields beyond what BaseState already tracks.

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

from dataclasses import dataclass

from lighthouse.bots.base_state import BaseState

############ CLASS ##############

@dataclass
class StockBotState(BaseState):
    """Strategy-specific state for the StockBot (currently identical to BaseState)."""
