"""
================================================================================
FUNDING MODEL - FIXED RATE
================================================================================

Simple funding rate model that returns the base funding rate unchanged.
No internal state, no modifications. Implements BaseFundingModel.
================================================================================
"""

from __future__ import annotations

####### IMPORTS ###################
from dataclasses import dataclass
from .base import BaseFundingModel

######### CLASS ###########
@dataclass
class FundingModel_Fixed(BaseFundingModel):
    """
    Funding rate model that applies a fixed base funding rate.

    No internal state or parameters. Simply returns base_funding_rate
    unchanged on every call.

    Methods:
        compute_rate(): returns base_funding_rate unchanged
    """

    def compute_rate(self, bar_idx: int, close_price: float, base_funding_rate: float) -> float:
        """
        Return the base funding rate unchanged.

        Args:
            bar_idx           : current bar index (unused)
            close_price       : current bar's close price (unused)
            base_funding_rate : instrument's base funding rate

        Returns:
            The base_funding_rate, unchanged.
        """
        return base_funding_rate
