"""
================================================================================
BASE FUNDING MODEL
================================================================================

Abstract base class for all funding rate models used by FundingEngine.

To implement a custom funding model, subclass BaseFundingModel and implement
compute_rate(). 
================================================================================
"""

######## IMPORTS ###################
from abc import ABC, abstractmethod

######## CLASS ###########
class BaseFundingModel(ABC):
    """
    Abstract interface for funding rate computation.

    FundingEngine calls compute_rate() each bar. The model is responsible for
    maintaining any internal state it needs between calls.

    Methods:
        compute_rate(): compute the effective funding rate for the current bar
    """

    @abstractmethod
    def compute_rate(self, bar_idx: int, close_price: float, base_funding_rate: float) -> float:
        """
        Compute the effective funding rate for the current bar.

        Called every bar by FundingEngine so internal model state evolves
        continuously even between settlement periods.

        Args:
            bar_idx           : current bar index (cursor position)
            close_price       : current bar's close price
            base_funding_rate : instrument's baseline funding rate from InstrumentSpec

        Returns:
            Signed effective funding rate.
        """
        ...
