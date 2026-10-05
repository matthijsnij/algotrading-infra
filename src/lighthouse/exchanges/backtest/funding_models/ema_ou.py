"""
================================================================================
FUNDING MODEL - EMA + ORNSTEIN-UHLENBECK NOISE
================================================================================

Synthetic funding rate model using EMA trend detection and Ornstein-Uhlenbeck
mean-reverting stochastic noise. Implements BaseFundingModel.

Rate components:
    1. Trend component:
       basis = (close - EMA) / EMA   (futures premium/discount proxy)
       trend = tanh(basis * basis_sensitivity) * base_funding_rate
       Bounded to [-base_funding_rate, +base_funding_rate] via tanh.

    2. OU noise:
       dX = -theta * X * dt + sigma * dW
       Zero-mean, mean-reverting noise on top of the trend.
       Disabled when ou_sigma == 0.0.
================================================================================
"""

from __future__ import annotations

####### IMPORTS ###################
import math
import numpy as np
from dataclasses import dataclass, field
from .base import BaseFundingModel

######## CLASS ###########
@dataclass
class FundingModel_EMA_OU(BaseFundingModel):
    """
    Funding rate model based on EMA trend + Ornstein-Uhlenbeck noise.

    Attributes:
        ema_period        : EMA window for price trend detection (number of bars)
        basis_sensitivity : tanh scaling factor for EMA basis → funding rate
        ou_theta          : OU mean-reversion speed [0, 1]; 0 = disabled
        ou_sigma          : OU process noise volatility; 0 = disabled
    
    Methods:
        compute_rate(): compute the synthetic funding rate for the current bar
    """
    ema_period:        int   = 120
    basis_sensitivity: float = 0.5
    ou_theta:          float = 0.1
    ou_sigma:          float = 0.05

    # Internal state
    _ema:      float = field(default=None, init=False, repr=False)
    _ou_state: float = field(default=0.0,  init=False, repr=False)

    def compute_rate(self, bar_idx: int, close_price: float, base_funding_rate: float) -> float:
        """
        Compute the synthetic funding rate for the current bar.

        Updates EMA and OU state on every call so the process evolves
        continuously between funding settlement periods.

        Args:
            bar_idx           : current bar index (unused; EMA is sequential)
            close_price       : current bar's close price
            base_funding_rate : instrument's base funding rate (bounds tanh output)

        Returns:
            Signed synthetic funding rate.
        """
        # EMA update
        alpha = 2.0 / (self.ema_period + 1)
        if self._ema is None:
            self._ema = close_price
        else:
            self._ema = alpha * close_price + (1.0 - alpha) * self._ema

        # Trend component: bounded by base_funding_rate via tanh
        basis = (close_price - self._ema) / self._ema if self._ema != 0.0 else 0.0
        trend = math.tanh(basis * self.basis_sensitivity) * base_funding_rate

        # OU noise update
        if self.ou_sigma > 0.0:
            dW = float(np.random.standard_normal())
            self._ou_state += -self.ou_theta * self._ou_state + self.ou_sigma * dW
        else:
            self._ou_state = 0.0

        return trend + self._ou_state
