"""
================================================================================
BACKTEST CONFIG
================================================================================

This file contains the BacktestConfig dataclass, which defines the immutable configuration parameters for the BacktestExchange simulation engine.

All configuration parameters are held here and passed by reference to each sub-engine.
================================================================================
"""
from __future__ import annotations
# ── Imports ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass

from .funding_models.base import BaseFundingModel
from .reporter import _parse_timeframe

# ── Dataclass ──────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    """
    Container for all backtest simulation configuration.
    
    Attributes:
        spread                 : bid/ask half-spread as fraction of price
        taker_fee              : taker fee rate (fraction of notional)
        maker_fee              : maker fee rate (fraction of notional)
        slippage               : additional adverse slippage on taker fills (fraction)
        latency_bars           : bars of latency before market orders fill (must be >= 1)
        partial_fill_fraction  : fraction of bar volume available for limit fills
        funding_model          : optional BaseFundingModel subclass (None disables funding)
        limit_fill_policy      : fill realism policy for limit orders.
                                 "through" (default) = conservative: fills only when price
                                 trades THROUGH the limit by one tick (buy: low < limit - tick;
                                 sell: high > limit + tick).  "touch" = legacy optimistic:
                                 fills when price touches the limit (low <= limit / high >= limit).
    """
    spread: float
    taker_fee: float
    maker_fee: float
    slippage: float
    latency_bars: int
    partial_fill_fraction: float
    funding_model: BaseFundingModel | None
    limit_fill_policy: str = "through"

    def __post_init__(self) -> None:
        """
        Validate configuration invariants.
        """
        if self.latency_bars < 1:
            raise ValueError(
                f"BacktestConfig: latency_bars must be >= 1 (got {self.latency_bars}). "
                f"latency_bars = 0 would cause look-ahead bias (orders placed after a bar closes "
                f"would fill at that bar's price). Use latency_bars = 1 for next-bar fills."
            )
        if self.limit_fill_policy not in ("through", "touch"):
            raise ValueError(
                f"BacktestConfig: limit_fill_policy must be 'through' or 'touch' "
                f"(got '{self.limit_fill_policy}')."
            )


# ── Dataclass ──────────────────────────────────────────────────────────────────
@dataclass
class MetricsConfig:
    """
    Container for reporting-only configuration: how Sharpe/Sortino/CAGR are computed.

    Kept separate from BacktestConfig (which is fill-simulation mechanics) because
    these fields affect only Reporter.get_results(), not the simulation itself.

    Attributes:
        metrics_timeframe : timeframe the equity curve is resampled to before computing
                            Sharpe/Sortino/return diagnostics, e.g. "1d" (default).
        risk_free_rate    : annual risk-free rate as a fraction (e.g. 0.02 = 2%). Default
                            0.0 (no adjustment). Note: for a leveraged
                            perp strategy where funding already embeds carry, subtracting
                            a risk-free rate double-counts; 0.0 is correct for a
                            fully-funded cash account.
        sortino_target    : annual minimum acceptable return (MAR) for the Sortino
                            downside calculation. None (default) falls back to
                            risk_free_rate.
    """
    metrics_timeframe: str            = "1d"
    risk_free_rate: float             = 0.0
    sortino_target: float | None   = None

    def __post_init__(self) -> None:
        """
        Validate configuration invariants.
        """
        _parse_timeframe(self.metrics_timeframe)  # raises ValueError if malformed

        if not (-1.0 <= self.risk_free_rate <= 1.0):
            raise ValueError(
                f"MetricsConfig: risk_free_rate must be a fraction in [-1.0, 1.0] "
                f"(got {self.risk_free_rate})."
            )
        if self.sortino_target is not None and not (-1.0 <= self.sortino_target <= 1.0):
            raise ValueError(
                f"MetricsConfig: sortino_target must be a fraction in [-1.0, 1.0] "
                f"(got {self.sortino_target})."
            )

