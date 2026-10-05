"""
================================================================================
LIGHTHOUSE
================================================================================
Infrastructure for researching, validating and running algorithmic trading
strategies.

Layers:
    domain/       : pure functions and value types over market data
    execution/    : exchange-agnostic trading verbs (data, orders, position)
    exchanges/    : BaseExchange contract plus live and simulated implementations
    data/         : historical data fetchers and sources
    bots/         : strategy base classes and lifecycle
    optimization/ : parameter sweeps and walk-forward analysis
    runtime/      : session execution, logging, monitoring
================================================================================
"""

__version__ = "0.1.0"
