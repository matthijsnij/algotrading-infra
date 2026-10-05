"""
================================================================================
BACKTEST STATE
================================================================================

This file contains the BacktestState class, which defines the shared mutable state for the BacktestExchange simulation engine.

All configuration, data, and runtime state is held here and passed by reference to each sub-engine.
================================================================================
"""

from __future__ import annotations

############# IMPORTS #############
import pandas as pd
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING: # Avoid circular import for type hints
    from .config import BacktestConfig, MetricsConfig

from lighthouse.domain.instruments import InstrumentSpec
from lighthouse.domain.calendars import TradingCalendar

################# CLASS ################
class BacktestState:
    """
    Single source of truth for all BacktestExchange state.

    Passed by reference to each sub-engine so that every component reads and
    writes to the same underlying objects without copying.
    """

    def __init__(
        self,
        df: pd.DataFrame,               
        ts_col: str,                        
        symbol: str, 
        instrument: InstrumentSpec,
        base_timeframe: str,
        initial_balance: float,
        config: "BacktestConfig", # in quotes because of TYPE_CHECKING import
        calendar: TradingCalendar | None = None,
        metrics_config: "MetricsConfig" | None = None,
    ) -> None:
        
        """
        Constructor. Initialize the BacktestState with all required configuration.
        
        Args:
            df              : OHLCV DataFrame for the backtest
            ts_col          : name of the timestamp column in df
            symbol          : market symbol
            instrument      : InstrumentSpec defining the instrument mechanics
            base_timeframe  : timeframe string of the input df 
            initial_balance : starting account balance in quote currency
            config          : BacktestConfig instance
            calendar        : TradingCalendar resolved from instrument.calendar; defaults
                              to CALENDAR_24_7 if omitted
            metrics_config  : MetricsConfig instance (reporting-only settings); defaults
                              to MetricsConfig() if omitted
        
        Properties:
            current_bar : OHLCV pd.Series for the current bar
        
        Methods:
            get_bar_timestamp(): return the timestamp for a given bar index
        """

        # ── Data (immutable after construction) ───────────────────────────────
        self.df                          = df # OHLCV df
        self.ts_col                      = ts_col # timestamp column name

        # ── Config (immutable after construction) ─────────────────────────────
        self.symbol                      = symbol # market symbol
        self.instrument                  = instrument # InstrumentSpec
        self.base_timeframe              = base_timeframe # timeframe of df
        self.config                      = config # BacktestConfig instance
        if calendar is None:
            from lighthouse.domain.calendars import CALENDAR_24_7 # local default to avoid forcing calendar on every caller
            calendar = CALENDAR_24_7
        self.calendar                    = calendar # TradingCalendar instance
        if metrics_config is None:
            from .config import MetricsConfig # local import to avoid circular import
            metrics_config = MetricsConfig() # if not passed, use default MetricsConfig
        self.metrics_config              = metrics_config # MetricsConfig instance
        self.pinned_periods_per_year: float | None = None # annualization factor for Sharpe/Sortino; set internally by walk-forward if pinning is needed
        
        # Extract trading mechanics from config for convenience
        self.spread                      = config.spread
        self.taker_fee                   = config.taker_fee
        self.maker_fee                   = config.maker_fee
        self.slippage                    = config.slippage
        self.latency_bars                = config.latency_bars
        self.partial_fill_fraction       = config.partial_fill_fraction
        self.limit_fill_policy           = config.limit_fill_policy

        # ── Balance ───────────────────────────────────────────────────────────
        self.initial_balance: float      = initial_balance # initial account balance in quote currency
        self.balance: dict[str, float]               = {"free": initial_balance, "used": 0.0, "total": initial_balance} # balance dict

        # ── Order books ───────────────────────────────────────────────────────
        self.open_orders: dict[str, dict[str, Any]]   = {}    # resting stop/limit orders
        self.filled_order_ids: set[str]     = set() # IDs of orders that were filled
        self.queued_market_orders: list[dict[str, Any]]     = []    # market orders awaiting latency fill

        # ── Position ──────────────────────────────────────────────────────────
        self.position: dict[str, Any] | None            = None  # current open position, or None if flat
        self.liquidation_price: float | None  = None  # liquidation price of the current position

        # ── Trade tracking ──────────────────────────────────
        self.open_trade: dict[str, Any] | None  = None  # in-progress trade metadata
        self.trade_log: list[dict[str, Any]]       = []    # completed trade records
        self.equity_curve: list[dict[str, Any]]    = []    # per-bar equity snapshots

        # ── Funding & fee accumulators ────────────────────────────────────────
        self.bars_since_funding: int     = 0 # number of bars since last funding payment/credit
        self.total_fees_paid: float      = 0.0 # total trading fees paid (entry + exit only)
        self.total_funding_paid: float   = 0.0 # total funding paid (negative = net received)

        # ── Order rejection tracking ──────────────────────────────────────────
        self.orders_rejected: int = 0 # count of orders rejected due to insufficient balance

        # ── Run state ─────────────────────────────────────────────────────────
        self.cursor: int                 = -1 # index of the current bar in df; starts at -1 so first advance() sets to 0
        self.finalized: bool             = False # True if the backtest has completed 

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def current_bar(self) -> pd.Series:
        """
        Get the OHLCV row for the current bar.
        
        Returns:
            pd.Series representing the current bar's OHLCV data.
        """
        return self.df.iloc[self.cursor]

    def get_bar_timestamp(self, bar_idx: int) -> pd.Timestamp:
        """
        Get the timestamp for the given bar index.

        Args:
            bar_idx (int): Index of the bar.

        Returns:
            pd.Timestamp representing the timestamp of the specified bar.
        """
        return self.df.iloc[bar_idx][self.ts_col]
