"""
================================================================================
BACKTEST EXCHANGE
================================================================================

This file contains the BacktestExchange class, which handles orchestration of the backtest simulation engine.

The BacktestExchange is a thin wrapper that delegates all simulation work to focused sub-engines that share a single BacktestState instance:
    - DataWindow: handles OHLCV data access 
    - OrderBook: handles order registration and queuing
    - FillEngine: handles position management and fill simulation
    - LiquidationEngine: handles liquidation checks and forced position closure (if applicable)
    - FundingEngine: handles funding rate application (if applicable)
    - Reporter: handles equity tracking and performance metrics
"""

from __future__ import annotations

####### IMPORTS ###########
import uuid
import pandas as pd
from typing import Any

from lighthouse.exchanges.base import BaseExchange
from lighthouse.domain.instruments import InstrumentSpec
from lighthouse.domain.calendars import create_calendar
from lighthouse.domain.enums import normalize_side
from lighthouse.domain.exchange_types import NormalizedPosition, NormalizedOrder
from lighthouse.domain.timeframe import Timeframe

from .state import BacktestState
from .data_window import DataWindow
from .order_book import OrderBook, TYPE_LIMIT, TYPE_STOP, TYPE_STOP_LIMIT
from .fill_engine import FillEngine
from .mechanics import LiquidationEngine, FundingEngine
from .config import BacktestConfig, MetricsConfig
from .reporter import Reporter, _parse_timeframe
from lighthouse.backtest_data.sources.base import validate_bar_continuity

########## CLASS ##########
class BacktestExchange(BaseExchange):
    """
    Simulated exchange for backtesting.

    Properties:
        cursor       : current bar index (0-based; -1 before first advance())
        current_bar  : OHLCV pd.Series for the current bar
        trade_log    : list of completed trade dicts
        equity_curve : list of per-bar {bar, timestamp, equity} snapshots

    Methods:
        advance()                 : advance the simulation by one base bar
        is_signal_bar_close()     : check if the current bar is the last bar of a signal timeframe
        finalize()                : finalize the backtest run 
        All BaseExchange methods
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_balance: float,
        instrument: InstrumentSpec,
        base_timeframe: str,
        config: BacktestConfig,
        metrics_config: MetricsConfig | None = None,
        pinned_periods_per_year: float | None = None,
    ) -> None:
        """
        Constructor.
        
        Params:
                df                  : OHLCV DataFrame with DatetimeIndex
                symbol              : market symbol
                initial_balance     : starting account equity
                instrument          : InstrumentSpec for the traded instrument
                base_timeframe      : timeframe of df
                config              : BacktestConfig object 
                metrics_config      : MetricsConfig object (reporting; defaults to MetricsConfig())
                pinned_periods_per_year : optional float for annualizing metrics (e.g. 365 for daily bars)
        """
        super().__init__(testnet=False)

        # Validate input df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"BacktestExchange: df must have a DatetimeIndex. "
                f"Got {type(df.index).__name__}. "
                f"Convert with: df.index = pd.to_datetime(df.index, utc=True)"
            )

        if df.index.tz is None:
            raise ValueError(
                "BacktestExchange: df index must be timezone-aware (UTC)."
                "Convert with: df.index = pd.to_datetime(df.index, utc=True)"
            )

        calendar = create_calendar(instrument.calendar)
        validate_bar_continuity(df, calendar, base_timeframe)

        # Validate config
        if config is None:
            raise ValueError(
                "BacktestExchange: config is required. "
                "Provide a BacktestConfig object with all parameters explicitly set."
            )
        metrics_config = metrics_config if metrics_config is not None else MetricsConfig()

        # metrics_timeframe must be at least as coarse as the base data, else resampling upsamples
        base_seconds, _    = _parse_timeframe(base_timeframe)
        metrics_seconds, _ = _parse_timeframe(metrics_config.metrics_timeframe)
        if metrics_seconds < base_seconds:
            raise ValueError(
                f"BacktestExchange: metrics_config.metrics_timeframe "
                f"({metrics_config.metrics_timeframe}) must be >= base_timeframe "
                f"({base_timeframe})."
            )

        # Shared state, passed by reference to all sub-engines
        self._state = BacktestState(
            df                        = df.copy().reset_index(drop=False),
            ts_col                    = df.index.name,
            symbol                    = symbol,
            instrument                = instrument,
            base_timeframe            = base_timeframe,
            initial_balance           = initial_balance,
            config                    = config,
            calendar                  = calendar,
            metrics_config            = metrics_config,
        )
        self._state.pinned_periods_per_year = pinned_periods_per_year

        # Initiate sub-engines
        self._fills    = FillEngine(self._state)
        self._book     = OrderBook(self._state, self._fills)
        self._funding  = FundingEngine(self._state)
        self._liq     = LiquidationEngine(self._state, close_position_fn=self._fills.close_position)
        self._data     = DataWindow(self._state)
        self._reporter = Reporter(self._state)

    # ── Runner interface ──────────────────────────────────────────────────────

    def advance(self) -> bool:
        """
        Advance the cursor by one bar and run all simulation events for that bar.

        Event sequence:
            1. Fill queued market orders at bar open  (latency simulation)
            2. Check resting stop / limit orders against bar OHLCV (SL before TP)
            3. Update mark price to bar close
            4. Apply funding rate if the interval has elapsed
            5. Check for liquidation at the updated mark price
            6. Append equity-curve snapshot

        Returns:
            True if the bar was processed; False when end-of-data is reached.
        """
        s = self._state
        s.cursor += 1
        if s.cursor >= len(s.df):
            s.cursor -= 1  # keep at last valid bar
            return False

        self._book.fill_queued_market_orders()  # step 1
        self._book.check_pending_orders()       # step 2
        self._liq.update_mark_price()           # step 3
        self._funding.apply_funding()           # step 4
        self._liq.check_liquidation()           # step 5
        self._reporter.record_equity()          # step 6
        return True

    def is_signal_bar_close(self, timeframe: str) -> bool:
        """
        Return True if the current bar is the last bar of its signal period.
        Wrapper around DataWindow.is_signal_bar_close() for external access.
        
        Args:
            timeframe : signal timeframe string
        
        Returns:
            True if the current bar is the last bar of its signal period; False otherwise.
        """
        return self._data.is_signal_bar_close(timeframe)

    def finalize(self) -> None:
        """
        Finalize the backtest run. Must be called after the simulation loop.

        1. Cancels all resting and queued orders.
        2. Force-closes any open position at the last bar's close (taker fee).
        3. Marks the run as finalized (idempotent — safe to call multiple times).
        """
        s = self._state
        if s.finalized:
            return

        # Clear resting and queued orders
        s.open_orders.clear()
        s.queued_market_orders.clear()

        # Close any open position at the last bar's close (taker fee)
        if s.position is not None and s.cursor >= 0:
            last_close = float(s.df.iloc[s.cursor]["close"])
            force_id   = str(uuid.uuid4())
            self._fills.close_position(last_close, order_id=force_id, is_maker=False, close_reason="finalize")

        # Set finalized flag to True, indicating that the backtest run is complete
        s.finalized = True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def cursor(self) -> int:
        """Current bar index (0-based)."""
        return self._state.cursor

    @property
    def current_bar(self) -> pd.Series:
        """OHLCV row for the current bar."""
        return self._state.current_bar

    # ── Reporting interface ───────────────────────────────────────────────────

    @property
    def trade_log(self) -> list[dict[str, Any]]:
        """
        Return trade log from internal state.
        """
        return list(self._state.trade_log)

    @property
    def equity_curve(self) -> list[dict[str, Any]]:
        """
        Return equity curve from internal state.
        """
        return list(self._state.equity_curve)

    def get_results(self) -> dict[str, Any]:
        """
        Return a summary dict of backtest performance metrics. Wrapper around Reporter.get_results() for external access.
        """
        return self._reporter.get_results()

    # ── BaseExchange: market data ─────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int) -> list:
        return self._data.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_bid_ask(self, symbol: str) -> dict[str, Any]:
        return self._data.fetch_bid_ask(symbol)

    # ── BaseExchange: account ─────────────────────────────────────────────────

    def fetch_balance(self) -> dict[str, dict[str, float]]:
        s = self._state
        return {
            s.instrument.quote_currency: {
                "free":  s.balance["free"],
                "used":  s.balance["used"],
                "total": s.balance["total"],
            }
        }

    # ── BaseExchange: positions ───────────────────────────────────────────────

    def fetch_open_positions(self, symbol: str) -> list[NormalizedPosition]:
        # Boundary conversion: internal position dict (with extras like
        # entry_price/mark_price/leverage) stays mutable; only the public
        # snapshot is emitted as a frozen NormalizedPosition.
        s = self._state
        if s.position is None:
            return []
        return [NormalizedPosition(
            symbol=s.position["symbol"],
            side=normalize_side(s.position["side"]),
            size=s.position["size"],
        )]

    def get_equity(self, symbols: list[str], quote_currency: str) -> float:
        """
        Return current account equity (balance total + unrealized PnL).

        One bot/exchange per backtest run, so `symbols`/`quote_currency` are
        ignored — equity is always expressed in this run's own quote currency.
        Mirrors the calculation in Reporter.record_equity().
        """
        s = self._state
        unrealized_pnl = 0.0
        if s.position is not None:
            mark  = s.position["mark_price"]
            entry = s.position["entry_price"]
            size  = s.position["size"]
            if s.position["side"] == "long":
                unrealized_pnl = (mark - entry) * size
            else:
                unrealized_pnl = (entry - mark) * size
        return s.balance["total"] + unrealized_pnl

    # ── BaseExchange: orders — internal dict → NormalizedOrder boundary conversion ──

    @staticmethod
    def _to_normalized_order(order: dict[str, Any]) -> NormalizedOrder:
        """
        Convert an OrderBook/FillEngine internal order dict to a NormalizedOrder.

        Internal dicts carry private keys (is_entry, fill_at_bar) which
        OrderBook already strips before returning; this only maps the
        remaining public keys onto the frozen NormalizedOrder shape.
        """
        return NormalizedOrder(
            id=order["id"],
            status=order["status"],
            symbol=order["symbol"],
            side=order["side"],
            type=order["type"],
            size=order["size"],
            price=order.get("price"),
            stop_price=order.get("stop_price"),
        )

    # ── BaseExchange: orders — market ─────────────────────────────────────────

    def place_buy_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.queue_market_order(symbol, "buy", size))

    def place_sell_market_order(self, symbol: str, size: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.queue_market_order(symbol, "sell", size))

    # ── BaseExchange: orders — limit ──────────────────────────────────────────

    def place_buy_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(symbol, "buy", TYPE_LIMIT, size, price=limit_price))

    def place_sell_limit_order(self, symbol: str, size: float, limit_price: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(symbol, "sell", TYPE_LIMIT, size, price=limit_price))

    # ── BaseExchange: orders — stop ───────────────────────────────────────────

    def place_buy_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(symbol, "buy", TYPE_STOP, size, stop_price=stop_price))

    def place_sell_stop_order(self, symbol: str, size: float, stop_price: float) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(symbol, "sell", TYPE_STOP, size, stop_price=stop_price))

    # ── BaseExchange: orders — stop-limit ────────────────────────────────────

    def place_buy_stop_limit_order(
        self, symbol: str, size: float, stop_price: float, limit_price: float
    ) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(
            symbol, "buy", TYPE_STOP_LIMIT, size,
            stop_price=stop_price, price=limit_price,
        ))

    def place_sell_stop_limit_order(
        self, symbol: str, size: float, stop_price: float, limit_price: float
    ) -> NormalizedOrder:
        return self._to_normalized_order(self._book.register_order(
            symbol, "sell", TYPE_STOP_LIMIT, size,
            stop_price=stop_price, price=limit_price,
        ))

    # ── BaseExchange: order management ───────────────────────────────────────

    def cancel_order(self, order_id: str, symbol: str) -> None:
        self._book.cancel_order(order_id)

    def cancel_all_orders(self, symbol: str) -> None:
        self._book.cancel_all_orders()

    def fetch_open_orders(self, symbol: str) -> list[NormalizedOrder]:
        return [self._to_normalized_order(o) for o in self._book.fetch_open_orders()]

    def fetch_order(self, order_id: str, symbol: str) -> NormalizedOrder:
        """Not applicable in backtesting. Orders fill synchronously within advance()."""
        raise NotImplementedError("fetch_order is not supported by BacktestExchange.")
