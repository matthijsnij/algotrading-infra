"""
================================================================================
PORTFOLIO MODULE
================================================================================
System-wide equity aggregator across exchanges, for Kelly/"system"-scope
position sizing and the drawdown monitor.

Live-only: a backtest has exactly one bot and one exchange, so exchange-local
equity already equals system equity by definition (see BaseBot.get_sizing_equity()
in bots/base_bot.py, and memories/repo/portfolio_system_equity_plan.md for the
full design rationale, including why a wall-clock TTL cache is safe here but
would be wrong against simulated backtest time).

Classes:
    Portfolio: registers exchanges and caches system-wide equity (TTL cache + lock)
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import threading
import time

from lighthouse.exchanges.base import BaseExchange
from lighthouse.utils.logging import get_logger

logger = get_logger(__name__)

############ PORTFOLIO ############

class Portfolio:
    """
    Aggregates equity across registered exchanges into a single reporting
    currency. Cross-currency aggregation is NOT performed — every registered
    exchange is expected to report in the same quote_currency (validated by
    the caller, e.g. runtime.live's DRAWDOWN_QUOTE_CURRENCY mismatch check).

    Attributes:
        quote_currency: single reporting currency all exchanges are summed in
        ttl_seconds:    cache lifetime in seconds before a fresh equity fetch
        _exchanges:     {name: (exchange, symbols, quote_currency)}
        _lock:          serializes cache reads/writes across bot/monitor/summary threads
        _cache:         {name: (equity, fetched_at monotonic timestamp)}
    """

    def __init__(self, quote_currency: str, ttl_seconds: float = 5.0) -> None:
        self.quote_currency = quote_currency
        self.ttl_seconds = ttl_seconds
        self._exchanges: dict[str, tuple[BaseExchange, list[str], str]] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, float]] = {}

    def register_exchange(
        self,
        name: str,
        exchange: BaseExchange,
        symbols: list[str],
        quote_currency: str | None = None,
    ) -> None:
        """
        Register an exchange this portfolio should include in system-wide equity.

        Args:
            name:           unique key for this exchange (e.g. exchange name string)
            exchange:       BaseExchange instance
            symbols:        symbols traded on this exchange, used to mark-to-market held assets
            quote_currency: currency this exchange's balances/prices are expressed in;
                            defaults to the portfolio's own quote_currency
        """
        self._exchanges[name] = (exchange, symbols, quote_currency or self.quote_currency)

    def exchange_equity(self, name: str) -> float:
        """Return one registered exchange's equity in its own quote currency, using the TTL cache."""
        if name not in self._exchanges:
            raise KeyError(f"Portfolio: no exchange registered under '{name}'.")

        with self._lock:
            now = time.monotonic()
            cached = self._cache.get(name)
            if cached is not None and (now - cached[1]) < self.ttl_seconds:
                return cached[0]

            exchange, symbols, quote_currency = self._exchanges[name]
            equity = exchange.get_equity(symbols, quote_currency)
            self._cache[name] = (equity, now)
            return equity

    def total_equity(self) -> float:
        """Return system-wide equity summed across all registered exchanges, in quote_currency."""
        return sum(self.exchange_equity(name) for name in self._exchanges)
