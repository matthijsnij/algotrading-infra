"""
================================================================================
TRADE LOG MODULE
================================================================================

Trade logger module for live trading.

Contains two classes:

  1. TradeRecord: dataclass defining the canonical schema for a closed trade.
  2. TradeLogger: writes TradeRecord instances to a per-bot CSV file in logs/trades/.

Used inside the bot's _close_trade method.

Notes:
    - The CSV file is created automatically on first write.
    - One file per strategy bot: logs/trades/{strategy_id}.csv

================================================================================
"""

from __future__ import annotations

############ IMPORTS ##############

import csv
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from lighthouse.utils.paths import logs_dir

############ CONSTANTS ##############

_LOG_DIR = logs_dir() / "trades"

############ SCHEMA ##############

@dataclass
class TradeRecord:
    """
    Canonical record of a single closed trade.

    All P&L and fee amounts are in settlement_currency.

    Nullable fields:
        entry_fee, exit_fee     — not all exchanges expose fee data on order responses.
        planned_sl_price,
        planned_tp_price        — not all strategies have to use explicit SL/TP orders.
        funding_paid            — None if instrument/exchange has no funding concept or
                                   payments could not be retrieved; 0.0 if applicable but
                                   no settlements occurred during the trade.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    trade_id:             str               # UUID generated at close
    bot_name:             str               # strategy bot class name
    exchange:             str               # exchange class name
    symbol:               str               # market symbol
    settlement_currency:  str               # currency of all P&L and fee fields

    # ── Entry ─────────────────────────────────────────────────────────────────
    side:                 str               # "long" or "short"
    entry_time:           datetime          # UTC fill time
    entry_price:          float             # fill price (quote per base)
    position_size:        float             # in base currency
    entry_fee:            float | None   # in settlement_currency; None if unavailable
    planned_sl_price:     float | None   # None if strategy does not use SL orders
    planned_tp_price:     float | None   # None if strategy does not use TP orders

    # ── Exit ──────────────────────────────────────────────────────────────────
    exit_time:            datetime          # UTC fill time; falls back to datetime.now() if exchange timestamp missing
    exit_price:           float             # fill price (quote per base); falls back to planned SL/TP price if fill price unavailable
    exit_reason:          str               # "tp" | "sl" | "emergency_halt" | "emergency_missing_orders" | "emergency_sl_cancelled" | "emergency_tp_cancelled"
    exit_fee:             float | None   # in settlement_currency; None if unavailable

    # ── Derived P&L (all in settlement_currency) ──────────────────────────────
    gross_pnl:            float             # (exit_price - entry_price) * size * direction_sign
    funding_paid:         float | None   # signed sum over [entry_time, exit_time]; positive = paid, negative = received; None if not applicable/unavailable
    net_pnl:              float             # gross_pnl - entry_fee - exit_fee - funding_paid
    net_pnl_pct:          float             # net_pnl / (entry_price * position_size)
    duration_seconds:     int               # (exit_time - entry_time).total_seconds()
    outcome:              str               # "win" | "loss" | "breakeven" | "unknown"


############ LOGGER ##############

class TradeLogger:
    """
    Appends TradeRecord instances to a CSV file in the logs/trades/ directory.

    The CSV file is created automatically on the first write, including the
    header row. Subsequent writes append without re-writing the header.

    File path: logs/trades/{strategy_id}.csv

    Methods:
        log(): Append a TradeRecord to the CSV trade log file.
    """

    def __init__(self, strategy_id: str, on_log: (Callable[["TradeRecord"], None]) | None = None) -> None:
        """
        Construct a new TradeLogger for the given strategy bot's strategy ID.

        Args:
            strategy_id: Strategy ID of the bot, used in the trade log filename.
            on_log: optional callback invoked with the record after every successful
                   write. Kept as a plain callable, not a concrete notifications type, so this module
                   has no dependency on the notifications layer.
        """
        # Create the logs directory if it doesn't exist, then set the file path and column names
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._path: Path = _LOG_DIR / f"{strategy_id.lower()}.csv"
        self._fieldnames: list[str] = [f.name for f in fields(TradeRecord)]
        self._on_log = on_log

    def log(self, record: TradeRecord) -> None:
        """
        Append a single TradeRecord as a new row to the CSV file.

        Creates the file and writes the header row on first call. On success,
        invokes the configured on_log callback (if any).

        Args:
            record: TradeRecord instance to log
        """
        # Check if the file already exists to determine whether to write the header row
        write_header = not self._path.exists()

        # Open in append mode so existing rows are never overwritten
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)

            # Write the column header exactly once, on file creation
            if write_header:
                writer.writeheader()

            writer.writerow(asdict(record))

        if self._on_log is not None:
            self._on_log(record)
