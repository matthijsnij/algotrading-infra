"""
================================================================================
UNIT TESTS FOR TRADE_LOG.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_trade_log.py -v

To run a specific test function:
    pytest tests/test_trade_log.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import csv
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from lighthouse.domain.trade_log import TradeRecord, TradeLogger

################### HELPERS ##########################

def _make_record(**overrides) -> TradeRecord:
    """Return a minimal valid TradeRecord, with any field overridable."""
    defaults = dict(
        trade_id="abc-123",
        bot_name="ExampleBot",
        exchange="Phemex",
        symbol="BTC/USD",
        settlement_currency="USD",
        side="long",
        entry_time=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        entry_price=100.0,
        position_size=1.0,
        entry_fee=0.1,
        planned_sl_price=95.0,
        planned_tp_price=110.0,
        exit_time=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        exit_price=110.0,
        exit_reason="tp",
        exit_fee=0.1,
        gross_pnl=10.0,
        funding_paid=None,
        net_pnl=9.8,
        net_pnl_pct=0.098,
        duration_seconds=3600,
        outcome="win",
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)

################### TESTS ##########################

# ── TradeLogger ────────────────────────────────────────────────────────

# file path; uses bot name lowercased as filename
def test_trade_logger_file_path(tmp_path):
    with patch("lighthouse.domain.trade_log._LOG_DIR", tmp_path):
        logger = TradeLogger("ExampleBot")
    assert logger._path == tmp_path / "examplebot.csv"

# header row; written exactly once on first log call
def test_trade_logger_header_written_once(tmp_path):
    with patch("lighthouse.domain.trade_log._LOG_DIR", tmp_path):
        logger = TradeLogger("ExampleBot")
        logger.log(_make_record())
        logger.log(_make_record())

    with open(logger._path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    # first row is the header, followed by two data rows — no duplicate header
    assert rows[0][0] == "trade_id"
    assert len(rows) == 3

# data row; field values are written correctly
def test_trade_logger_row_values(tmp_path):
    with patch("lighthouse.domain.trade_log._LOG_DIR", tmp_path):
        logger = TradeLogger("ExampleBot")
        logger.log(_make_record(trade_id="xyz-999", outcome="loss", net_pnl=-5.0))

    with open(logger._path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    assert row["trade_id"] == "xyz-999"
    assert row["outcome"] == "loss"
    assert float(row["net_pnl"]) == pytest.approx(-5.0)


# ── on_log callback wiring ──────────────────────────────────────────────────

# no on_log callback configured (default None): no crash, nothing to assert on
def test_trade_logger_no_on_log_is_noop(tmp_path):
    with patch("lighthouse.domain.trade_log._LOG_DIR", tmp_path):
        logger = TradeLogger("ExampleBot")
        logger.log(_make_record())  # must not raise

# on_log configured: it's called with the record after a successful write
def test_trade_logger_calls_on_log_on_successful_write(tmp_path):
    on_log = MagicMock()
    with patch("lighthouse.domain.trade_log._LOG_DIR", tmp_path):
        logger = TradeLogger("ExampleBot", on_log=on_log)
        record = _make_record()
        logger.log(record)

    on_log.assert_called_once_with(record)

