"""
================================================================================
UNIT TESTS FOR STATE_STORE.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_state_store.py -v

To run a specific test function:
    pytest tests/test_state_store.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import sqlite3
from datetime import datetime, timezone

import pytest

from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.bots.examples.crypto_bot.state import CryptoBotState
from lighthouse.domain.enums import Side
from lighthouse.domain.exchange_types import Fee
from lighthouse.runtime.state_store import (
    build_bot_key,
    clear_halted_bot,
    default_db_filename,
    init_db,
    init_run_metadata,
    list_bot_states,
    load_state,
    save_state,
)

################### TESTS ##########################

# ── init_db() ─────────────────────────────────────────────────────────

# creates the db file and bot_state table
def test_init_db_creates_table(tmp_path):
    db_path = tmp_path / "sub" / "lighthouse.db"
    init_db(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bot_state'"
        ).fetchall()
    assert tables

# calling init_db() twice does not raise
def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)
    init_db(db_path)  # should not raise


# ── save_state() / load_state() round-trip ───────────────────────────

# BaseState round-trips every field type (enum, datetime, dataclass, None, primitives)
def test_round_trip_base_state(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode           = BotMode.WAITING_FILL
    state.side           = Side.LONG
    state.lifetime_tick_count      = 42
    state.last_closed_bar = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state.last_exit_bar   = datetime(2023, 12, 31, tzinfo=timezone.utc)
    state.halt_reason     = "some reason"
    state.position_size   = 0.5
    state.entry_limit_order_id = "order-1"
    state.sl_order_id     = "sl-1"
    state.tp_order_id     = None
    state.entry_time      = datetime(2024, 1, 2, tzinfo=timezone.utc)
    state.entry_price     = 100.5
    state.entry_fee       = Fee(cost=0.1, currency="USDT")
    state.lifetime_trade_count     = 3
    state.halt_restartable = False

    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)
    loaded = load_state(db_path, "crypto_bot:BTC/USDT:USDT", BaseState)

    assert loaded is not None
    assert loaded.mode == BotMode.WAITING_FILL
    assert loaded.side == Side.LONG
    assert loaded.lifetime_tick_count == 42
    assert loaded.last_closed_bar == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert loaded.last_exit_bar == datetime(2023, 12, 31, tzinfo=timezone.utc)
    assert loaded.halt_reason == "some reason"
    assert loaded.position_size == 0.5
    assert loaded.entry_limit_order_id == "order-1"
    assert loaded.sl_order_id == "sl-1"
    assert loaded.tp_order_id is None
    assert loaded.entry_time == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert loaded.entry_price == 100.5
    assert loaded.entry_fee == Fee(cost=0.1, currency="USDT")
    assert loaded.lifetime_trade_count == 3
    assert loaded.halt_restartable is False

# CryptoBotState's subclass-specific fields (sl_price, tp_price) round-trip too
def test_round_trip_example_state(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = CryptoBotState()
    state.mode      = BotMode.IN_POSITION
    state.sl_price  = 99.0
    state.tp_price  = 110.0

    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)
    loaded = load_state(db_path, "crypto_bot:BTC/USDT:USDT", CryptoBotState)

    assert isinstance(loaded, CryptoBotState)
    assert loaded.mode == BotMode.IN_POSITION
    assert loaded.sl_price == 99.0
    assert loaded.tp_price == 110.0

# saving the same bot_key twice overwrites the row (INSERT OR REPLACE)
def test_save_state_overwrites_existing_row(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.lifetime_tick_count = 1
    save_state(db_path, "bot:SYM", state)

    state.lifetime_tick_count = 2
    save_state(db_path, "bot:SYM", state)

    loaded = load_state(db_path, "bot:SYM", BaseState)
    assert loaded.lifetime_tick_count == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM bot_state WHERE bot_key = ?", ("bot:SYM",)).fetchall()
    assert rows[0][0] == 1


# ── load_state() missing cases ────────────────────────────────────────

# no row for that bot_key -> None
def test_load_state_missing_key_returns_none(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    assert load_state(db_path, "nonexistent:SYM", BaseState) is None

# db file doesn't exist yet (first run) -> None, no crash
def test_load_state_missing_db_file_returns_none(tmp_path):
    db_path = tmp_path / "never_created.db"

    assert load_state(db_path, "bot:SYM", BaseState) is None


# ── forward/backward compatibility ────────────────────────────────────

# unknown/extra keys in the stored JSON are ignored, not raised
def test_deserialize_ignores_unknown_fields(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO bot_state (bot_key, state_json, updated_at) VALUES (?, ?, ?)",
            ("bot:SYM", '{"lifetime_tick_count": 7, "some_removed_field": "gone"}', "2024-01-01T00:00:00+00:00"),
        )

    loaded = load_state(db_path, "bot:SYM", BaseState)
    assert loaded is not None
    assert loaded.lifetime_tick_count == 7


# ── default_db_filename() ──────────────────────────────────────────────

# filename encodes exchange name (lowercased) and live/testnet mode
def test_default_db_filename_naming():
    assert default_db_filename("phemex", False) == "phemex_live.db"
    assert default_db_filename("phemex", True) == "phemex_testnet.db"
    assert default_db_filename("PHEMEX", True) == "phemex_testnet.db"


# ── init_run_metadata() ────────────────────────────────────────────────

# fresh db: first call creates the table and inserts a row silently
def test_init_run_metadata_fresh_db_inserts_row(tmp_path):
    db_path = tmp_path / "phemex_live.db"
    init_db(db_path)

    init_run_metadata(db_path, "phemex", False)  # should not raise

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT exchange_name, testnet FROM run_metadata WHERE id = 1").fetchone()
    assert row == ("phemex", 0)

# second call with the same exchange/testnet is a no-op
def test_init_run_metadata_matching_call_is_noop(tmp_path):
    db_path = tmp_path / "phemex_live.db"
    init_db(db_path)

    init_run_metadata(db_path, "phemex", False)
    init_run_metadata(db_path, "phemex", False)  # should not raise

# second call with a different exchange raises RuntimeError
def test_init_run_metadata_different_exchange_raises(tmp_path):
    db_path = tmp_path / "shared.db"
    init_db(db_path)
    init_run_metadata(db_path, "phemex", False)

    with pytest.raises(RuntimeError, match="phemex"):
        init_run_metadata(db_path, "binance", False)

# second call with a different testnet flag raises RuntimeError
def test_init_run_metadata_different_mode_raises(tmp_path):
    db_path = tmp_path / "shared.db"
    init_db(db_path)
    init_run_metadata(db_path, "phemex", False)

    with pytest.raises(RuntimeError, match="live"):
        init_run_metadata(db_path, "phemex", True)


# ── build_bot_key() ─────────────────────────────────────────────────────

# derived from the registry name (bot_config["name"]), not any class name
def test_build_bot_key_uses_registry_name():
    assert build_bot_key({"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}) == "crypto_bot:BTC/USDT:USDT"


# ── list_bot_states() ────────────────────────────────────────────────────

# db file doesn't exist yet -> [], no crash
def test_list_bot_states_missing_db_returns_empty(tmp_path):
    db_path = tmp_path / "never_created.db"
    assert list_bot_states(db_path) == []

# db exists but has no rows -> []
def test_list_bot_states_empty_store_returns_empty(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)
    assert list_bot_states(db_path) == []

# summarizes every row: identifier, mode, restartable, halt reason, last-updated
def test_list_bot_states_summarizes_rows(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    idle_state = BaseState()
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", idle_state)

    halted_state = BaseState()
    halted_state.mode = BotMode.HALTED
    halted_state.halt_restartable = False
    halted_state.halt_reason = "exceeded max restart attempts"
    save_state(db_path, "stock_bot:AAPL", halted_state)

    summaries = {s.bot_key: s for s in list_bot_states(db_path)}
    assert set(summaries) == {"crypto_bot:BTC/USDT:USDT", "stock_bot:AAPL"}

    idle_summary = summaries["crypto_bot:BTC/USDT:USDT"]
    assert idle_summary.mode == "idle"
    assert idle_summary.restartable is True
    assert idle_summary.halt_reason is None

    halted_summary = summaries["stock_bot:AAPL"]
    assert halted_summary.mode == "halted"
    assert halted_summary.restartable is False
    assert halted_summary.halt_reason == "exceeded max restart attempts"


# ── clear_halted_bot() ───────────────────────────────────────────────────

# db file doesn't exist yet -> not found, clear message
def test_clear_halted_bot_missing_db(tmp_path):
    db_path = tmp_path / "never_created.db"
    result = clear_halted_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    assert result.found is False
    assert result.cleared is False
    assert "No state store found" in result.message

# unknown bot_key -> not found, store untouched
def test_clear_halted_bot_unknown_key(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    result = clear_halted_bot(db_path, "nonexistent:SYM")

    assert result.found is False
    assert result.cleared is False
    assert "No bot found" in result.message

# HALTED + non-restartable -> cleared back to a restartable IDLE state
def test_clear_halted_bot_clears_halted_row(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode = BotMode.HALTED
    state.halt_restartable = False
    state.halt_reason = "exceeded max restart attempts"
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    result = clear_halted_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    assert result.found is True
    assert result.cleared is True
    assert "Restart the bot's process" in result.message

    loaded = load_state(db_path, "crypto_bot:BTC/USDT:USDT", BaseState)
    assert loaded.mode == BotMode.IDLE
    assert loaded.halt_restartable is True
    assert loaded.halt_reason is None

# a non-halted bot is refused, store left untouched
def test_clear_halted_bot_refuses_non_halted(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode = BotMode.IN_POSITION
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    result = clear_halted_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    assert result.found is True
    assert result.cleared is False
    assert "not halted" in result.message

    loaded = load_state(db_path, "crypto_bot:BTC/USDT:USDT", BaseState)
    assert loaded.mode == BotMode.IN_POSITION  # untouched

# clearing only touches mode/halt_restartable/halt_reason — every other field survives (no data loss)
def test_clear_halted_bot_preserves_unrelated_fields(tmp_path):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = CryptoBotState()
    state.mode = BotMode.HALTED
    state.halt_restartable = False
    state.lifetime_tick_count = 42
    state.sl_price = 99.0
    state.tp_price = 110.0
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    clear_halted_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    loaded = load_state(db_path, "crypto_bot:BTC/USDT:USDT", CryptoBotState)
    assert loaded.mode == BotMode.IDLE
    assert loaded.lifetime_tick_count == 42
    assert loaded.sl_price == 99.0
    assert loaded.tp_price == 110.0
