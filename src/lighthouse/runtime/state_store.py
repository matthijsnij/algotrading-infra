"""
================================================================================
STATE STORE
================================================================================
SQLite-backed persistence for per-bot live state, so that mode, trade-in-
progress fields, and lifetime counters survive a watchdog restart or a full
process restart. Fully opt-in: callers that never pass a db_path (e.g.
backtesting) never touch this module.

One row per bot, keyed by a "<bot name>:<symbol>" string built by
build_bot_key() from the bot's registry name (not its class name — renaming a
bot class must not orphan its persisted state mid-position). State is stored
as a self-describing JSON blob so BaseState/subclass field additions or
removals don't require a schema migration — unknown keys on load are ignored,
missing keys keep the dataclass default.

A bot stuck HALTED with halt_restartable=False needs an operator to clear it
back to a restartable IDLE state before its process is restarted — see
runtime/state_cli.py for the supported way to do this (list/clear), instead
of hand-editing the db file.

Functions:
    build_bot_key()        : build a bot's storage identifier from its config
    default_db_filename()  : build the exchange+mode-namespaced db filename
    init_db()              : create the bot_state table if it doesn't exist yet
    init_run_metadata()    : stamp/verify the db's exchange+mode metadata row
    save_state()           : upsert a bot's current state as JSON
    load_state()           : load and reconstruct a bot's state, or None if not found
    list_bot_states()      : summarize every bot in the store (identifier, mode,
                            restartable, last-updated, halt reason)
    clear_halted_bot()     : clear a single halted bot back to a restartable IDLE state
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import dataclasses
import importlib
import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from lighthouse.bots.base_state import BaseState, BotMode

############ CONSTANTS ############

def default_db_filename(exchange_name: str, testnet: bool) -> str:
    """Build the exchange+mode-namespaced default db filename, e.g. 'phemex_testnet.db'."""
    mode = "testnet" if testnet else "live"
    return f"{exchange_name.lower()}_{mode}.db"


def build_bot_key(bot_config: dict) -> str:
    """
    Build a bot's state-store identifier from its config: "<registry name>:<symbol>".

    Derived from bot_config["name"] (the registry key, e.g. "crypto_bot") rather than
    the bot's class name, so renaming a bot class doesn't orphan its persisted state.
    """
    return f"{bot_config['name']}:{bot_config['symbol']}"

############ SERIALIZATION HELPERS ############

def _serialize_value(v: Any) -> Any:
    """Convert a single state field value into a JSON-safe representation."""
    if isinstance(v, Enum):
        return {"__enum__": f"{type(v).__module__}.{type(v).__qualname__}", "value": v.value}
    if isinstance(v, datetime):
        return {"__datetime__": v.isoformat()}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            "__dataclass__": f"{type(v).__module__}.{type(v).__qualname__}",
            "fields": {f.name: _serialize_value(getattr(v, f.name)) for f in dataclasses.fields(v)},
        }
    return v


def _deserialize_value(v: Any) -> Any:
    """Reverse _serialize_value(): reconstruct enums/datetimes/dataclasses, passthrough otherwise."""
    if isinstance(v, dict) and "__enum__" in v:
        module_name, _, class_name = v["__enum__"].rpartition(".")
        enum_cls = getattr(importlib.import_module(module_name), class_name)
        return enum_cls(v["value"])
    if isinstance(v, dict) and "__datetime__" in v:
        return datetime.fromisoformat(v["__datetime__"])
    if isinstance(v, dict) and "__dataclass__" in v:
        module_name, _, class_name = v["__dataclass__"].rpartition(".")
        dc_cls = getattr(importlib.import_module(module_name), class_name)
        kwargs = {k: _deserialize_value(val) for k, val in v["fields"].items()}
        return dc_cls(**kwargs)
    return v


def _serialize_state(state: BaseState) -> str:
    """Serialize a BaseState (or subclass) instance to a JSON string."""
    fields = dataclasses.fields(state)
    payload = {f.name: _serialize_value(getattr(state, f.name)) for f in fields}
    return json.dumps(payload)


def _deserialize_state(state_json: str, state_cls: type) -> BaseState:
    """Reconstruct a state_cls instance from a JSON string produced by _serialize_state()."""
    obj = state_cls()
    for key, value in json.loads(state_json).items():
        if not hasattr(obj, key):
            continue  # forward/backward compat — ignore unknown or removed fields
        setattr(obj, key, _deserialize_value(value))
    return obj

############ PUBLIC API ############

def init_db(db_path: Path) -> None:
    """
    Create the bot_state table if it doesn't already exist.

    Ensures db_path's parent directory exists. Safe to call multiple times.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_state (
                bot_key    TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def init_run_metadata(db_path: Path, exchange_name: str, testnet: bool) -> None:
    """
    Stamp or verify the db's run_metadata row (single row, exchange_name + testnet).

    Creates the table if missing. If no row exists yet, inserts one for the
    current run. If a row already exists and either field differs from the
    current run's exchange/mode, raises — this catches accidentally pointing
    a live run's paths.state_dir at a testnet db (or vice versa).

    Raises:
        RuntimeError: existing run_metadata row doesn't match exchange_name/testnet
    """
    exchange_name = exchange_name.lower()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                exchange_name TEXT NOT NULL,
                testnet       INTEGER NOT NULL,
                updated_at    TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT exchange_name, testnet FROM run_metadata WHERE id = 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO run_metadata (id, exchange_name, testnet, updated_at) VALUES (1, ?, ?, ?)",
                (exchange_name, int(testnet), datetime.now(timezone.utc).isoformat()),
            )
            return

        stored_exchange, stored_testnet = row[0], bool(row[1])
        if stored_exchange != exchange_name or stored_testnet != testnet:
            stored_mode = "testnet" if stored_testnet else "live"
            requested_mode = "testnet" if testnet else "live"
            raise RuntimeError(
                f"State db '{db_path}' was previously stamped for exchange="
                f"'{stored_exchange}' mode={stored_mode}, but this run requested "
                f"exchange='{exchange_name}' mode={requested_mode}. "
                "Point paths.state_dir elsewhere, or delete this db file if the "
                "switch is intentional."
            )


def save_state(db_path: Path, bot_key: str, state: BaseState) -> None:
    """Upsert the given bot's current state as a JSON blob, keyed by bot_key."""
    state_json = _serialize_state(state)
    updated_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (bot_key, state_json, updated_at) VALUES (?, ?, ?)",
            (bot_key, state_json, updated_at),
        )


def load_state(db_path: Path, bot_key: str, state_cls: type) -> BaseState | None:
    """
    Load and reconstruct the persisted state for bot_key.

    Returns None if db_path doesn't exist yet (first run) or no row is found
    for bot_key (new bot).
    """
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT state_json FROM bot_state WHERE bot_key = ?", (bot_key,)
        ).fetchone()
    if row is None:
        return None
    return _deserialize_state(row[0], state_cls)

############ INSPECTION / RECOVERY ############

@dataclasses.dataclass
class BotStateSummary:
    """One row's worth of at-a-glance status, as reported by list_bot_states()."""
    bot_key:     str
    mode:        str
    restartable: bool
    halt_reason: str | None
    updated_at:  str


@dataclasses.dataclass
class ClearResult:
    """Outcome of clear_halted_bot(): whether the row existed, was cleared, and why."""
    bot_key: str
    found:   bool
    cleared: bool
    message: str


def _mode_value(payload: dict) -> str:
    """Read the raw BotMode string out of a state_json payload without full deserialization."""
    mode_field = payload.get("mode")
    if isinstance(mode_field, dict) and "value" in mode_field:
        return mode_field["value"]
    return str(mode_field)


def list_bot_states(db_path: Path) -> list[BotStateSummary]:
    """
    Summarize every bot currently in the store.

    Returns [] if db_path doesn't exist yet or the store has no rows — callers
    (see runtime/state_cli.py) are expected to report that distinction clearly
    rather than treating an empty list as a failure.
    """
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT bot_key, state_json, updated_at FROM bot_state ORDER BY bot_key"
        ).fetchall()
    return [
        BotStateSummary(
            bot_key=bot_key,
            mode=_mode_value(payload := json.loads(state_json)),
            restartable=bool(payload.get("halt_restartable", True)),
            halt_reason=payload.get("halt_reason"),
            updated_at=updated_at,
        )
        for bot_key, state_json, updated_at in rows
    ]


def clear_halted_bot(db_path: Path, bot_key: str) -> ClearResult:
    """
    Clear a single halted bot back to a restartable IDLE state.

    Refuses (store left untouched) unless the bot's persisted mode is HALTED —
    clearing a running bot's state would be a silent no-op overwritten on its
    next tick, which would only manufacture confusing incidents. Does not
    restart anything: the caller must restart the bot's process for the clear
    to take effect.
    """
    if not db_path.exists():
        return ClearResult(bot_key, found=False, cleared=False, message=f"No state store found at '{db_path}'.")

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT state_json FROM bot_state WHERE bot_key = ?", (bot_key,)
        ).fetchone()
        if row is None:
            return ClearResult(bot_key, found=False, cleared=False, message=f"No bot found with key '{bot_key}'.")

        payload = json.loads(row[0])
        mode = _mode_value(payload)
        if mode != BotMode.HALTED.value:
            return ClearResult(
                bot_key, found=True, cleared=False,
                message=f"Bot '{bot_key}' is not halted (mode={mode}) — refusing to clear.",
            )

        halt_reason = payload.get("halt_reason")
        payload["mode"] = _serialize_value(BotMode.IDLE)
        payload["halt_restartable"] = True
        payload["halt_reason"] = None
        conn.execute(
            "UPDATE bot_state SET state_json = ?, updated_at = ? WHERE bot_key = ?",
            (json.dumps(payload), datetime.now(timezone.utc).isoformat(), bot_key),
        )

    return ClearResult(
        bot_key, found=True, cleared=True,
        message=(
            f"Bot '{bot_key}' cleared: HALTED ({halt_reason}) -> restartable IDLE. "
            "Restart the bot's process for this to take effect."
        ),
    )
