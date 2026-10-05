"""
================================================================================
STATE STORE CLI
================================================================================
Operator-facing entry point for inspecting and clearing persisted bot state
(runtime/state_store.py) from a terminal, without hand-editing the SQLite
file directly.

Usage:
    python -m lighthouse.runtime.state_cli list --db-path state/phemex_live.db
    python -m lighthouse.runtime.state_cli clear --db-path state/phemex_live.db --bot-key crypto_bot:BTC/USDT:USDT

Listing shows every bot's identifier (see state_store.build_bot_key()), mode,
whether the watchdog may restart it, when it was last updated, and its halt
reason — the diagnosis path for "one of my bots hasn't traded since Friday".

Clearing only ever acts on a bot that is currently HALTED, resetting it to a
restartable IDLE state. It does not itself resume anything: the bot's process
must be restarted for the clear to take effect.

Functions:
    list_bots()  : print every bot in the store, or a clear message if the
                  store is missing or empty
    clear_bot()  : clear a single halted bot back to IDLE, or print why not
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import argparse
from pathlib import Path

from lighthouse.runtime.state_store import clear_halted_bot, list_bot_states

############ COMMANDS ############

def list_bots(db_path: Path) -> None:
    """Print every bot in the state store, or a clear message if it's missing/empty."""
    if not db_path.exists():
        print(f"No state store found at '{db_path}'.")
        return

    summaries = list_bot_states(db_path)
    if not summaries:
        print(f"State store '{db_path}' has no bots.")
        return

    for s in summaries:
        print(
            f"  {s.bot_key} | mode={s.mode} restartable={s.restartable} "
            f"updated_at={s.updated_at} halt_reason={s.halt_reason}"
        )


def clear_bot(db_path: Path, bot_key: str) -> None:
    """Clear a single halted bot back to a restartable IDLE state, or explain why not."""
    result = clear_halted_bot(db_path, bot_key)
    print(result.message)

############ ENTRY POINT ############

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and clear persisted bot state without hand-editing SQLite"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="List every bot in the state store")
    list_parser.add_argument("--db-path", type=str, required=True, help="Path to the state store SQLite db")

    clear_parser = subparsers.add_parser("clear", help="Clear a single halted bot back to IDLE")
    clear_parser.add_argument("--db-path", type=str, required=True, help="Path to the state store SQLite db")
    clear_parser.add_argument("--bot-key", type=str, required=True, help="Bot's state-store identifier (name:symbol)")

    args = parser.parse_args()
    db_path = Path(args.db_path)

    if args.action == "list":
        list_bots(db_path)
    else:
        clear_bot(db_path, args.bot_key)
