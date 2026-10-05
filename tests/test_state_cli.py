"""
================================================================================
UNIT TESTS FOR RUNTIME/STATE_CLI.PY

To run all tests in this file:
    pytest tests/test_state_cli.py -v
================================================================================
"""

################### IMPORTS ##########################

from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.runtime.state_cli import clear_bot, list_bots
from lighthouse.runtime.state_store import init_db, save_state

################### TESTS ##########################

# ── list_bots() ──────────────────────────────────────────────────────

# store doesn't exist yet -> clear "no store" message, not a crash
def test_list_bots_missing_store_reports_clearly(tmp_path, capsys):
    db_path = tmp_path / "never_created.db"

    list_bots(db_path)

    out = capsys.readouterr().out
    assert "No state store found" in out

# store exists but has no bots -> clear "empty" message
def test_list_bots_empty_store_reports_clearly(tmp_path, capsys):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    list_bots(db_path)

    out = capsys.readouterr().out
    assert "no bots" in out

# store has bots -> each printed with identifier, mode, restartable, halt reason
def test_list_bots_prints_every_bot(tmp_path, capsys):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode = BotMode.HALTED
    state.halt_restartable = False
    state.halt_reason = "exceeded max restart attempts"
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    list_bots(db_path)

    out = capsys.readouterr().out
    assert "crypto_bot:BTC/USDT:USDT" in out
    assert "mode=halted" in out
    assert "restartable=False" in out
    assert "exceeded max restart attempts" in out


# ── clear_bot() ──────────────────────────────────────────────────────

# clearing a halted bot prints the outcome and the restart reminder
def test_clear_bot_clears_halted_bot(tmp_path, capsys):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode = BotMode.HALTED
    state.halt_restartable = False
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    clear_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    out = capsys.readouterr().out
    assert "cleared" in out
    assert "Restart the bot's process" in out

# refusing a non-halted bot prints the explanation
def test_clear_bot_refuses_non_halted(tmp_path, capsys):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    state = BaseState()
    state.mode = BotMode.IN_POSITION
    save_state(db_path, "crypto_bot:BTC/USDT:USDT", state)

    clear_bot(db_path, "crypto_bot:BTC/USDT:USDT")

    out = capsys.readouterr().out
    assert "not halted" in out

# unknown bot_key prints "not found"
def test_clear_bot_unknown_key(tmp_path, capsys):
    db_path = tmp_path / "lighthouse.db"
    init_db(db_path)

    clear_bot(db_path, "nonexistent:SYM")

    out = capsys.readouterr().out
    assert "No bot found" in out
