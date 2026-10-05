"""
================================================================================
UNIT TESTS FOR LIVE.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_live.py -v

To run a specific test function:
    pytest tests/test_live.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import threading
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.base_state import BaseState, BotMode
from lighthouse.runtime.live import command_listener, main, _check_duplicate_symbols
from lighthouse.runtime.state_store import default_db_filename, init_db, init_run_metadata, load_state

################### HELPERS ##########################

class StubBot:
    """Minimal bot double — just enough for command_listener to call halt() on."""
    def __init__(self):
        self.halt = MagicMock()

################### TESTS ##########################

# distinct symbols across bot specs: no error
def test_check_duplicate_symbols_no_duplicates():
    bot_specs = [
        {"class": type("BotA", (), {}), "config": {"symbol": "BTC/USDT:USDT"}},
        {"class": type("BotB", (), {}), "config": {"symbol": "ETH/USDT:USDT"}},
    ]
    _check_duplicate_symbols(bot_specs)  # should not raise

# two bot specs targeting the same symbol raises RuntimeError
def test_check_duplicate_symbols_raises_on_duplicate():
    bot_specs = [
        {"class": type("BotA", (), {}), "config": {"symbol": "BTC/USDT:USDT"}},
        {"class": type("BotB", (), {}), "config": {"symbol": "BTC/USDT:USDT"}},
    ]
    with pytest.raises(RuntimeError, match="BTC/USDT:USDT"):
        _check_duplicate_symbols(bot_specs)

# "stop <botname>" halts the matching bot with restartable=False (deliberate manual stop)
def test_command_listener_stop_botname_passes_restartable_false():
    bot = StubBot()
    threads = [MagicMock()]
    shutdown_event = threading.Event()
    logger = MagicMock()

    with patch("builtins.input", side_effect=["stop stubbot", EOFError]):
        command_listener([bot], threads, shutdown_event, logger)

    bot.halt.assert_called_once()
    assert bot.halt.call_args.kwargs["restartable"] is False

# "list" does not raise when a bot's thread slot is None (halted-during-reconcile alignment fix)
def test_command_listener_list_handles_none_thread():
    bot = StubBot()
    threads = [None]
    shutdown_event = threading.Event()
    logger = MagicMock()

    with patch("builtins.input", side_effect=["list", EOFError]):
        command_listener([bot], threads, shutdown_event, logger)  # should not raise


# ── main(): end-to-end state persistence wiring ──────────────────────────────

@dataclass
class TrivialState(BaseState):
    """BaseState + one extra counter, incremented once per on_tick() call."""
    custom_counter: int = field(default=0)


class TrivialBot(BaseBot):
    """Minimal concrete bot — RUNTIME WIRING is under test here, not strategy logic."""
    def __init__(self, exchange, config, notifier=None, portfolio=None) -> None:
        super().__init__(exchange, config, notifier=notifier, portfolio=portfolio)
        self.state = TrivialState()

    def on_tick(self, df: pd.DataFrame) -> None:
        self.state.custom_counter += 1


BOT_KEY = "trivial_bot:BTC/USDT:USDT"


def _make_context(tmp_path: Path) -> dict:
    """Build a context dict by hand (skip real YAML/pydantic loading)."""
    return {
        "exchange_name": "phemex",
        "market_type": "swap",
        "testnet": False,
        "risk_settings": {
            "max_drawdown_pct": 5.0,
            "quote_currency": "USDT",
            "check_interval": 60,
            "consecutive_limit": 5,
            "max_bot_restarts": 3,
            "restart_backoff_seconds": 30.0,
            "restart_backoff_max_seconds": 600.0,
            "restart_healthy_reset_seconds": 3600.0,
        },
        "bot_spec": {
            "class": TrivialBot,
            "config": {
                "name": "trivial_bot",
                "symbol": "BTC/USDT:USDT",
                "quote_currency": "USDT",
                "timeframe": "1h",
                "limit": 10,
                "scan_interval": 60,
                "position_interval": 10,
                "max_consecutive_errors": 5,
            },
        },
        "logging_level": "INFO",
        "logs_dir": tmp_path / "logs",
        "state_dir": tmp_path / "state",
    }


def _run_main_with_stop_all(context: dict) -> None:
    """Run main() to completion, patching every exchange-touching call."""
    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    with patch("lighthouse.runtime.live.create_exchange", return_value=exchange), \
         patch("lighthouse.runtime.live.resolve_credentials", return_value={}), \
         patch("lighthouse.runtime.live.check_credentials_not_shared"), \
         patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch("lighthouse.runtime.live.time.sleep"), \
         patch("builtins.input", side_effect=["stop all", EOFError]):
        main(context)


# After a full main() run + "stop all" shutdown, the state db exists and reflects HALTED
def test_main_persists_state_on_shutdown(tmp_path):
    context = _make_context(tmp_path)

    _run_main_with_stop_all(context)

    db_path = context["state_dir"] / default_db_filename("phemex", False)
    assert db_path.exists()

    loaded = load_state(db_path, BOT_KEY, TrivialState)
    assert loaded is not None
    assert loaded.mode == BotMode.HALTED
    assert loaded.custom_counter >= 1  # run() always executes tick() at least once


# A second main() call reusing the same state_dir picks up the persisted state rather
# than starting fresh — the closest thing to a full-process-restart test without a
# real subprocess. HALTED+restartable=True (the "stop all" shutdown default) is
# auto-cleared to IDLE, so the second run's bot resumes ticking and its counter keeps
# climbing from where the first run left off, rather than resetting to 0.
def test_main_second_run_resumes_from_persisted_state(tmp_path):
    context = _make_context(tmp_path)

    _run_main_with_stop_all(context)
    db_path = context["state_dir"] / default_db_filename("phemex", False)
    first_run_state = load_state(db_path, BOT_KEY, TrivialState)
    assert first_run_state.mode == BotMode.HALTED
    assert first_run_state.halt_restartable is True  # "stop all" uses halt()'s default

    _run_main_with_stop_all(context)
    second_run_state = load_state(db_path, BOT_KEY, TrivialState)

    # Only provable if the loaded state (not a fresh default) was actually used:
    # a fresh TrivialState would restart custom_counter at 0 -> 1, never exceeding run 1.
    assert second_run_state.custom_counter > first_run_state.custom_counter


# main() propagates check_credentials_not_shared's RuntimeError before touching the exchange
def test_main_raises_when_credentials_shared(tmp_path):
    context = _make_context(tmp_path)

    with patch("lighthouse.runtime.live.check_credentials_not_shared", side_effect=RuntimeError("shared creds")), \
         patch("lighthouse.runtime.live.create_exchange") as mock_create_exchange, \
         patch("lighthouse.runtime.live.resolve_credentials", return_value={}):
        with pytest.raises(RuntimeError, match="shared creds"):
            main(context)

    mock_create_exchange.assert_not_called()


# main() aborts startup when the state dir's run_metadata doesn't match this run's exchange/mode
def test_main_raises_on_run_metadata_mismatch(tmp_path):
    context = _make_context(tmp_path)
    db_path = context["state_dir"] / default_db_filename("phemex", False)
    init_db(db_path)
    init_run_metadata(db_path, "binance", False)  # stamped for a different exchange

    with patch("lighthouse.runtime.live.check_credentials_not_shared"), \
         patch("lighthouse.runtime.live.create_exchange") as mock_create_exchange, \
         patch("lighthouse.runtime.live.resolve_credentials", return_value={}):
        with pytest.raises(RuntimeError, match="binance"):
            main(context)

    mock_create_exchange.assert_not_called()


# ── main(): notifier wiring ───────────────────────────────────────────────────

# create_notifier() is built from context["alerting_settings"]; notify_startup/shutdown/close
# are called at the expected points in a normal "stop all" run.
def test_main_notifies_startup_and_shutdown(tmp_path):
    context = _make_context(tmp_path)
    context["alerting_settings"] = {"enabled": False, "channel": "telegram", "categories": {}}
    notifier = MagicMock()

    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    with patch("lighthouse.runtime.live.create_exchange", return_value=exchange), \
         patch("lighthouse.runtime.live.resolve_credentials", return_value={}), \
         patch("lighthouse.runtime.live.check_credentials_not_shared"), \
         patch("lighthouse.runtime.live.create_notifier", return_value=notifier) as mock_create_notifier, \
         patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch("lighthouse.runtime.live.time.sleep"), \
         patch("builtins.input", side_effect=["stop all", EOFError]):
        main(context)

    mock_create_notifier.assert_called_once_with(context["alerting_settings"], credentials=None)
    notifier.notify_startup.assert_called_once_with("phemex", ["TrivialBot"])
    notifier.notify_shutdown.assert_called_once()
    notifier.close.assert_called_once()


# alerting enabled with telegram: credentials are resolved in main() (not inside
# create_notifier(), which has no dependency on the config layer) and passed through.
def test_main_resolves_telegram_credentials_when_alerting_enabled(tmp_path):
    context = _make_context(tmp_path)
    context["alerting_settings"] = {"enabled": True, "channel": "telegram", "categories": {}}
    notifier = MagicMock()
    fake_credentials = {"bot_token": "tok", "chat_id": "chat"}

    df = pd.DataFrame(
        {"open": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    )
    exchange = MagicMock()
    exchange.get_equity.return_value = 1000.0
    with patch("lighthouse.runtime.live.create_exchange", return_value=exchange), \
         patch("lighthouse.runtime.live.resolve_credentials", return_value={}), \
         patch("lighthouse.runtime.live.check_credentials_not_shared"), \
         patch("lighthouse.runtime.live.resolve_telegram_credentials", return_value=fake_credentials) as mock_resolve, \
         patch("lighthouse.runtime.live.create_notifier", return_value=notifier) as mock_create_notifier, \
         patch("lighthouse.bots.base_bot.get_open_positions", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_open_orders", return_value=[]), \
         patch("lighthouse.bots.base_bot.get_ohlcv_df", return_value=df), \
         patch("lighthouse.bots.base_bot.get_bid_ask", return_value={"bid": 1, "ask": 2}), \
         patch("lighthouse.runtime.live.time.sleep"), \
         patch("builtins.input", side_effect=["stop all", EOFError]):
        main(context)

    mock_resolve.assert_called_once()
    mock_create_notifier.assert_called_once_with(context["alerting_settings"], credentials=fake_credentials)

