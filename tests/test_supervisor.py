"""
================================================================================
UNIT TESTS FOR SUPERVISOR.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_supervisor.py -v

To run a specific test function:
    pytest tests/test_supervisor.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import threading
import time
from unittest.mock import MagicMock, patch

from lighthouse.bots.base_state import BotMode
from lighthouse.runtime.supervisor import (
    BotRestartInfo,
    init_restart_state,
    prepare_bot,
    supervise_bots,
)

################### HELPERS ##########################

HEALTHY_RESET = 3600.0  # default-sized reset window; individual tests override where needed


class FakeState:
    """Minimal state double — real attribute access, no MagicMock magic."""
    def __init__(self, mode=BotMode.IDLE, halt_restartable=True, halt_reason=None):
        self.mode = mode
        self.halt_restartable = halt_restartable
        self.halt_reason = halt_reason


class FakeBot:
    """Minimal bot double with a real .state and a controllable reconcile()."""
    reconcile_halts = False  # set True on the class to simulate a halt-on-reconcile bug

    def __init__(self, exchange=None, config=None, notifier=None, portfolio=None):
        self.exchange = exchange
        self.config = config
        self.notifier = notifier
        self.portfolio = portfolio
        self.state = FakeState(mode=BotMode.IDLE, halt_restartable=True)
        self._stop_event = threading.Event()
        self._db_path = None
        self._bot_key = None
        self.persist_calls = 0

    def attach_persistence(self, db_path, bot_key, persist_state_fn) -> None:
        self._db_path = db_path
        self._bot_key = bot_key
        self._persist_state_fn = persist_state_fn

    def reconcile(self) -> None:
        if self.reconcile_halts:
            self.state.mode = BotMode.HALTED
            self.state.halt_reason = "reconcile re-halted"

    def persist_state(self) -> None:
        self.persist_calls += 1

    def run(self) -> None:
        # Block until _stop_event is set, simulating a real bot's run loop
        self._stop_event.wait()


def _dead_thread() -> threading.Thread:
    """A thread that has already finished — thread.is_alive() is False."""
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def _alive_thread() -> threading.Thread:
    """A thread that blocks forever (until process exit) — thread.is_alive() is True."""
    t = threading.Thread(target=threading.Event().wait, daemon=True)
    t.start()
    return t


def _make_halted_bot(restartable: bool = True) -> FakeBot:
    bot = FakeBot()
    bot.state.mode = BotMode.HALTED
    bot.state.halt_restartable = restartable
    bot.state.halt_reason = "test halt"
    return bot


################### TESTS ##########################

# init_restart_state() returns index-aligned defaults
def test_init_restart_state_returns_n_defaults():
    states = init_restart_state(3)
    assert len(states) == 3
    assert all(isinstance(s, BotRestartInfo) for s in states)
    assert all(s.attempts == 0 and s.healthy_since is None for s in states)


# alive thread: untouched, healthy_since gets recorded, no restart
def test_supervise_alive_thread_untouched():
    bot = FakeBot()
    threads = [_alive_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert bots[0] is bot  # untouched
    assert states[0].attempts == 0
    assert states[0].healthy_since is not None
    logger.critical.assert_not_called()


# dead + HALTED + restartable + under cap -> restart happens
def test_supervise_restarts_halted_restartable_bot():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert bots[0] is not bot  # replaced with a fresh instance
    assert isinstance(bots[0], FakeBot)
    assert threads[0] is not None and threads[0].is_alive()
    assert states[0].attempts == 1
    logger.info.assert_called()


# dead + HALTED + halt_restartable=False -> no restart
def test_supervise_does_not_restart_non_restartable_halt():
    bot = _make_halted_bot(restartable=False)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert bots[0] is bot  # untouched
    assert states[0].attempts == 0
    logger.critical.assert_not_called()


# dead + not HALTED (framework bug) -> no restart, CRITICAL logged
def test_supervise_dead_not_halted_logs_critical_no_restart():
    bot = FakeBot()  # mode stays IDLE — thread died without calling halt()
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert bots[0] is bot  # untouched
    assert states[0].attempts == 0
    logger.critical.assert_called_once()


# backoff window blocks an immediate second restart attempt
def test_supervise_backoff_blocks_immediate_retry():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()
    shutdown_event = threading.Event()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=shutdown_event, logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )
    assert states[0].attempts == 1

    # Simulate the freshly-restarted bot dying again immediately.
    threads[0] = _dead_thread()
    bots[0].state.mode = BotMode.HALTED
    bots[0].state.halt_restartable = True

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=shutdown_event, logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    # Still within the 30s backoff window — no second attempt yet.
    assert states[0].attempts == 1


# cap exhaustion logs CRITICAL exactly once, not on every tick
def test_supervise_cap_exhaustion_logs_once():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}}]
    states = init_restart_state(1)
    states[0].attempts = 3  # already at the cap
    logger = MagicMock()

    for _ in range(3):
        supervise_bots(
            bots, threads, bot_specs, exchange=None, restart_states=states,
            shutdown_event=threading.Event(), logger=logger,
            max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
            healthy_reset_seconds=HEALTHY_RESET,
        )

    assert bots[0] is bot  # never restarted
    logger.critical.assert_called_once()


# cap exhaustion marks the persisted state non-restartable exactly once (ADR 0006 / issue #28)
def test_supervise_cap_exhaustion_marks_state_non_restartable():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}}]
    states = init_restart_state(1)
    states[0].attempts = 3  # already at the cap
    logger = MagicMock()

    for _ in range(3):
        supervise_bots(
            bots, threads, bot_specs, exchange=None, restart_states=states,
            shutdown_event=threading.Event(), logger=logger,
            max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
            healthy_reset_seconds=HEALTHY_RESET,
        )

    assert bots[0] is bot  # never restarted, never re-halted
    assert bot.state.halt_restartable is False
    assert bot.persist_calls == 1  # written once, not on every tick


# cap exhaustion goes through bot.persist_state(), not the halt path (no duplicate halt alert)
def test_supervise_cap_exhaustion_does_not_call_halt():
    bot = _make_halted_bot(restartable=True)
    bot.halt = MagicMock(side_effect=AssertionError("halt() must not be called on exhaustion"))
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}}]
    states = init_restart_state(1)
    states[0].attempts = 3  # already at the cap
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    bot.halt.assert_not_called()


# sustained healthy uptime resets the attempt counter
def test_supervise_healthy_reset_zeroes_attempts():
    bot = FakeBot()
    threads = [_alive_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    states[0].attempts = 2
    states[0].cap_exhausted_logged = True
    states[0].healthy_since = time.monotonic() - 100.0  # already "healthy" for 100s
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=10.0,  # much shorter than the 100s elapsed above
    )

    assert states[0].attempts == 0
    assert states[0].cap_exhausted_logged is False
    assert states[0].healthy_since is None
    logger.info.assert_called()


# reconcile() re-halting the new instance immediately does not start a thread
def test_supervise_reconcile_rehalt_does_not_start_thread():
    class RehaltingBot(FakeBot):
        reconcile_halts = True

    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": RehaltingBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert isinstance(bots[0], RehaltingBot)
    assert bots[0].state.mode == BotMode.HALTED
    assert threads[0] is None  # no thread started
    assert states[0].attempts == 1
    logger.error.assert_called_once()


# ── prepare_bot() ─────────────────────────────────────────────────────────────

BOT_CONFIG = {"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}

# db_path=None: behaves exactly as today — no persistence attributes touched, no load/save
def test_prepare_bot_no_db_path_behaves_like_before():
    with patch("lighthouse.runtime.supervisor.load_state") as mock_load:
        bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path=None)

    mock_load.assert_not_called()
    assert bot._db_path is None
    assert bot._bot_key is None
    assert bot.persist_calls == 0
    assert bot.state.mode == BotMode.IDLE  # fresh default state


# db_path given, no persisted row (load_state returns None): fresh state used, snapshot saved
def test_prepare_bot_with_db_path_no_persisted_row():
    fake_db_path = "fake/lighthouse.db"
    with patch("lighthouse.runtime.supervisor.load_state", return_value=None) as mock_load, \
         patch("lighthouse.runtime.supervisor.save_state") as mock_save:
        bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path=fake_db_path)

    mock_load.assert_called_once_with(fake_db_path, "crypto_bot:BTC/USDT:USDT", FakeState)
    assert bot._db_path == fake_db_path
    assert bot._bot_key == "crypto_bot:BTC/USDT:USDT"
    assert bot._persist_state_fn is mock_save  # injected so Bots never imports runtime.state_store
    assert bot.state.mode == BotMode.IDLE  # fresh default, nothing to load
    assert bot.persist_calls == 1  # post-reconcile snapshot


# db_path given, persisted row exists: loaded state is assigned to the bot
def test_prepare_bot_loads_persisted_state():
    persisted = FakeState(mode=BotMode.WAITING_FILL, halt_restartable=True)
    with patch("lighthouse.runtime.supervisor.load_state", return_value=persisted):
        bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path="fake/lighthouse.db")

    assert bot.state is persisted
    assert bot.state.mode == BotMode.WAITING_FILL


# persisted HALTED + restartable=True: auto-cleared to IDLE before assignment
def test_prepare_bot_auto_clears_restartable_halt():
    persisted = FakeState(mode=BotMode.HALTED, halt_restartable=True, halt_reason="watchdog restart")
    with patch("lighthouse.runtime.supervisor.load_state", return_value=persisted):
        bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path="fake/lighthouse.db")

    assert bot.state.mode == BotMode.IDLE
    assert bot.state.halt_reason is None


# persisted HALTED + restartable=False: stays HALTED (manual clear required)
def test_prepare_bot_keeps_non_restartable_halt():
    persisted = FakeState(mode=BotMode.HALTED, halt_restartable=False, halt_reason="manual stop")
    with patch("lighthouse.runtime.supervisor.load_state", return_value=persisted):
        bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path="fake/lighthouse.db")

    assert bot.state.mode == BotMode.HALTED
    assert bot.state.halt_reason == "manual stop"


# supervise_bots() forwards db_path through to prepare_bot() on restart
def test_supervise_bots_forwards_db_path_to_restart():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": BOT_CONFIG}]
    states = init_restart_state(1)
    logger = MagicMock()
    persisted = FakeState(mode=BotMode.WAITING_FILL, halt_restartable=True)

    with patch("lighthouse.runtime.supervisor.load_state", return_value=persisted) as mock_load:
        supervise_bots(
            bots, threads, bot_specs, exchange=None, restart_states=states,
            shutdown_event=threading.Event(), logger=logger,
            max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
            healthy_reset_seconds=HEALTHY_RESET,
            db_path="fake/lighthouse.db",
        )

    mock_load.assert_called_once_with("fake/lighthouse.db", "crypto_bot:BTC/USDT:USDT", FakeState)
    assert bots[0].state is persisted
    assert bots[0].state.mode == BotMode.WAITING_FILL


# prepare_bot() forwards notifier through to the bot constructor
def test_prepare_bot_forwards_notifier():
    notifier = MagicMock()
    bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path=None, notifier=notifier)

    assert bot.notifier is notifier


# prepare_bot() forwards portfolio through to the bot constructor
def test_prepare_bot_forwards_portfolio():
    portfolio = MagicMock()
    bot = prepare_bot(FakeBot, BOT_CONFIG, exchange=None, db_path=None, portfolio=portfolio)

    assert bot.portfolio is portfolio


# ── notifier wiring in supervise_bots() ──────────────────────────────────────

# successful restart -> notify_restart_success() called with (bot_name, attempt, max_attempts)
def test_supervise_restart_success_notifies():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()
    notifier = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
        notifier=notifier,
    )

    notifier.notify_restart_success.assert_called_once_with("FakeBot", 1, 3)


# restart attempt raises an exception -> notify_restart_failed_attempt() called
def test_supervise_restart_exception_notifies_failed_attempt():
    class ExplodingBot(FakeBot):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": ExplodingBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()
    notifier = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
        notifier=notifier,
    )

    notifier.notify_restart_failed_attempt.assert_called_once()
    args = notifier.notify_restart_failed_attempt.call_args[0]
    assert args[0] == "FakeBot"
    assert args[1] == 1
    assert args[2] == 3
    assert "boom" in args[3]


# reconcile() re-halting the new instance immediately -> notify_restart_failed_attempt() called
def test_supervise_reconcile_rehalt_notifies_failed_attempt():
    class RehaltingBot(FakeBot):
        reconcile_halts = True

    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": RehaltingBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()
    notifier = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
        notifier=notifier,
    )

    notifier.notify_restart_failed_attempt.assert_called_once()
    args = notifier.notify_restart_failed_attempt.call_args[0]
    assert args[0] == "RehaltingBot"
    assert args[1] == 1
    assert args[2] == 3
    assert "reconcile re-halted" in args[3]


# cap exhaustion -> notify_restart_exhausted() called exactly once, not on every tick
def test_supervise_cap_exhaustion_notifies_once():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {"name": "crypto_bot", "symbol": "BTC/USDT:USDT"}}]
    states = init_restart_state(1)
    states[0].attempts = 3  # already at the cap
    logger = MagicMock()
    notifier = MagicMock()

    for _ in range(3):
        supervise_bots(
            bots, threads, bot_specs, exchange=None, restart_states=states,
            shutdown_event=threading.Event(), logger=logger,
            max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
            healthy_reset_seconds=HEALTHY_RESET,
            notifier=notifier,
        )

    notifier.notify_restart_exhausted.assert_called_once_with("FakeBot", "crypto_bot:BTC/USDT:USDT", 3)


# notifier=None (default): no crash on any restart-outcome path
def test_supervise_no_notifier_is_safe():
    bot = _make_halted_bot(restartable=True)
    threads = [_dead_thread()]
    bots = [bot]
    bot_specs = [{"class": FakeBot, "config": {}}]
    states = init_restart_state(1)
    logger = MagicMock()

    supervise_bots(
        bots, threads, bot_specs, exchange=None, restart_states=states,
        shutdown_event=threading.Event(), logger=logger,
        max_restarts=3, backoff_seconds=30.0, backoff_max_seconds=600.0,
        healthy_reset_seconds=HEALTHY_RESET,
    )

    assert states[0].attempts == 1  # restart still happened, no notifier required
