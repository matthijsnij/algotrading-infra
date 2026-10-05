"""
================================================================================
SUPERVISOR MODULE
================================================================================
Watchdog that auto-restarts halted bot threads within the currently running
live-trading process (NOT a full process restart — see runtime/live.py).

Only bots that halted themselves (state.mode == HALTED) AND whose halt was
marked restartable (state.halt_restartable, set via BaseBot.halt(...,
restartable=...)) are eligible. Deliberate system/manual halts (circuit
breaker, "stop <botname>") set restartable=False and are left alone. Threads
that die WITHOUT calling halt() (a framework bug) are CRITICAL-logged only —
never auto-restarted.

Classes:
    BotRestartInfo : per-bot watchdog bookkeeping (attempts, backoff, health)

Functions:
    init_restart_state()  : build a fresh list of BotRestartInfo, index-aligned
                            with the bots/threads lists in runtime.live.main()
    has_pending_restart()  : whether a dead bot is still within its restart budget —
                            used by runtime.live.main() so a bot merely waiting out
                            its backoff window isn't mistaken for a terminal shutdown
    prepare_bot()          : construct a bot, load its persisted state (if db_path is
                            given), and reconcile it against the exchange — shared by
                            runtime.live.main()'s initial startup and this module's
                            watchdog restarts so both pick up persisted state the same way
    supervise_bots()      : called ~once per second from the main health-check
                            loop; restarts eligible halted bots with exponential
                            backoff, a max-attempt cap, and a healthy-uptime
                            reset of the attempt counter. Accepts an optional lock
                            shared with runtime.live's command listener and drawdown
                            monitor so bots/threads mutations don't interleave.
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lighthouse.bots.base_state import BotMode
from lighthouse.runtime.state_store import build_bot_key, load_state, save_state

############ RESTART STATE ############

@dataclass
class BotRestartInfo:
    """
    Per-bot watchdog bookkeeping. One instance per bot, index-aligned with
    bots/threads/bot_specs in runtime.live.main().

    Attributes:
        attempts:             number of restart attempts made since the last healthy reset
        next_retry_at:        time.monotonic() timestamp before which no new attempt is made
        healthy_since:        time.monotonic() timestamp the thread was first seen alive
                              since the last restart; None while dead or already reset
        cap_exhausted_logged: whether the "max restart attempts exceeded" CRITICAL log
                              has already been emitted (logged once, not on every tick)
    """
    attempts: int = 0
    next_retry_at: float = 0.0
    healthy_since: float | None = None
    cap_exhausted_logged: bool = False


def init_restart_state(n: int) -> list:
    """Return a fresh list of n BotRestartInfo instances, all in their default state."""
    return [BotRestartInfo() for _ in range(n)]


def has_pending_restart(bot: Any, info: BotRestartInfo, max_restarts: int) -> bool:
    """
    Whether a dead bot may still be auto-restarted later (halted, restartable,
    and under the attempt cap) — as opposed to terminally dead.
    """
    state = bot.state
    return (
        state is not None
        and state.mode == BotMode.HALTED
        and state.halt_restartable
        and info.attempts < max_restarts
    )


############ BOT PREPARATION ############

def prepare_bot(
    bot_class: Any,
    bot_config: dict,
    exchange: Any,
    db_path: Path | None,
    notifier: Any | None = None,
    portfolio: Any | None = None,
) -> Any:
    """
    Construct a bot, load its persisted state (if any), and reconcile it against the exchange.

    Shared by runtime.live.main()'s initial startup and supervise_bots()'s watchdog
    restarts, so both pick up persisted state (lifetime_tick_count, trade-in-progress
    fields, etc.) the same way rather than always starting from a bare default state.

    If db_path is None, persistence is fully skipped (e.g. backtests never call this
    with a db_path) — behaves exactly like the old inline
    `bot_class(exchange, bot_config); bot.reconcile()` pattern.

    If db_path is given and a persisted row exists for this bot: HALTED rows with
    halt_restartable=True are auto-cleared to IDLE (a fresh process/watchdog restart is
    a strong enough signal to retry) before being assigned to the new bot. HALTED rows
    with halt_restartable=False are loaded as-is (still HALTED) — reconcile() and
    run()'s HALTED guard then keep this bot from starting, requiring a manual DB clear.

    Args:
        bot_class: BaseBot subclass to instantiate
        bot_config: this bot's config dict
        exchange:  shared BaseExchange instance
        db_path:   path to the SQLite state store, or None to disable persistence
        notifier:  optional Notifier for alerting; forwarded to the bot's constructor
        portfolio: optional runtime.portfolio.Portfolio for system-wide equity sizing;
                  forwarded to the bot's constructor (live-only, None in backtests)

    Returns:
        The prepared (and reconciled) bot instance.
    """
    bot = bot_class(exchange, bot_config, notifier=notifier, portfolio=portfolio)

    if db_path is not None:
        bot_key = build_bot_key(bot_config)
        loaded = load_state(db_path, bot_key, type(bot.state))
        if loaded is not None:
            if loaded.mode == BotMode.HALTED and loaded.halt_restartable:
                loaded.mode = BotMode.IDLE
                loaded.halt_reason = None
            bot.state = loaded
        bot.attach_persistence(db_path, bot_key, save_state)

    bot.reconcile()

    if db_path is not None:
        bot.persist_state()  # snapshot immediately post-reconcile

    return bot


############ SUPERVISOR ############

def supervise_bots(
    bots: list,
    threads: list,
    bot_specs: list,
    exchange: Any,
    restart_states: list,
    shutdown_event: threading.Event,
    logger,
    *,
    max_restarts: int,
    backoff_seconds: float,
    backoff_max_seconds: float,
    healthy_reset_seconds: float,
    lock: threading.Lock | None = None,
    db_path: Path | None = None,
    notifier: Any | None = None,
    portfolio: Any | None = None,
) -> None:
    """
    Inspect each bot/thread pair and restart eligible halted bots in place.

    Mutates `bots`, `threads`, and `restart_states` in place (index-aligned) —
    does not return a value. Intended to be called once per health-check tick
    (~1s) from runtime.live.main(), immediately before the "all threads dead"
    check.

    Args:
        bots:            list of current bot instances (mutated in place on restart)
        threads:         list of corresponding threads, or None if never started
                        (mutated in place on restart)
        bot_specs:       list of {"class": ..., "config": ...} dicts used to
                        re-instantiate a bot at the same index
        exchange:        shared BaseExchange instance passed to new bot instances
        restart_states:  list of BotRestartInfo, index-aligned with bots (mutated)
        shutdown_event:  threading.Event; restart attempts are skipped once set
        logger:          logger for restart/health/exhaustion messages
        max_restarts:          restart attempts allowed before giving up (0 disables)
        backoff_seconds:       initial delay before the first restart attempt
        backoff_max_seconds:   cap on the exponential backoff delay
        healthy_reset_seconds: sustained healthy uptime required to reset attempts to 0
        lock:                  optional lock shared with the command listener and drawdown
                              monitor; held while inspecting/mutating bots so a restart can't
                              interleave with a manual/drawdown halt sweep. No locking if None.
        db_path:               path to the SQLite state store passed through to prepare_bot()
                              on each restart, or None to disable persistence.
        notifier:              optional Notifier for alerting; forwarded to prepare_bot() on
                              each restart and used for the three restart-outcome notifications.
        portfolio:             optional runtime.portfolio.Portfolio forwarded to prepare_bot() on
                              each restart (live-only, None in backtests).
    """
    now = time.monotonic()

    with lock or nullcontext():
        for i, (bot, thread, spec, info) in enumerate(zip(bots, threads, bot_specs, restart_states)):
            alive = thread is not None and thread.is_alive()

            if alive:
                if info.healthy_since is None:
                    info.healthy_since = now
                elif now - info.healthy_since >= healthy_reset_seconds:
                    info.attempts = 0
                    info.cap_exhausted_logged = False
                    info.healthy_since = None
                    logger.info(
                        "%s has been healthy for >= %.0fs — restart attempt counter reset.",
                        bot.__class__.__name__, healthy_reset_seconds,
                    )
                continue

            # Thread is dead (or was never started).
            state = bot.state
            if state is None or state.mode != BotMode.HALTED:
                logger.critical("%s thread died unexpectedly.", bot.__class__.__name__)
                continue

            if not state.halt_restartable:
                continue  # deliberate/system halt — watchdog must not fight it

            if shutdown_event.is_set():
                continue  # system is shutting down

            if now < info.next_retry_at:
                continue  # still within the backoff window

            if info.attempts >= max_restarts:
                if not info.cap_exhausted_logged:
                    bot_key = build_bot_key(spec["config"])
                    logger.critical(
                        "%s exceeded max restart attempts (%d) — manual intervention required. state_key=%s",
                        bot.__class__.__name__, max_restarts, bot_key,
                    )
                    if notifier is not None:
                        notifier.notify_restart_exhausted(bot.__class__.__name__, bot_key, max_restarts)
                    # Make the halt terminal (ADR 0006): a later process start must not
                    # resurrect this bot with a full budget. Written directly here, not via
                    # bot.halt() — the bot is already halted/dead, so routing through halt()
                    # would re-run exchange cleanup and fire a duplicate halt alert.
                    state.halt_restartable = False
                    bot.persist_state()
                    info.cap_exhausted_logged = True
                continue

            info.attempts += 1
            try:
                new_bot = prepare_bot(spec["class"], spec["config"], exchange, db_path, notifier=notifier, portfolio=portfolio)
            except Exception as exc:
                logger.error(
                    "%s restart attempt %d/%d raised an exception: %s",
                    bot.__class__.__name__, info.attempts, max_restarts, exc,
                )
                if notifier is not None:
                    notifier.notify_restart_failed_attempt(
                        bot.__class__.__name__, info.attempts, max_restarts, str(exc),
                    )
                info.next_retry_at = now + min(backoff_seconds * 2 ** (info.attempts - 1), backoff_max_seconds)
                continue

            if new_bot.state is not None and new_bot.state.mode == BotMode.HALTED:
                # Reconciliation halted the new instance immediately — don't start a thread.
                bots[i] = new_bot
                threads[i] = None
                logger.error(
                    "%s restart attempt %d/%d: reconciliation re-halted the bot immediately (%s).",
                    new_bot.__class__.__name__, info.attempts, max_restarts, new_bot.state.halt_reason,
                )
                if notifier is not None:
                    notifier.notify_restart_failed_attempt(
                        new_bot.__class__.__name__, info.attempts, max_restarts,
                        f"reconciliation re-halted: {new_bot.state.halt_reason}",
                    )
                info.next_retry_at = now + min(backoff_seconds * 2 ** (info.attempts - 1), backoff_max_seconds)
                continue

            new_thread = threading.Thread(target=new_bot.run, daemon=True)
            new_thread.start()
            bots[i] = new_bot
            threads[i] = new_thread
            info.healthy_since = now
            logger.info(
                "%s restarted (attempt %d/%d).",
                new_bot.__class__.__name__, info.attempts, max_restarts,
            )
            if notifier is not None:
                notifier.notify_restart_success(new_bot.__class__.__name__, info.attempts, max_restarts)
            info.next_retry_at = now + min(backoff_seconds * 2 ** (info.attempts - 1), backoff_max_seconds)
