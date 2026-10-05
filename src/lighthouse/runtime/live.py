"""
================================================================================
LIVE TRADING RUNNER
================================================================================

Entry point for running live trading from a config file.

Usage:
    python -m lighthouse.runtime.live --config examples/live.yaml
    python -m lighthouse.runtime.live --config examples/testnet.yaml

Config file (YAML):
    See examples/live.yaml for the full schema.
    Key sections:
        paths: logs_dir, state_dir (SQLite live-state store; null disables persistence)
        logging: log level
        exchange: name (phemex), market_type (swap/spot/margin)
        risk: max_drawdown_pct, drawdown_quote_currency, check_interval, consecutive_errors_limit,
              max_bot_restarts, restart_backoff_seconds, restart_backoff_max_seconds,
              restart_healthy_reset_seconds
        bots: single bot config (name, symbol, timeframe, strategy params)
        testnet: true routes to the exchange's sandbox using LIGHTHOUSE_<EXCHANGE>_TESTNET_API_KEY/SECRET

Commands (during live trading):
    Ctrl+C          — graceful shutdown of all bots
    list            — print all bots and their thread status
    stop all        — halt all bots
    stop <botname>  — halt specific bot by class name

State persistence:
    If paths.state_dir is set (default "state"), a SQLite db
    (<state_dir>/<exchange>_<mode>.db, e.g. phemex_live.db / phemex_testnet.db,
    see runtime/state_store.py::default_db_filename()) is created at
    startup and each bot's state is saved after every tick and on halt() via
    prepare_bot() / BaseBot.persist_state(). A run_metadata row stamps the db
    with the exchange/mode on first use and is checked on every startup —
    pointing a live run's state_dir at a testnet db (or vice versa) raises
    instead of silently mixing state. On the next startup or watchdog
    restart, prepare_bot() loads it back so counters/trade-in-progress fields
    survive — HALTED+restartable rows are auto-cleared to IDLE, HALTED+
    non-restartable rows stay HALTED (bot skipped at startup until the row is
    manually cleared). Set paths.state_dir: null to disable persistence entirely.

Watchdog:
    A bot thread that dies after calling halt() with restartable=True (the
    default) is auto-restarted in place by runtime.supervisor.supervise_bots(),
    with exponential backoff and a max-attempt cap (see RiskConfig). Deliberate
    halts (manual stop, drawdown circuit breaker) set restartable=False and are
    left alone. This only restarts a bot's thread within this running process —
    it does not survive a full process restart.

Functions:
    command_listener()        : reads commands from stdin
    _check_duplicate_symbols(): raise if two bot specs target the same symbol
    main()                    : runs the full lifecycle of live trading
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

# Fixed imports
import argparse
import logging
import signal
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from lighthouse.utils.logging import get_logger, init_logging
from lighthouse.runtime.monitor import drawdown_monitor
from lighthouse.runtime.summary import periodic_summary
from lighthouse.runtime.supervisor import init_restart_state, supervise_bots, has_pending_restart, prepare_bot
from lighthouse.runtime.state_store import init_db, init_run_metadata, default_db_filename
from lighthouse.runtime.portfolio import Portfolio
from lighthouse.exchanges.factory import create_exchange
from lighthouse.bots.base_state import BotMode
from lighthouse.config.loader import load_live_config
from lighthouse.config.credentials import (
    resolve_credentials,
    check_credentials_not_shared,
    resolve_telegram_credentials,
)
from lighthouse.config.convert import config_to_live_context
from lighthouse.notifications.factory import create_notifier

############ VALIDATION ############

def _check_duplicate_symbols(bot_specs: list[dict]) -> None:
    """
    Raise RuntimeError if two or more bot specs target the same symbol.

    The exchange nets positions per symbol, so running more than one bot on the
    same symbol/account makes per-position PnL and funding attribution
    impossible — this is a hard startup guard, not a soft warning.

    Args:
        bot_specs : list of bot spec dicts, each with a "class" and "config" (symbol)
    """
    specs_by_symbol: dict[str, list[str]] = {}
    for spec in bot_specs:
        specs_by_symbol.setdefault(spec["config"]["symbol"], []).append(spec["class"].__name__)
    duplicate_symbols = {sym: names for sym, names in specs_by_symbol.items() if len(names) > 1}
    if duplicate_symbols:
        raise RuntimeError(
            f"Duplicate symbols across bot specs: {duplicate_symbols}. "
            "The exchange nets positions per symbol, so running more than one bot on "
            "the same symbol/account makes per-position PnL and funding attribution "
            "impossible. Use one bot per symbol per account."
        )

############ COMMAND LISTENER ############

def command_listener(
    bots: list,
    threads: list,
    shutdown_event: threading.Event,
    logger: logging.Logger,
    lock: threading.Lock | None = None,
) -> None:
    """
    Runs in a daemon thread. This function reads commands from stdin (standard input) and dispatches them.

    Args:
        bots: list of bot instances to control
        threads: list of corresponding bot threads to monitor
        shutdown_event: threading.Event to signal main thread to initiate shutdown
        lock: optional lock shared with the watchdog and drawdown monitor, held while
              reading/acting on bots so a restart can't interleave. No locking if None.

    Commands:
        list            — print all bots and their thread alive status.
        stop all        — halt all bots and trigger a graceful system shutdown.
        stop <botname>  — halt a specific bot by class name (case-insensitive).
    """
    while True: # continually listen for commands until shutdown_event is set or stdin is closed
        try:
            cmd = input().strip().lower() # strip and lowercase cmd input
        except EOFError:
            # stdin closed (e.g. piped input exhausted or terminal detached)
            logger.warning("Command listener: stdin closed. Terminal commands are no longer available.")
            break

        if cmd == "list":   # list all bots and their thread alive status
            with lock or nullcontext():
                for bot, thread in zip(bots, threads):
                    status = "alive" if thread is not None and thread.is_alive() else "dead"
                    print(f"  {bot.__class__.__name__}: {status}")

        elif cmd == "stop all":     # halt all bots and trigger graceful shutdown
            logger.info("Command received: stop all.")
            shutdown_event.set()  # signals main thread to run graceful shutdown

        elif cmd.startswith("stop "):   # halt specific bot by class name
            target = cmd[5:].strip()
            with lock or nullcontext():
                matches = [b for b in bots if b.__class__.__name__.lower() == target]
                if matches:
                    for bot in matches:
                        logger.info("Command received: stop %s.", bot.__class__.__name__)
                        bot.halt("Manual stop via command listener.", level="info", restartable=False)
                else:
                    print(f"  No bot found with name '{target}'. Use 'list' to see available bots.")

        else:
            print(f"  Unknown command '{cmd}'. Available: list, stop all, stop <botname>.")

############ MAIN ############

def main(context: dict) -> None:
    """
    Runs the full lifecycle of the live trading system.
    
    Args:
        context: dict from _config_to_live_context() with exchange, bot, risk settings
    """
    # Extract context values
    exchange_name = context["exchange_name"]
    market_type = context["market_type"]
    testnet = context["testnet"]
    risk_settings = context["risk_settings"]
    bot_spec = context["bot_spec"]
    state_dir = context.get("state_dir")  # None disables state persistence
    
    MAX_DRAWDOWN_PCT = risk_settings["max_drawdown_pct"]
    DRAWDOWN_QUOTE_CURRENCY = risk_settings["quote_currency"]
    DRAWDOWN_CHECK_INTERVAL = risk_settings["check_interval"]
    DRAWDOWN_CONSECUTIVE_LIMIT = risk_settings["consecutive_limit"]
    
    init_logging(
        "lighthouse-live",
        level=getattr(logging, context["logging_level"]),
        log_dir=context["logs_dir"],
    )
    logger = get_logger("main")

    check_credentials_not_shared(exchange_name)

    # ── State persistence ─────────────────────────────────────────────────────
    db_path: Path | None = None
    if state_dir is not None:
        db_path = Path(state_dir) / default_db_filename(exchange_name, testnet)
        init_db(db_path)
        init_run_metadata(db_path, exchange_name, testnet)
        logger.info("State persistence enabled | db_path=%s", db_path)
    else:
        logger.warning("State persistence disabled (paths.state_dir is null) — no state survives a restart.")

    if testnet:
        logger.warning("Running in TESTNET mode against %s.", exchange_name)

    # ── Notifier (alerting) ───────────────────────────────────────────────────
    # Credentials are resolved here (Runtime already depends on Config) and passed
    # in already-resolved, so notifications/factory.py has no dependency on Config.
    alerting_settings = context.get("alerting_settings", {})
    telegram_credentials = None
    if alerting_settings.get("enabled") and alerting_settings.get("channel", "telegram") == "telegram":
        telegram_credentials = resolve_telegram_credentials()
    notifier = create_notifier(alerting_settings, credentials=telegram_credentials)

    # ── Create exchange instance(s) ──────────────────────────────────────────────
    credentials = resolve_credentials(exchange_name, testnet=testnet)
    exchange = create_exchange(exchange_name, credentials=credentials, market_type=market_type, testnet=testnet)
    logger.info("Exchange instance created: %s (market_type=%s, testnet=%s)", exchange_name, market_type, testnet)

    # ── Create bot instance(s) ────────────────────────────────────────────────
    # prepare_bot() also loads any persisted state and reconciles the bot against
    # the exchange (same helper the watchdog uses to restart bots — see supervisor.py).
    # Portfolio is live-only (registered here, not in backtests — see runtime/portfolio.py);
    # bots default to "exchange" equity_scope and only consult it when configured "system".
    bot_class = bot_spec["class"]
    bot_config = bot_spec["config"]
    portfolio = Portfolio(quote_currency=DRAWDOWN_QUOTE_CURRENCY)
    portfolio.register_exchange(exchange_name, exchange, [bot_config["symbol"]])
    bot = prepare_bot(bot_class, bot_config, exchange, db_path, notifier=notifier, portfolio=portfolio)
    logger.info("%s instance created and reconciled.", bot_class.__name__)

    # ── Collect all bots ──────────────────────────────────────────────────────
    bots = [bot]
    bot_specs = [bot_spec]

    # ── Validate no two bots trade the same symbol ────────────────────────────
    _check_duplicate_symbols(bot_specs)

    # ── Validate quote currencies match DRAWDOWN_QUOTE_CURRENCY ──────────────
    mismatches = {
        bot.config["symbol"]: bot.config["quote_currency"]
        for bot in bots
        if bot.config["quote_currency"] != DRAWDOWN_QUOTE_CURRENCY
    }
    if mismatches:
        raise RuntimeError(
            f"Quote currency mismatch — DRAWDOWN_QUOTE_CURRENCY is '{DRAWDOWN_QUOTE_CURRENCY}' "
            f"but the following symbols use a different quote currency: {mismatches}. "
            "Equity cannot be summed across different currencies. "
            "Update DRAWDOWN_QUOTE_CURRENCY or align all symbols to the same quote currency."
        )

    # ── Fetch starting equity for drawdown monitor ────────────────────────────
    # (Reconciliation already happened inside prepare_bot(), above.)
    starting_equity = portfolio.total_equity()
    logger.info(
        "Starting portfolio equity for drawdown monitor: %.4f %s",
        starting_equity, DRAWDOWN_QUOTE_CURRENCY,
    )
    if starting_equity <= 0:
        raise RuntimeError(
            f"Starting portfolio equity is {starting_equity} — "
            "drawdown monitor cannot function with zero or negative equity. "
            "Check account balance and DRAWDOWN_QUOTE_CURRENCY setting."
        )

    # ── Shutdown event ────────────────────────────────────────────────────────
    shutdown_event = threading.Event() # signal for main thread to initiate shutdown

    # ── Shared lock ───────────────────────────────────────────────────────────
    # Guards bots/threads so the watchdog's restarts can't interleave with a
    # manual stop or a drawdown-triggered halt sweep from the other threads.
    bots_lock = threading.Lock()

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    def _handle_sigterm(signum, frame) -> None:
        logger.warning("SIGTERM received. Initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ── Notify startup ────────────────────────────────────────────────────────
    notifier.notify_startup(exchange_name, [b.__class__.__name__ for b in bots])

    # ── Start bot threads ─────────────────────────────────────────────────────
    threads = []
    restart_states = init_restart_state(len(bots))
    summary_threads = []
    for bot in bots:
        if bot.state is not None and bot.state.mode == BotMode.HALTED:
            logger.warning(
                "Skipping thread start for %s — halted during reconciliation: %s",
                bot.__class__.__name__, bot.state.halt_reason,
            )
            threads.append(None)  # keep bots/threads index-aligned
            continue
        thread = threading.Thread(target=bot.run, daemon=True) # daemon=True means all threads will exit when main thread exits
        thread.start()
        threads.append(thread)
        logger.info("Thread started for %s.", bot.__class__.__name__)

        summary_interval = bot.config.get("summary_interval_seconds")
        if summary_interval:
            summary_thread = threading.Thread(
                target=periodic_summary,
                args=(bot, notifier, summary_interval, shutdown_event),
                daemon=True,
            )
            summary_thread.start()
            summary_threads.append(summary_thread)
            logger.info("Periodic summary thread started for %s (interval=%ds).", bot.__class__.__name__, summary_interval)

    # ── Start drawdown monitor thread ─────────────────────────────────────────
    equity_fn = portfolio.total_equity
    monitor_thread = threading.Thread(
        target=drawdown_monitor,
        args=(equity_fn, bots, shutdown_event, starting_equity),
        kwargs={
            "max_drawdown_pct":  MAX_DRAWDOWN_PCT,
            "check_interval":    DRAWDOWN_CHECK_INTERVAL,
            "consecutive_limit": DRAWDOWN_CONSECUTIVE_LIMIT,
            "lock":              bots_lock,
            "notifier":          notifier,
        },
        daemon=True,
    )
    monitor_thread.start()
    logger.info("Drawdown monitor thread started.")

    # ── Start command listener thread ─────────────────────────────────────────
    listener_thread = threading.Thread(target=command_listener, args=(bots, threads, shutdown_event, logger, bots_lock), daemon=True) # background thread for command listener
    listener_thread.start()

    logger.info("System running. Press Ctrl+C to stop all, or type 'list' / 'stop <botname>'.")

    # ── Keep main thread alive ────────────────────────────────────────────────
    try:
        while not shutdown_event.is_set():
            # ── Restart eligible halted bots; CRITICAL-log unexpectedly dead ones ──
            supervise_bots(
                bots, threads, bot_specs, exchange, restart_states, shutdown_event, logger,
                max_restarts=risk_settings["max_bot_restarts"],
                backoff_seconds=risk_settings["restart_backoff_seconds"],
                backoff_max_seconds=risk_settings["restart_backoff_max_seconds"],
                healthy_reset_seconds=risk_settings["restart_healthy_reset_seconds"],
                lock=bots_lock,
                db_path=db_path,
                notifier=notifier,
                portfolio=portfolio,
            )

            # ── Check if threads still up (bots merely waiting out a restart backoff don't count) ──
            all_dead = all(
                (thread is None or not thread.is_alive())
                and not has_pending_restart(bot, info, risk_settings["max_bot_restarts"])
                for bot, thread, info in zip(bots, threads, restart_states)
            )

            if all_dead: # no thread alive and no bot left with a pending restart -> shut down.
                shutdown_event.set()
                break

            time.sleep(1) # sleep briefly


        # ── Loop exited ─────────────────────────────────────

        # Either:
        # - shutdown_event was received
        # - all bot threads have stopped

        # Halt any bots that are not already halted, then wait for all threads to finish.
        all_threads_dead = all(t is None or not t.is_alive() for t in threads)
        if all_threads_dead:
            logger.info("All bot threads have stopped. Shutting down system...")
        else:
            logger.info("Stop all command received. Halting all bots...")

        with bots_lock:
            for bot in bots:
                if bot.state is None or bot.state.mode != BotMode.HALTED:
                    bot.halt("Manual stop via command listener.", level="info")
        for thread in threads:
            if thread is not None:
                thread.join()
        logger.info("All bots stopped. System shut down cleanly.")
        notifier.notify_shutdown("all threads stopped" if all_threads_dead else "manual stop")
        notifier.close()

    # ── Shutdown via Ctrl+C (KeyboardInterrupt) ─────────────────────────────────────────────────────
   
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Halting all bots...")
        with bots_lock:
            for bot in bots:
                bot.halt("Manual stop via KeyboardInterrupt.", level="info")

        # Wait for all threads to finish their current tick before exiting.
        for thread in threads:
            if thread is not None:
                thread.join()

        logger.info("All bots stopped. System shut down cleanly.")
        notifier.notify_shutdown("KeyboardInterrupt")
        notifier.close()

    # ── Shutdown: unexpected main thread crash ─────────────────────
    
    # Catches any unhandled exception in the health-check loop.
    # Re-raises the exception so the process exits with a non-zero exit code.
    except Exception as exc:
        logger.critical("Main thread crashed unexpectedly: %s", exc, exc_info=True)
        with bots_lock:
            for bot in bots:
                bot.halt(f"Main thread crash: {exc}")
        for thread in threads:
            if thread is not None:
                thread.join()
        logger.critical("All bots stopped after main thread crash.")
        notifier.notify_shutdown(f"main thread crash: {exc}")
        notifier.close()
        raise

# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run live trading from a YAML config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to live.yaml or testnet.yaml config file"
    )
    args = parser.parse_args()
    
    # Load config and convert to context
    config_path = Path(args.config)
    config = load_live_config(config_path)
    context = config_to_live_context(config, config_path.resolve().parent)
    
    # Run main loop
    main(context)
