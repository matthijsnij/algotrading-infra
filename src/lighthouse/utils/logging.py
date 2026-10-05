"""
================================================================================
LOGGER MODULE
================================================================================

Wrapper around Python's built-in logging module providing a two-tier file
logging system:

- System-wide log:  logs/events/algotrading_YYYY-MM-DD.log — captures all output from
                    every bot, core module, and the orchestration layer.
- Per-bot logs:     logs/events/<BotName>_YYYY-MM-DD.log   — one file per bot, containing
                    only that bot's own log lines.

All log files rotate at UTC midnight; each day gets a fresh file with the date
already in the name, no renaming takes place.  Output is also written to the
console.

Functions:
    init_logging():    configure root logger; call once at startup.
    get_logger()       return a named logger for a module.
    add_bot_file_handler(): attach a dedicated file handler to a bot logger.
================================================================================
"""

from __future__ import annotations

################# IMPORTS ##################
import logging
import time
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path

from lighthouse.utils.paths import logs_dir

# ── Configuration ────────────────────────────────────────────────────────

# Set by init_logging(); default level/dir used only if it's called with no overrides.
LOG_DIR: Path | None = None
# Timestamps are always UTC; the trailing 'Z' marks Zulu (UTC) time unambiguously.
LOG_FORMAT = "%(asctime)sZ | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT   = "%Y-%m-%d %H:%M:%S"


def _make_formatter() -> logging.Formatter:
    """Build the shared log formatter, forcing timestamps to UTC."""
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT)
    formatter.converter = time.gmtime  # render %(asctime)s in UTC, not local time
    return formatter

# ── Internal state ─────────────────────────────────────────────────────────────

_initialized = False


class _DatedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A rotating file handler where the active log file always carries today's
    UTC date in its name:  ``<name>_YYYY-MM-DD.log``.

    At UTC midnight the current file is closed and a new one is opened for the
    next date.  No renaming takes place, each day's file is created fresh.

    Methods:
        doRollover() : override parent method to use date in filename
    """

    def __init__(self, log_dir: Path, name: str, **kwargs) -> None:
        """
        Construct a new handler for the given bot name, writing to log_dir.

        Args:
            log_dir: Directory where log files are stored.
            name:    Bot class name used in the log filename.
            **kwargs: Additional keyword arguments passed to the parent TimedRotatingFileHandler
        """
        self._log_dir  = log_dir # set log directory
        self._name = name # set name for filename generation
        today    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d") # current date in UTC
        filename = log_dir / f"{name}_{today}.log" # name_YYYY-MM-DD.log
        super().__init__(str(filename), when="midnight", utc=True, backupCount=0, **kwargs) # initialize parent with custom filename and rotation settings

    def doRollover(self) -> None:
        """
        Override the parent method to close the current file and open a new one.
        """
        if self.stream:
            self.stream.close()
            self.stream = None  
        # Name the new file after the date that starts at self.rolloverAt
        new_date         = datetime.fromtimestamp(self.rolloverAt, tz=timezone.utc).strftime("%Y-%m-%d")
        self.baseFilename = str(self._log_dir / f"{self._name}_{new_date}.log")
        self.stream      = self._open()
        self.rolloverAt += self.interval

# ── Public API ─────────────────────────────────────────────────────────────────

def init_logging(session_name: str, level: int = logging.DEBUG, log_dir: Path | None = None) -> None:
    """
    Configure the root logger once with a system-wide log file.

    Must be called once at process startup (from main.py) before any
    get_logger() calls.  Subsequent calls do nothing.

    The active log file is named ``<session_name>_YYYY-MM-DD.log`` inside LOG_DIR
    and captures output from every component (bots, core modules, orchestration).
    At UTC midnight the handler closes the current file and opens a new one
    named after the next date.

    Args:
        session_name: Label used in the log filename.
        level: root logger level (defaults to DEBUG; pass logging.INFO etc. in production).
        log_dir: directory to write log files to. Defaults to <repo>/logs/events.
    """
    global _initialized, LOG_DIR
    if _initialized:
        return

    LOG_DIR = log_dir if log_dir is not None else logs_dir() / "events"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = _make_formatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler, active file is always <session_name>_YYYY-MM-DD.log
    file_handler = _DatedRotatingFileHandler(LOG_DIR, session_name, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Root logger, all child loggers inherit these handlers
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for the given module.

    Usage (in any module):
        from lighthouse.utils.logging import get_logger
        logger = get_logger(__name__)

    Args:
        name: Name of the logger, typically ``__name__`` of the calling module.
    """
    if not _initialized:
        raise RuntimeError(
            "init_logging() must be called before get_logger(). "
            "Call init_logging(session_name) once at process startup in main.py."
        )
    return logging.getLogger(name)


def add_bot_file_handler(bot_name: str) -> None:
    """
    Attach a dedicated log file handler to the named bot's logger.

    Messages logged by the bot propagate up to the root logger as normal
    (so they appear in the main algotrading log), but are also written to
    ``<bot_name>_YYYY-MM-DD.log``, a file that contains only that bot's output.

    Must be called after init_logging(). Calling it again with the same
    bot_name is a no-op (guards against duplicate handlers).

    Args:
        bot_name: Use the logger name used by the bot (i.e. the bot class name).
    """
    if not _initialized:
        raise RuntimeError(
            "init_logging() must be called before add_bot_file_handler(). "
            "Call init_logging() once at process startup in main.py."
        )

    logger = logging.getLogger(bot_name)

    # Guard against duplicate handlers if called more than once for the same bot
    if any(isinstance(h, _DatedRotatingFileHandler) for h in logger.handlers):
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = _make_formatter()
    handler   = _DatedRotatingFileHandler(LOG_DIR, bot_name, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)