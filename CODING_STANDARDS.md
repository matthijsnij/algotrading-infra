# Coding Standards

The rules code in this repo must follow. These are enforceable standards, not
suggestions: a change that breaks one of them is a defect even if it works.

Scope: everything under `src/lighthouse/` and `tests/`. Project context, roadmap and
working style live in `.github/copilot-instructions.md`; this file is only about how
code is written.

## Language and typing

- **Python 3.11+.** `requires-python = ">=3.11"`. Use builtin generics (`dict[str, int]`,
  `list[Foo]`), never `typing.Dict` / `typing.List`.
- **Every function signature is fully type-hinted**, including `-> None` returns.
- **Use PEP 604 unions** (`X | None`), never `Optional[X]` or `Union[X, Y]`.
- **Any module using PEP 604 annotations must start with `from __future__ import
  annotations`**, placed after the module docstring and before all other imports.
  Without it, dataclass fields and parameter annotations raise `TypeError` at runtime on
  some evaluation paths. This is not optional.

## Imports

- **Absolute, rooted at `lighthouse`, for anything crossing a package boundary**:
  `from lighthouse.execution.position import ...`. Never a bare `from execution import ...`.
- **Single-dot sibling imports are allowed *within* a self-contained subpackage**
  (`from .state import BacktestState` inside `exchanges/backtest/`). `exchanges/backtest/`
  and `backtest_data/` are deliberately built this way; do not "fix" them to absolute.
  Parent-traversing relative imports (`from ..domain import ...`) are never allowed.
- **Grouped** into logical blocks (stdlib / third-party / lighthouse), separated by blank
  lines, under the `############ IMPORTS ############` banner.

## File layout

- **Every module opens with a boxed banner docstring** summarising the module and its
  public API:

  ```python
  """
  ================================================================================
  MODULE NAME
  ================================================================================
  Short description.

  Functions:
      foo() : one-line description
  ================================================================================
  """
  ```

  Keep the listed API in sync when you add or remove a public function.

- **Section separators** mark structure inside a file: `############ IMPORTS ############`
  for major blocks, `# ── subsection ──────────` (box-drawing characters) for finer
  grouping.

## Docstrings

- **Google style, always.** `docs/conf.py` runs `sphinx.ext.napoleon` with
  `napoleon_google_docstring = True` and `napoleon_numpy_docstring = False`, so NumPy
  (`Parameters\n----------`) and reST (`:param x:`) styles silently render wrong in the
  API docs. Using them is a defect, not a preference.
- **Every public function, method and class has a docstring** with a one-line summary,
  then the applicable sections in this order:

  ```python
  def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
      """
      Average True Range over `period` bars.

      Args:
          df: OHLCV frame with columns open, high, low, close, volume
          period: Lookback in bars

      Returns:
          Series of ATR values, indexed like `df`

      Raises:
          ValueError: `period` is not positive, or `df` is too short
      """
  ```

- **Document every raised exception type in `Raises:`.** Callers rely on this; the
  exchange and config layers raise a lot and it is not inferable from the signature.
- Private helpers (`_name`) need at least the one-line summary; `Args:`/`Returns:` are
  optional for them.

## Layer boundaries

These are architectural invariants. Breaking one is a design defect, not a style nit.

- **`domain/` is pure**: stateless functions over DataFrames and value types. No I/O, no
  network, no exchange calls, no logging side effects, no clock reads.
- **Bots never touch exchange objects directly.** They call the exchange-agnostic helpers
  in `execution/` (`data.py`, `orders.py`, `position.py`).
- **`BaseBot` holds zero strategy logic.** It owns orchestration only: tick loop, data
  fetch, error handling, lifecycle. Strategy logic belongs in the subclass's `on_tick()`
  / `on_halt()`. Per-bot mutable state subclasses `BaseState`.
- **`config/convert.py` is the sole pydantic-to-engine translation layer**, and stays
  free of exchange-library imports (`ccxt` must not appear there).

## No leakage of example-specific details

Phemex/ccxt and the bots under `bots/examples/` are *illustrations* of the general
interfaces, not the design target.

- Their names, quirks and assumptions must not appear — in code, comments, or docstrings
  — outside `exchanges/live/phemex.py`, `bots/examples/`, and their own tests.
- Shared code (`execution/`, `domain/`, `runtime/`, `config/`, `exchanges/base.py`,
  `exchanges/factory.py`, `bots/base_bot.py`, `bots/base_state.py`, `bots/registry.py`)
  depends only on the abstract `BaseExchange` / `BaseBot` / `BaseState` interfaces.
- If a design question can only be answered by looking at how Phemex or an example bot
  does something, **generalize the interface** — do not copy the example's specifics into
  shared code.

## Extension points

- **Registry + factory, never `if`/`elif` chains.** Pluggable components (exchanges, bots,
  data sources, funding models, notifiers) are added by registering in the relevant
  registry dict. Branching on a name string to pick an implementation is a defect.
- Live exchanges register in `exchanges/factory.py` (`_EXCHANGE_REGISTRY`) and are built
  via `create_exchange(exchange_name, *, credentials, market_type, testnet=False)`, where
  `market_type` is required. `BacktestExchange` is deliberately *not* in the registry; it
  is constructed directly in `runtime/backtest.py`.
- Bots register in `bots/registry.py`.

## Domain value types

- **Sides**: use the `Side` enum and `normalize_side()` from `domain/enums.py`. Never
  compare raw side strings; `"buy"`/`"long"` normalise to `"long"`, case-insensitively.
- **Symbols** have two forms, and they are not interchangeable:
  - trading symbol: `"BTC/USDT:USDT"` (ccxt perpetual notation)
  - datasource symbol: `"BTCUSDT"` (no slash)

  Quote currency is parsed as `symbol.split("/")[1].split(":")[0]`.
- **OHLCV** is always a `pd.DataFrame` with columns `open, high, low, close, volume`.

## Configuration

- **Bot configs are plain dicts**, not dataclasses or pydantic models — see
  `bots/examples/crypto_bot/config.py` and its `run_config`. Keys are documented with
  aligned inline `#` comments. New bot configs follow this pattern.
- The one sanctioned exception is `BacktestConfig`, a dataclass in
  `exchanges/backtest/config.py`.
- **All storage locations come from config.** No hardcoded paths; resolve through
  `utils/paths.py`.

## Logging

- Obtain loggers with `utils.logging.get_logger(name)` after `init_logging(...)`.
- **Use lazy `%s` formatting**: `logger.info("fetched %s bars", n)`. Never f-strings or
  `.format()` inside a logging call.
- Bots attach a per-bot file handler via `add_bot_file_handler`.

## Secrets and credentials

This repo holds keys that can move real money. These rules are not negotiable.

- **All API keys and secrets resolve through `config/credentials.py`** — environment
  variables named `LIGHTHOUSE_<EXCHANGE>_API_KEY` / `_API_SECRET` (and the `_TESTNET_`
  variants). Never hardcode a secret in source, never put one in a YAML config, never
  commit one. Files under `examples/` contain placeholders and env-var *names* only.
- **Never log a secret, at any level including DEBUG.** Do not pass a credentials dict, an
  exchange client holding credentials, or a signed request payload to a logger. Log the
  env-var name or a masked value, never the value itself.
- **Never let a secret reach an exception message or traceback.** When wrapping an exchange
  error, build the message from the operation, symbol and status — not the request body.
- Credentials are passed explicitly as arguments (`create_exchange(..., credentials=...)`,
  `create_notifier(settings, credentials=...)`). Do not read env vars ad hoc from other
  modules.

## Failure handling

- Bots count consecutive tick errors and `halt(...)` once `max_consecutive_errors` is
  reached; halting triggers `on_halt()` (emergency close + cleanup). **Preserve this
  safety flow** when editing bot lifecycle code — it is what stops a broken bot from
  trading.
- Do not swallow exceptions to keep a tick loop alive. Let the consecutive-error counter
  do its job.

## Concurrency

- `halt()` can be called from another thread (command listener, drawdown monitor) while
  the bot's own thread is mid-`tick()`. **Every mutate-then-persist of bot state in
  `tick()` and `halt()` must be wrapped in `with self._state_lock:`** (`BaseBot`). Without
  it, a stale `tick()` snapshot can commit to SQLite *after* `halt()`'s `HALTED` write and
  silently revert it to `IDLE`. Preserve this lock discipline in any change touching
  `tick()`, `halt()`, or state persistence.

## Tests

- **One test file per source module**, named `test_<module>.py`. The layout is flat at the
  top of `tests/` for most layers (`runtime/supervisor.py` → `tests/test_supervisor.py`,
  `execution/positions.py` → `tests/test_positions.py`), with subfolders only for
  `test_bots/`, `test_signals/`, `test_notifications/`, `exchanges/` and `backtest/`.
  Put a new test file where its siblings already live rather than inventing a nested path.
- Name tests `test_<function>_<case>()`.
- **Every test carries a short `#` comment above it** describing the case being covered
  (see `tests/test_positions.py`).
- **Fixtures are deterministic and hand-constructed** so assertions are exact (e.g. a
  constant true range so ATR is exactly 2.0). When adding a fixture to `conftest.py`,
  document the arithmetic that makes the assertion exact, matching the existing ones.
- Mock exchange and network access with `MagicMock` / `patch`. Domain functions are pure
  and are tested directly against DataFrame fixtures — do not mock them.
- **pytest + `unittest.mock` only.** No additional test frameworks.
- **Always run tests with the venv interpreter**: `.\venv\Scripts\python.exe -m pytest tests/ -v` (optionally `-k "substring"`). Never bare `pytest`/`python` — the system interpreter on PATH has no deps installed.
