"""
================================================================================
ROLLING WALK-FORWARD HARNESS 
================================================================================

Rolling walk-forward optimization on top of the reusable run_single() core.

The data is carved into a sequence of chronological (train, test) windows.  A
fixed-size train window slides forward by a fixed step; each test window sits
strictly after its train window.  On every window the parameter grid is swept on
the train segment only, the best variant is selected by the objective, and that
frozen parameter set is run exactly once on the test segment.  The optimizer
never sees test data, so out-of-sample (OOS) results are structurally free of
look-ahead / peeking bias.

Each returned bundle is the OOS run_single() bundle for one window, augmented
with the window metadata, the full set of train sweep bundles, and the selected
train bundle, enough for the aggregate reporter to build cross-run
summaries, plateau tables and parameter-stability tables.

wf_config keys:
    train_len    : number of train bars per window
    test_len     : number of test bars per window
    step         : bars to slide the train window forward between windows
    warmup_bars  : leading bars prepended to each segment for indicator history
                   (excluded from that segment's metrics; see run_single)

Functions:
    rolling_splits()   : carve a DataFrame into chronological (train, test, meta) windows
    run_walk_forward() : sweep-on-train / evaluate-once-on-test across all windows
================================================================================
"""

############ IMPORTS ############
import dataclasses
import pandas as pd
from typing import Any, Callable
from lighthouse.runtime.backtest import run_single
from lighthouse.exchanges.backtest.config import MetricsConfig
from lighthouse.exchanges.backtest.reporter import _resample_equity
from lighthouse.optimization.grid import build_grid
from lighthouse.optimization.objective import select_best_by_score

############ SPLITTING ############

def rolling_splits(
    df: pd.DataFrame,
    train_len: int,
    test_len: int,
    step: int,
    warmup_bars: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """
    Carve a DataFrame into rolling walk-forward (train, test, meta) windows.

    A fixed-size train window of 'train_len' base bars slides forward by
    'step' bars.  The test window of 'test_len' base bars starts on the bar
    immediately after the train window ends, so test data is always strictly
    after train data with no overlap between a window's train live segment and
    its test live segment.

    Each slice carries a leading warmup prefix of `warmup_bars` bars so the bot
    can accumulate indicator history before its live segment begins (matching
    run_single()'s warmup contract).  The test warmup prefix is drawn from the
    bars immediately preceding the test live segment (i.e. the tail of the train
    window); this is history the bot is allowed to see and introduces no
    look-ahead.

    Args:
        df          : OHLCV DataFrame with a chronological DatetimeIndex.
        train_len   : number of train bars per window.
        test_len    : number of test bars per window.
        step        : bars to advance the train window between windows (> 0).
        warmup_bars : leading warmup bars prepended to each slice (>= 0).

    Returns:
        A list of (train_slice, test_slice, meta) tuples in chronological order.
        Each slice includes its warmup prefix.  `meta` describes the window:
            window        : zero-based window index
            train_start   : timestamp of first train bar
            train_end     : timestamp of last train bar
            test_start    : timestamp of first test bar
            test_end      : timestamp of last test bar
            train_len     : live train bar count
            test_len      : live test bar count
            warmup_bars   : warmup prefix length

        Returns [] if the data is too short to fit even one full window.

    Raises:
        ValueError : if train_len, test_len or step is <= 0, or warmup_bars < 0.
    """
    # Validate input parameters 
    if train_len <= 0:
        raise ValueError(f"rolling_splits: train_len must be > 0 (got {train_len}).")
    if test_len <= 0:
        raise ValueError(f"rolling_splits: test_len must be > 0 (got {test_len}).")
    if step <= 0:
        raise ValueError(f"rolling_splits: step must be > 0 (got {step}).")
    if warmup_bars < 0:
        raise ValueError(f"rolling_splits: warmup_bars must be >= 0 (got {warmup_bars}).")

    n = len(df)
    splits: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []

    # First train live segment must leave room for its warmup prefix
    first_train_live_start = warmup_bars

    window = 0
    while True:
        # Resolve this window's boundaries (in df row coordinates (integer))
        train_live_start = first_train_live_start + window * step
        test_live_start  = train_live_start + train_len
        test_live_end    = test_live_start + test_len   # exclusive

        # Stop once the test live segment no longer fits inside the data
        if test_live_end > n:
            break

        # Slice train and test, each including its warmup prefix
        train_slice = df.iloc[train_live_start - warmup_bars : train_live_start + train_len]
        test_slice  = df.iloc[test_live_start - warmup_bars : test_live_end]

        # Window metadata (live-segment boundaries, inclusive timestamps)
        meta: dict[str, Any] = {
            "window":      window,
            "train_start": df.index[train_live_start],
            "train_end":   df.index[train_live_start + train_len - 1],
            "test_start":  df.index[test_live_start],
            "test_end":    df.index[test_live_end - 1],
            "train_len":   train_len,
            "test_len":    test_len,
            "warmup_bars": warmup_bars,
        }

        splits.append((train_slice, test_slice, meta))
        window += 1

    return splits


############ HARNESS ############

def _make_variant_run_cfg(
    run_config: dict[str, Any],
    backtest_overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a per-variant copy of run_config with the backtest overrides applied.

    Args:
        run_config         : template run config (must hold a "backtest_config"
                             BacktestConfig dataclass plus the fields run_single
                             reads: symbol, initial_balance, instrument, timeframe).
        backtest_overrides : swept BacktestConfig field overrides for this variant.

    Returns:
        A shallow copy of run_config with "backtest_config" replaced by a new
        BacktestConfig carrying the overrides.
    """
    backtest_config = dataclasses.replace(run_config["backtest_config"], **backtest_overrides)
    return {**run_config, "backtest_config": backtest_config}


def _merge_bot_config(
    template_bot_config: dict[str, Any],
    bot_overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge swept bot overrides onto the template bot config.

    Args:
        template_bot_config : template bot config dict.
        bot_overrides   : swept bot-param overrides for this variant.

    Returns:
        A new bot config dict which is the template with the overrides applied.

    Raises:
        ValueError : if an override key is absent from template_bot_config.
    """
    unknown = set(bot_overrides) - set(template_bot_config)
    if unknown:
        raise ValueError(
            f"run_walk_forward: bot sweep override(s) {sorted(unknown)} are not "
            f"keys of the template_bot_config; available: {sorted(template_bot_config)}."
        )
    return {**template_bot_config, **bot_overrides}


def _pin_periods_per_year(df: pd.DataFrame, run_config: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the annualization factor from the full dataset's calendar span and store it
    in run_config so every walk-forward fold's Sharpe/Sortino uses the same annualization
    (per-fold empirical derivation would jitter with each fold's span, breaking fold
    comparability).

    No-op if the full span is too short to derive a value (fewer than 2 resampled periods).

    Args:
        df         : full OHLCV DataFrame (chronological DatetimeIndex).
        run_config : template run config

    Returns:
        run_config with "_pinned_periods_per_year" key added (or unchanged if span too short).
    """
    metrics_config = run_config.get("metrics_config")

    metrics_timeframe = metrics_config.metrics_timeframe if metrics_config is not None else MetricsConfig().metrics_timeframe

    # Reuse the reporter's resampling math on a constant dummy "equity" series so the
    # derived periods_per_year reflects only the data's timeframe/calendar span.
    dummy_equity_curve = [{"bar": i, "timestamp": ts, "equity": 1.0} for i, ts in enumerate(df.index)]
    _, periods_per_year = _resample_equity(dummy_equity_curve, metrics_timeframe)
    if periods_per_year <= 0:
        return run_config

    return {**run_config, "_pinned_periods_per_year": periods_per_year}


def run_walk_forward(
    df: pd.DataFrame,
    bot_class: type,
    template_bot_config: dict[str, Any],
    run_config: dict[str, Any],
    sweep_spec: dict[str, dict[str, list[Any]]],
    wf_config: dict[str, Any],
    score_fn: Callable[[dict[str, Any]], float],
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run a rolling walk-forward optimization and return one OOS bundle per window.

    For every (train, test) window produced by rolling_splits():
        1. Expand sweep_spec into (bot_overrides, backtest_overrides) grid points.
        2. Run each grid point on the TRAIN segment (optimizer sees train only).
        3. Select the best train bundle by the objective (primary + tie-breakers).
        4. Run the winning frozen parameters exactly ONCE on the TEST segment.

    The optimizer never touches test data, so the returned test metrics are a
    genuine out-of-sample estimate.

    Args:
        df              : full OHLCV DataFrame with a chronological DatetimeIndex.
        bot_class       : bot class to instantiate (subclass of BaseBot).
        template_bot_config : template bot config; swept bot params override its keys.
        run_config         : template run config with the fields run_single reads
                          (symbol, initial_balance, instrument, timeframe,
                          backtest_config); swept backtest params override the
                          BacktestConfig fields. If its "metrics_config" leaves
                          periods_per_year unset, it is auto-pinned to a single
                          value derived from the full `df`'s calendar span so
                          Sharpe/Sortino annualization is comparable across windows
                          (an explicit override is left untouched).
        sweep_spec      : parameter sweep specification (see optimization.grid);
                          must be non-empty (use run_single() directly for a single run).
        wf_config       : walk-forward config (see module docstring): train_len,
                          test_len, step, warmup_bars.
        score_fn        : scoring function; higher is always better. Called on every train-sweep bundle; the
                          bundle with the highest score is selected per window.
                          Use the ready-made scorers from optimization.objective or supply your own.
        verbose         : if True, print per-window progress to stdout.

    Returns:
        A list of OOS result bundles, one per window, in chronological order.
        Each bundle is the TEST-segment run_single() bundle augmented with:
            window     : the rolling_splits() window meta
            train_runs : all train sweep bundles evaluated on this window
            selected   : the winning train bundle chosen by the objective
        Returns [] if the data is too short to fit even one window.

    Raises:
        ValueError : if sweep_spec is empty or None (use run_single() for a single run);
                     or on invalid window sizing (via rolling_splits); or on unknown bot
                     sweep override keys.
    """
    # ── Resolve window sizing ─────────────────────────────────────────────────
    train_len   = wf_config["train_len"]
    test_len    = wf_config["test_len"]
    step        = wf_config["step"]
    warmup_bars = wf_config.get("warmup_bars", 0)

    # Pin periods_per_year across all windows so fold Sharpe/Sortino annualization is comparable
    run_config = _pin_periods_per_year(df, run_config)

    # ── Carve chronological train/test windows ────────────────────────────────
    splits = rolling_splits(df, train_len, test_len, step, warmup_bars)

    if verbose:
        print(f"\nWalk-forward: {len(splits)} window(s) "
              f"(train={train_len}, test={test_len}, step={step}, warmup={warmup_bars})")

    # Validate sweep_spec before expanding, should be non-empty for walk-forward
    if not sweep_spec:
        raise ValueError(
            "run_walk_forward: sweep_spec is empty or None. "
            "A walk-forward requires at least one parameter to sweep; "
            "use run_single() directly for a single run without sweeping."
        )
    grid = build_grid(sweep_spec)

    oos_bundles: list[dict[str, Any]] = []

    # Loop over each (train, test) window, sweep on train, evaluate once on test
    for train_slice, test_slice, meta in splits:
        window = meta["window"]

        if verbose:
            print(f"\n── Window {window}: "
                  f"train {meta['train_start']} → {meta['train_end']} | "
                  f"test {meta['test_start']} → {meta['test_end']} | "
                  f"{len(grid)} variant(s)")

        # ── 1. Sweep the grid on TRAIN only ─────────────────────────────────
        train_bundles: list[dict[str, Any]] = []

        # Loop over grid points
        for bot_overrides, backtest_overrides in grid:

            # Build the per-run bot config and run config with overrides applied
            bot_config      = _merge_bot_config(template_bot_config, bot_overrides)
            variant_run_config = _make_variant_run_cfg(run_config, backtest_overrides)

            # Capture the window metadata for this run
            train_meta = {**meta, "label": f"window{window}:train", "segment": "train"}

            # Run the bot on the training data
            bundle = run_single(
                df           = train_slice,
                bot_class    = bot_class,
                bot_config   = bot_config,
                run_config   = variant_run_config,
                warmup_bars  = warmup_bars,
                segment_meta = train_meta,
                verbose      = False,
            )
            train_bundles.append(bundle)

        # ── 2. Select the best variant by the objective ───────────────────────
        best = select_best_by_score(train_bundles, score_fn)

        # ── 3. Freeze best params and run once on TEST ────────────────────────
        best_bot_config      = best["bot_config"]
        best_backtest_config = best["backtest_config"]
        test_run_config      = {**run_config, "backtest_config": best_backtest_config}

        # Capture test window metadata for this run
        test_meta = {**meta, "label": f"window{window}:test", "segment": "test"}

        # Run the best variant on the test data once
        oos_bundle = run_single(
            df           = test_slice,
            bot_class    = bot_class,
            bot_config   = best_bot_config,
            run_config   = test_run_config,
            warmup_bars  = warmup_bars,
            segment_meta = test_meta,
            verbose      = False,
        )

        # ── Augment with window / sweep provenance for aggregate reporting ────
        oos_bundle["window"]     = meta
        oos_bundle["train_runs"] = train_bundles
        oos_bundle["selected"]   = best

        if verbose:
            train_score = score_fn(best["metrics"])
            oos_trades = oos_bundle["metrics"].get("total_trades")
            oos_sharpe = oos_bundle["metrics"].get("sharpe_ratio")
            print(f"   train score={train_score:.4f} | OOS sharpe={oos_sharpe:.4f}"
                  f" | OOS trades={oos_trades}")

        oos_bundles.append(oos_bundle)

    return oos_bundles
