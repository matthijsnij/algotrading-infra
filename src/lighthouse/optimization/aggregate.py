"""
================================================================================
AGGREGATE REPORTING
================================================================================

Persist walk-forward results to disk and print a cross-run console summary.

Consumes the list of OOS bundles returned by optimization.walk_forward.run_walk_forward()
(one bundle per walk-forward window). Each bundle is the run_single() result for the
test segment, augmented with:
    window      : rolling_splits() meta (window index, train/test timestamps, etc.)
    train_runs  : all train-sweep run_single() bundles evaluated on that window
    selected    : the single winning train bundle chosen by the objective

The OOS equity curve is additionally chained across windows into a continuous
"rebased_equity" series (see _rebased_equity_with_drawdown()) since each window's
raw equity independently resets to ~initial_balance; this powers both the
oos_equity_curve.csv columns and the console drawdown scalars.

Functions:
    write_results()  : save per-run data + summary CSVs/trade-logs to an output directory
    print_summary()  : print a formatted cross-run summary to stdout
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import dataclasses
import math
import os
from typing import Any
import pandas as pd

from lighthouse.domain.timeframe import Timeframe
from lighthouse.exchanges.backtest.reporter import _resample_equity

############ CONSTANTS ############

# Below these thresholds Sharpe/Sortino/CAGR/Calmar are statistically unreliable
_MIN_RELIABLE_RETURN_PERIODS = 30   # resampled return observations for Sharpe/Sortino
_MIN_RELIABLE_YEARS          = 0.5  # wall-clock span for CAGR/Calmar

############ HELPERS ############

# ── BacktestConfig serialisation ─────────────────────────────────────────────

def _backtest_cfg_to_dict(config: Any) -> dict[str, Any]:
    """
    Flatten a BacktestConfig dataclass to a plain {field: value} dict.

    Non-primitive field values (e.g. a FundingModel dataclass) are converted
    to their repr() string so they can appear in CSV columns without error.

    Args:
        config : BacktestConfig dataclass instance.

    Returns:
        A flat dict representation of the BacktestConfig dataclass.
    """
    # Check if the config is already a plain dict (e.g. from a test bundle); if so, return it directly
    if not (dataclasses.is_dataclass(config) and not isinstance(config, type)):
        return dict(config)

    result: dict[str, Any] = {}

    # Loop over the dataclass fields and extract their values
    for field in dataclasses.fields(config):
        val = getattr(config, field.name)
        if isinstance(val, (int, float, str, bool, type(None))):
            result[field.name] = val
        else:
            result[field.name] = repr(val)
    return result


# ── Swept-param discovery ─────────────────────────────────────────────────────

def _identify_swept_params(train_runs: list[dict[str, Any]]) -> dict[str, str]:
    """
    Return the parameters that actually vary across train_runs.

    Compares bot_config dicts and BacktestConfig fields across every run in the
    list; a parameter is considered swept if it takes more than one distinct value.

    Args:
        train_runs : list of run_single() bundles from one walk-forward window.

    Returns:
        An ordered dict containing the swept parameters, mapping "bot.<param>" or "backtest.<param>" to its section
        string ("bot" or "backtest").  Empty if no parameters vary (e.g. single run).
    """
    if not train_runs:
        return {}

    swept: dict[str, str] = {}

    # Filter swept bot params
    first_bot = train_runs[0]["bot_config"]
    for key in first_bot:
        values = {run["bot_config"].get(key) for run in train_runs} # set
        if len(values) > 1:
            swept[f"bot.{key}"] = "bot"

    # Filter swept backtest params
    first_bt = _backtest_cfg_to_dict(train_runs[0]["backtest_config"])
    for key in first_bt:
        values = set()
        for run in train_runs:
            bt_dict = _backtest_cfg_to_dict(run["backtest_config"])
            values.add(bt_dict.get(key))
        if len(values) > 1:
            swept[f"backtest.{key}"] = "backtest"

    return swept


def _get_selected_param_val(
    bundle: dict[str, Any],
    full_key: str,
    section: str,
) -> Any:
    """
    Read a single swept param value from the selected (winning) train bundle.

    Args:
        bundle   : OOS bundle (has a "selected" sub-bundle).
        full_key : "bot.<param>" or "backtest.<param>".
        section  : "bot" or "backtest".

    Returns:
        The param value from the selected bundle's bot_config / backtest_config.
    """
    # Get the param name without the "bot." or "backtest." prefix
    param = full_key.split(".", 1)[1]

    # Get the selected bundle from the OOS bundle
    sel = bundle["selected"]
    if section == "bot":
        return sel["bot_config"].get(param)

    # Transform BacktestConfig to a dict then get the param value
    bt_dict = _backtest_cfg_to_dict(sel["backtest_config"])
    return bt_dict.get(param)


# ── Continuous OOS equity chaining ────────────────────────────────────────────

def _rebased_equity_with_drawdown(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chain per-window OOS equity curves into one continuous rebased series.

    Each window's raw "equity" curve independently resets to ~initial_balance
    (every window is its own run_single() call), which would create artificial
    jumps if the raw curves were simply concatenated. This instead carries
    forward the RETURN RATIO (not absolute P&L) at each bar: the series is
    seeded with the first window's own first raw equity value, and every
    subsequent bar multiplies the running "carry" value by that bar's raw
    return ratio. Within any single window rebased_equity == equity exactly;
    only window boundaries chain the previous window's last value forward
    (that boundary bar itself contributes zero return, since no calendar gap
    is modeled as P&L). A running-max drawdown_pct is then computed on the
    now-continuous series.

    Assumes bundles are chronological and OOS test windows are non-overlapping
    (same assumption already made by _print_deployment_recommendation); prints
    a warning (does not raise) if consecutive windows' test ranges overlap.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().

    Returns:
        List of {window, bar, timestamp, equity, rebased_equity, drawdown_pct}
        rows, one per raw equity-curve bar across all windows. Empty if no
        bundle has a non-empty equity_curve.
    """
    carry: float | None = None
    rows: list[dict[str, Any]] = []
    prev_test_end = None

    # Loop over windows in order, chaining each window's raw curve onto the running carry value
    for bundle in bundles:
        meta = bundle["window"]

        if prev_test_end is not None and meta["test_start"] <= prev_test_end:
            print(
                f"_rebased_equity_with_drawdown: window {meta['window']} OOS test range "
                f"overlaps the previous window — rebased_equity chaining assumes "
                f"non-overlapping OOS windows; treat drawdown/rebased_equity with caution."
            )
        prev_test_end = meta["test_end"]

        curve = bundle["equity_curve"]
        if not curve:
            continue

        prev_raw = None
        for entry in curve:
            raw_eq = entry["equity"]
            if prev_raw is None:
                if carry is None:
                    carry = raw_eq  # seed with the very first window's own first raw value
                # else: carry already holds the continuation value from the prior window's
                # last bar — this window's first bar contributes zero return by construction
            else:
                carry = carry * (raw_eq / prev_raw)
            rows.append({
                "window":         meta["window"],
                "bar":            entry["bar"],
                "timestamp":      entry["timestamp"],
                "equity":         raw_eq,
                "rebased_equity": carry,
            })
            prev_raw = raw_eq

    # Global running-max drawdown on the now-continuous rebased_equity series
    peak = None
    for row in rows:
        peak = row["rebased_equity"] if peak is None else max(peak, row["rebased_equity"])
        row["drawdown_pct"] = ((row["rebased_equity"] - peak) / peak * 100.0) if peak > 0 else 0.0

    return rows


def _drawdown_scalar_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract scalar drawdown stats from rebased-equity rows.

    Locates the single deepest drawdown episode (the row with the most
    negative drawdown_pct) and walks backward to its most recent preceding
    peak (drawdown_pct == 0) to measure the episode's duration.

    Args:
        rows : rows produced by _rebased_equity_with_drawdown(); must already
               carry a "drawdown_pct" key.

    Returns:
        {
            "max_drawdown_pct":     positive magnitude of the deepest drawdown (%),
                                    0.0 if rows is empty,
            "max_dd_duration":      pd.Timedelta between the peak and the trough
                                    of the deepest episode; pd.Timedelta(0) if empty,
            "current_drawdown_pct": rows[-1]["drawdown_pct"] if still underwater
                                    (< -1e-9) at the last OOS bar, else None,
        }
    """
    if not rows:
        return {
            "max_drawdown_pct":     0.0,
            "max_dd_duration":      pd.Timedelta(0),
            "current_drawdown_pct": None,
        }

    # Locate the deepest drawdown (most negative drawdown_pct)
    trough_idx = min(range(len(rows)), key=lambda i: rows[i]["drawdown_pct"])
    trough     = rows[trough_idx]

    # Walk backward from the trough to the most recent peak (drawdown_pct == 0)
    peak_idx = trough_idx
    for i in range(trough_idx, -1, -1):
        if rows[i]["drawdown_pct"] == 0.0:
            peak_idx = i
            break
    peak = rows[peak_idx]

    return {
        "max_drawdown_pct":     abs(trough["drawdown_pct"]),
        "max_dd_duration":      trough["timestamp"] - peak["timestamp"],
        "current_drawdown_pct": rows[-1]["drawdown_pct"] if rows[-1]["drawdown_pct"] < -1e-9 else None,
    }


# ── Pooled OOS return distribution ────────────────────────────────────────────

def _pooled_oos_returns(bundles: list[dict[str, Any]]) -> pd.Series:
    """
    Concatenate each window's resampled OOS returns into one pooled series.

    Mirrors Reporter.get_results()'s own resampling exactly (per-window,
    metrics_timeframe-aligned returns via the same _resample_equity()) so
    pooled skew/kurtosis are consistent with the existing Sharpe/Sortino
    methodology.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().

    Returns:
        pd.Series of pooled simple returns (index reset); empty float series
        if no window has both a metrics_timeframe and a non-empty equity_curve.
    """
    parts: list[pd.Series] = []

    # Loop over windows and resample each window's own equity curve, using its own metrics_timeframe
    for bundle in bundles:
        tf = bundle["metrics"].get("metrics_timeframe")
        if not tf or not bundle["equity_curve"]:
            continue
        returns, _ = _resample_equity(bundle["equity_curve"], tf)
        if len(returns):
            parts.append(returns)

    return pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)


# ── Walk-forward efficiency ───────────────────────────────────────────────────

def _compute_wfe(bundles: list[dict[str, Any]]) -> float | None:
    """
    Walk-forward efficiency: mean(OOS Sharpe) / mean(selected train Sharpe).

    Matches the overfitting-flag convention already used in
    _print_deployment_recommendation (OOS vs. train Sharpe comparison).

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().

    Returns:
        The WFE ratio, or None if either side has no usable Sharpe values, or
        the mean train Sharpe is ~0 (degenerate ratio).
    """
    oos_sharpes = [
        b["metrics"]["sharpe_ratio"] for b in bundles if "sharpe_ratio" in b["metrics"]
    ]
    train_sharpes = [
        b["selected"]["metrics"]["sharpe_ratio"]
        for b in bundles
        if "selected" in b and "sharpe_ratio" in b["selected"]["metrics"]
    ]
    if not oos_sharpes or not train_sharpes:
        return None

    mean_oos   = sum(oos_sharpes) / len(oos_sharpes)
    mean_train = sum(train_sharpes) / len(train_sharpes)
    if abs(mean_train) < 1e-9:
        return None  # undefined / degenerate train Sharpe

    return mean_oos / mean_train


# ── Funding vs. price P&L split ────────────────────────────────────────────────

def _funding_share_summary(bundles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Sum price P&L vs. funding P&L vs. net P&L across all OOS windows.

    Auto-detects whether funding was active at all (any bundle with a nonzero
    total_funding_paid); returns None to hide the block entirely for spot /
    non-perp instruments where funding never applies.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().

    Returns:
        {"price_pnl", "funding_pnl", "net_pnl", "share_pct"} summed/derived
        across all windows, or None if funding was never active. "share_pct" is
        the funding P&L's share of net P&L (None if net_pnl ~ 0).
    """
    if all(abs(b["metrics"].get("total_funding_paid", 0.0)) < 1e-9 for b in bundles):
        return None

    total_funding_paid = sum(b["metrics"].get("total_funding_paid", 0.0) for b in bundles)
    total_price_pnl    = sum(b["metrics"]["gross_pnl"] for b in bundles)
    total_net_pnl      = sum(b["metrics"]["net_pnl"] for b in bundles)
    funding_pnl        = -total_funding_paid  # sign flip: positive = net funding income

    share_pct = (funding_pnl / total_net_pnl * 100.0) if abs(total_net_pnl) > 1e-9 else None

    return {
        "price_pnl":   total_price_pnl,
        "funding_pnl": funding_pnl,
        "net_pnl":     total_net_pnl,
        "share_pct":   share_pct,
    }


############ WRITE RESULTS ############

def write_results(
    bundles: list[dict[str, Any]],
    out_dir: str,
    equity_curve_timeframe: Timeframe | None = Timeframe.parse("1d"),
) -> None:
    """
    Persist walk-forward results to out_dir.

    Output layout:
        <out_dir>/
            train_runs_metrics.csv         - every train-sweep run across all windows
            best_params_per_window.csv     — one row per window: selected params + OOS metrics
            oos_equity_curve.csv           — OOS equity chained continuously across windows
                                             via rebased_equity + a globally-computed
                                             drawdown_pct (resampled to equity_curve_timeframe
                                             unless None); raw "equity" is kept unchanged
                                             (resets per window) for backward compatibility
            param_stability.csv            — how often each swept param value was selected
            trade_log_per_window/
                window_0000_<TESTSTART>_<TESTEND>.csv   — OOS trade log for window 0
                window_0001_<TESTSTART>_<TESTEND>.csv   — OOS trade log for window 1
                ...
            plateau_tables/
                window_0000_<TESTSTART>_<TESTEND>_bot_<param>.csv        — metric vs bot param, window 0
                window_0000_<TESTSTART>_<TESTEND>_backtest_<param>.csv   — metric vs backtest param, window 0
                ...

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.
        equity_curve_timeframe : Timeframe applied per window to resample the OOS
            equity curve before stitching (e.g. Timeframe.parse("4h")); None writes
            the raw per-bar curve (window, bar, timestamp, equity, rebased_equity,
            drawdown_pct).
    """

    os.makedirs(out_dir, exist_ok=True)
    _write_train_runs_metrics(bundles, out_dir)
    _write_best_params_per_window(bundles, out_dir)
    _write_oos_equity_curve(bundles, out_dir, equity_curve_timeframe)
    _write_param_stability(bundles, out_dir)
    _write_trade_logs(bundles, out_dir)
    _write_plateau_tables(bundles, out_dir)


# ── train_runs_metrics.csv ─────────────────────────────────────────────────────

def _write_train_runs_metrics(bundles: list[dict[str, Any]], out_dir: str) -> None:
    """
    Write train_runs_metrics.csv: every train-sweep run across every window. 

    Only includes train runs. The output contains full parameter sets 
    for both bot_config and backtest_config, plus the metrics for that run.

    Columns: window | segment | train_start | train_end | test_start | test_end |
             bot.<param>... | backtest.<param>... | <metric>...

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.

    Raises:
        KeyError : if a bundle is missing its required "train_runs" key.
    """
    rows: list[dict[str, Any]] = []

    # Loop over the result bundles for all windows
    for bundle in bundles:

        # Get window index and boundary timestamps for this bundle
        meta = bundle["window"]
        window_idx = meta["window"]

        # Require the train_runs key; an empty list is allowed (window contributes no rows)
        if "train_runs" not in bundle:
            raise KeyError("_write_train_runs_metrics: bundle missing required 'train_runs' key.")

        # Loop over all train-sweep runs for this window
        for run in bundle["train_runs"]:
            row: dict[str, Any] = {
                "window":      window_idx,
                "segment":     "train",
                "train_start": meta["train_start"],
                "train_end":   meta["train_end"],
                "test_start":  meta["test_start"],
                "test_end":    meta["test_end"],
            }

            # Add bot_config and backtest_config fields, prefixed with "bot." and "backtest."
            for k, v in run["bot_config"].items():
                row[f"bot.{k}"] = v
            for k, v in _backtest_cfg_to_dict(run["backtest_config"]).items():
                row[f"backtest.{k}"] = v

            # Add metrics for the run
            row.update(run["metrics"])
            rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "train_runs_metrics.csv"), index=False)


# ── best_params_per_window.csv ────────────────────────────────────────────────

def _write_best_params_per_window(bundles: list[dict[str, Any]], out_dir: str) -> None:
    """
    Write best_params_per_window.csv: one row per window.

    Columns: window | train_start | train_end | test_start | test_end |
             best_bot.<param>... | best_backtest.<param>... | oos.<metric>...
    
    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.
    """
    rows: list[dict[str, Any]] = []

    # Loop over the result bundles for all windows
    for bundle in bundles:
        meta = bundle["window"]
        row: dict[str, Any] = {
            "window":      meta["window"],
            "train_start": meta["train_start"],
            "train_end":   meta["train_end"],
            "test_start":  meta["test_start"],
            "test_end":    meta["test_end"],
        }

        # Get the winning train run bundle
        sel = bundle["selected"]

        # Add the selected bot_config and backtest_config fields, prefixed with "best_bot." and "best_backtest."
        for k, v in sel["bot_config"].items():
            row[f"best_bot.{k}"] = v
        for k, v in _backtest_cfg_to_dict(sel["backtest_config"]).items():
            row[f"best_backtest.{k}"] = v

        # Add the OOS metrics for this window, prefixed with "oos."
        for k, v in bundle["metrics"].items():
            row[f"oos.{k}"] = v
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "best_params_per_window.csv"), index=False
    )


# ── oos_equity_curve.csv ──────────────────────────────────────────────────────

def _write_oos_equity_curve(
    bundles: list[dict[str, Any]],
    out_dir: str,
    equity_curve_timeframe: Timeframe | None,
) -> None:
    """
    Write oos_equity_curve.csv: OOS equity chained continuously across all windows.

    Snapshots come from _rebased_equity_with_drawdown(), which chains the raw
    per-window "equity" curves (each window independently resets to
    ~initial_balance) into a continuous "rebased_equity" series plus a
    globally-computed "drawdown_pct".

    When equity_curve_timeframe is None the raw per-bar curve is written with
    columns: window | bar | timestamp | equity | rebased_equity | drawdown_pct.

    Otherwise each window's rows are resampled to the given pandas rule (last
    value per period) before stitching, dropping the bar index. Resampling is
    done per window so periods never straddle a window boundary. Columns:
    window | timestamp | equity | rebased_equity | drawdown_pct.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.
        equity_curve_timeframe : Timeframe resample rule; None writes raw.

    Raises:
        KeyError : if a bundle is missing its required "equity_curve" key.
    """
    # Require the equity_curve key on every bundle before chaining (empty lists are allowed)
    for bundle in bundles:
        if "equity_curve" not in bundle:
            raise KeyError("_write_oos_equity_curve: bundle missing required 'equity_curve' key.")

    rebased_rows = _rebased_equity_with_drawdown(bundles)

    if equity_curve_timeframe is None:
        # Raw per-bar curve: rebased rows already carry window/bar/timestamp/equity/
        # rebased_equity/drawdown_pct in the right column order
        pd.DataFrame(rebased_rows).to_csv(
            os.path.join(out_dir, "oos_equity_curve.csv"), index=False
        )
        return

    columns = ["window", "timestamp", "equity", "rebased_equity", "drawdown_pct"]
    if not rebased_rows:
        pd.DataFrame(columns=columns).to_csv(
            os.path.join(out_dir, "oos_equity_curve.csv"), index=False
        )
        return

    rows: list[dict[str, Any]] = []
    df_all = pd.DataFrame(rebased_rows)

    # Resample each window's rows independently so periods never straddle a window boundary
    for window_idx, group in df_all.groupby("window", sort=False):
        g = group.set_index(pd.DatetimeIndex(group["timestamp"]))
        resampled = g.resample(equity_curve_timeframe.pandas_freq)[
            ["equity", "rebased_equity", "drawdown_pct"]
        ].last().dropna()
        for ts, vals in resampled.iterrows():
            rows.append({
                "window":         window_idx,
                "timestamp":      ts,
                "equity":         vals["equity"],
                "rebased_equity": vals["rebased_equity"],
                "drawdown_pct":   vals["drawdown_pct"],
            })

    pd.DataFrame(rows, columns=columns).to_csv(
        os.path.join(out_dir, "oos_equity_curve.csv"), index=False
    )


# ── param_stability.csv ───────────────────────────────────────────────────────

def _write_param_stability(bundles: list[dict[str, Any]], out_dir: str) -> None:
    """
    Write param_stability.csv: how often each swept param value was selected.

    Across all windows, discovers every parameter that was swept (varied across
    train runs) and counts how many windows selected each distinct value, giving
    a view of how stable the optimizer's parameter choices are over time.

    Columns: param | value | count | total_windows | frequency

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.

    Raises:
        KeyError : if a bundle is missing its required "train_runs" key.
    """
    columns = ["param", "value", "count", "total_windows", "frequency"]
    total_windows = len(bundles)

    # Discover every parameter that was swept in any window
    all_swept: dict[str, str] = {}

    # Loop over all windows and collect the swept parameters from each window's train runs
    for bundle in bundles:
        if "train_runs" not in bundle:
            raise KeyError("_write_param_stability: bundle missing required 'train_runs' key.")
        all_swept.update(_identify_swept_params(bundle["train_runs"]))

    # Count how often each (param, value) was selected across windows
    rows: list[dict[str, Any]] = []
    for full_key, section in all_swept.items():
        counts: dict[Any, int] = {}
        for bundle in bundles:
            val = _get_selected_param_val(bundle, full_key, section)
            counts[val] = counts.get(val, 0) + 1

        for value, count in counts.items():
            rows.append({
                "param":         full_key,
                "value":         value,
                "count":         count,
                "total_windows": total_windows,
                "frequency":     count / total_windows if total_windows else 0.0,
            })

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(out_dir, "param_stability.csv"), index=False)


# ── trade_log_per_window/ ─────────────────────────────────────────────────────

def _write_trade_logs(bundles: list[dict[str, Any]], out_dir: str) -> None:
    """
    Write one CSV per OOS window under trade_log_per_window/.

    Each file name is window_<NNNN>_<TESTSTART>_<TESTEND>.csv (window index
    zero-padded to 4 digits; test start/end dates as YYYYMMDD).

    Only the OOS (test-segment) trade log is written; train-sweep trade logs are
    intentionally excluded as they are intermediate optimization artifacts.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.

    Raises:
        KeyError : if a bundle is missing its required "trade_log" key.
    """

    # Create the trade_log_per_window/ subdirectory if it doesn't exist, to keep the trade logs organized and separate from the other output files.
    trade_dir = os.path.join(out_dir, "trade_log_per_window")
    os.makedirs(trade_dir, exist_ok=True)

    # Loop over the result bundles for all windows and write each trade log to a separate CSV file
    for bundle in bundles:
        meta = bundle["window"]
        window_idx = meta["window"]

        # Require the trade_log key; an empty list is allowed (window contributes no file)
        if "trade_log" not in bundle:
            raise KeyError("_write_trade_logs: bundle missing required 'trade_log' key.")
        trades = bundle["trade_log"]

        # Skip writing a CSV if there are no trades for this window, and print a message to inform the user.
        if not trades:
            print(f"_write_trade_logs: window {window_idx} has no trades; skipping CSV.")
            continue

        ts = meta["test_start"]
        te = meta["test_end"]
        path = os.path.join(
            trade_dir,
            f"window_{window_idx:04d}_{ts:%Y%m%d}_{te:%Y%m%d}.csv",
        )
        pd.DataFrame(trades).to_csv(path, index=False)


# ── plateau_tables/ ───────────────────────────────────────────────────────────

def _write_plateau_tables(bundles: list[dict[str, Any]], out_dir: str) -> None:
    """
    Write per-window plateau tables: one CSV per swept param per window.

    Each table contains the average metrics across all train-sweep runs that
    share the same value for that parameter, giving a view of how metrics change
    as the parameter varies.

    File names: window_<NNNN>_<TESTSTART>_<TESTEND>_<section>_<param>.csv
    (test start/end dates as YYYYMMDD).

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
        out_dir : root output directory; created if absent.

    Raises:
        KeyError : if a bundle is missing its required "train_runs" key.
    """

    # Create the plateau_tables/ subdirectory if it doesn't exist, to keep the plateau tables organized and separate from the other output files.
    plateau_dir = os.path.join(out_dir, "plateau_tables")
    os.makedirs(plateau_dir, exist_ok=True)

    # Loop over the result bundles for all windows and write each plateau table to a separate CSV file
    for bundle in bundles:

        meta = bundle["window"]
        window_idx = meta["window"]

        # Require the train_runs key; an empty list is allowed (window contributes no tables)
        if "train_runs" not in bundle:
            raise KeyError("_write_plateau_tables: bundle missing required 'train_runs' key.")
        train_runs = bundle["train_runs"]
        if not train_runs:
            continue

        # Test-segment date span for the file name
        ts = meta["test_start"]
        te = meta["test_end"]

        # Get the swept parameters for this window (those that vary across train runs)
        swept = _identify_swept_params(train_runs)

        # Loop over each swept parameter and compute the average metrics for each distinct value of that parameter
        for full_key, section in swept.items():

            # Get the parameter name without the "bot." or "backtest." prefix
            param = full_key.split(".", 1)[1]

            # Loop over all train runs and collect the metrics for each distinct value of the parameter
            rows: list[dict[str, Any]] = []
            for run in train_runs:

                # Grab the parameter value from the appropriate section (bot_config or backtest_config)
                if section == "bot":
                    val = run["bot_config"].get(param)
                else:
                    bt_dict = _backtest_cfg_to_dict(run["backtest_config"])
                    val = bt_dict.get(param)

                # Add a row for this run, containing the parameter value and the metrics
                row: dict[str, Any] = {"param_value": val}
                row.update(run["metrics"])
                rows.append(row)

            df = pd.DataFrame(rows)
            # Only keep numeric columns (metrics) for averaging, excluding the param_value column
            numeric_cols = [
                c for c in df.columns
                if c != "param_value" and pd.api.types.is_numeric_dtype(df[c])
            ]
            # Group by the parameter value and compute the mean of each metric, resetting the index to get a flat DataFrame
            summary = df.groupby("param_value")[numeric_cols].mean().reset_index()

            # Sanitize the parameter name for the file name by replacing slashes with underscores, and construct the output file path
            safe_param = param.replace("/", "_").replace("\\", "_")
            path = os.path.join(
                plateau_dir,
                f"window_{window_idx:04d}_{ts:%Y%m%d}_{te:%Y%m%d}_{section}_{safe_param}.csv",
            )
            summary.to_csv(path, index=False)


############ PRINT SUMMARY ############

def print_summary(bundles: list[dict[str, Any]]) -> None:
    """
    Print a cross-run console summary of walk-forward OOS results.

    Sections printed:
        1. Aggregate OOS statistics (mean ± std across windows), plus
           Walk-Forward Efficiency, pooled-return skew/kurtosis, drawdown
           scalars from the continuous rebased OOS equity curve, and a
           funding-vs-price P&L split block (only when funding was active)
        2. Per-window table: window, dates, selected params, OOS metrics
        3. Param stability: how often each value was selected across windows
        4. Deployment recommendation: most recent window's best params + red flags

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
    """
    if not bundles:
        print("print_summary: no bundles to summarise.")
        return

    _print_aggregate_oos(bundles)
    _print_per_window_table(bundles)
    _print_param_stability(bundles)
    _print_deployment_recommendation(bundles)


# ── Aggregate OOS ──────────────────────────────────────────────────────────

def _print_aggregate_oos(bundles: list[dict[str, Any]]) -> None:
    """
    Print mean ± std for key OOS metrics across all windows, plus:
        - Walk-Forward Efficiency (mean OOS Sharpe / mean selected train Sharpe)
        - Pooled OOS return skewness + excess kurtosis (min 4 pooled observations)
        - Drawdown scalars (max %, max duration, current if still underwater)
          computed on the continuous rebased OOS equity curve
        - A funding-vs-price P&L split block, only printed when funding was
          actually active this run (auto-detected; hidden otherwise)

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
    """
    metrics_of_interest = [
        ("sharpe_ratio",        "OOS Sharpe"),
        ("sortino_ratio",       "OOS Sortino"),
        ("calmar_ratio",        "OOS Calmar"),
        ("return_pct",          "OOS Return (%)"),
        ("cagr",                "OOS CAGR"),
        ("max_drawdown_pct",    "OOS Max DD (%)"),
        ("total_trades",        "OOS Trade Count"),
        ("win_rate",            "OOS Win Rate"),
        ("profit_factor",       "OOS Profit Factor"),
        ("expectancy",          "OOS Expectancy"),
    ]

    width = 64
    print("\n" + "=" * width)
    print("  WALK-FORWARD AGGREGATE (OOS)")
    print("=" * width)
    print(f"  Windows : {len(bundles)}")
    print()

    # Loop over all metrics of interest and compute mean ± std across windows
    for key, label in metrics_of_interest:
        vals = [
            b["metrics"][key]
            for b in bundles
            if key in b["metrics"] and b["metrics"][key] is not None
            and not (isinstance(b["metrics"][key], float) and math.isnan(b["metrics"][key])) # skip NaN values
            and not (isinstance(b["metrics"][key], float) and math.isinf(b["metrics"][key])) # skip Inf values
        ]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
        std_val  = variance ** 0.5
        print(f"  {label:<26}  mean={mean_val:>10.4f}  std={std_val:>9.4f}")

    print()

    # ── Walk-forward efficiency (Sharpe-based) ──────────────────────────────
    wfe = _compute_wfe(bundles)
    if wfe is not None:
        print(f"  {'Walk-Forward Efficiency (Sharpe)':<26}  {wfe:>10.4f}")
    else:
        print(f"  {'Walk-Forward Efficiency (Sharpe)':<26}  N/A (degenerate train Sharpe)")

    # ── Pooled OOS return distribution shape ────────────────────────────────
    pooled = _pooled_oos_returns(bundles)
    if len(pooled) >= 4:
        print(f"  {'Return Skewness':<26}  {pooled.skew():>10.4f}")
        print(f"  {'Return Excess Kurtosis':<26}  {pooled.kurt():>10.4f}")
    else:
        print(f"  Insufficient pooled return observations ({len(pooled)}) for skew/kurtosis.")

    # ── Drawdown scalars on the continuous rebased equity curve ────────────
    dd_rows   = _rebased_equity_with_drawdown(bundles)
    dd_stats  = _drawdown_scalar_stats(dd_rows)
    dd_days   = dd_stats["max_dd_duration"].total_seconds() / 86400.0
    print(f"  {'OOS Max Drawdown (%)':<26}  {dd_stats['max_drawdown_pct']:>10.4f}")
    print(f"  {'OOS Max DD Duration':<26}  {dd_days:>10.2f} days")
    if dd_stats["current_drawdown_pct"] is not None:
        print(f"  Current OOS Drawdown: {dd_stats['current_drawdown_pct']:.2f}% (still underwater)")

    print("=" * width)

    # ── Funding vs. price P&L split (only when funding was actually active) ─
    funding = _funding_share_summary(bundles)
    if funding is not None:
        print("\n  Funding vs Price P&L (OOS, summed across windows)")
        print(f"    Price P&L    : {funding['price_pnl']:.4f}")
        print(f"    Funding P&L  : {funding['funding_pnl']:.4f}")
        print(f"    Net P&L      : {funding['net_pnl']:.4f}")
        if funding["share_pct"] is not None:
            print(f"    Funding Share: {funding['share_pct']:.2f}%")
        else:
            print("    Funding Share: N/A (net P&L ~ 0)")
        print("=" * width)



# ── Per-window table ───────────────────────────────────────────────────────

def _print_per_window_table(bundles: list[dict[str, Any]]) -> None:
    """
    Print selected params and OOS metrics for every window.

    Args:
        bundles : list of OOS bundles returned by run_walk_forward().
    """
    width = 80
    print("\n" + "=" * width)
    print("  PER-WINDOW RESULTS")
    print("=" * width)

    # Loop over all windows and print the selected params and OOS metrics 
    for bundle in bundles:
        meta       = bundle["window"]
        w          = meta["window"]
        oos_m      = bundle["metrics"]
        if "train_runs" not in bundle:
            raise KeyError("_print_per_window_table: bundle missing required 'train_runs' key.")
        train_runs = bundle["train_runs"]
        swept      = _identify_swept_params(train_runs)

        # Selected param values for this window
        param_parts: list[str] = []
        for full_key, section in swept.items():
            val = _get_selected_param_val(bundle, full_key, section)
            param = full_key.split(".", 1)[1]
            param_parts.append(f"{param}={val}")
        params_display = ", ".join(param_parts) if param_parts else "(no sweep)"

        # Key OOS metrics
        sharpe = oos_m.get("sharpe_ratio",     float("nan"))
        ret    = oos_m.get("return_pct",        float("nan"))
        dd     = oos_m.get("max_drawdown_pct",  float("nan"))
        trades = oos_m.get("total_trades",      0)

        flags: list[str] = []
        if trades == 0:
            flags.append("ZERO TRADES")

        flag_str = f"  [!] {', '.join(flags)}" if flags else ""

        print(f"  Window {w:>2} | {meta['test_start']} → {meta['test_end']}")
        print(f"    Params : {params_display}")
        print(
            f"    OOS    : sharpe={sharpe:.4f}  return={ret:.4f}%"
            f"  dd={dd:.4f}%  trades={trades}{flag_str}"
        )

    print("=" * width)


# ── Param stability ────────────────────────────────────────────────────────

def _print_param_stability(bundles: list[dict[str, Any]]) -> None:
    """
    Print how often each param value was selected as best across windows.

    Only parameters that actually vary across any window are shown.
    """
    # Collect all swept param keys across all windows
    all_swept: dict[str, str] = {}
    for bundle in bundles:
        if "train_runs" not in bundle:
            raise KeyError("_print_param_stability: bundle missing required 'train_runs' key.")
        all_swept.update(_identify_swept_params(bundle["train_runs"]))

    if not all_swept:
        return

    width = 64
    print("\n" + "=" * width)
    print("  PARAM STABILITY  (selection frequency across windows)")
    print("=" * width)

    # Loop over all swept params and count how many times each value was selected across windows
    for full_key in sorted(all_swept):
        section = all_swept[full_key]
        counts: dict[Any, int] = {}

        for bundle in bundles:
            val = _get_selected_param_val(bundle, full_key, section)
            counts[val] = counts.get(val, 0) + 1

        total    = len(bundles)
        best_val = max(counts, key=lambda v: counts[v])

        print(f"\n  {full_key}:")
        # Sort the values by count descending and print each value with its count and a marker for the most stable value
        for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            marker = "  <-- most stable" if val == best_val else ""
            print(f"    {str(val):<22}  {cnt:>2}/{total} windows{marker}")

    print("\n" + "=" * width)


# ── Deployment recommendation ──────────────────────────────────────────────

def _print_deployment_recommendation(bundles: list[dict[str, Any]]) -> None:
    """
    Print the most recent window's best params and flag potential problems.

    Red flags raised:
        - Zero trades in the latest OOS window
        - OOS Sharpe < 0 while train Sharpe > 0 (overfitting signal)
        - OOS Sharpe < 50 % of train Sharpe
        - Latest OOS window has < _MIN_RELIABLE_RETURN_PERIODS resampled return periods
          (Sharpe/Sortino statistically unreliable)
        - Latest OOS window spans < _MIN_RELIABLE_YEARS years (CAGR/Calmar unstable)
        - Selected train Sharpe is below the pure-noise threshold for the trial count
        - Latest window params differ from the most-stable combo across earlier windows
        - High param variance: a param took a different value in > 60 % of windows

    Also reports the SE*sqrt(2*ln N) Sharpe threshold attainable from pure noise
    alone for the number of grid trials swept on train (Bailey & Lopez de Prado).
    """
    width = 64
    print("\n" + "=" * width)
    print("  DEPLOYMENT RECOMMENDATION")
    print("=" * width)

    latest     = bundles[-1]
    meta       = latest["window"]
    sel        = latest["selected"]
    oos_m      = latest["metrics"]
    train_m    = sel["metrics"]
    if "train_runs" not in latest:
        raise KeyError("_print_deployment_recommendation: bundle missing required 'train_runs' key.")
    train_runs = latest["train_runs"]
    swept      = _identify_swept_params(train_runs)

    # Compute most-stable combo (majority-vote per param across all windows)
    stable_combo: dict[str, Any] = {}
    for full_key, section in swept.items():
        counts: dict[Any, int] = {}
        for bundle in bundles:
            val = _get_selected_param_val(bundle, full_key, section)
            counts[val] = counts.get(val, 0) + 1
        stable_combo[full_key] = max(counts, key=lambda v: counts[v])

    print(f"\n  Latest window {meta['window']}: {meta['test_start']} → {meta['test_end']}")
    print("\n  Recommended params (from most recent window):")

    # Compare the latest window's selected params to the most-stable combo and note any mismatches
    stability_mismatches: list[str] = []
    for full_key, section in swept.items():
        val        = _get_selected_param_val(latest, full_key, section)
        stable_val = stable_combo.get(full_key)
        param      = full_key.split(".", 1)[1]
        if val == stable_val:
            note = " (matches stable)"
        else:
            note = f"  [MISMATCH — stable={stable_val}]"
            stability_mismatches.append(full_key)
        print(f"    {full_key}: {val}{note}")

    if not swept:
        print("    (no swept params)")

    # Issues / red flags
    red_flags: list[str] = []

    oos_trades = oos_m.get("total_trades", 0)
    if oos_trades == 0:
        red_flags.append(
            "Latest OOS window has ZERO trades — bot may be overly selective or data is thin."
        )

    oos_sharpe   = oos_m.get("sharpe_ratio",  float("nan"))
    train_sharpe = train_m.get("sharpe_ratio", float("nan"))
    if not (math.isnan(oos_sharpe) or math.isnan(train_sharpe)):
        if train_sharpe > 0 and oos_sharpe < 0:
            red_flags.append(
                f"OOS Sharpe ({oos_sharpe:.3f}) is negative while train Sharpe"
                f" ({train_sharpe:.3f}) was positive — strong overfitting signal."
            )
        elif train_sharpe > 0 and oos_sharpe < train_sharpe * 0.5:
            red_flags.append(
                f"OOS Sharpe ({oos_sharpe:.3f}) is < 50% of train Sharpe"
                f" ({train_sharpe:.3f}) — possible overfitting."
            )

    # Sample-size reliability: too few resampled return periods makes Sharpe/Sortino noisy
    n_return_periods = oos_m.get("n_return_periods")
    if n_return_periods is not None and n_return_periods < _MIN_RELIABLE_RETURN_PERIODS:
        red_flags.append(
            f"Latest OOS window has only {n_return_periods} resampled return period(s)"
            f" (< {_MIN_RELIABLE_RETURN_PERIODS}) — Sharpe/Sortino are statistically unreliable."
        )

    # Short calendar span: CAGR/Calmar's annualization exponent amplifies noise on short windows
    years_elapsed = oos_m.get("years_elapsed")
    if years_elapsed is not None and years_elapsed < _MIN_RELIABLE_YEARS:
        red_flags.append(
            f"Latest OOS window spans only {years_elapsed:.2f} year(s)"
            f" (< {_MIN_RELIABLE_YEARS}) — CAGR/Calmar are unstable on short windows."
        )

    # Selection-noise threshold: with N grid trials of true Sharpe 0, the best of N draws has
    # E[max] ~ SE*sqrt(2*ln N) where SE ~ sqrt(1/T_years) (Bailey & Lopez de Prado).
    n_trials    = len(train_runs)
    train_years = train_m.get("years_elapsed")
    if train_years is not None and train_years > 0 and n_trials > 1:
        noise_threshold = math.sqrt(1.0 / train_years) * math.sqrt(2.0 * math.log(n_trials))
        print(
            f"\n  Selection noise threshold: {n_trials} grid trial(s) over {train_years:.2f}y of"
            f" train data can attain a Sharpe of up to ~{noise_threshold:.3f} from pure noise alone."
        )
        if not math.isnan(train_sharpe) and train_sharpe < noise_threshold:
            red_flags.append(
                f"Selected train Sharpe ({train_sharpe:.3f}) is below the pure-noise threshold"
                f" (~{noise_threshold:.3f}) for {n_trials} grid trial(s) — may not reflect genuine edge."
            )

    if stability_mismatches:
        red_flags.append(
            "Latest window params differ from the most-stable combo on: "
            + ", ".join(stability_mismatches)
            + "."
        )

    # High param variance: param took distinct values in > 60 % of windows
    n_windows = len(bundles)
    for full_key, section in swept.items():
        vals      = [_get_selected_param_val(b, full_key, section) for b in bundles]
        n_unique  = len(set(str(v) for v in vals))
        if n_unique > max(1, n_windows * 0.6):
            red_flags.append(
                f"High param variance: '{full_key}' took {n_unique} distinct values"
                f" across {n_windows} windows."
            )

    if red_flags:
        print("\n  [!] POSSIBLE ISSUES:")
        for flag in red_flags:
            print(f"    - {flag}")
    else:
        print("\n  No issues detected.")

    print("=" * width + "\n")
