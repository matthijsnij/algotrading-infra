"""
================================================================================
UNIT TESTS FOR optimization/aggregate.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_aggregate.py -v

To run a specific test function:
    pytest tests/test_aggregate.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import dataclasses
import os
import pandas as pd
import pytest
from lighthouse.domain.timeframe import Timeframe
from lighthouse.optimization.aggregate import (
    write_results,
    print_summary,
    _identify_swept_params,
    _backtest_cfg_to_dict,
    _get_selected_param_val,
    _print_deployment_recommendation,
    _print_aggregate_oos,
    _print_per_window_table,
    _print_param_stability,
    _write_oos_equity_curve,
    _write_plateau_tables,
    _rebased_equity_with_drawdown,
    _drawdown_scalar_stats,
    _pooled_oos_returns,
    _compute_wfe,
    _funding_share_summary,
    _MIN_RELIABLE_RETURN_PERIODS,
    _MIN_RELIABLE_YEARS,
)

################### HELPERS ##########################

def _equity_curve(day: str) -> list[dict]:
    """
    Build an intraday equity curve spanning two calendar days starting at 'day'.

    Two snapshots on the first day and two on the second, so a "1D" resample
    (last value per day) yields exactly 2 rows.
    """
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1)
    return [
        {"bar": 0, "timestamp": d0,                              "equity": 1000.0},
        {"bar": 1, "timestamp": d0 + pd.Timedelta(hours=12),    "equity": 1010.0},
        {"bar": 2, "timestamp": d1,                              "equity": 1020.0},
        {"bar": 3, "timestamp": d1 + pd.Timedelta(hours=12),    "equity": 1030.0},
    ]


def _daily_equity_curve(day: str, values: list[float]) -> list[dict]:
    """One equity snapshot per calendar day (day, day+1, ...), for pooled-returns tests."""
    d0 = pd.Timestamp(day)
    return [
        {"bar": i, "timestamp": d0 + pd.Timedelta(days=i), "equity": v}
        for i, v in enumerate(values)
    ]


def _train_run(breakout_window: int, sharpe: float) -> dict:
    """A single train-sweep bundle with one swept bot param and a dict backtest_config."""
    return {
        "bot_config":      {"breakout_window": breakout_window, "atr_period": 14},
        "backtest_config": {"spread": 0.001},
        "metrics":         {"sharpe_ratio": sharpe, "return_pct": sharpe * 2, "total_trades": 5},
    }


def _bundle(window: int, day: str, selected_bw: int, with_trades: bool, sweep: bool = True) -> dict:
    """
    Build a minimal OOS bundle for one walk-forward window.

    Args:
        window      : window index.
        day         : first test-day date string (test spans two days).
        selected_bw : breakout_window value of the selected train run.
        with_trades : whether the OOS trade_log is non-empty.
        sweep       : if True, provide two train runs (param varies); else one.
    """
    test_start = pd.Timestamp(day)
    test_end   = test_start + pd.Timedelta(days=1)

    # If sweep is True, provide two train runs with different breakout_window values; else one run.
    if sweep:
        train_runs = [_train_run(10, 1.0), _train_run(20, 2.0)]
    else:
        train_runs = [_train_run(selected_bw, 1.0)]

    selected = next(r for r in train_runs if r["bot_config"]["breakout_window"] == selected_bw)

    trades = [{"pnl": 5.0}, {"pnl": -2.0}] if with_trades else []

    return {
        "window": {
            "window":      window,
            "train_start": test_start - pd.Timedelta(days=2),
            "train_end":   test_start - pd.Timedelta(days=1),
            "test_start":  test_start,
            "test_end":    test_end,
            "train_len":   10,
            "test_len":    5,
            "warmup_bars": 0,
        },
        "train_runs":      train_runs,
        "selected":        selected,
        "bot_config":      selected["bot_config"],
        "backtest_config": {"spread": 0.001},
        "metrics":         {"sharpe_ratio": 1.5, "return_pct": 3.0, "total_trades": len(trades)},
        "equity_curve":    _equity_curve(day),
        "trade_log":       trades,
    }


@pytest.fixture
def bundles() -> list[dict]:
    """Two-window OOS bundle list: window 0 selects bw=10 w/ trades, window 1 selects bw=20 w/o trades."""
    return [
        _bundle(0, "2021-01-01", selected_bw=10, with_trades=True),
        _bundle(1, "2021-01-03", selected_bw=20, with_trades=False),
    ]

################### TESTS ##########################

# ── write_results — files & dirs ─────────────────────────────────────────────

# write_results creates all expected top-level files and subdirectories
def test_write_results_creates_all_outputs(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    for name in ("train_runs_metrics.csv", "best_params_per_window.csv",
                 "oos_equity_curve.csv", "param_stability.csv"):
        assert os.path.isfile(os.path.join(out, name))
    assert os.path.isdir(os.path.join(out, "trade_log_per_window"))
    assert os.path.isdir(os.path.join(out, "plateau_tables"))


# train_runs_metrics.csv carries window + train/test date columns
def test_train_runs_metrics_has_date_columns(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    df = pd.read_csv(os.path.join(out, "train_runs_metrics.csv"))
    for col in ("window", "train_start", "train_end", "test_start", "test_end"):
        assert col in df.columns
    # 2 windows × 2 train runs each = 4 rows
    assert len(df) == 4


# best_params_per_window.csv has one row per window
def test_best_params_per_window_shape(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    df = pd.read_csv(os.path.join(out, "best_params_per_window.csv"))
    assert len(df) == 2
    assert "window" in df.columns


# ── oos_equity_curve.csv ─────────────────────────────────────────────────────

# default "1D" resample → last-per-day rows, columns window|timestamp|equity|
# rebased_equity|drawdown_pct; window 0 is the first window so rebased == raw
# exactly; window 1's carried-forward rebased is higher than its own raw restart
def test_oos_equity_curve_resampled_default(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    df = pd.read_csv(os.path.join(out, "oos_equity_curve.csv"))
    assert list(df.columns) == ["window", "timestamp", "equity", "rebased_equity", "drawdown_pct"]
    # each window spans 2 calendar days → 2 rows/window → 4 rows total
    assert len(df) == 4
    # last equity value of the first day is kept (1010.0), not the first (1000.0)
    assert df.iloc[0]["equity"] == 1010.0

    w0 = df[df["window"] == 0]
    w1 = df[df["window"] == 1]
    # window 0 is the very first window → rebased_equity == raw equity exactly
    assert (w0["rebased_equity"] == w0["equity"]).all()
    # window 1 restarts its own raw equity at ~1000, but its rebased value carries
    # forward from window 0's higher ending equity, so rebased > raw here
    assert (w1["rebased_equity"] > w1["equity"]).all()
    # this fixture's equity is monotonically increasing end-to-end once rebased →
    # no drawdown anywhere in the stitched curve
    assert (df["drawdown_pct"] == 0.0).all()


# equity_curve_timeframe=None → raw per-bar curve, columns window|bar|timestamp|
# equity|rebased_equity|drawdown_pct; first bar of the first window is the seed,
# so rebased_equity == equity exactly there
def test_oos_equity_curve_raw_when_none(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out, equity_curve_timeframe=None)
    df = pd.read_csv(os.path.join(out, "oos_equity_curve.csv"))
    assert list(df.columns) == ["window", "bar", "timestamp", "equity", "rebased_equity", "drawdown_pct"]
    # 4 snapshots per window × 2 windows = 8 rows
    assert len(df) == 8
    first_row = df.iloc[0]
    assert first_row["rebased_equity"] == first_row["equity"] == 1000.0


# equity_curve_timeframe takes a Timeframe (not a raw pandas rule string)
def test_oos_equity_curve_accepts_timeframe_instance(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out, equity_curve_timeframe=Timeframe.parse("1d"))
    df = pd.read_csv(os.path.join(out, "oos_equity_curve.csv"))
    assert list(df.columns) == ["window", "timestamp", "equity", "rebased_equity", "drawdown_pct"]
    assert len(df) == 4


# _write_oos_equity_curve() called directly (not via write_results) produces the resampled file
def test_write_oos_equity_curve_direct_call(tmp_path, bundles):
    out = str(tmp_path)
    _write_oos_equity_curve(bundles, out, Timeframe.parse("1d"))
    df = pd.read_csv(os.path.join(out, "oos_equity_curve.csv"))
    assert list(df.columns) == ["window", "timestamp", "equity", "rebased_equity", "drawdown_pct"]
    assert len(df) == 4


# _write_oos_equity_curve() raises directly when a bundle is missing 'equity_curve'
def test_write_oos_equity_curve_direct_call_missing_key_raises(tmp_path):
    with pytest.raises(KeyError, match="equity_curve"):
        _write_oos_equity_curve([{}], str(tmp_path), None)


# ── trade logs ───────────────────────────────────────────────────────────────

# trade-log filenames embed YYYYMMDD dates; empty logs produce no file
def test_trade_log_filenames_and_skip(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    trade_dir = os.path.join(out, "trade_log_per_window")
    files = os.listdir(trade_dir)
    # only window 0 has trades → exactly one file
    assert len(files) == 1
    fname = files[0]
    assert fname.startswith("window_0000_")
    assert "20210101" in fname and "20210102" in fname


# ── plateau tables ───────────────────────────────────────────────────────────

# plateau filenames embed dates + section + param
def test_plateau_filenames_contain_dates_and_param(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    plateau_dir = os.path.join(out, "plateau_tables")
    files = os.listdir(plateau_dir)
    # 2 windows × 1 swept param = 2 tables
    assert len(files) == 2
    for fname in files:
        assert "_bot_breakout_window.csv" in fname
    assert any("20210101" in f and "20210102" in f for f in files)


# _write_plateau_tables() called directly (not via write_results) writes one CSV per swept param per window
def test_write_plateau_tables_direct_call(tmp_path, bundles):
    out = str(tmp_path)
    _write_plateau_tables(bundles, out)
    plateau_dir = os.path.join(out, "plateau_tables")
    files = os.listdir(plateau_dir)
    assert len(files) == 2
    for fname in files:
        assert "_bot_breakout_window.csv" in fname


# _write_plateau_tables() raises directly when a bundle is missing 'train_runs'
def test_write_plateau_tables_direct_call_missing_key_raises(tmp_path):
    with pytest.raises(KeyError, match="train_runs"):
        _write_plateau_tables([{"window": {"window": 0}}], str(tmp_path))


# ── param_stability.csv ──────────────────────────────────────────────────────

# param_stability.csv has expected columns and valid frequencies
def test_param_stability_columns_and_frequency(tmp_path, bundles):
    out = str(tmp_path)
    write_results(bundles, out)
    df = pd.read_csv(os.path.join(out, "param_stability.csv"))
    assert list(df.columns) == ["param", "value", "count", "total_windows", "frequency"]
    assert (df["frequency"] >= 0).all() and (df["frequency"] <= 1).all()
    assert (df["total_windows"] == 2).all()
    # bw=10 selected once, bw=20 selected once → two rows, each frequency 0.5
    assert set(df["value"]) == {10, 20}
    assert (df["frequency"] == 0.5).all()


# no swept params → param_stability.csv is empty but keeps its columns
def test_param_stability_empty_when_no_sweep(tmp_path):
    bundles = [
        _bundle(0, "2021-01-01", selected_bw=10, with_trades=True, sweep=False),
        _bundle(1, "2021-01-03", selected_bw=10, with_trades=True, sweep=False),
    ]
    out = str(tmp_path)
    write_results(bundles, out)
    df = pd.read_csv(os.path.join(out, "param_stability.csv"))
    assert list(df.columns) == ["param", "value", "count", "total_windows", "frequency"]
    assert len(df) == 0


# ── missing-key guards ───────────────────────────────────────────────────────

# bundle missing train_runs → KeyError
def test_write_results_missing_train_runs_raises(tmp_path, bundles):
    del bundles[0]["train_runs"]
    with pytest.raises(KeyError, match="train_runs"):
        write_results(bundles, str(tmp_path))


# bundle missing equity_curve → KeyError
def test_write_results_missing_equity_curve_raises(tmp_path, bundles):
    del bundles[0]["equity_curve"]
    with pytest.raises(KeyError, match="equity_curve"):
        write_results(bundles, str(tmp_path))


# bundle missing trade_log → KeyError
def test_write_results_missing_trade_log_raises(tmp_path, bundles):
    del bundles[0]["trade_log"]
    with pytest.raises(KeyError, match="trade_log"):
        write_results(bundles, str(tmp_path))


# ── helpers ──────────────────────────────────────────────────────────────────

# _identify_swept_params flags only the param that varies across train runs
def test_identify_swept_params_detects_variation():
    runs = [_train_run(10, 1.0), _train_run(20, 2.0)]
    swept = _identify_swept_params(runs)
    assert swept == {"bot.breakout_window": "bot"}


# _identify_swept_params returns empty when nothing varies
def test_identify_swept_params_empty_when_constant():
    runs = [_train_run(10, 1.0), _train_run(10, 2.0)]
    swept = _identify_swept_params(runs)
    assert swept == {}


# _backtest_cfg_to_dict passes a plain dict through unchanged
def test_backtest_cfg_to_dict_dict_passthrough():
    assert _backtest_cfg_to_dict({"spread": 0.001}) == {"spread": 0.001}


# _backtest_cfg_to_dict flattens a dataclass, repr-ing non-primitive fields
def test_backtest_cfg_to_dict_dataclass():
    @dataclasses.dataclass
    class Cfg:
        spread: float
        model: object

    cfg = Cfg(spread=0.001, model=object())
    out = _backtest_cfg_to_dict(cfg)
    assert out["spread"] == 0.001
    assert isinstance(out["model"], str)  # non-primitive → repr string


# _get_selected_param_val reads the selected bundle's bot param
def test_get_selected_param_val_reads_selected():
    bundle = _bundle(0, "2021-01-01", selected_bw=20, with_trades=True)
    val = _get_selected_param_val(bundle, "bot.breakout_window", "bot")
    assert val == 20


# ── _print_deployment_recommendation — new metrics-reliability warnings ──────

# n_return_periods below _MIN_RELIABLE_RETURN_PERIODS flags Sharpe/Sortino as unreliable
def test_deployment_recommendation_flags_low_return_periods(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True)]
    bundles[-1]["metrics"]["n_return_periods"] = _MIN_RELIABLE_RETURN_PERIODS - 1

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "resampled return period" in out
    assert "Sharpe/Sortino are statistically unreliable" in out


# n_return_periods at/above the threshold does not raise the flag
def test_deployment_recommendation_no_flag_for_sufficient_return_periods(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True)]
    bundles[-1]["metrics"]["n_return_periods"] = _MIN_RELIABLE_RETURN_PERIODS

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "resampled return period" not in out


# years_elapsed below _MIN_RELIABLE_YEARS flags CAGR/Calmar as unstable
def test_deployment_recommendation_flags_short_span(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True)]
    bundles[-1]["metrics"]["years_elapsed"] = _MIN_RELIABLE_YEARS - 0.1

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "CAGR/Calmar are unstable on short windows" in out


# Selection-noise threshold: N=2 train trials over 1.0y → SE=sqrt(1/1)=1, threshold =
# 1 * sqrt(2*ln(2)) ≈ 1.1774. Selected train Sharpe is 1.0 (bw=10 per _train_run), below
# the threshold, so the "may not reflect genuine edge" flag must fire.
def test_deployment_recommendation_flags_sharpe_below_noise_threshold(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True)]
    for run in bundles[-1]["train_runs"]:
        run["metrics"]["years_elapsed"] = 1.0

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "Selection noise threshold" in out
    assert "1.177" in out
    assert "below the pure-noise threshold" in out


# Same N/years as above, but the selected train Sharpe (2.0, bw=20) clears the ≈1.1774
# threshold, so the threshold is still reported but no red flag is raised for it.
def test_deployment_recommendation_reports_noise_threshold_without_flag(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=20, with_trades=True)]
    for run in bundles[-1]["train_runs"]:
        run["metrics"]["years_elapsed"] = 1.0

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "Selection noise threshold" in out
    assert "below the pure-noise threshold" not in out


# Single train run (no sweep) → fewer than 2 trials → noise threshold is not computed/printed
def test_deployment_recommendation_skips_noise_threshold_for_single_trial(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True, sweep=False)]
    bundles[-1]["train_runs"][0]["metrics"]["years_elapsed"] = 1.0

    _print_deployment_recommendation(bundles)
    out = capsys.readouterr().out
    assert "Selection noise threshold" not in out


# ── _rebased_equity_with_drawdown ─────────────────────────────────────────────

# Window 0's rebased_equity matches raw equity exactly bar-for-bar; window 1's first
# bar carries window 0's ending value forward with zero return, then applies its own
# +0.8% ratio on top (rebased = 10315 * 1.008 = 10397.72, per the plan's worked example)
def test_rebased_equity_matches_raw_within_first_window_and_chains_ratio():
    t0 = pd.Timestamp("2021-01-01")
    bundle0 = {
        "window": {"window": 0, "test_start": t0, "test_end": t0 + pd.Timedelta(days=1)},
        "equity_curve": [
            {"bar": 0, "timestamp": t0,                           "equity": 10000.0},
            {"bar": 1, "timestamp": t0 + pd.Timedelta(hours=12), "equity": 10315.0},
        ],
    }
    t1 = t0 + pd.Timedelta(days=1)
    bundle1 = {
        "window": {"window": 1, "test_start": t1, "test_end": t1 + pd.Timedelta(days=1)},
        "equity_curve": [
            {"bar": 0, "timestamp": t1,                           "equity": 10000.0},
            {"bar": 1, "timestamp": t1 + pd.Timedelta(hours=12), "equity": 10080.0},
        ],
    }

    rows = _rebased_equity_with_drawdown([bundle0, bundle1])
    assert len(rows) == 4

    # window 0: rebased == raw exactly
    assert rows[0]["equity"] == rows[0]["rebased_equity"] == 10000.0
    assert rows[1]["equity"] == rows[1]["rebased_equity"] == 10315.0

    # window 1 bar 0: carries window 0's ending value forward, raw equity resets to 10000
    assert rows[2]["equity"] == 10000.0
    assert rows[2]["rebased_equity"] == pytest.approx(10315.0)

    # window 1 bar 1: rebased = 10315 * (10080/10000) = 10315 * 1.008 = 10397.52
    assert rows[3]["equity"] == 10080.0
    assert rows[3]["rebased_equity"] == pytest.approx(10397.52)


# monotonically increasing rebased curve → drawdown_pct is 0.0 at every bar
def test_rebased_equity_drawdown_zero_when_monotonic_increasing():
    t0 = pd.Timestamp("2021-01-01")
    bundle = {
        "window": {"window": 0, "test_start": t0, "test_end": t0 + pd.Timedelta(days=1)},
        "equity_curve": [
            {"bar": 0, "timestamp": t0,                           "equity": 1000.0},
            {"bar": 1, "timestamp": t0 + pd.Timedelta(hours=12), "equity": 1100.0},
        ],
    }
    rows = _rebased_equity_with_drawdown([bundle])
    assert all(row["drawdown_pct"] == 0.0 for row in rows)


# empty equity_curve on every bundle → no rows, no crash
def test_rebased_equity_empty_when_no_curves():
    t0 = pd.Timestamp("2021-01-01")
    bundle = {
        "window": {"window": 0, "test_start": t0, "test_end": t0 + pd.Timedelta(days=1)},
        "equity_curve": [],
    }
    assert _rebased_equity_with_drawdown([bundle]) == []


# overlapping OOS test ranges across consecutive windows print a caution warning
def test_rebased_equity_warns_on_overlapping_test_windows(capsys):
    t0 = pd.Timestamp("2021-01-01")
    bundle0 = {
        "window": {"window": 0, "test_start": t0, "test_end": t0 + pd.Timedelta(days=2)},
        "equity_curve": [{"bar": 0, "timestamp": t0, "equity": 1000.0}],
    }
    # window 1 starts before window 0 ends → overlap
    t1 = t0 + pd.Timedelta(days=1)
    bundle1 = {
        "window": {"window": 1, "test_start": t1, "test_end": t1 + pd.Timedelta(days=1)},
        "equity_curve": [{"bar": 0, "timestamp": t1, "equity": 1000.0}],
    }

    _rebased_equity_with_drawdown([bundle0, bundle1])
    out = capsys.readouterr().out
    assert "overlaps the previous window" in out


# ── _drawdown_scalar_stats ────────────────────────────────────────────────────

def _dd_rows(pcts: list[float]) -> list[dict]:
    """Hand-built rows with only the fields _drawdown_scalar_stats reads."""
    t0 = pd.Timestamp("2021-01-01")
    return [
        {"timestamp": t0 + pd.Timedelta(days=i), "drawdown_pct": p}
        for i, p in enumerate(pcts)
    ]


# single deepest episode (-10.0% at day 2), recovered by the last bar (0.0%) →
# current_drawdown_pct is None; peak is the last 0.0% bar strictly before the trough
def test_drawdown_scalar_stats_recovered_episode():
    rows = _dd_rows([0.0, -5.0, -10.0, -3.0, 0.0])
    stats = _drawdown_scalar_stats(rows)
    assert stats["max_drawdown_pct"] == pytest.approx(10.0)
    assert stats["max_dd_duration"] == pd.Timedelta(days=2)
    assert stats["current_drawdown_pct"] is None


# same episode shape, but still underwater at the final bar → current_drawdown_pct is set
def test_drawdown_scalar_stats_still_underwater():
    rows = _dd_rows([0.0, -5.0, -10.0, -3.0])
    stats = _drawdown_scalar_stats(rows)
    assert stats["max_drawdown_pct"] == pytest.approx(10.0)
    assert stats["current_drawdown_pct"] == pytest.approx(-3.0)


# empty rows → all-zero/None, no crash
def test_drawdown_scalar_stats_empty_rows():
    stats = _drawdown_scalar_stats([])
    assert stats["max_drawdown_pct"] == 0.0
    assert stats["max_dd_duration"] == pd.Timedelta(0)
    assert stats["current_drawdown_pct"] is None


# ── _pooled_oos_returns ───────────────────────────────────────────────────────

# 2 windows, each with a 3-point daily equity curve and its own metrics_timeframe →
# pooled series is the exact concatenation of each window's own resampled returns
def test_pooled_oos_returns_concatenates_manual_resample():
    curve_a = _daily_equity_curve("2021-01-01", [1000.0, 1010.0, 1020.0])
    curve_b = _daily_equity_curve("2021-02-01", [2000.0, 2020.0, 2000.0])
    bundles = [
        {"metrics": {"metrics_timeframe": "1d"}, "equity_curve": curve_a},
        {"metrics": {"metrics_timeframe": "1d"}, "equity_curve": curve_b},
    ]

    pooled = _pooled_oos_returns(bundles)

    expected_a = [(1010.0 - 1000.0) / 1000.0, (1020.0 - 1010.0) / 1010.0]
    expected_b = [(2020.0 - 2000.0) / 2000.0, (2000.0 - 2020.0) / 2020.0]
    assert len(pooled) == 4
    assert pooled.tolist() == pytest.approx(expected_a + expected_b)


# a window missing metrics_timeframe, or with an empty equity_curve, contributes nothing
def test_pooled_oos_returns_skips_windows_without_timeframe_or_curve():
    curve_a = _daily_equity_curve("2021-01-01", [1000.0, 1010.0, 1020.0])
    bundles = [
        {"metrics": {"metrics_timeframe": "1d"}, "equity_curve": curve_a},
        {"metrics": {}, "equity_curve": curve_a},                       # no timeframe
        {"metrics": {"metrics_timeframe": "1d"}, "equity_curve": []},   # no curve
    ]
    pooled = _pooled_oos_returns(bundles)
    assert len(pooled) == 2


# no usable windows at all → empty float series
def test_pooled_oos_returns_empty_when_no_usable_windows():
    pooled = _pooled_oos_returns([{"metrics": {}, "equity_curve": []}])
    assert pooled.empty
    assert pooled.dtype == float


# ── _compute_wfe ──────────────────────────────────────────────────────────────

# mean(OOS sharpe)/mean(train sharpe) = (1.0+3.0)/2 / (2.0+4.0)/2 = 2.0/3.0
def test_compute_wfe_exact_ratio():
    bundles = [
        {"metrics": {"sharpe_ratio": 1.0}, "selected": {"metrics": {"sharpe_ratio": 2.0}}},
        {"metrics": {"sharpe_ratio": 3.0}, "selected": {"metrics": {"sharpe_ratio": 4.0}}},
    ]
    wfe = _compute_wfe(bundles)
    assert wfe == pytest.approx(2.0 / 3.0)


# mean train Sharpe ~0 → degenerate ratio → None
def test_compute_wfe_none_when_train_sharpe_degenerate():
    bundles = [
        {"metrics": {"sharpe_ratio": 1.0}, "selected": {"metrics": {"sharpe_ratio": 1.0}}},
        {"metrics": {"sharpe_ratio": 1.0}, "selected": {"metrics": {"sharpe_ratio": -1.0}}},
    ]
    assert _compute_wfe(bundles) is None


# no bundles/usable sharpe values → None
def test_compute_wfe_none_when_no_sharpe_values():
    assert _compute_wfe([{"metrics": {}, "selected": {"metrics": {}}}]) is None


# ── _funding_share_summary ────────────────────────────────────────────────────

# funding never active (total_funding_paid ~ 0 across all windows) → None, block hidden
def test_funding_share_summary_none_when_inactive():
    bundles = [
        {"metrics": {"total_funding_paid": 0.0, "gross_pnl": 100.0, "net_pnl": 90.0}},
        {"metrics": {"total_funding_paid": 0.0, "gross_pnl": 50.0,  "net_pnl": 45.0}},
    ]
    assert _funding_share_summary(bundles) is None


# nonzero funding → exact price/funding/net split + share_pct
# total_funding_paid = 15+5=20 (cost paid) → funding_pnl = -20 (net cost, sign-flipped)
# total_price_pnl = 100+50=150, total_net_pnl = 80+40=120, share_pct = -20/120*100
def test_funding_share_summary_exact_split():
    bundles = [
        {"metrics": {"total_funding_paid": 15.0, "gross_pnl": 100.0, "net_pnl": 80.0}},
        {"metrics": {"total_funding_paid": 5.0,  "gross_pnl": 50.0,  "net_pnl": 40.0}},
    ]
    result = _funding_share_summary(bundles)
    assert result["price_pnl"]   == pytest.approx(150.0)
    assert result["funding_pnl"] == pytest.approx(-20.0)
    assert result["net_pnl"]     == pytest.approx(120.0)
    assert result["share_pct"]   == pytest.approx(-20.0 / 120.0 * 100.0)


# net_pnl ~ 0 → share_pct is None (undefined denominator) instead of raising
def test_funding_share_summary_share_pct_none_when_net_pnl_zero():
    bundles = [{"metrics": {"total_funding_paid": 10.0, "gross_pnl": 10.0, "net_pnl": 0.0}}]
    result = _funding_share_summary(bundles)
    assert result["share_pct"] is None


# ── _print_aggregate_oos — new console blocks ────────────────────────────────

# WFE and drawdown scalar lines always appear; funding block is absent when inactive
def test_print_aggregate_oos_shows_wfe_and_drawdown_hides_funding(capsys, bundles):
    _print_aggregate_oos(bundles)
    out = capsys.readouterr().out
    assert "Walk-Forward Efficiency" in out
    assert "OOS Max Drawdown" in out
    assert "OOS Max DD Duration" in out
    assert "Funding vs Price P&L" not in out


# funding block appears once funding is detected as active on any window
def test_print_aggregate_oos_shows_funding_block_when_active(capsys, bundles):
    bundles[0]["metrics"]["total_funding_paid"] = 12.0
    bundles[0]["metrics"]["gross_pnl"] = 100.0
    bundles[0]["metrics"]["net_pnl"]   = 80.0
    bundles[1]["metrics"]["total_funding_paid"] = 0.0
    bundles[1]["metrics"]["gross_pnl"] = 50.0
    bundles[1]["metrics"]["net_pnl"]   = 40.0

    _print_aggregate_oos(bundles)
    out = capsys.readouterr().out
    assert "Funding vs Price P&L" in out
    assert "Price P&L" in out and "Funding P&L" in out


# print_summary runs end-to-end without raising on the standard bundles fixture
def test_print_summary_smoke(capsys, bundles):
    print_summary(bundles)
    out = capsys.readouterr().out
    assert "WALK-FORWARD AGGREGATE" in out
    assert "PER-WINDOW RESULTS" in out


# ── _print_per_window_table — direct call ────────────────────────────────────

# _print_per_window_table() called directly prints window header, selected
# params, and OOS metrics, flagging the window with zero OOS trades
def test_print_per_window_table_direct_call(capsys, bundles):
    _print_per_window_table(bundles)
    out = capsys.readouterr().out

    assert "PER-WINDOW RESULTS" in out
    assert "breakout_window=10" in out
    assert "breakout_window=20" in out
    assert "ZERO TRADES" in out  # window 1 has no OOS trades


# ── _print_param_stability — direct call ─────────────────────────────────────

# _print_param_stability() called directly prints per-value selection frequency
# across windows, marking the most-selected value as the most stable one
def test_print_param_stability_direct_call(capsys, bundles):
    _print_param_stability(bundles)
    out = capsys.readouterr().out

    assert "PARAM STABILITY" in out
    assert "breakout_window" in out
    assert "most stable" in out


# _print_param_stability() prints nothing when no swept params exist across any window
def test_print_param_stability_direct_call_no_output_when_no_sweep(capsys):
    bundles = [_bundle(0, "2021-01-01", selected_bw=10, with_trades=True, sweep=False)]

    _print_param_stability(bundles)

    assert capsys.readouterr().out == ""


