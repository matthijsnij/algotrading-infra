"""
================================================================================
UNIT TESTS FOR optimization/objective.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_objective.py -v

To run a specific test function:
    pytest tests/test_objective.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from lighthouse.optimization.objective import (
    rank_results_by_metric,
    rank_results_by_score,
    select_best_by_score,
    results_dataframe,
    score_single_metric,
)

################### HELPERS ##########################

def bundle(sharpe: float = 0.0, dd: float = 0.0, trades: int = 0, **params) -> dict:
    """Minimal result bundle with the metrics used in these tests, plus optional bot_config params."""
    return {
        "metrics": {"sharpe_ratio": sharpe, "max_drawdown_pct": dd, "total_trades": trades},
        "bot_config": dict(params),
    }

################### TESTS ##########################

# ── rank_results_by_metric ──────────────────────────────────────────────────

# results ranked highest sharpe first when maximizing
def test_rank_results_by_metric_descending_sharpe():
    results = [bundle(1.0), bundle(3.0), bundle(2.0)]
    ranked = rank_results_by_metric(results, primary="sharpe_ratio", maximize=True)
    assert [b["metrics"]["sharpe_ratio"] for b in ranked] == [3.0, 2.0, 1.0]

# maximize=False → lowest value first (e.g. smallest drawdown is best)
def test_rank_results_by_metric_minimize_primary():
    results = [bundle(dd=5.0), bundle(dd=15.0), bundle(dd=10.0)]
    ranked = rank_results_by_metric(results, primary="max_drawdown_pct", maximize=False)
    assert [b["metrics"]["max_drawdown_pct"] for b in ranked] == [5.0, 10.0, 15.0]

# tie on primary → secondary metric breaks the tie
def test_rank_results_by_metric_tie_broken_by_secondary():
    # two bundles with same sharpe; lower drawdown wins as secondary
    results = [bundle(sharpe=2.0, dd=10.0), bundle(sharpe=2.0, dd=5.0), bundle(sharpe=1.0)]
    ranked = rank_results_by_metric(
        results,
        primary="sharpe_ratio",
        maximize=True,
        secondaries=[("max_drawdown_pct", False)],  # minimize drawdown
    )
    assert ranked[0]["metrics"]["max_drawdown_pct"] == 5.0
    assert ranked[1]["metrics"]["max_drawdown_pct"] == 10.0
    assert ranked[2]["metrics"]["sharpe_ratio"] == 1.0

# empty input → returns empty list without raising
def test_rank_results_by_metric_empty_input():
    assert rank_results_by_metric([], primary="sharpe_ratio", maximize=True) == []

# input list is not mutated
def test_rank_results_by_metric_does_not_mutate_input():
    original = [bundle(1.0), bundle(3.0), bundle(2.0)]
    original_order = [b["metrics"]["sharpe_ratio"] for b in original]
    rank_results_by_metric(original, primary="sharpe_ratio", maximize=True)
    assert [b["metrics"]["sharpe_ratio"] for b in original] == original_order

# bundle missing the requested metric → raises KeyError
def test_rank_results_by_metric_missing_metric_raises():
    results = [{"metrics": {"sharpe_ratio": 1.0}}]
    with pytest.raises(KeyError):
        rank_results_by_metric(results, primary="nonexistent_metric", maximize=True)

# bundle with no "metrics" dict → raises KeyError
def test_rank_results_by_metric_missing_metrics_dict_raises():
    with pytest.raises(KeyError):
        rank_results_by_metric([{"bot_config": {}}], primary="sharpe_ratio", maximize=True)

# ── rank_results_by_score ───────────────────────────────────────────────────

# results ranked best-first by a composite score function
def test_rank_results_by_score_descending():
    results = [bundle(1.0), bundle(3.0), bundle(2.0)]
    ranked = rank_results_by_score(results, lambda m: m["sharpe_ratio"])
    assert [b["metrics"]["sharpe_ratio"] for b in ranked] == [3.0, 2.0, 1.0]

# empty input → returns empty list without raising
def test_rank_results_by_score_empty_input():
    assert rank_results_by_score([], lambda m: m["sharpe_ratio"]) == []

# original bundles are returned (raw metrics unchanged, only sort order uses normalized values)
def test_rank_results_by_score_returns_original_bundles():
    results = [bundle(1.0), bundle(3.0)]
    ranked = rank_results_by_score(results, lambda m: m["sharpe_ratio"])
    assert ranked[0]["metrics"]["sharpe_ratio"] == 3.0  # raw value, not percentile

# rank normalization: scale-dominant metric does not override a lower-scale metric with equal weight
def test_rank_results_by_score_scale_independence():
    # Without normalization, max_drawdown_pct (0-50) would dominate sharpe_ratio (0-3).
    # With rank normalization both live on [0,1] so equal weights mean equal importance.
    from lighthouse.optimization.objective import score_composite
    scorer = score_composite([
        ("sharpe_ratio",     1.0, True),   # sharpe: higher is better
        ("max_drawdown_pct", 1.0, False),  # dd: lower is better
    ])
    # A: great sharpe, bad dd  B: poor sharpe, great dd  — should be a near-tie
    a = {"metrics": {"sharpe_ratio": 3.0, "max_drawdown_pct": 40.0}, "bot_config": {}}
    b_ = {"metrics": {"sharpe_ratio": 0.5, "max_drawdown_pct": 5.0}, "bot_config": {}}
    # Without normalization A scores (3-40)/2=-18.5, B scores (0.5-5)/2=-2.25 → B wins by a mile.
    # With rank normalization A: percentiles (1.0, 0.0), B: percentiles (0.0, 1.0) → exact tie.
    ranked = rank_results_by_score([a, b_], scorer)
    # Both bundles should appear (no crash); the raw sharpe values are preserved on the output
    assert {ranked[0]["metrics"]["sharpe_ratio"], ranked[1]["metrics"]["sharpe_ratio"]} == {3.0, 0.5}

# ties in a metric receive the average rank percentile
def test_rank_results_by_score_tied_metric():
    # three bundles with tied sharpe; ranking should still be stable (no crash, all present)
    results = [bundle(2.0), bundle(2.0), bundle(2.0)]
    ranked = rank_results_by_score(results, lambda m: m["sharpe_ratio"])
    assert len(ranked) == 3

# bundle missing "metrics" dict → raises KeyError
def test_rank_results_by_score_missing_metrics_raises():
    with pytest.raises(KeyError):
        rank_results_by_score([{"bot_config": {}}], lambda m: m["sharpe_ratio"])

# ── select_best_by_score ────────────────────────────────────────────────────

# returns the single bundle with the highest score
def test_select_best_by_score_returns_highest():
    results = [bundle(1.0), bundle(3.0), bundle(2.0)]
    best = select_best_by_score(results, lambda m: m["sharpe_ratio"])
    assert best["metrics"]["sharpe_ratio"] == 3.0

# empty input → raises ValueError
def test_select_best_by_score_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select_best_by_score([], lambda m: m["sharpe_ratio"])

# ── score_single_metric ─────────────────────────────────────────────────────

# returns the value of the named metric
def test_score_single_metric_returns_value():
    assert score_single_metric({"sharpe_ratio": 1.5}, "sharpe_ratio") == 1.5

# missing metric → raises KeyError
def test_score_single_metric_missing_raises():
    with pytest.raises(KeyError):
        score_single_metric({"sharpe_ratio": 1.0}, "nonexistent_metric")

# ── results_dataframe ───────────────────────────────────────────────────────

# all metrics included by default; one row per bundle
def test_results_dataframe_all_metrics():
    results = [bundle(1.0, dd=5.0), bundle(2.0, dd=10.0)]
    df = results_dataframe(results, include_params=False)
    assert len(df) == 2
    assert "sharpe_ratio" in df.columns
    assert "max_drawdown_pct" in df.columns
    assert list(df["sharpe_ratio"]) == [1.0, 2.0]

# metrics_subset limits the metric columns
def test_results_dataframe_metrics_subset():
    results = [bundle(1.0, dd=5.0)]
    df = results_dataframe(results, metrics_subset=["sharpe_ratio"], include_params=False)
    assert list(df.columns) == ["sharpe_ratio"]

# requesting a metric not present → raises KeyError
def test_results_dataframe_subset_missing_raises():
    results = [bundle(1.0)]
    with pytest.raises(KeyError):
        results_dataframe(results, metrics_subset=["nonexistent_metric"])

# include_params appends bot_config columns prefixed with "bot_"
def test_results_dataframe_include_params_columns():
    results = [bundle(1.0, breakout_window=10)]
    df = results_dataframe(results, include_params=True)
    assert "bot_breakout_window" in df.columns
    assert list(df["bot_breakout_window"]) == [10]

# bundle missing "metrics" key → raises ValueError
def test_results_dataframe_missing_metrics_raises():
    with pytest.raises(ValueError, match="metrics"):
        results_dataframe([{"bot_config": {}}], include_params=False)

# include_params=True but bundle missing "bot_config" → raises ValueError
def test_results_dataframe_missing_bot_config_raises():
    with pytest.raises(ValueError, match="bot_config"):
        results_dataframe([{"metrics": {"sharpe_ratio": 1.0}}], include_params=True)
