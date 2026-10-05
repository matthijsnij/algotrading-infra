"""
================================================================================
OBJECTIVE
================================================================================

Selection and ranking of backtest result bundles by a performance objective.

A "bundle" is the dict returned by backtesting.run_single(); its "metrics" key
holds the metrics dict from Reporter.get_results() (sharpe_ratio, return_pct,
max_drawdown_pct, total_trades, etc.).  These helpers pick the best bundle (or a
full ranking) for a single primary metric, with optional secondary metrics used
only as tie-breakers.

Metric direction:
    Higher is better -> maximize=True
    Lower is better  -> maximize=False

Functions:
    rank_results_by_metric()          : sort bundles best-first by a single named metric (+ tie-breakers)
    rank_results_by_score() : sort bundles best-first by a composite score function
    select_best_by_score()  : return the single best bundle by a composite score function
    results_dataframe()     : build a DataFrame of metrics (and optionally params) for all bundles

Scoring functions (pass to select_best_by_score / rank_results_by_score / run_walk_forward):
    score_single_metric()  : baseline scorer using a single named metric of choice
    score_sharpe()         : annualized Sharpe ratio
    score_sortino()        : annualized Sortino ratio 
    score_calmar()         : CAGR / max-drawdown 
    score_profit_factor()  : gross wins / gross losses
    score_composite()      : factory; returns a weighted multi-metric scorer callable
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############
import math
import pandas as pd
from typing import Any, Callable


############ HELPERS ############

def _metric(bundle: dict[str, Any], name: str) -> float:
    """
    Read a metric value from a result bundle, validating its presence.

    Args:
        bundle : result bundle dict (must contain a "metrics" sub-dict)
        name   : metric key to read

    Returns:
        The metric value as a float.

    Raises:
        KeyError : if the bundle has no "metrics" dict or the metric is absent.
    """
    metrics = bundle.get("metrics")
    if metrics is None:
        raise KeyError("objective: result bundle has no 'metrics' dict.")
    if name not in metrics:
        raise KeyError(
            f"objective: metric '{name}' not found in bundle metrics; "
            f"available: {sorted(metrics)}."
        )
    return metrics[name]


def _sort_key(
    bundle: dict[str, Any],
    ordering: list[tuple[str, bool]],
) -> tuple[float, ...]:
    """
    Build a descending sort key so that max() / sorted(reverse=True) yields the
    best bundle first.

    Each (metric, maximize) pair contributes its value when maximize is True, or
    its negation when maximize is False, so that a single "largest is best"
    comparison honours mixed metric directions.

    Args:
        bundle   : result bundle dict
        ordering : list of (metric_name, maximize) pairs, primary first

    Returns:
        A tuple usable as a "largest is best" sort key.
    """
    return tuple(
        (_metric(bundle, name) if maximize else -_metric(bundle, name))
        for name, maximize in ordering
    )


def _rank_normalize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return a parallel list of metrics dicts with every numeric metric replaced
    by its rank percentile across the population (0.0 = lowest, 1.0 = highest).
        - Ties receive the average of their rank positions.
        - A single-bundle population maps every metric to 0.5.
        - Original bundles are not mutated.

    Args:
        results : list of result bundles (from run_single() runs)
    
    Returns:
        A new list of metrics dicts with rank-percentile values.
    """
    N = len(results)
    all_metrics = [b["metrics"] for b in results]
    normalized = [dict(m) for m in all_metrics]

    # Handle the single-bundle case: every metric is 0.5
    if N == 1:
        for key in all_metrics[0]:
            normalized[0][key] = 0.5
        return normalized

    # Loop over all metrics in the first bundle; assumes all bundles have the same keys
    for key in all_metrics[0]:

        # Get all values for this metric across the population
        values = [m[key] for m in all_metrics]

        # Skip non-numeric metrics
        if not all(isinstance(v, (int, float)) for v in values):
            continue

        # Compute rank percentiles
        percentiles = pd.Series(values).rank(method='average', pct=True)
        for idx in range(N):
            normalized[idx][key] = percentiles.iloc[idx]

    return normalized


############ RANKING ############

def rank_results_by_metric(
    results: list[dict[str, Any]],
    primary: str,
    maximize: bool,
    secondaries: list[tuple[str, bool]] | None = None,
) -> list[dict[str, Any]]:
    """
    Return the result bundles sorted best-first by the primary metric, using the
    secondary metrics only to break ties.

    The input list is not mutated; a new sorted list is returned.

    Args:
        results     : list of result bundles (from run_single() runs)
        primary     : primary metric key to rank by
        maximize    : True if a higher primary value is better; False if lower is better
        secondaries : optional list of (metric_name, maximize) tie-breakers,
                      applied in order after the primary metric

    Returns:
        A new list of bundles ordered best-first.  Empty input yields [].
    """
    if not results:
        return []
    ordering: list[tuple[str, bool]] = [(primary, maximize)] + list(secondaries or [])
    return sorted(results, key=lambda b: _sort_key(b, ordering), reverse=True)


def rank_results_by_score(
    results: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """
    Return the result bundles sorted best-first by the passed score function.

    Each bundle's metrics are rank-percentile normalized across the population
    before being passed to score_fn, so metric scale differences do not affect
    the weighting.  The original bundles (with raw metrics) are returned; only
    the sort order is derived from normalized values.

    Useful for post-hoc inspection when you want the full ranking ordered by the
    same objective that drove train-time selection in run_walk_forward().
    The input list is not mutated; a new sorted list is returned.

    Args:
        results  : list of result bundles (from run_single() runs).
        score_fn : scoring function callable; receives rank-percentile metrics.

    Returns:
        A new list of bundles ordered best-first. Empty input yields [].
    """
    if not results:
        return []
    for i, bundle in enumerate(results):
        if "metrics" not in bundle:
            raise KeyError(
                f"rank_results_by_score: bundle at index {i} has no 'metrics' dict."
            )
    norm_metrics = _rank_normalize(results)
    scored = sorted(
        zip(norm_metrics, results),
        key=lambda pair: score_fn(pair[0]),
        reverse=True,
    )
    return [bundle for _, bundle in scored]

############ SELECTION ############

def select_best_by_score(
    results: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """
    Return the single best result bundle according to the passed score function.

    Calls rank_results_by_score() and returns the first bundle in the sorted list.

    Args:
        results  : non-empty list of result bundles (from run_single() runs).
        score_fn : scoring function callable

    Returns:
        The bundle with the highest score.

    Raises:
        ValueError : if results is empty.
    """
    if not results:
        raise ValueError("select_best_by_score: results list is empty; nothing to select.")
    return rank_results_by_score(results, score_fn)[0]

############ INSPECTION ############

def results_dataframe(
    results: list[dict[str, Any]],
    metrics_subset: list[str] | None = None,
    include_params: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per result bundle for easy inspection.

    Args:
        results        : list of result bundles (from run_single() runs)
        metrics_subset : optional subset of metric keys to include; None includes all
        include_params : if True, append bot_config columns (prefixed "bot_")

    Returns:
        A DataFrame with one row per bundle, and columns for the requested metrics and optionally the bot_config parameters.

    Raises:
        KeyError   : if a requested column name is not present in a bundle's metrics.
        ValueError : if a bundle has no "metrics" dict, or include_params is True
                     and a bundle has no "bot_config" dict.
    """
    rows: list[dict[str, Any]] = []
    for bundle in results:
        if "metrics" not in bundle:
            raise ValueError("results_dataframe: bundle missing required 'metrics' key.")
        metrics = bundle["metrics"]

        if metrics_subset is not None:
            # Check that all requested metrics are present in the metrics dict
            missing = [m for m in metrics_subset if m not in metrics]
            if missing:
                raise KeyError(
                    f"results_dataframe: metric(s) {missing} not found in bundle;"
                    f"available: {sorted(metrics)}."
                )
            # Create a row dict with only the requested metrics
            row: dict[str, Any] = {m: metrics[m] for m in metrics_subset}
        else:
            # Create a row dict with all metrics
            row = dict(metrics)

        if include_params:
            # Include bot_config columns, prefixed with "bot_"
            if "bot_config" not in bundle:
                raise ValueError(
                    "results_dataframe: bundle missing required 'bot_config' key "
                    "(required when include_params is True)."
                )
            bot_config = bundle["bot_config"]
            for k, v in bot_config.items():
                row[f"bot_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)

############ SCORING CALLABLES ############

def score_single_metric(metrics: dict[str, Any], metric: str) -> float:
    """
    Single-metric baseline scorer: returns the value of the specified metric from the metrics dict.

    Args:
        metrics : metrics dict from a result bundle (Reporter.get_results()).
        metric  : the key of the metric to return.

    Returns:
        The value of the specified metric from the metrics dict.

    Raises:
        KeyError : if the specified metric is not present in the metrics dict.
    """
    if metric not in metrics:
        raise KeyError(
            f"score_single_metric: metric '{metric}' not found in metrics dict;"
            f"available: {sorted(metrics)}."
        )

    return metrics[metric]


# ── Named single-metric scorers ─────────────────────────────────────────────

def score_sharpe(metrics: dict[str, Any]) -> float:
    """
    Score by annualized Sharpe ratio.

    Args:
        metrics : metrics dict from a result bundle (Reporter.get_results()).

    Returns:
        sharpe_ratio value.
    """
    return score_single_metric(metrics, "sharpe_ratio")


def score_sortino(metrics: dict[str, Any]) -> float:
    """
    Score by annualized Sortino ratio.

    Args:
        metrics : metrics dict from a result bundle (Reporter.get_results()).

    Returns:
        sortino_ratio value.
    """
    return score_single_metric(metrics, "sortino_ratio")


def score_calmar(metrics: dict[str, Any]) -> float:
    """
    Score by Calmar ratio (CAGR / max drawdown).

    Args:
        metrics : metrics dict from a result bundle (Reporter.get_results()).

    Returns:
        calmar_ratio value.
    """
    return score_single_metric(metrics, "calmar_ratio")


def score_profit_factor(metrics: dict[str, Any]) -> float:
    """
    Score by profit factor (gross wins / gross losses).

    Args:
        metrics : metrics dict from a result bundle (Reporter.get_results()).

    Returns:
        profit_factor value.
    """
    return score_single_metric(metrics, "profit_factor")


# ── Composite factory ────────────────────────────────────────────────────────

def score_composite(
    components: list[tuple[str, float, bool]],
) -> Callable[[dict[str, Any]], float]:
    """
    Factory that returns a weighted multi-metric scorer callable.

    Call once to build the scorer, then pass the returned callable to
    select_best_by_score / rank_results_by_score / run_walk_forward.

    This function expects normalized metrics (rank percentiles) as input, such that scoring is scale-invariant.

    Each component is a (metric_name, weight, maximize) triple:
        metric_name : key in the metrics dict
        weight      : relative importance; only ratios matter; weights are
                      normalized internally so they sum to 1.
        maximize    : True if a higher metric value is better;
                      False if a lower value is better.
                      For maximize=False the metric is negated before weighting.
    Args:
        components : non-empty list of (metric_name, weight, maximize) tuples.

    Returns:
        A callable with signature (metrics: dict) -> float.

    Raises:
        ValueError : if components is empty or all weights are zero.
    """
    # Validate the components list 
    if not components:
        raise ValueError("score_composite: components list must not be empty.")

    for i, (name, weight, maximize) in enumerate(components):
        if not isinstance(name, str):
            raise TypeError(
                f"score_composite: component {i} metric_name must be str; got {type(name).__name__}."
            )
        if not isinstance(weight, (int, float)):
            raise TypeError(
                f"score_composite: component {i} weight must be int or float; got {type(weight).__name__}."
            )
        if not isinstance(maximize, bool):
            raise TypeError(
                f"score_composite: component {i} maximize must be bool; got {type(maximize).__name__}."
            )
        if weight < 0:
            raise ValueError(
                f"score_composite: component {i} weight must be non-negative; got {weight}."
            )

    total_weight = sum(w for _, w, _ in components)
    if total_weight <= 0:
        raise ValueError(
            "score_composite: sum of weights must be positive; "
            f"got {total_weight}."
        )

    # Store the normalized weights and maximize flags for use in the scorer
    normalized: list[tuple[str, float, bool]] = [
        (name, weight / total_weight, maximize)
        for name, weight, maximize in components
    ]

    # Define the scorer function that computes the weighted score based on the metrics dict
    def _scorer(metrics: dict[str, Any]) -> float:
        total = 0.0
        for name, weight, maximize in normalized:
            if name not in metrics:
                raise KeyError(
                    f"score_composite: metric '{name}' not found in metrics dict; "
                    f"available: {sorted(metrics)}."
                )
            value: float = metrics[name]
            total += weight * (value if maximize else -value)
        return total

    return _scorer