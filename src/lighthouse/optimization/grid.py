"""
================================================================================
PARAMETER GRID FOR BACKTEST RUNS
================================================================================

Build a parameter grid (full Cartesian product) from a sweep specification.
Each grid point is a pair of override dicts: one for bot params, one for
backtest params. The walk-forward harness merges these onto their respective template configs
before calling run_single().  build_grid() knows nothing about the templates;
it only expands the spec into override pairs.

Sweep spec shape (both sections optional, one must be non-empty with at least two values per parameter):
    {
        "bot":      {"<bot_param>":      [v1, v2, ...], ...},
        "backtest": {"<backtest_param>": [v1, v2, ...], ...},
    }

Functions:
    count_variants() : number of variants a sweep spec expands to (Cartesian product)
    build_grid()     : expand a sweep spec into (bot_overrides, backtest_overrides) pairs
================================================================================
"""

############ IMPORTS ############
import itertools
from typing import Any

############ HELPERS ############

def _split_spec(
    sweep_spec: dict[str, dict[str, list[Any]]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """
    Split a sweep spec into its bot and backtest sub-specs, validating shape.

    Args:
        sweep_spec : sweep specification 

    Returns:
        (bot_spec, backtest_spec) where each is a {param: [values]} dict
        (empty dict if that section is absent).

    Raises:
        ValueError : if a swept parameter maps to an empty list of values or to
                     fewer than two distinct values, or if the spec contains
                     unknown top-level sections.
    """
    # Check for unknown top-level sections (only "bot" and "backtest" are allowed)
    known = {"bot", "backtest"}
    unknown = set(sweep_spec) - known
    if unknown:
        raise ValueError(
            f"build_grid: unknown sweep_spec sections {sorted(unknown)}; "
            f"expected only {sorted(known)}."
        )

    # Extract bot and backtest sub-specs, defaulting to empty dicts if absent
    bot_spec: dict[str, list[Any]] = dict(sweep_spec.get("bot", {}))
    backtest_spec: dict[str, list[Any]] = dict(sweep_spec.get("backtest", {}))

    # Validate that at least one section has parameters to sweep
    if not bot_spec and not backtest_spec:
        raise ValueError(
            "build_grid: sweep_spec has no parameters to sweep. "
            "Use run_single() directly for a single run."
        )

    # Validate that each parameter maps to a non-empty list of values
    for section, spec in (("bot", bot_spec), ("backtest", backtest_spec)):
        for param, values in spec.items():
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise ValueError(
                    f"build_grid: sweep_spec['{section}']['{param}'] must be a "
                    f"non-empty list of values (got {values!r})."
                )
            # A sweep needs at least two distinct values to actually vary the param
            if len(set(values)) < 2:
                raise ValueError(
                    f"build_grid: sweep_spec['{section}']['{param}'] must have at "
                    f"least two distinct values to sweep (got {values!r})."
                )

    return bot_spec, backtest_spec


def _product(spec: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Expand a {param: [values]} spec into the Cartesian product of assignments.

    Args:
        spec : {param: [values]} dict

    Returns:
        A list of {param: value} dicts, one per combination.  An empty spec
        yields a single empty assignment [{}] (the neutral element).
    """
    params = list(spec.keys())

    # Cartesian product of all parameter value lists
    combos = itertools.product(*(spec[p] for p in params))

    # Build a list of dicts, one per combination, mapping each param to its value
    return [dict(zip(params, combo)) for combo in combos]


############ GRID ############

def count_variants(sweep_spec: dict[str, dict[str, list[Any]]]) -> int:
    """
    Return the number of variants the sweep spec expands to (Cartesian product
    of every bot and backtest parameter's value count).

    Args:
        sweep_spec : sweep specification (see module docstring)

    Returns:
        Total variant count.
    """
    # Split the sweep spec into bot and backtest sections
    bot_spec, backtest_spec = _split_spec(sweep_spec)

    # Count the total number of variants by multiplying the lengths of all value lists
    total = 1
    for values in {**bot_spec, **backtest_spec}.values():
        total *= len(values)
    return total


def build_grid(
    sweep_spec: dict[str, dict[str, list[Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Expand a sweep spec into override-pair tuples for the full parameter grid.

    Each tuple is (bot_overrides, backtest_overrides): plain dicts containing
    only the swept parameter values for that grid point. The walk-forward
    harness merges these onto its template configs before calling run_single():
        bot_config      = {**template_bot_config, **bot_overrides}
        backtest_config = dataclasses.replace(template_backtest_config, **backtest_overrides)

    build_grid() has no knowledge of the template configs; it only expands the
    spec. Template merging and validation happen in the harness.

    Args:
        sweep_spec : sweep specification (see module docstring)

    Returns:
        A list of (bot_overrides, backtest_overrides) tuples, one per grid point.
        Order is deterministic (bot params vary slowest).
    
    Raises:
        ValueError : if a swept parameter maps to an empty list of values or to
                     fewer than two distinct values. Raises through _split_spec().
    """
    bot_spec, backtest_spec = _split_spec(sweep_spec)

    bot_assignments      = _product(bot_spec)
    backtest_assignments = _product(backtest_spec)

    return [
        (bot_assignment, backtest_assignment)
        for bot_assignment      in bot_assignments
        for backtest_assignment in backtest_assignments
    ]
