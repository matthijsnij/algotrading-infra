"""
================================================================================
UNIT TESTS FOR optimization/grid.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_grid.py -v

To run a specific test function:
    pytest tests/test_grid.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from lighthouse.optimization.grid import build_grid, count_variants

################### FIXTURES ##########################

BOT_SPEC      = {"bot": {"breakout_window": [10, 20, 30], "atr_period": [14, 21]}}
BACKTEST_SPEC = {"backtest": {"spread": [0.0005, 0.001]}}
COMBINED_SPEC = {**BOT_SPEC, **BACKTEST_SPEC}

################### TESTS ##########################

# ── count_variants ────────────────────────────────────────────────────────

# 3 × 2 bot params → 6 variants
def test_count_variants_bot_only():
    assert count_variants(BOT_SPEC) == 6

# 2 backtest values → 2 variants
def test_count_variants_backtest_only():
    assert count_variants(BACKTEST_SPEC) == 2

# 3 × 2 × 2 combined → 12 variants
def test_count_variants_combined():
    assert count_variants(COMBINED_SPEC) == 12

# ── build_grid — shape and content ────────────────────────────────────────

# combined spec → 12 tuples, each a (bot_overrides, backtest_overrides) pair
def test_build_grid_length_matches_count_variants():
    grid = build_grid(COMBINED_SPEC)
    assert len(grid) == count_variants(COMBINED_SPEC)

# override dicts contain ONLY the swept keys — no template bleed-through
def test_build_grid_override_dicts_contain_only_swept_keys():
    grid = build_grid(COMBINED_SPEC)
    for bot_ov, bt_ov in grid:
        assert set(bot_ov.keys()) == {"breakout_window", "atr_period"}
        assert set(bt_ov.keys()) == {"spread"}

# bot-only spec → backtest overrides are always empty dicts
def test_build_grid_bot_only_backtest_overrides_empty():
    grid = build_grid(BOT_SPEC)
    assert all(bt_ov == {} for _, bt_ov in grid)

# backtest-only spec → bot overrides are always empty dicts
def test_build_grid_backtest_only_bot_overrides_empty():
    grid = build_grid(BACKTEST_SPEC)
    assert all(bot_ov == {} for bot_ov, _ in grid)

# bot params vary slowest: first len(backtest) rows share the same bot assignment
def test_build_grid_bot_params_vary_slowest():
    grid = build_grid(COMBINED_SPEC)
    backtest_count = count_variants(BACKTEST_SPEC)  # 2
    first_bot_ov = grid[0][0]
    for bot_ov, _ in grid[:backtest_count]:
        assert bot_ov == first_bot_ov

# all value combinations appear exactly once (no duplicates, no missing)
def test_build_grid_all_combinations_present():
    grid = build_grid(COMBINED_SPEC)
    seen = set()
    for bot_ov, bt_ov in grid:
        key = (bot_ov["breakout_window"], bot_ov["atr_period"], bt_ov["spread"])
        assert key not in seen, f"duplicate combination: {key}"
        seen.add(key)
    assert len(seen) == 12

# ── build_grid — validation ────────────────────────────────────────────────

# empty spec → raises ValueError
def test_build_grid_empty_spec_raises():
    with pytest.raises(ValueError, match="no parameters to sweep"):
        build_grid({})

# unknown top-level section → raises ValueError
def test_build_grid_unknown_section_raises():
    with pytest.raises(ValueError, match="unknown sweep_spec sections"):
        build_grid({"strategy": {"x": [1, 2]}})

# empty values list for a param → raises ValueError
def test_build_grid_empty_values_list_raises():
    with pytest.raises(ValueError):
        build_grid({"bot": {"breakout_window": []}})

# single-value list for a param → raises ValueError (nothing to sweep)
def test_split_single_value_raises():
    with pytest.raises(ValueError, match="at least two distinct values"):
        build_grid({"bot": {"breakout_window": [10]}})

# list with duplicate values (< 2 distinct) → raises ValueError
def test_split_duplicate_values_raises():
    with pytest.raises(ValueError, match="at least two distinct values"):
        build_grid({"backtest": {"spread": [0.001, 0.001]}})
