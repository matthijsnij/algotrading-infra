"""
================================================================================
UNIT TESTS FOR optimization/walk_forward.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_walk_forward.py -v

To run a specific test function:
    pytest tests/test_walk_forward.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.optimization.walk_forward import rolling_splits, run_walk_forward, _pin_periods_per_year
from lighthouse.bots.examples.crypto_bot.bot import CryptoBot
from lighthouse.bots.examples.crypto_bot.config import crypto_config
from lighthouse.exchanges.backtest.config import BacktestConfig, MetricsConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

def make_df(n_bars: int) -> pd.DataFrame:
    """Return an OHLCV DataFrame with a UTC DatetimeIndex (hourly bars)."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n_bars)],
        "high":   [101.0 + i for i in range(n_bars)],
        "low":    [ 99.0 + i for i in range(n_bars)],
        "close":  [100.0 + i for i in range(n_bars)],
        "volume": [1000.0]   * n_bars,
    }, index=idx)

################### TESTS ##########################

# ── rolling_splits — window construction ─────────────────────────────────────

# known sizing yields the expected number of windows
def test_rolling_splits_window_count():
    # n=30, warmup=3: test_live_end = 18 + 5*window <= 30 → windows 0,1,2 → 3 windows
    splits = rolling_splits(make_df(30), train_len=10, test_len=5, step=5, warmup_bars=3)
    assert len(splits) == 3
    assert [meta["window"] for _, _, meta in splits] == [0, 1, 2]


# test segment always starts strictly after its train segment (no live overlap)
def test_rolling_splits_test_after_train():
    splits = rolling_splits(make_df(30), train_len=10, test_len=5, step=5, warmup_bars=3)
    for _, _, meta in splits:
        assert meta["test_start"] > meta["train_end"]


# each slice carries its warmup prefix (train_len + warmup, test_len + warmup)
def test_rolling_splits_warmup_prefix_lengths():
    warmup, train_len, test_len = 3, 10, 5
    splits = rolling_splits(make_df(30), train_len=train_len, test_len=test_len, step=5, warmup_bars=warmup)
    for train_slice, test_slice, _ in splits:
        assert len(train_slice) == warmup + train_len
        assert len(test_slice) == warmup + test_len


# the train live segment begins exactly at the meta train_start timestamp
def test_rolling_splits_train_live_start_matches_meta():
    warmup = 3
    splits = rolling_splits(make_df(30), train_len=10, test_len=5, step=5, warmup_bars=warmup)
    for train_slice, _, meta in splits:
        # first live bar is the one immediately after the warmup prefix
        assert train_slice.index[warmup] == meta["train_start"]


# consecutive windows advance the train start by exactly `step` bars
def test_rolling_splits_step_advances_windows():
    df = make_df(30)
    splits = rolling_splits(df, train_len=10, test_len=5, step=5, warmup_bars=3)
    starts = [meta["train_start"] for _, _, meta in splits]
    # 5 hourly bars between consecutive train starts
    assert starts[1] - starts[0] == pd.Timedelta(hours=5)
    assert starts[2] - starts[1] == pd.Timedelta(hours=5)


# data too short for even one full window → empty list
def test_rolling_splits_short_data_returns_empty():
    splits = rolling_splits(make_df(10), train_len=10, test_len=5, step=5, warmup_bars=3)
    assert splits == []


# ── rolling_splits — validation ──────────────────────────────────────────────

# train_len <= 0 → ValueError
def test_rolling_splits_invalid_train_len_raises():
    with pytest.raises(ValueError, match="train_len"):
        rolling_splits(make_df(30), train_len=0, test_len=5, step=5)


# test_len <= 0 → ValueError
def test_rolling_splits_invalid_test_len_raises():
    with pytest.raises(ValueError, match="test_len"):
        rolling_splits(make_df(30), train_len=10, test_len=0, step=5)


# step <= 0 → ValueError
def test_rolling_splits_invalid_step_raises():
    with pytest.raises(ValueError, match="step"):
        rolling_splits(make_df(30), train_len=10, test_len=5, step=0)


# warmup_bars < 0 → ValueError
def test_rolling_splits_negative_warmup_raises():
    with pytest.raises(ValueError, match="warmup_bars"):
        rolling_splits(make_df(30), train_len=10, test_len=5, step=5, warmup_bars=-1)


# ── run_walk_forward — smoke test ────────────────────────────────────────────

# a tiny end-to-end walk-forward returns one augmented OOS bundle per window
def test_run_walk_forward_smoke():
    df = make_df(60)

    run_config = {
        "symbol":          "BTC/USDT:USDT",
        "initial_balance": 10_000.0,
        "instrument": InstrumentSpec(
            quote_currency="USDT", can_short=True, leverage=10.0,
            maintenance_margin=0.05, has_liquidation=False,
            base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
        ),
        "timeframe": "1h",
        "backtest_config": BacktestConfig(
            spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
            latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
        ),
    }
    wf_config = {"train_len": 20, "test_len": 10, "step": 10, "warmup_bars": 20}
    sweep_spec = {"bot": {"breakout_window": [10, 20]}}

    bundles = run_walk_forward(
        df                  = df,
        bot_class           = CryptoBot,
        template_bot_config = crypto_config,
        run_config          = run_config,
        sweep_spec          = sweep_spec,
        wf_config           = wf_config,
        score_fn            = lambda m: m.get("total_trades", 0),
        verbose             = False,
    )

    expected_windows = len(rolling_splits(df, 20, 10, 10, 20))
    assert len(bundles) == expected_windows
    for bundle in bundles:
        assert "window" in bundle
        assert "train_runs" in bundle
        assert "selected" in bundle
        assert "metrics" in bundle
        # 2 grid points swept per window
        assert len(bundle["train_runs"]) == 2

# ── _pin_periods_per_year — cross-fold annualization pinning ─────────────────

# 60 hourly bars → default "1d" metrics_timeframe resamples to 3 daily bins (day1 full,
# day2 full, day3 partial with 12 bars); 2 return periods span exactly 2 calendar days
# (2024-01-01 → 2024-01-03) → periods_per_year = 2 / 2 * 365.25 = 365.25.
def test_pin_periods_per_year_derives_from_full_span():
    df = make_df(60)
    run_config = {"symbol": "BTC/USDT:USDT"}

    pinned = _pin_periods_per_year(df, run_config)

    assert pinned["_pinned_periods_per_year"] == pytest.approx(365.25)
    assert pinned["symbol"] == "BTC/USDT:USDT"  # other keys preserved


# with metrics_timeframe="1h" no bars are dropped (data is already hourly): 60 bars → 59
# return periods spanning exactly 59 hours (2.4583...d) → 59 / (59/24) * 365.25 = 24 * 365.25 = 8766.0
def test_pin_periods_per_year_respects_custom_timeframe():
    df = make_df(60)
    run_config = {"metrics_config": MetricsConfig(metrics_timeframe="1h")}

    pinned = _pin_periods_per_year(df, run_config)

    assert pinned["_pinned_periods_per_year"] == pytest.approx(8766.0)


# too little data to derive a stable span (fewer than 2 resampled periods) → no pinning
def test_pin_periods_per_year_noop_when_span_too_short():
    df = make_df(1)
    run_config = {"symbol": "BTC/USDT:USDT"}

    pinned = _pin_periods_per_year(df, run_config)

    assert "_pinned_periods_per_year" not in pinned


# ── run_walk_forward — periods_per_year is pinned identically across windows ─

# Same 60-bar dataset/config as test_run_walk_forward_smoke; with no metrics_config
# supplied, every window's OOS "periods_per_year" must equal the empirically-derived
# constant for the FULL dataset (365.25, per test_pin_periods_per_year_derives_from_full_span)
# rather than each window's own (shorter, jittery) span.
def test_run_walk_forward_pins_periods_per_year_across_windows():
    df = make_df(60)

    run_config = {
        "symbol":          "BTC/USDT:USDT",
        "initial_balance": 10_000.0,
        "instrument": InstrumentSpec(
            quote_currency="USDT", can_short=True, leverage=10.0,
            maintenance_margin=0.05, has_liquidation=False,
            base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
        ),
        "timeframe": "1h",
        "backtest_config": BacktestConfig(
            spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
            latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
        ),
    }
    wf_config = {"train_len": 20, "test_len": 10, "step": 10, "warmup_bars": 20}
    sweep_spec = {"bot": {"breakout_window": [10, 20]}}

    bundles = run_walk_forward(
        df                  = df,
        bot_class           = CryptoBot,
        template_bot_config = crypto_config,
        run_config          = run_config,
        sweep_spec          = sweep_spec,
        wf_config           = wf_config,
        score_fn            = lambda m: m.get("total_trades", 0),
        verbose             = False,
    )

    assert len(bundles) > 1  # need multiple windows for cross-window comparison to be meaningful
    for bundle in bundles:
        assert bundle["metrics"]["periods_per_year"] == pytest.approx(365.25)
    assert len({b["metrics"]["periods_per_year"] for b in bundles}) == 1