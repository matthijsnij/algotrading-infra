"""
================================================================================
UNIT TESTS FOR exchanges/backtest/reporter.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_reporter.py -v

To run a specific test function:
    pytest tests/backtest/test_reporter.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.reporter import Reporter, _parse_timeframe, _resample_equity
from lighthouse.exchanges.backtest.state import BacktestState
from lighthouse.exchanges.backtest.config import BacktestConfig, MetricsConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

INITIAL_BALANCE = 10_000.0


def make_df(n_bars: int = 10) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with timestamp as a column."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC"),
        "open":      [100.0] * n_bars,
        "high":      [101.0] * n_bars,
        "low":       [ 99.0] * n_bars,
        "close":     [100.0] * n_bars,
        "volume":    [1000.0] * n_bars,
    })


def make_state(cursor: int = 0, metrics_config: MetricsConfig = None) -> BacktestState:
    """Return a minimal BacktestState for reporter tests."""
    config = BacktestConfig(
        spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
        latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
    )
    instrument = InstrumentSpec(
        quote_currency="USDT", can_short=True, leverage=1.0,
        maintenance_margin=0.0, has_liquidation=False,
        base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
    )
    state = BacktestState(
        df=make_df(), ts_col="timestamp", symbol="BTC/USDT",
        instrument=instrument, base_timeframe="1h",
        initial_balance=INITIAL_BALANCE, config=config,
        metrics_config=metrics_config,
    )
    state.cursor = cursor
    return state


def make_equity_curve(equities: list[float], freq: str = "1h") -> list[dict]:
    """Return an equity curve with evenly spaced tz-aware UTC timestamps."""
    stamps = pd.date_range("2024-01-01", periods=len(equities), freq=freq, tz="UTC")
    return [
        {"bar": i, "timestamp": stamps[i], "equity": eq}
        for i, eq in enumerate(equities)
    ]


def make_trade(net_pnl: float, pnl: float = None, total_fees: float = 0.0) -> dict:
    """Return a minimal trade log entry with the given p&l values."""
    if pnl is None:
        pnl = net_pnl + total_fees
    return {
        "symbol":       "BTC/USDT",
        "side":         "long",
        "entry_price":  100.0,
        "exit_price":   100.0,
        "size":         1.0,
        "entry_bar":    0,
        "exit_bar":     1,
        "entry_ts":     pd.Timestamp("2024-01-01", tz="UTC"),
        "exit_ts":      pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        "pnl":          pnl,
        "entry_fee":    0.0,
        "exit_fee":     total_fees,
        "funding_paid": 0.0,
        "total_fees":   total_fees,
        "net_pnl":      net_pnl,
        "return_pct":   0.0,
        "close_reason": "exit_order",
    }

################### TESTS ##########################

# ── record_equity() ────────────────────────────────────────────────────────

# no position: equity == balance["total"]
def test_record_equity_no_position():
    state    = make_state()
    reporter = Reporter(state)

    reporter.record_equity()

    assert state.equity_curve[0]["equity"] == pytest.approx(INITIAL_BALANCE)


# long position: equity includes unrealized p&l
def test_record_equity_long_unrealized_pnl():
    state    = make_state()
    reporter = Reporter(state)
    state.position = {
        "side":        "long",
        "size":        2.0,
        "entry_price": 100.0,
        "mark_price":  110.0,  # +10 per unit → unrealized_pnl = 20.0
    }

    reporter.record_equity()

    expected_equity = state.balance["total"] + (110.0 - 100.0) * 2.0
    assert state.equity_curve[0]["equity"] == pytest.approx(expected_equity)


# short position: equity includes unrealized p&l (inverted direction)
def test_record_equity_short_unrealized_pnl():
    state    = make_state()
    reporter = Reporter(state)
    state.position = {
        "side":        "short",
        "size":        2.0,
        "entry_price": 100.0,
        "mark_price":  90.0,   # -10 per unit short → unrealized_pnl = +20.0
    }

    reporter.record_equity()

    expected_equity = state.balance["total"] + (100.0 - 90.0) * 2.0
    assert state.equity_curve[0]["equity"] == pytest.approx(expected_equity)

# ── get_results() ────────────────────────────────────────────────────────

# zero trades: total_trades=0, win_rate=0, profit_factor=inf, net_pnl=0
def test_get_results_zero_trades():
    state    = make_state()
    reporter = Reporter(state)

    results = reporter.get_results()

    assert results["total_trades"]  == 0
    assert results["win_rate"]      == 0.0
    assert results["net_pnl"]       == 0.0
    assert results["profit_factor"] == float("inf")


# correct win_rate, gross_pnl, net_pnl, return_pct from a known set of trades
def test_get_results_pnl_metrics():
    state    = make_state()
    reporter = Reporter(state)
    # 2 winning trades (net_pnl=10 each), 1 losing trade (net_pnl=-4)
    state.trade_log = [
        make_trade(net_pnl=10.0, pnl=10.5, total_fees=0.5),
        make_trade(net_pnl=10.0, pnl=10.5, total_fees=0.5),
        make_trade(net_pnl=-4.0, pnl=-3.5, total_fees=0.5),
    ]

    results = reporter.get_results()

    assert results["total_trades"]  == 3
    assert results["winning_trades"] == 2
    assert results["losing_trades"]  == 1
    assert results["win_rate"]      == pytest.approx(round(2 / 3, 4))
    assert results["gross_pnl"]     == pytest.approx(10.5 + 10.5 - 3.5)   # sum of pnl
    assert results["net_pnl"]       == pytest.approx(10.0 + 10.0 - 4.0)   # sum of net_pnl
    assert results["return_pct"]    == pytest.approx((10.0 + 10.0 - 4.0) / INITIAL_BALANCE * 100)


# profit_factor = inf when there are no losing trades
def test_get_results_profit_factor_inf_no_losses():
    state    = make_state()
    reporter = Reporter(state)
    state.trade_log = [make_trade(net_pnl=5.0)]

    results = reporter.get_results()

    assert results["profit_factor"] == float("inf")


# max_drawdown_pct tracks the largest peak-to-trough decline in the equity curve
def test_get_results_max_drawdown():
    state    = make_state()
    reporter = Reporter(state)
    # equity rises to 12000, drops to 9000 → drawdown = (12000-9000)/12000 = 25%
    # then recovers to 11000 → smaller drawdown, does not affect max
    state.equity_curve = make_equity_curve([10_000.0, 12_000.0, 9_000.0, 11_000.0])

    results = reporter.get_results()

    expected_dd = (12_000.0 - 9_000.0) / 12_000.0 * 100
    assert results["max_drawdown_pct"] == pytest.approx(expected_dd)


# sharpe_ratio = 0 when equity is perfectly flat (zero std dev of returns)
def test_get_results_sharpe_zero_flat_equity():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0] * 5)

    results = reporter.get_results()

    assert results["sharpe_ratio"] == 0.0


# _parse_timeframe accepts an integer count + m/h/d/w unit and returns (seconds, pandas alias)
def test_parse_timeframe_valid():
    assert _parse_timeframe("1m") == (60, "1min")
    assert _parse_timeframe("5m") == (300, "5min")
    assert _parse_timeframe("4h") == (14_400, "4h")
    assert _parse_timeframe("1d") == (86_400, "1D")
    assert _parse_timeframe("2w") == (1_209_600, "2W")


# a malformed timeframe string raises rather than silently falling back
def test_parse_timeframe_invalid_raises():
    with pytest.raises(ValueError, match="unrecognized timeframe"):
        _parse_timeframe("bogus")


# returns terminate at ruin instead of resuming afterwards: equity
# 10000 -> 8000 -> 0 -> 5000 -> 6000 yields exactly [-0.2, -1.0].
# The old filter dropped only the undefined 0 -> 5000 step and then
# appended a spurious +0.2 "recovery" return from a wiped-out account.
# Uses daily-spaced snapshots so each bar is already its own resample bucket
# (metrics_timeframe defaults to "1d"), keeping the expected math identical
# to the pre-resampling bar-level calculation.
def test_get_results_returns_truncate_at_ruin():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve(
        [10_000.0, 8_000.0, 0.0, 5_000.0, 6_000.0], freq="1D"
    )

    returns  = [-0.2, -1.0]
    mean_r   = sum(returns) / 2
    var_r    = sum((r - mean_r) ** 2 for r in returns) / 1
    std_r    = var_r ** 0.5
    ppy      = 365.25 # 2 daily return periods spanning 2 days → empirically 365.25/yr
    expected = round(mean_r / std_r * (ppy ** 0.5), 4)

    results = reporter.get_results()

    assert results["sharpe_ratio"] == pytest.approx(expected)
    assert results["n_return_periods"] == 2


# expectancy = net_pnl / total_trades
def test_get_results_expectancy():
    state    = make_state()
    reporter = Reporter(state)
    # net_pnl = 10 + 10 - 4 = 16; 3 trades → expectancy = 16/3
    state.trade_log = [
        make_trade(net_pnl=10.0),
        make_trade(net_pnl=10.0),
        make_trade(net_pnl=-4.0),
    ]

    results = reporter.get_results()

    assert results["expectancy"] == pytest.approx(round(16.0 / 3, 4))


# expectancy = 0 when there are no trades
def test_get_results_expectancy_zero_trades():
    state    = make_state()
    reporter = Reporter(state)

    results = reporter.get_results()

    assert results["expectancy"] == 0.0


# max_consecutive_losses counts the longest unbroken streak of net_pnl <= 0
def test_get_results_max_consecutive_losses():
    state    = make_state()
    reporter = Reporter(state)
    # sequence: L L W L L L W → max streak = 3
    state.trade_log = [
        make_trade(net_pnl=-1.0),
        make_trade(net_pnl=-2.0),
        make_trade(net_pnl= 5.0),
        make_trade(net_pnl=-3.0),
        make_trade(net_pnl=-4.0),
        make_trade(net_pnl=-5.0),
        make_trade(net_pnl= 1.0),
    ]

    results = reporter.get_results()

    assert results["max_consecutive_losses"] == 3


# max_consecutive_losses = 0 when all trades are winners
def test_get_results_max_consecutive_losses_all_wins():
    state    = make_state()
    reporter = Reporter(state)
    state.trade_log = [make_trade(net_pnl=5.0), make_trade(net_pnl=3.0)]

    results = reporter.get_results()

    assert results["max_consecutive_losses"] == 0


# time_in_market = sum(exit-entry) / len(equity_curve)
def test_get_results_time_in_market():
    state    = make_state()
    reporter = Reporter(state)
    # equity curve has 10 bars; trades cover 3+3=6 bars → 6/10 = 0.6
    state.equity_curve = make_equity_curve([10_000.0] * 10)
    t0 = make_trade(net_pnl=1.0); t0["entry_bar"] = 0; t0["exit_bar"] = 3
    t1 = make_trade(net_pnl=1.0); t1["entry_bar"] = 5; t1["exit_bar"] = 8
    state.trade_log = [t0, t1]

    results = reporter.get_results()

    assert results["time_in_market"] == pytest.approx(0.6)


# time_in_market = 0 with no trades
def test_get_results_time_in_market_no_trades():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0] * 5)

    results = reporter.get_results()

    assert results["time_in_market"] == 0.0


# recovery_factor = net_pnl / max_absolute_drawdown
def test_get_results_recovery_factor():
    state    = make_state()
    reporter = Reporter(state)
    # peak=12000, trough=9000 → max_dd_abs=3000; net_pnl=1500 → RF=0.5
    state.equity_curve = make_equity_curve([10_000.0, 12_000.0, 9_000.0, 11_000.0])
    state.trade_log = [make_trade(net_pnl=1_500.0)]

    results = reporter.get_results()

    assert results["recovery_factor"] == pytest.approx(1_500.0 / 3_000.0)


# recovery_factor = inf when there is no drawdown
def test_get_results_recovery_factor_no_drawdown():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 11_000.0])
    state.trade_log = [make_trade(net_pnl=1_000.0)]

    results = reporter.get_results()

    assert results["recovery_factor"] == float("inf")


# sortino_ratio = 0 when all returns are non-negative (no downside)
# daily spacing so metrics_timeframe="1d" resamples 1:1 with the bar-level curve
def test_get_results_sortino_zero_no_downside():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 10_200.0, 10_300.0], freq="1D"
    )

    results = reporter.get_results()

    assert results["sortino_ratio"] == 0.0


# sortino_ratio matches hand-computed value when there are negative returns.
# Daily spacing so metrics_timeframe="1d" resamples 1:1 with the bar-level curve,
# keeping the expected math identical to the pre-resampling bar-level calculation.
def test_get_results_sortino_with_negative_returns():
    state    = make_state()
    reporter = Reporter(state)
    # equity: 10000 → 10100 → 9900 → 10200
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 9_900.0, 10_200.0], freq="1D"
    )
    # compute expected sortino using the same formula
    r0 = (10_100.0 - 10_000.0) / 10_000.0
    r1 = ( 9_900.0 - 10_100.0) / 10_100.0
    r2 = (10_200.0 -  9_900.0) /  9_900.0
    returns  = [r0, r1, r2]
    mean_r   = sum(returns) / 3
    d_var    = sum(min(r, 0.0) ** 2 for r in returns) / 3
    d_std    = d_var ** 0.5
    ppy      = 365.25 # 3 daily return periods spanning 3 days → empirically 365.25/yr
    expected = round(mean_r / d_std * (ppy ** 0.5), 4)

    results = reporter.get_results()

    assert results["sortino_ratio"] == pytest.approx(expected)


# cagr = 0 when final equity equals initial balance
def test_get_results_cagr_flat():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([INITIAL_BALANCE, INITIAL_BALANCE])

    results = reporter.get_results()

    assert results["cagr"] == pytest.approx(0.0)


# cagr uses wall-clock elapsed time: (final/initial)^(1/years) - 1, years = span / 365.25d
def test_get_results_cagr_known_growth():
    state    = make_state()
    reporter = Reporter(state)
    # 4 hourly snapshots → span = 3h (first to last), not 4 bars
    state.equity_curve = make_equity_curve(
        [10_000.0 + i * 25.0 for i in range(4)]
    )
    final_eq = 10_000.0 + 3 * 25.0  # 10075
    years    = (3 * 3_600) / (365.25 * 86_400)
    expected = round((final_eq / INITIAL_BALANCE) ** (1.0 / years) - 1.0, 6)

    results = reporter.get_results()

    assert results["cagr"] == pytest.approx(expected)


# total loss of capital → cagr = -1.0 (not 0.0, which would read as "flat")
def test_get_results_cagr_ruin():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 5_000.0, 0.0])

    results = reporter.get_results()

    assert results["cagr"] == pytest.approx(-1.0)


# calmar_ratio = 0 when max drawdown is zero
def test_get_results_calmar_no_drawdown():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 10_100.0])

    results = reporter.get_results()

    assert results["calmar_ratio"] == 0.0


# calmar_ratio = cagr / (max_drawdown_pct / 100)
def test_get_results_calmar_matches_formula():
    state    = make_state()
    reporter = Reporter(state)
    # drawdown present: peak=12000, trough=9000 → max_dd_pct=25%
    # final equity = 11000 → compute expected cagr then calmar
    state.equity_curve = make_equity_curve([10_000.0, 12_000.0, 9_000.0, 11_000.0])
    years        = (3 * 3_600) / (365.25 * 86_400)   # 4 hourly snapshots → 3h span
    cagr         = (11_000.0 / INITIAL_BALANCE) ** (1.0 / years) - 1.0
    max_dd_pct   = (12_000.0 - 9_000.0) / 12_000.0 * 100
    expected     = round(cagr / (max_dd_pct / 100.0), 4)

    results = reporter.get_results()

    assert results["calmar_ratio"] == pytest.approx(expected)


# _resample_equity drops empty resampled periods rather than forward-filling
# (ffill would inject spurious zero returns for calendar gaps like weekends).
# Days 3-4 have no snapshots and are dropped, not synthesized.
def test_resample_equity_drops_gaps_no_ffill():
    curve = [
        {"bar": 0, "timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "equity": 10_000.0},
        {"bar": 1, "timestamp": pd.Timestamp("2024-01-02", tz="UTC"), "equity": 10_100.0},
        {"bar": 2, "timestamp": pd.Timestamp("2024-01-05", tz="UTC"), "equity": 10_300.0},
    ]

    returns, periods_per_year = _resample_equity(curve, "1d")

    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx((10_100.0 - 10_000.0) / 10_000.0)
    assert returns.iloc[1] == pytest.approx((10_300.0 - 10_100.0) / 10_100.0)
    assert periods_per_year > 0.0


# fewer than 2 resampled points → empty returns, periods_per_year = 0.0
def test_resample_equity_insufficient_points_returns_empty():
    curve = [{"bar": 0, "timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "equity": 10_000.0}]

    returns, periods_per_year = _resample_equity(curve, "1d")

    assert returns.empty
    assert periods_per_year == 0.0


# an empty equity curve is handled the same as an insufficient one
def test_resample_equity_empty_curve_returns_empty():
    returns, periods_per_year = _resample_equity([], "1d")

    assert returns.empty
    assert periods_per_year == 0.0


# metrics_timeframe reports the resampling target used for sharpe/sortino/return diagnostics
def test_get_results_metrics_timeframe_key():
    state    = make_state()
    reporter = Reporter(state)

    results = reporter.get_results()

    assert results["metrics_timeframe"] == "1d"


# fewer than 2 resampled points (all bars fall in the same calendar day) → guard returns 0.0
def test_get_results_resample_guard_insufficient_points():
    state    = make_state()
    reporter = Reporter(state)
    # 3 hourly bars, all within 2024-01-01 → resamples to a single "1d" bucket
    state.equity_curve = make_equity_curve([10_000.0, 10_100.0, 9_900.0], freq="1h")

    results = reporter.get_results()

    assert results["n_return_periods"] == 0
    assert results["periods_per_year"] == 0.0
    assert results["sharpe_ratio"] == 0.0
    assert results["sortino_ratio"] == 0.0


# periods_per_year is derived empirically from the resampled series' calendar span,
# not a hard-coded 24/7 constant; a gapless daily curve → ~365.25/yr regardless of length
def test_get_results_periods_per_year_empirical_gapless_daily():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0 + i for i in range(10)], freq="1D")

    results = reporter.get_results()

    assert results["n_return_periods"] == 9
    assert results["periods_per_year"] == pytest.approx(365.25)


# regression (no code change): a Mon-Fri equity curve annualizes to ~252/yr because
# _resample_equity already drops empty (weekend) periods rather than forward-filling;
# this locks in that pre-existing behaviour as part of the gap-support feature
def test_get_results_periods_per_year_mon_fri_equity_curve():
    state    = make_state()
    reporter = Reporter(state)
    n      = 60  # 12 trading weeks
    stamps = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    state.equity_curve = [
        {"bar": i, "timestamp": stamps[i], "equity": 10_000.0 + i} for i in range(n)
    ]

    results = reporter.get_results()

    assert results["n_return_periods"] == n - 1
    assert results["periods_per_year"] == pytest.approx(252, rel=0.1)


# years_elapsed is the wall-clock span between first and last equity snapshot
def test_get_results_years_elapsed():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 10_100.0, 10_200.0], freq="1D")

    results = reporter.get_results()

    expected = round((2 * 86_400) / (365.25 * 86_400), 6)
    assert results["years_elapsed"] == pytest.approx(expected)


# return_autocorr_lag1 matches pandas Series.autocorr(lag=1) on the resampled returns
def test_get_results_return_autocorr_lag1():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 10_150.0, 10_300.0], freq="1D"
    )
    r0 = (10_100.0 - 10_000.0) / 10_000.0
    r1 = (10_150.0 - 10_100.0) / 10_100.0
    r2 = (10_300.0 - 10_150.0) / 10_150.0
    expected = round(pd.Series([r0, r1, r2]).autocorr(lag=1), 4)

    results = reporter.get_results()

    assert results["return_autocorr_lag1"] == pytest.approx(expected)


# return_autocorr_lag1 = 0.0 when there are 2 or fewer return periods (undefined)
def test_get_results_return_autocorr_lag1_insufficient_periods():
    state    = make_state()
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 10_100.0], freq="1D")

    results = reporter.get_results()

    assert results["return_autocorr_lag1"] == 0.0


# risk_free_rate = 0.0 (default) reproduces backward-compatible Sharpe (no RF adjustment)
def test_get_results_sharpe_zero_rf_matches_default():
    curve = [10_000.0, 10_100.0, 9_900.0, 10_200.0]
    state_default = make_state()
    state_default.equity_curve = make_equity_curve(curve, freq="1D")
    state_rf_zero = make_state(metrics_config=MetricsConfig(risk_free_rate=0.0))
    state_rf_zero.equity_curve = make_equity_curve(curve, freq="1D")

    results_default = Reporter(state_default).get_results()
    results_rf_zero = Reporter(state_rf_zero).get_results()

    assert results_rf_zero["sharpe_ratio"] == results_default["sharpe_ratio"]


# a positive risk_free_rate lowers Sharpe by exactly the expected per-period amount
def test_get_results_sharpe_with_risk_free_rate():
    rf = 0.05
    mc = MetricsConfig(risk_free_rate=rf)
    state    = make_state(metrics_config=mc)
    reporter = Reporter(state)
    # 3 daily returns → 3 resampled periods, ppy = 365.25 (gapless daily)
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 9_900.0, 10_200.0], freq="1D"
    )
    r0 = (10_100.0 - 10_000.0) / 10_000.0
    r1 = ( 9_900.0 - 10_100.0) / 10_100.0
    r2 = (10_200.0 -  9_900.0) /  9_900.0
    returns = [r0, r1, r2]
    mean_r  = sum(returns) / 3
    var_r   = sum((r - mean_r) ** 2 for r in returns) / 2  # sample std, n-1
    std_r   = var_r ** 0.5
    ppy     = 365.25
    rf_p    = (1.0 + rf) ** (1.0 / ppy) - 1.0
    expected = round((mean_r - rf_p) / std_r * (ppy ** 0.5), 4)

    results = reporter.get_results()

    assert results["sharpe_ratio"] == pytest.approx(expected)


# sortino_target overrides the MAR used for the downside deviation and numerator
def test_get_results_sortino_with_target():
    target = 0.01
    mc = MetricsConfig(sortino_target=target)
    state    = make_state(metrics_config=mc)
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 9_900.0, 10_200.0], freq="1D"
    )
    r0 = (10_100.0 - 10_000.0) / 10_000.0
    r1 = ( 9_900.0 - 10_100.0) / 10_100.0
    r2 = (10_200.0 -  9_900.0) /  9_900.0
    returns = [r0, r1, r2]
    mean_r  = sum(returns) / 3
    d_var   = sum(min(r - target, 0.0) ** 2 for r in returns) / 3
    d_std   = d_var ** 0.5
    ppy     = 365.25
    expected = round((mean_r - target) / d_std * (ppy ** 0.5), 4)

    results = reporter.get_results()

    assert results["sortino_ratio"] == pytest.approx(expected)


# sortino_target=None (default) falls back to the risk-free rate as the MAR
def test_get_results_sortino_target_defaults_to_risk_free_rate():
    rf = 0.03
    mc = MetricsConfig(risk_free_rate=rf, sortino_target=None)
    state    = make_state(metrics_config=mc)
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 9_900.0, 10_200.0], freq="1D"
    )
    ppy  = 365.25
    rf_p = (1.0 + rf) ** (1.0 / ppy) - 1.0

    results = reporter.get_results()

    mc_explicit = MetricsConfig(sortino_target=rf_p)
    state_explicit = make_state(metrics_config=mc_explicit)
    state_explicit.equity_curve = make_equity_curve(
        [10_000.0, 10_100.0, 9_900.0, 10_200.0], freq="1D"
    )
    results_explicit = Reporter(state_explicit).get_results()

    assert results["sortino_ratio"] == pytest.approx(results_explicit["sortino_ratio"])


# metrics_timeframe in MetricsConfig changes the resampling target reported back
def test_get_results_custom_metrics_timeframe():
    mc = MetricsConfig(metrics_timeframe="1h")
    state    = make_state(metrics_config=mc)
    reporter = Reporter(state)
    state.equity_curve = make_equity_curve([10_000.0, 10_100.0, 9_900.0], freq="1h")

    results = reporter.get_results()

    assert results["metrics_timeframe"] == "1h"
    assert results["n_return_periods"] == 2
