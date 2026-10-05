"""
================================================================================
REPORTER
================================================================================

This file contains the Reporter class, which records per-bar equity snapshots and computes summary statistics at the end of the backtest.

Helper functions:
    _parse_timeframe()  : parse a timeframe string into (seconds, pandas resample alias)
    _resample_equity()  : resample the equity curve to a target timeframe and compute
                           simple returns + empirically-derived periods_per_year
================================================================================
"""

######### IMPORTS #########
import pandas as pd
from typing import Any
from lighthouse.domain.timeframe import Timeframe
from .state import BacktestState

######### HELPER FUNCTIONS #########
def _parse_timeframe(timeframe: str) -> tuple[int, str]:
    """
    Parse a timeframe string into (seconds, pandas resample alias).

    Args:
        timeframe : string matching ^(\\d+)([mhdw])$, e.g. "1m", "4h", "1d", "1w"

    Returns:
        (seconds, pandas_alias), e.g. "1d" -> (86400, "1D")

    Raises:
        ValueError: if timeframe does not match the expected pattern.
    """
    try:
        tf = Timeframe.parse(timeframe)
    except ValueError as exc:
        raise ValueError(f"Reporter: unrecognized timeframe {timeframe!r}. {exc}") from exc

    return tf.seconds, tf.pandas_freq


def _resample_equity(
    equity_curve: list[dict[str, Any]],
    metrics_timeframe: str,
) -> tuple[pd.Series, float]:
    """
    Resample the bar-level equity curve to `metrics_timeframe` and compute simple returns.

    Empty resampled periods are dropped (NOT forward-filled), so calendar gaps (e.g.
    weekends) do not inject spurious zero returns. The return series terminates at the
    first non-positive equity value (account ruin); later values are undefined.

    Args:
        equity_curve      : list of {bar, timestamp, equity} snapshots
        metrics_timeframe : timeframe string parsed by _parse_timeframe(), e.g. "1d"

    Returns:
        (returns, periods_per_year). `returns` is a pd.Series of simple returns indexed
        by the resampled timestamps (empty if fewer than 2 usable points).
        `periods_per_year` is derived empirically as
        len(returns) / (calendar span of returns in days / 365.25); 0.0 if empty.
    """
    if len(equity_curve) < 2:
        return pd.Series(dtype=float), 0.0

    _, alias = _parse_timeframe(metrics_timeframe)

    df     = pd.DataFrame(equity_curve)
    series = df.set_index("timestamp")["equity"].resample(alias).last().dropna()

    if len(series) < 2:
        return pd.Series(dtype=float), 0.0

    values  = series.tolist()
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            break # account is wiped out; later returns are undefined
        returns.append((values[i] - prev) / prev)

    if len(returns) < 2:
        return pd.Series(dtype=float), 0.0

    returns_series = pd.Series(returns, index=series.index[1:1 + len(returns)])

    span_days        = (returns_series.index[-1] - series.index[0]).total_seconds() / 86400.0
    periods_per_year = (len(returns_series) / span_days * 365.25) if span_days > 0 else 0.0

    return returns_series, periods_per_year

########## CLASS #########
class Reporter:
    """
    Records per-bar equity snapshots and computes summary statistics.

    Methods:
        record_equity(): append an equity snapshot for the current bar (step 6 of advance())
        get_results(): compute and return the full performance summary dict
    """

    def __init__(self, state: BacktestState) -> None:
        """
        Constructor. Initialize the Reporter with a reference to the shared BacktestState instance.

        Args:
            state : shared BacktestState instance
        """
        self._s = state

    def record_equity(self) -> None:
        """
        Append a per-bar equity snapshot to the equity curve.

        Equity = balance["total"] + unrealized_pnl so the curve reflects true
        account value during open positions.

        Called as step 6 of advance().
        """
        s              = self._s
        unrealized_pnl = 0.0

        if s.position is not None:
            mark  = s.position["mark_price"]
            entry = s.position["entry_price"]
            size  = s.position["size"]
            if s.position["side"] == "long":
                unrealized_pnl = (mark - entry) * size
            else:
                unrealized_pnl = (entry - mark) * size

        s.equity_curve.append({
            "bar":       s.cursor,
            "timestamp": s.get_bar_timestamp(s.cursor),
            "equity":    s.balance["total"] + unrealized_pnl,
        })

    def get_results(self) -> dict[str, Any]:
        """
        Return a summary of backtest performance.

        Returns:
            A dictionary containing the following keys and their corresponding values:

                total_trades            : number of completed trades
                winning_trades          : trades where net_pnl > 0
                losing_trades           : trades where net_pnl <= 0
                win_rate                : winning / total  (0-1)
                gross_pnl               : sum of raw position p&l before fees
                total_fees              : total fees paid (entry + exit)
                net_pnl                 : gross_pnl minus all fees and funding costs
                return_pct              : net_pnl / initial_balance * 100
                avg_win                 : mean net_pnl of winning trades
                avg_loss                : mean net_pnl of losing trades
                profit_factor           : sum(wins) / |sum(losses)|; inf if no losses
                max_drawdown_pct        : maximum equity decline during the run in %
                sharpe_ratio            : annualized Sharpe ratio from returns on the
                                          equity curve resampled to
                                          `metrics_config.metrics_timeframe` (uses sample
                                          std; scales by sqrt(periods_per_year)); excess
                                          return is over the per-period risk-free rate
                                          derived from `metrics_config.risk_free_rate`
                                          (0.0 by default); 0.0 if fewer than 2 resampled
                                          return periods
                sortino_ratio           : annualized Sortino ratio (downside deviation,
                                          population std of min(r - MAR, 0)) on the same
                                          resampled returns. MAR is
                                          `metrics_config.sortino_target` if set, else the
                                          per-period risk-free rate
                metrics_timeframe       : timeframe the equity curve was resampled to for
                                          sharpe/sortino/return diagnostics
                n_return_periods        : number of resampled return observations used for
                                          sharpe/sortino
                periods_per_year        : annualization factor; derived empirically from the
                                          resampled series' calendar span (not a fixed
                                          24/7 assumption), or overridden internally by
                                          walk-forward fold pinning
                return_autocorr_lag1    : lag-1 autocorrelation of the resampled returns;
                                          a high positive value flags that even the
                                          resampled Sharpe may be inflated by serial
                                          correlation. 0.0 if fewer than 3 return periods
                cagr                    : compound annual growth rate as a fraction
                                          (e.g. 0.15 = 15%); uses wall-clock elapsed
                                          time between first and last equity snapshot.
                                          -1.0 if the account was wiped out; inf if the
                                          extrapolation overflows on a very short window
                years_elapsed           : wall-clock span of the equity curve in years
                                          (first to last snapshot)
                calmar_ratio            : cagr / (max_drawdown_pct / 100); 0 if no drawdown
                expectancy              : avg net_pnl per trade (net_pnl / total_trades)
                max_consecutive_losses  : longest streak of trades with net_pnl <= 0
                time_in_market          : fraction of equity-curve bars with an open position
                recovery_factor         : net_pnl / max absolute drawdown in quote currency; inf if no DD
                total_funding_paid      : cumulative net funding cost across all bars
                final_equity            : closing account equity
        """
        s      = self._s
        trades = s.trade_log
        n      = len(trades)

        gross_pnl  = sum(t["pnl"]       for t in trades)
        total_fees = sum(t["total_fees"] for t in trades)
        net_pnl    = sum(t["net_pnl"]   for t in trades)

        wins   = [t for t in trades if t["net_pnl"] >  0]
        losses = [t for t in trades if t["net_pnl"] <= 0]

        win_rate  = len(wins) / n if n > 0 else 0.0
        avg_win   = sum(t["net_pnl"] for t in wins)   / len(wins)   if wins   else 0.0
        avg_loss  = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0.0

        sum_wins   = sum(t["net_pnl"] for t in wins)
        sum_losses = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else float("inf")

        # ── Drawdown (pct + absolute, single pass) ────────────────────────────
        max_dd_pct = 0.0
        max_dd_abs = 0.0
        if s.equity_curve:
            equities = [e["equity"] for e in s.equity_curve]
            peak = equities[0]
            for eq in equities:
                peak      = max(peak, eq)
                dd_abs    = peak - eq
                dd_pct    = (dd_abs / peak * 100.0) if peak > 0 else 0.0
                max_dd_abs = max(max_dd_abs, dd_abs)
                max_dd_pct = max(max_dd_pct, dd_pct)

        # ── Sharpe and Sortino from resampled equity returns ──────────────────
        mc                                   = s.metrics_config
        metrics_timeframe                    = mc.metrics_timeframe
        resampled_returns, periods_per_year  = _resample_equity(s.equity_curve, metrics_timeframe)
        if s.pinned_periods_per_year is not None:
            periods_per_year = s.pinned_periods_per_year  # override with walk-forward pinned value if set
        n_return_periods                    = len(resampled_returns)

        sharpe               = 0.0
        sortino              = 0.0
        return_autocorr_lag1 = 0.0
        if n_return_periods > 1:
            mean_r = resampled_returns.mean()

            # Per-period risk-free rate, derived geometrically from the annual rate
            rf_p = (1.0 + mc.risk_free_rate) ** (1.0 / periods_per_year) - 1.0 if periods_per_year > 0 else 0.0
            mar  = mc.sortino_target if mc.sortino_target is not None else rf_p

            # Sharpe: sample std (divide by n-1); rf shifts the numerator only
            std_r  = resampled_returns.std()
            sharpe = ((mean_r - rf_p) / std_r * (periods_per_year ** 0.5)) if std_r > 0 else 0.0

            # Sortino: population downside deviation relative to the MAR (target)
            downside_var = (resampled_returns - mar).clip(upper=0.0).pow(2).mean()
            downside_std = downside_var ** 0.5
            sortino = ((mean_r - mar) / downside_std * (periods_per_year ** 0.5)) if downside_std > 0 else 0.0

            if n_return_periods > 2:
                autocorr             = resampled_returns.autocorr(lag=1)
                return_autocorr_lag1 = autocorr if pd.notna(autocorr) else 0.0


        # ── Years elapsed (wall-clock span of the equity curve) ───────────────
        years_elapsed = 0.0
        if len(s.equity_curve) >= 2:
            elapsed       = s.equity_curve[-1]["timestamp"] - s.equity_curve[0]["timestamp"]
            years_elapsed = elapsed.total_seconds() / (365.25 * 86400)

        # ── CAGR ──────────────────────────────────────────────────────────────
        cagr = 0.0
        if years_elapsed > 0 and s.initial_balance > 0:
            final_eq = s.equity_curve[-1]["equity"]
            ratio    = final_eq / s.initial_balance
            if ratio <= 0:
                cagr = -1.0
            else:
                try:
                    cagr = ratio ** (1.0 / years_elapsed) - 1.0
                except OverflowError:
                    cagr = float("inf") # gain over a window far shorter than a year

        # ── Calmar = CAGR / max_drawdown (both as fractions) ──────────────────
        calmar = 0.0
        if max_dd_pct > 0:
            calmar = cagr / (max_dd_pct / 100.0)

        # ── Expectancy (avg net_pnl per trade) ────────────────────────────────
        expectancy = net_pnl / n if n > 0 else 0.0

        # ── Max consecutive losses ────────────────────────────────────────────
        max_consec_losses = 0
        consec            = 0
        for t in trades:
            if t["net_pnl"] <= 0:
                consec           += 1
                max_consec_losses = max(max_consec_losses, consec)
            else:
                consec = 0

        # ── Time in market (fraction of bars with open position) ──────────────
        time_in_market = 0.0
        total_bars     = len(s.equity_curve)
        if total_bars > 0:
            bars_in_market = sum(t["exit_bar"] - t["entry_bar"] for t in trades)
            time_in_market = min(bars_in_market / total_bars, 1.0) # timeframe invariant

        # ── Recovery factor (net_pnl / max absolute drawdown) ─────────────────
        recovery_factor = (net_pnl / max_dd_abs) if max_dd_abs > 0 else float("inf")

        return {
            "total_trades":             n,
            "winning_trades":           len(wins),
            "losing_trades":            len(losses),
            "win_rate":                 round(win_rate, 4),
            "gross_pnl":                round(gross_pnl, 4),
            "total_fees":               round(total_fees, 4),
            "net_pnl":                  round(net_pnl, 4),
            "return_pct":               round(net_pnl / s.initial_balance * 100, 4),
            "avg_win":                  round(avg_win, 4),
            "avg_loss":                 round(avg_loss, 4),
            "profit_factor":            round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
            "max_drawdown_pct":         round(max_dd_pct, 4),
            "sharpe_ratio":             round(sharpe, 4),
            "sortino_ratio":            round(sortino, 4),
            "metrics_timeframe":        metrics_timeframe,
            "n_return_periods":         n_return_periods,
            "periods_per_year":         round(periods_per_year, 4),
            "return_autocorr_lag1":     round(return_autocorr_lag1, 4),
            "cagr":                     round(cagr, 6),
            "years_elapsed":            round(years_elapsed, 6),
            "calmar_ratio":             round(calmar, 4),
            "expectancy":               round(expectancy, 4),
            "max_consecutive_losses":   max_consec_losses,
            "time_in_market":           round(time_in_market, 4),
            "recovery_factor":          round(recovery_factor, 4) if recovery_factor != float("inf") else float("inf"),
            "total_funding_paid":       round(s.total_funding_paid, 4),
            "final_equity":             round(s.balance["total"], 4),
        }
