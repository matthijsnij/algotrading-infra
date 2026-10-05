"""
================================================================================
BACKTESTING RUNNER
================================================================================

Entry point for running a backtest from a config file.

Usage:
    python -m lighthouse.runtime.backtest --config examples/backtest.yaml

Config file (YAML):
    See examples/backtest.yaml for the full schema and defaults.
    Key sections:
        paths: data_dir, logs_dir, results_dir
        logging: log level
        backtest: start_date, end_date, symbol_datasource, initial_balance, warmup_bars
        costs: spread, fees, slippage, latency_bars, fill policy
        funding: funding model (fixed, historical, or ema_ou)
        metrics: sharpe, sortino, risk-free rate settings
        bots: list of bot configs (currently single-bot backtest only)

Functions:
    run_single()            : run one backtest on a pre-loaded DataFrame
    run()                   : load config, data, and run backtest
================================================================================"""

from __future__ import annotations

############ IMPORTS ############
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Any
from lighthouse.exchanges.backtest.exchange import BacktestExchange
from lighthouse.backtest_data.sources.csv import CSVDataSource
from lighthouse.execution.data import get_ohlcv_df
from lighthouse.config.loader import load_backtest_config
from lighthouse.config.convert import config_to_run_config, build_funding_model
from lighthouse.domain.timeframe import Timeframe
from lighthouse.utils.logging import init_logging

############ RUNNER ############

def run(config_path: str) -> dict[str, Any]:
    """
    Load config from YAML file and run a single backtest.

    Steps:
        1. Load BacktestConfig from YAML file
        2. Convert to run_config dict
        3. Load OHLCV data via CSVDataSource
        4. Attach the funding model (needs the loaded data's timestamps)
        5. Call run_single() with the bot_spec and warmup_bars
        6. Print results summary
        7. Return the full result bundle

    Args:
        config_path: Path to backtest.yaml config file

    Returns:
        Result bundle dict from run_single().
    """
    
    # ── 1. Load and validate config ───────────────────────────────────────────
    config_path = Path(config_path)
    config = load_backtest_config(config_path)
    run_config = config_to_run_config(config, config_path.resolve().parent)

    init_logging(
        "lighthouse-backtest",
        level=getattr(logging, run_config["logging_level"]),
        log_dir=run_config["logs_dir"],
    )

    # ── 2. Load data via CSVDataSource ────────────────────────────────────────
    data_source = CSVDataSource(str(run_config["data_dir"]))
    df = data_source.load(
        symbol    = run_config["symbol_datasource"],
        timeframe = run_config["timeframe"],
        start_date     = run_config.get("start_date"),
        end_date = run_config.get("end_date"),
    )

    # ── 3. Attach the funding model now that bar timestamps are known ────────
    run_config["backtest_config"].funding_model = build_funding_model(
        run_config["funding_config"], run_config["data_dir"], df.index,
    )

    # ── 4. Run simulation ─────────────────────────────────────────────────────
    bot_spec = run_config["bot_spec"]
    bundle   = run_single(
        df          = df,
        bot_class   = bot_spec["class"],
        bot_config  = bot_spec["config"],
        run_config  = run_config,
        warmup_bars = run_config["warmup_bars"]
    )

    # ── 5. Print and return ───────────────────────────────────────────────────
    _print_results(bundle["metrics"])
    return bundle

def _print_results(results: dict) -> None:
    """
    Print a formatted summary of backtest results.

    Args:
        results : dict returned by exchange.get_results()
    """
    width = 40
    print("\n" + "=" * width)
    print("  BACKTEST RESULTS")
    print("=" * width)
    for key, value in results.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:<22} {value:>10.4f}")
        else:
            print(f"  {label:<22} {value:>10}")
    print("=" * width + "\n")


############ RUN SINGLE ############

def run_single(
    df: pd.DataFrame,
    bot_class: type,
    bot_config: dict[str, Any],
    run_config: dict[str, Any],
    warmup_bars: int,
    segment_meta: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute one backtest on a pre-loaded DataFrame slice and return a result bundle.

    This is the reusable core used by both the manual run() entrypoint and the
    walk-forward / parameter-sweep harnesses.  Data loading and slicing happen
    before calling this function; run_single() owns only simulation.

    Args:
        df           : OHLCV DataFrame with DatetimeIndex (UTC), columns open/high/low/close/volume.
                       May include a warmup prefix of warmup_bars rows before the live segment.
        bot_class    : bot class to instantiate (must be a subclass of BaseBot)
        bot_config   : bot configuration dict passed to the bot constructor
        run_config   : dict with all run config parameters combined
        warmup_bars  : number of leading bars to advance without calling on_tick(). Lets the bot's
                       indicator history accumulate without opening positions. Warmup bars are
                       excluded from the returned equity_curve and trade_log.
        segment_meta : optional metadata dict about this segment (label, fold index, dates, etc.);
                       auto-populated from df if not provided.
        verbose      : if True, print progress and results to stdout

    Returns:
        A result bundle dict with keys:
            metrics         : performance metrics dict (from Reporter.get_results())
            equity_curve    : list of per-bar {bar, timestamp, equity} dicts (post-warmup only)
            trade_log       : list of completed trade dicts (post-warmup entries only)
            bot_config      : the bot config dict used in this run
            backtest_config : the BacktestConfig instance used in this run
            segment_meta    : metadata about this data segment

    Raises:
        ValueError: bot_config["scan_interval"] is not a whole multiple of the
            base timeframe (run_config["timeframe"]) — see ADR 0001.
    """
    # ── Validate cadence: scan_interval must divide cleanly into base bars ────
    base_seconds  = Timeframe.parse(run_config["timeframe"]).seconds
    scan_interval = float(bot_config["scan_interval"])
    if scan_interval % base_seconds != 0:
        raise ValueError(
            f"run_single(): bot scan_interval ({scan_interval}s) must be a whole "
            f"multiple of the backtest base_timeframe '{run_config['timeframe']}' "
            f"({base_seconds}s), so scan boundaries land on base-bar closes."
        )
    bars_per_scan = int(scan_interval // base_seconds)

    # ── Resolve segment metadata ──────────────────────────────────────────────
    if segment_meta is None:
        live_bars  = max(0, len(df) - warmup_bars)
        live_start = df.index[warmup_bars] if warmup_bars < len(df) else None
        live_end   = df.index[-1]          if len(df) > 0            else None
        segment_meta = {
            "label":       "single_run",
            "start":       live_start,
            "end":         live_end,
            "total_bars":  len(df),
            "warmup_bars": warmup_bars,
            "live_bars":   live_bars,
        }

    # ── Construct exchange ────────────────────────────────────────────────────
    exchange = BacktestExchange(
        df              = df,
        symbol          = run_config["symbol"],
        initial_balance = run_config["initial_balance"],
        instrument      = run_config["instrument"],
        base_timeframe  = run_config["timeframe"],
        config          = run_config["backtest_config"],
        metrics_config  = run_config.get("metrics_config"),
        pinned_periods_per_year = run_config.get("_pinned_periods_per_year"),
    )

    # ── Construct bot ─────────────────────────────────────────────────────────
    bot = bot_class(exchange, bot_config)

    # ── Run the simulation loop ───────────────────────────────────────────────
    total_bars = len(df)

    if verbose:
        label = segment_meta.get("label", "run")
        warmup_note = f", warmup={warmup_bars}" if warmup_bars else ""
        print(f"\nRunning backtest [{label}]: {total_bars} bars "
              f"({run_config['timeframe']} | {run_config['symbol']}){warmup_note}")

    bar = 0
    while exchange.advance():
        bar += 1

        if bar <= warmup_bars:
            # Warmup: advance simulation without invoking the bot so indicator
            # history can accumulate.  No trades open during this period.
            pass
        else:
            # Live segment: poll on scan_interval boundaries, matching live's
            # cadence (BaseBot.run()) instead of gating on signal-bar closes.
            live_bar_idx = bar - warmup_bars - 1
            if live_bar_idx % bars_per_scan == 0:
                ohlcv = get_ohlcv_df(
                    exchange, run_config["symbol"],
                    bot_config["timeframe"], bot_config["limit"],
                )
                bot.on_tick(ohlcv)

        if verbose and bar % 500 == 0:
            pct = bar / total_bars * 100
            print(f"  Progress: {bar}/{total_bars} bars ({pct:.1f}%)")

    if verbose:
        print(f"  Progress: {bar}/{total_bars} bars (100.0%)")

    # ── Finalize ──────────────────────────────────────────────────────────────
    exchange.finalize()

    # ── Build result bundle ───────────────────────────────────────────────────
    metrics      = exchange.get_results()
    equity_curve = [e for e in exchange.equity_curve if e["bar"] >= warmup_bars]
    trade_log    = [t for t in exchange.trade_log    if t["entry_bar"] >= warmup_bars]

    return {
        "metrics":         metrics,
        "equity_curve":    equity_curve,
        "trade_log":       trade_log,
        "bot_config":      bot_config,
        "backtest_config": run_config["backtest_config"],
        "segment_meta":    segment_meta,
    }


####### ENTRY POINT ########
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a backtest from a YAML config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to backtest.yaml config file"
    )
    args = parser.parse_args()
    
    run(args.config)
