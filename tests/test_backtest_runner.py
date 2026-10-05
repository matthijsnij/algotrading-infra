"""
================================================================================
UNIT TESTS FOR runtime/backtest.py — run_single() execution-model cadence

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_backtest_runner.py -v

To run a specific test function:
    pytest tests/test_backtest_runner.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.runtime.backtest import run_single
from lighthouse.bots.base_bot import BaseBot
from lighthouse.bots.base_state import BaseState
from lighthouse.exchanges.backtest.config import BacktestConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

class _RecordingBot(BaseBot):
    """Minimal bot that records the exchange cursor at every on_tick() call."""

    last_instance: "_RecordingBot | None" = None

    def __init__(self, exchange, config):
        super().__init__(exchange, config)
        self.state = BaseState()
        self.tick_cursors: list[int] = []
        _RecordingBot.last_instance = self

    def on_tick(self, df: pd.DataFrame) -> None:
        self.tick_cursors.append(self.exchange.cursor)


def make_df(n_bars: int, freq: str = "5min") -> pd.DataFrame:
    """Return an OHLCV DataFrame with a UTC DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open":   [100.0] * n_bars,
        "high":   [101.0] * n_bars,
        "low":    [99.0]  * n_bars,
        "close":  [100.0] * n_bars,
        "volume": [1000.0] * n_bars,
    }, index=idx)


def make_run_config(timeframe: str = "5m") -> dict:
    return {
        "symbol":          "BTC/USDT:USDT",
        "initial_balance": 10_000.0,
        "instrument": InstrumentSpec(
            quote_currency="USDT", can_short=True, leverage=10.0,
            maintenance_margin=0.05, has_liquidation=False,
            base_funding_rate=0.0, funding_interval_bars=8, settlement="linear",
        ),
        "timeframe": timeframe,
        "backtest_config": BacktestConfig(
            spread=0.0, taker_fee=0.001, maker_fee=0.0005, slippage=0.0,
            latency_bars=1, partial_fill_fraction=1.0, funding_model=None,
        ),
    }


def make_bot_config(timeframe: str = "5m", scan_interval: float = 900, limit: int = 3) -> dict:
    return {
        "symbol":             "BTC/USDT:USDT",
        "timeframe":          timeframe,
        "limit":              limit,
        "scan_interval":      scan_interval,
        "position_interval":  60,
    }

################### TESTS ##########################

# ── cadence: on_tick() fires on scan_interval boundaries, not every bar ──────

# scan_interval (900s) is 3x the 5m (300s) base timeframe → on_tick() fires
# every 3rd base bar, starting immediately with the first live bar
def test_run_single_ticks_on_scan_interval_boundaries():
    df = make_df(12)
    bot_config = make_bot_config(scan_interval=900)

    bundle = run_single(
        df=df, bot_class=_RecordingBot, bot_config=bot_config,
        run_config=make_run_config(), warmup_bars=0, verbose=False,
    )

    assert _RecordingBot.last_instance.tick_cursors == [0, 3, 6, 9]
    assert bundle["metrics"] is not None


# scan_interval == base_timeframe (the common single-bar-per-scan case) ticks
# every bar, matching the old is_signal_bar_close()-gated behaviour
def test_run_single_ticks_every_bar_when_scan_interval_equals_base():
    df = make_df(4)
    bot_config = make_bot_config(scan_interval=300)

    run_single(
        df=df, bot_class=_RecordingBot, bot_config=bot_config,
        run_config=make_run_config(), warmup_bars=0, verbose=False,
    )

    assert _RecordingBot.last_instance.tick_cursors == [0, 1, 2, 3]


# warmup bars never trigger on_tick(); cadence counting starts at the first live bar
def test_run_single_ticks_skip_warmup_bars():
    df = make_df(9)
    bot_config = make_bot_config(scan_interval=900)  # every 3rd bar

    run_single(
        df=df, bot_class=_RecordingBot, bot_config=bot_config,
        run_config=make_run_config(), warmup_bars=3, verbose=False,
    )

    # first live bar is cursor=3; cadence restarts counting from there
    assert _RecordingBot.last_instance.tick_cursors == [3, 6]


# ── validation: scan_interval must be a whole multiple of base_timeframe ─────

def test_run_single_raises_when_scan_interval_not_multiple_of_base_timeframe():
    df = make_df(4)
    bot_config = make_bot_config(scan_interval=400)  # not a multiple of 300s

    with pytest.raises(ValueError, match="scan_interval"):
        run_single(
            df=df, bot_class=_RecordingBot, bot_config=bot_config,
            run_config=make_run_config(), warmup_bars=0, verbose=False,
        )
