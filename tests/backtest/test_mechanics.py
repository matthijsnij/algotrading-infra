"""
================================================================================
UNIT TESTS FOR exchanges/backtest/mechanics.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_mechanics.py -v

To run a specific test function:
    pytest tests/backtest/test_mechanics.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.mechanics import calc_liquidation_price, LiquidationEngine, FundingEngine
from lighthouse.exchanges.backtest.funding_models.base import BaseFundingModel
from lighthouse.exchanges.backtest.state import BacktestState
from lighthouse.exchanges.backtest.config import BacktestConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

def make_df(n_bars: int = 10) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with timestamp as a column."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_bars, freq="1h"),
        "open":      [100.0 + i for i in range(n_bars)],
        "high":      [101.0 + i for i in range(n_bars)],
        "low":       [ 99.0 + i for i in range(n_bars)],
        "close":     [100.0 + i for i in range(n_bars)],
        "volume":    [1000.0]   * n_bars,
    })


def make_instrument(leverage: float = 10.0, maintenance_margin: float = 0.05, has_liquidation: bool = True) -> InstrumentSpec:
    """Return a minimal InstrumentSpec with configurable liquidation-relevant fields."""
    return InstrumentSpec(
        quote_currency        = "USDT",
        can_short             = True,
        leverage              = leverage,
        maintenance_margin    = maintenance_margin,
        has_liquidation       = has_liquidation,
        base_funding_rate     = 0.0,
        funding_interval_bars = 8,
        settlement            = "linear",
    )


def make_state(leverage: float = 10.0, maintenance_margin: float = 0.05, has_liquidation: bool = True) -> BacktestState:
    """Return a BacktestState at cursor=0 for liquidation engine tests."""
    config = BacktestConfig(
        spread                = 0.0,
        taker_fee             = 0.0006,
        maker_fee             = 0.0002,
        slippage              = 0.0,
        latency_bars          = 1,
        partial_fill_fraction = 1.0,
        funding_model         = None,
    )
    state = BacktestState(
        df              = make_df(),
        ts_col          = "timestamp",
        symbol          = "BTC/USDT",
        instrument      = make_instrument(leverage=leverage, maintenance_margin=maintenance_margin, has_liquidation=has_liquidation),
        base_timeframe  = "1h",
        initial_balance = 10_000.0,
        config          = config,
    )
    state.cursor = 0  # position at first bar so current_bar is valid
    return state


def open_position(state: BacktestState, side: str, entry_price: float, liq_price: float) -> None:
    """Manually inject an open position and liquidation price into state."""
    state.position = {
        "symbol":      state.symbol,
        "side":        side,
        "size":        1.0,
        "entry_price": entry_price,
        "mark_price":  entry_price,
        "leverage":    state.instrument.leverage,
    }
    state.liquidation_price = liq_price

################### TESTS: LiquidationEngine ##########################

# ── calc_liquidation_price() ────────────────────────────────────────────────────────

# long position: entry * (1 - 1/leverage + maintenance_margin)
def test_calc_liquidation_price_long():
    liq_price = calc_liquidation_price(entry_price=1000.0, side="long", leverage=10.0, maintenance_margin=0.05)
    expected  = 1000.0 * (1.0 - 1.0 / 10.0 + 0.05)  # = 950.0
    assert liq_price == pytest.approx(expected)


# short position: entry * (1 + 1/leverage - maintenance_margin)
def test_calc_liquidation_price_short():
    liq_price = calc_liquidation_price(entry_price=1000.0, side="short", leverage=10.0, maintenance_margin=0.05)
    expected  = 1000.0 * (1.0 + 1.0 / 10.0 - 0.05)  # = 1050.0
    assert liq_price == pytest.approx(expected)

# ── LiquidationEngine.update_mark_price() ────────────────────────────────────────────────────────

# updates position mark_price to current bar close
def test_update_mark_price_with_open_position():
    state      = make_state()
    engine     = LiquidationEngine(state, close_position_fn=lambda *a, **kw: None)
    open_position(state, side="long", entry_price=100.0, liq_price=50.0)

    engine.update_mark_price()

    expected_close = float(state.df.iloc[0]["close"])  # close of bar 0 = 100.0
    assert state.position["mark_price"] == expected_close


# no-op when flat: no error, position stays None
def test_update_mark_price_when_flat():
    state  = make_state()
    engine = LiquidationEngine(state, close_position_fn=lambda *a, **kw: None)

    engine.update_mark_price()  # should not raise

    assert state.position is None

# ── LiquidationEngine.check_liquidation() ────────────────────────────────────────────────────────

# long position: mark_price <= liquidation_price triggers force-close
def test_check_liquidation_long_breach():
    state  = make_state()
    closed = []
    def mock_close(price, order_id, is_maker, close_reason="liquidation"):
        state.position         = None
        state.liquidation_price = None
        closed.append({"price": price, "close_reason": close_reason})

    engine = LiquidationEngine(state, close_position_fn=mock_close)
    open_position(state, side="long", entry_price=1000.0, liq_price=950.0)
    state.position["mark_price"] = 940.0  # below liq price

    engine.check_liquidation()

    assert len(closed) == 1
    assert closed[0]["close_reason"] == "liquidation"
    assert closed[0]["price"] == pytest.approx(950.0)


# short position: mark_price >= liquidation_price triggers force-close
def test_check_liquidation_short_breach():
    state  = make_state()
    closed = []
    def mock_close(price, order_id, is_maker, close_reason="liquidation"):
        state.position          = None
        state.liquidation_price = None
        closed.append({"price": price, "close_reason": close_reason})

    engine = LiquidationEngine(state, close_position_fn=mock_close)
    open_position(state, side="short", entry_price=1000.0, liq_price=1050.0)
    state.position["mark_price"] = 1060.0  # above liq price

    engine.check_liquidation()

    assert len(closed) == 1
    assert closed[0]["close_reason"] == "liquidation"
    assert closed[0]["price"] == pytest.approx(1050.0)


# no-op when flat: close_position_fn is never called
def test_check_liquidation_when_flat():
    state  = make_state()
    closed = []
    engine = LiquidationEngine(state, close_position_fn=lambda *a, **kw: closed.append(1))

    engine.check_liquidation()

    assert closed == []


# no-op when liquidation_price is None (instrument doesn't support liquidation)
def test_check_liquidation_no_liquidation_price():
    state  = make_state(has_liquidation=False)
    closed = []
    engine = LiquidationEngine(state, close_position_fn=lambda *a, **kw: closed.append(1))
    open_position(state, side="long", entry_price=1000.0, liq_price=None)
    state.position["mark_price"] = 1.0  

    engine.check_liquidation()

    assert closed == []


# after liquidation: all open and queued orders are cleared
def test_check_liquidation_clears_orders():
    state  = make_state()
    def mock_close(price, order_id, is_maker, close_reason="liquidation"):
        state.position          = None
        state.liquidation_price = None

    engine = LiquidationEngine(state, close_position_fn=mock_close)
    open_position(state, side="long", entry_price=1000.0, liq_price=950.0)
    state.position["mark_price"] = 940.0

    state.open_orders           = {"order-1": {}, "order-2": {}}
    state.queued_market_orders  = [{"id": "order-3"}]

    engine.check_liquidation()

    assert state.open_orders          == {}
    assert state.queued_market_orders == []

################### HELPERS: FundingEngine ##########################

class FixedRateModel(BaseFundingModel):
    """Minimal funding model that always returns a fixed rate"""
    def __init__(self, rate: float):
        self._rate = rate
    def compute_rate(self, bar_idx, close_price, base_funding_rate) -> float:
        return self._rate


def make_funding_state(rate: float, pos_side: str, funding_interval_bars: int = 1) -> BacktestState:
    """Return a state with a fixed-rate funding model, an open position, and cursor at settlement."""
    model = FixedRateModel(rate)
    config = BacktestConfig(
        spread                = 0.0,
        taker_fee             = 0.0,
        maker_fee             = 0.0,
        slippage              = 0.0,
        latency_bars          = 1,
        partial_fill_fraction = 1.0,
        funding_model         = model,
    )
    instrument = InstrumentSpec(
        quote_currency        = "USDT",
        can_short             = True,
        leverage              = 10.0,
        maintenance_margin    = 0.05,
        has_liquidation       = True,
        base_funding_rate     = 0.01,
        funding_interval_bars = funding_interval_bars,
        settlement            = "linear",
        has_funding           = True,
    )
    state = BacktestState(
        df              = make_df(),
        ts_col          = "timestamp",
        symbol          = "BTC/USDT",
        instrument      = instrument,
        base_timeframe  = "1h",
        initial_balance = 10_000.0,
        config          = config,
    )
    state.cursor = 0
    state.bars_since_funding = funding_interval_bars - 1  # so next bar is settlement
    open_position(state, side=pos_side, entry_price=100.0, liq_price=50.0)
    state.position["mark_price"] = 100.0
    return state


################### TESTS ##########################

# ── FundingEngine.apply_funding() ────────────────────────────────────────────────────────

# no-op when funding_model is None: balance unchanged
def test_apply_funding_no_model():
    state  = make_state()  # funding_model=None by default
    engine = FundingEngine(state)
    open_position(state, side="long", entry_price=100.0, liq_price=50.0)
    initial_total = state.balance["total"]

    engine.apply_funding()

    assert state.balance["total"] == initial_total


# no-op when has_funding is False: guard returns before any settlement
def test_apply_funding_zero_base_rate():
    instrument = InstrumentSpec(
        quote_currency        = "USDT",
        can_short             = True,
        leverage              = 10.0,
        maintenance_margin    = 0.05,
        has_liquidation       = True,
        base_funding_rate     = 0.01,
        funding_interval_bars = 1,
        settlement            = "linear",
        has_funding           = False,      # instrument has no funding, triggers the guard
    )
    config = BacktestConfig(
        spread=0.0, taker_fee=0.0, maker_fee=0.0, slippage=0.0,
        latency_bars=1, partial_fill_fraction=1.0,
        funding_model=FixedRateModel(0.01), # model present but has_funding guard fires first
    )
    state = BacktestState(
        df=make_df(), ts_col="timestamp", symbol="BTC/USDT",
        instrument=instrument, base_timeframe="1h",
        initial_balance=10_000.0, config=config,
    )
    state.cursor = 0
    open_position(state, side="long", entry_price=100.0, liq_price=50.0)
    engine = FundingEngine(state)
    initial_total = state.balance["total"]

    engine.apply_funding()

    assert state.balance["total"] == initial_total


# no-op before settlement interval elapses: balance unchanged mid-interval
def test_apply_funding_before_interval():
    state  = make_funding_state(rate=0.01, pos_side="long", funding_interval_bars=3)
    state.bars_since_funding = 0  # far from settlement, needs 3 bars total
    engine = FundingEngine(state)
    initial_total = state.balance["total"]

    engine.apply_funding()  # bars_since_funding becomes 1, interval is 3 → no settlement

    assert state.balance["total"] == initial_total


# long pays when rate > 0: balance decreases by size * mark * rate
def test_apply_funding_long_pays_positive_rate():
    rate   = 0.01
    state  = make_funding_state(rate=rate, pos_side="long")  # size=1.0, mark=100.0
    engine = FundingEngine(state)
    initial_total = state.balance["total"]
    expected_cost = 1.0 * 100.0 * rate  # = 1.0

    engine.apply_funding()

    assert state.balance["total"]    == pytest.approx(initial_total - expected_cost)
    assert state.total_funding_paid  == pytest.approx(expected_cost)


# long receives when rate < 0: balance increases
def test_apply_funding_long_receives_negative_rate():
    rate   = -0.01
    state  = make_funding_state(rate=rate, pos_side="long")
    engine = FundingEngine(state)
    initial_total  = state.balance["total"]
    expected_credit = 1.0 * 100.0 * abs(rate)  # = 1.0

    engine.apply_funding()

    assert state.balance["total"]   == pytest.approx(initial_total + expected_credit)
    assert state.total_funding_paid == pytest.approx(-expected_credit)  # negative = net received


# short pays when rate < 0: balance decreases
def test_apply_funding_short_pays_negative_rate():
    rate   = -0.01
    state  = make_funding_state(rate=rate, pos_side="short")
    engine = FundingEngine(state)
    initial_total = state.balance["total"]
    expected_cost = 1.0 * 100.0 * abs(rate)  # = 1.0

    engine.apply_funding()

    assert state.balance["total"]   == pytest.approx(initial_total - expected_cost)
    assert state.total_funding_paid == pytest.approx(expected_cost)


# no-op when flat at settlement: interval elapses but position is None → balance unchanged
def test_apply_funding_flat_at_settlement():
    state          = make_funding_state(rate=0.01, pos_side="long")
    state.position = None  # go flat after state setup, before settlement fires
    engine         = FundingEngine(state)
    initial_total  = state.balance["total"]

    engine.apply_funding()

    assert state.balance["total"] == initial_total
