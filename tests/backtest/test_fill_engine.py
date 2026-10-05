"""
================================================================================
UNIT TESTS FOR exchanges/backtest/fill_engine.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_fill_engine.py -v

To run a specific test function:
    pytest tests/backtest/test_fill_engine.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.fill_engine import FillEngine
from lighthouse.exchanges.backtest.state import BacktestState
from lighthouse.exchanges.backtest.config import BacktestConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

# Test constants used to derive exact expected values in assertions
TAKER_FEE  = 0.001   # 0.1%
MAKER_FEE  = 0.0005  # 0.05%
LEVERAGE   = 10.0
INITIAL_BALANCE = 10_000.0

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

def make_state() -> BacktestState:
    """Return a BacktestState at cursor=0 with clean balance for fill engine tests."""
    config = BacktestConfig(
        spread                = 0.0,
        taker_fee             = TAKER_FEE,
        maker_fee             = MAKER_FEE,
        slippage              = 0.0,
        latency_bars          = 1,
        partial_fill_fraction = 1.0,
        funding_model         = None,
    )
    instrument = InstrumentSpec(
        quote_currency        = "USDT",
        can_short             = True,
        leverage              = LEVERAGE,
        maintenance_margin    = 0.05,
        has_liquidation       = False,  # disabled: keeps tests focused on fill logic
        base_funding_rate     = 0.0,
        funding_interval_bars = 8,
        settlement            = "linear",
    )
    state = BacktestState(
        df              = make_df(),
        ts_col          = "timestamp",
        symbol          = "BTC/USDT",
        instrument      = instrument,
        base_timeframe  = "1h",
        initial_balance = INITIAL_BALANCE,
        config          = config,
    )
    state.cursor = 0
    return state

def make_order(side: str = "buy", size: float = 1.0, is_entry: bool = True) -> dict:
    """Return a minimal order dict for routing into FillEngine."""
    return {
        "id":         "test-order-id",
        "status":     "open",
        "symbol":     "BTC/USDT",
        "side":       side,
        "type":       "market",
        "size":       size,
        "price":      None,
        "stop_price": None,
        "is_entry":   is_entry,
    }

def setup_open_position(
    state: BacktestState,
    side: str,
    entry_price: float,
    size: float = 1.0,
    entry_fee: float = 0.0,
    funding_paid: float = 0.0,
) -> None:
    """
    Manually inject an open position into state, adjusting balance to reflect margin used.
    Use this to set up state for close_position tests without going through execute_market_fill.
    """
    margin = (size * entry_price) / state.instrument.leverage
    state.position = {
        "symbol":      state.symbol,
        "side":        side,
        "size":        size,
        "entry_price": entry_price,
        "mark_price":  entry_price,
        "leverage":    state.instrument.leverage,
    }
    state.open_trade = {
        "symbol":       state.symbol,
        "side":         side,
        "entry_price":  entry_price,
        "size":         size,
        "entry_bar":    state.cursor,
        "entry_ts":     state.get_bar_timestamp(state.cursor),
        "entry_fee":    entry_fee,
        "funding_paid": funding_paid,
    }
    state.balance["free"] -= margin
    state.balance["used"] += margin

################### TESTS ##########################

# ── execute_market_fill() ────────────────────────────────────────────────────────

# opens long: correct balance deduction (margin + taker fee), position fields populated
def test_execute_market_fill_opens_long():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="buy", size=1.0)

    # size=1.0, price=100.0 → notional=100, fee=0.1, margin=10, required=10.1
    engine.execute_market_fill(order, fill_price=100.0)

    assert state.position is not None
    assert state.position["side"]        == "long"
    assert state.position["entry_price"] == 100.0
    assert state.position["size"]        == 1.0
    assert state.balance["free"]         == pytest.approx(INITIAL_BALANCE - 10.0 - 0.1)
    assert state.balance["used"]         == pytest.approx(10.0)
    assert state.balance["total"]        == pytest.approx(INITIAL_BALANCE - 0.1)


# opens short: same mechanics, position side is "short"
def test_execute_market_fill_opens_short():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="sell", size=1.0)

    engine.execute_market_fill(order, fill_price=100.0)

    assert state.position is not None
    assert state.position["side"] == "short"


# rejected when free balance < margin + fee: orders_rejected incremented, no position opened
def test_execute_market_fill_rejected_insufficient_margin():
    state                = make_state()
    state.balance["free"] = 5.0  # less than required 10.1
    engine               = FillEngine(state)
    order                = make_order(side="buy", size=1.0)

    engine.execute_market_fill(order, fill_price=100.0)

    assert state.position       is None
    assert state.orders_rejected == 1


# reversal: opposing position closes first, new one opens with opposite side
def test_execute_market_fill_reversal():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="long", entry_price=100.0, size=1.0)
    order  = make_order(side="sell", size=1.0)

    engine.execute_market_fill(order, fill_price=100.0)

    assert state.position is not None
    assert state.position["side"] == "short"
    assert len(state.trade_log)    == 1  # the reversal close was recorded
    assert state.trade_log[0]["close_reason"] == "reversal"

# ── execute_limit_fill() ────────────────────────────────────────────────────────

# uses maker fee instead of taker fee: lower balance deduction
def test_execute_limit_fill_uses_maker_fee():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="buy", size=1.0)

    # size=1.0, price=100.0 → notional=100, maker_fee=0.05 (vs taker 0.1)
    engine.execute_limit_fill(order, fill_price=100.0)

    assert state.balance["total"] == pytest.approx(INITIAL_BALANCE - 0.05)
    assert state.position is not None
    assert state.position["side"] == "long"

# ── close_position() ────────────────────────────────────────────────────────

# long profit: p&l = (exit - entry) * size, balance updated correctly, trade log appended
def test_close_position_long_profit():
    state  = make_state()
    engine = FillEngine(state)
    # Open at 100, entry_fee = 100 * 0.001 = 0.1
    setup_open_position(state, side="long", entry_price=100.0, size=1.0, entry_fee=0.1)
    initial_total = state.balance["total"]

    # Close at 110 (taker): pnl=10, exit_fee=110*0.001=0.11
    engine.close_position(fill_price=110.0, order_id="close-1", is_maker=False)

    trade = state.trade_log[0]
    assert state.position               is None
    assert state.balance["used"]        == pytest.approx(0.0)
    assert state.balance["total"]       == pytest.approx(initial_total + 10.0 - 0.11)
    assert trade["pnl"]                 == pytest.approx(10.0)
    assert trade["exit_fee"]            == pytest.approx(0.11)
    assert trade["net_pnl"]             == pytest.approx(10.0 - 0.1 - 0.11)  # pnl - entry_fee - exit_fee


# long loss: p&l is negative, balance decreases
def test_close_position_long_loss():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="long", entry_price=100.0, size=1.0, entry_fee=0.1)
    initial_total = state.balance["total"]

    # Close at 90: pnl=-10, exit_fee=90*0.001=0.09
    engine.close_position(fill_price=90.0, order_id="close-1", is_maker=False)

    trade = state.trade_log[0]
    assert state.balance["total"] == pytest.approx(initial_total - 10.0 - 0.09)
    assert trade["pnl"]           == pytest.approx(-10.0)


# short profit: p&l = (entry - exit) * size
def test_close_position_short_profit():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="short", entry_price=100.0, size=1.0, entry_fee=0.1)
    initial_total = state.balance["total"]

    # Close at 90 (short profits when price falls): pnl=10, exit_fee=0.09
    engine.close_position(fill_price=90.0, order_id="close-1", is_maker=False)

    trade = state.trade_log[0]
    assert state.balance["total"] == pytest.approx(initial_total + 10.0 - 0.09)
    assert trade["pnl"]           == pytest.approx(10.0)


# net_pnl correctly deducts accumulated funding_paid from the trade
def test_close_position_net_pnl_includes_funding():
    state  = make_state()
    engine = FillEngine(state)
    # funding_paid=2.0 simulates accumulated funding charges during the trade
    setup_open_position(state, side="long", entry_price=100.0, size=1.0, entry_fee=0.1, funding_paid=2.0)

    # Close at 110: pnl=10, entry_fee=0.1, exit_fee=0.11, funding=2.0
    engine.close_position(fill_price=110.0, order_id="close-1", is_maker=False)

    trade = state.trade_log[0]
    assert trade["funding_paid"] == pytest.approx(2.0)
    assert trade["net_pnl"]      == pytest.approx(10.0 - 0.1 - 0.11 - 2.0)


# close_reason is stored in the trade log
def test_close_position_close_reason_stored():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="long", entry_price=100.0, size=1.0)

    engine.close_position(fill_price=100.0, order_id="close-1", is_maker=False, close_reason="liquidation")

    assert state.trade_log[0]["close_reason"] == "liquidation"

# ── fill_pending_order() ────────────────────────────────────────────────────────

# flat + is_entry=True + is_maker=True → routes to execute_limit_fill (maker fee applied)
def test_fill_pending_order_flat_entry_limit():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="buy", size=1.0, is_entry=True)

    engine.fill_pending_order(order, fill_price=100.0, is_maker=True)

    # Maker fee = 0.05 (not taker 0.1) confirms limit fill was used
    assert state.position               is not None
    assert state.balance["total"]       == pytest.approx(INITIAL_BALANCE - 0.05)


# flat + is_entry=True + is_maker=False → routes to execute_market_fill (taker fee applied)
def test_fill_pending_order_flat_entry_market():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="buy", size=1.0, is_entry=True)

    engine.fill_pending_order(order, fill_price=100.0, is_maker=False)

    # Taker fee = 0.1 confirms market fill was used
    assert state.position               is not None
    assert state.balance["total"]       == pytest.approx(INITIAL_BALANCE - 0.1)


# flat + is_entry=False (orphaned exit) → silent discard, no position opened, no error
def test_fill_pending_order_orphaned_exit_discarded():
    state  = make_state()
    engine = FillEngine(state)
    order  = make_order(side="sell", size=1.0, is_entry=False)

    engine.fill_pending_order(order, fill_price=100.0, is_maker=False)  # should not raise

    assert state.position  is None
    assert state.trade_log == []


# position open + is_entry=False → closes position
def test_fill_pending_order_exit_closes_position():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="long", entry_price=100.0, size=1.0)
    order  = make_order(side="sell", size=1.0, is_entry=False)

    engine.fill_pending_order(order, fill_price=100.0, is_maker=False)

    assert state.position  is None
    assert len(state.trade_log) == 1


# position open + is_entry=True → raises ValueError (overlapping entry logic)
def test_fill_pending_order_entry_with_open_position_raises():
    state  = make_state()
    engine = FillEngine(state)
    setup_open_position(state, side="long", entry_price=100.0, size=1.0)
    order  = make_order(side="buy", size=1.0, is_entry=True)

    with pytest.raises(ValueError, match="entry order"):
        engine.fill_pending_order(order, fill_price=100.0, is_maker=False)
