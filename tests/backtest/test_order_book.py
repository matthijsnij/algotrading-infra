"""
================================================================================
UNIT TESTS FOR exchanges/backtest/order_book.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/backtest/test_order_book.py -v

To run a specific test function:
    pytest tests/backtest/test_order_book.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.exchanges.backtest.order_book import OrderBook, TYPE_LIMIT, TYPE_STOP, TYPE_STOP_LIMIT
from lighthouse.exchanges.backtest.fill_engine import FillEngine
from lighthouse.exchanges.backtest.state import BacktestState
from lighthouse.exchanges.backtest.config import BacktestConfig
from lighthouse.domain.instruments import InstrumentSpec

################### HELPERS ##########################

TAKER_FEE       = 0.001
MAKER_FEE       = 0.0005
LEVERAGE        = 10.0
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


def make_state(
    can_short:             bool  = True,
    spread:                float = 0.0,
    slippage:              float = 0.0,
    partial_fill_fraction: float = 1.0,
    latency_bars:          int   = 1,
    limit_fill_policy:     str   = "through",
    tick_size:             float = 0.01,
) -> BacktestState:
    """Return a BacktestState at cursor=0 for order book tests."""
    config = BacktestConfig(
        spread                = spread,
        taker_fee             = TAKER_FEE,
        maker_fee             = MAKER_FEE,
        slippage              = slippage,
        latency_bars          = latency_bars,
        partial_fill_fraction = partial_fill_fraction,
        funding_model         = None,
        limit_fill_policy     = limit_fill_policy,
    )
    instrument = InstrumentSpec(
        quote_currency        = "USDT",
        can_short             = can_short,
        leverage              = LEVERAGE,
        maintenance_margin    = 0.05,
        has_liquidation       = False,
        base_funding_rate     = 0.0,
        funding_interval_bars = 8,
        settlement            = "linear",
        tick_size             = tick_size,
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


def make_engines(state: BacktestState) -> tuple[FillEngine, OrderBook]:
    """Return a (FillEngine, OrderBook) pair sharing the same state."""
    fill_engine = FillEngine(state)
    book        = OrderBook(state, fill_engine)
    return fill_engine, book


def set_bar(
    state:  BacktestState,
    open_:  float,
    high:   float,
    low:    float,
    close:  float  = 100.0,
    volume: float  = 1000.0,
) -> None:
    """Override OHLCV values at the current cursor row for targeted trigger testing."""
    idx = state.cursor
    state.df.loc[idx, "open"]   = open_
    state.df.loc[idx, "high"]   = high
    state.df.loc[idx, "low"]    = low
    state.df.loc[idx, "close"]  = close
    state.df.loc[idx, "volume"] = volume


def inject_long_position(state: BacktestState, entry_price: float = 100.0, size: float = 1.0) -> None:
    """Directly inject an open long position and matching open_trade into state."""
    margin = (size * entry_price) / state.instrument.leverage
    state.position = {
        "symbol":      state.symbol,
        "side":        "long",
        "size":        size,
        "entry_price": entry_price,
        "mark_price":  entry_price,
        "leverage":    state.instrument.leverage,
    }
    state.open_trade = {
        "symbol":       state.symbol,
        "side":         "long",
        "entry_price":  entry_price,
        "size":         size,
        "entry_bar":    state.cursor,
        "entry_ts":     state.get_bar_timestamp(state.cursor),
        "entry_fee":    0.0,
        "funding_paid": 0.0,
    }
    state.balance["free"] -= margin
    state.balance["used"] += margin

################### TESTS ##########################

# ── queue_market_order() ────────────────────────────────────────────────────────

# fill_at_bar = cursor + latency_bars
def test_queue_market_order_fill_at_bar():
    state        = make_state(latency_bars=2)
    state.cursor = 3
    _, book      = make_engines(state)

    order = book.queue_market_order("BTC/USDT", "buy", 1.0)

    assert order["fill_at_bar"] == 5  # 3 + 2


# raises when can_short=False and selling while flat (no open long to close)
def test_queue_market_order_raises_when_cannot_short():
    state   = make_state(can_short=False)
    _, book = make_engines(state)

    with pytest.raises(ValueError):
        book.queue_market_order("BTC/USDT", "sell", 1.0)

# ── fill_queued_market_orders() ────────────────────────────────────────────────────────

# buy fills at open * (1 + spread + slippage)
def test_fill_queued_market_orders_buy_fill_price():
    state   = make_state(spread=0.001, slippage=0.001)
    _, book = make_engines(state)
    set_bar(state, open_=100.0, high=101.0, low=99.0)

    order = book.queue_market_order("BTC/USDT", "buy", 1.0)
    order["fill_at_bar"] = 0  # force fill immediately at cursor=0
    book.fill_queued_market_orders()

    # fill_price = 100.0 * (1 + 0.001 + 0.001) = 100.2
    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(100.0 * 1.002)


# sell fills at open * (1 - spread - slippage)
def test_fill_queued_market_orders_sell_fill_price():
    state   = make_state(spread=0.001, slippage=0.001)
    _, book = make_engines(state)
    set_bar(state, open_=100.0, high=101.0, low=99.0)

    order = book.queue_market_order("BTC/USDT", "sell", 1.0)
    order["fill_at_bar"] = 0
    book.fill_queued_market_orders()

    # fill_price = 100.0 * (1 - 0.001 - 0.001) = 99.8
    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(100.0 * 0.998)


# order with fill_at_bar > cursor stays queued (latency not yet elapsed)
def test_fill_queued_market_orders_respects_latency():
    state   = make_state(latency_bars=1)
    _, book = make_engines(state)

    book.queue_market_order("BTC/USDT", "buy", 1.0)  # fill_at_bar=1, cursor=0
    book.fill_queued_market_orders()

    assert len(state.queued_market_orders) == 1   # still pending
    assert state.position                  is None

# ── register_order() ────────────────────────────────────────────────────────

# is_entry=True when flat, is_entry=False when position is open
def test_register_order_is_entry_flag():
    state   = make_state()
    _, book = make_engines(state)

    order_flat = book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=95.0)
    assert order_flat["is_entry"] is True

    inject_long_position(state)
    order_with_position = book.register_order("BTC/USDT", "sell", TYPE_STOP, size=1.0, stop_price=90.0)
    assert order_with_position["is_entry"] is False


# unsupported order_type raises ValueError
def test_register_order_invalid_type_raises():
    state   = make_state()
    _, book = make_engines(state)

    with pytest.raises(ValueError, match="unsupported order_type"):
        book.register_order("BTC/USDT", "buy", "market", size=1.0)

# ── check_pending_orders() ────────────────────────────────────────────────────────

# buy stop triggers when high >= stop_price; gap fill: max(stop, open) * (1 + slippage)
def test_check_pending_orders_buy_stop_gap_protection():
    state   = make_state(slippage=0.001)
    _, book = make_engines(state)
    # Bar gaps up above stop: open=108, high=115, stop=105 → fill=max(105,108)*1.001
    set_bar(state, open_=108.0, high=115.0, low=107.0)
    book.register_order("BTC/USDT", "buy", TYPE_STOP, size=1.0, stop_price=105.0)

    book.check_pending_orders()

    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(108.0 * 1.001)


# sell stop triggers when low <= stop_price; gap fill: min(stop, open) * (1 - slippage)
def test_check_pending_orders_sell_stop_gap_protection():
    state   = make_state(slippage=0.001)
    _, book = make_engines(state)
    inject_long_position(state)
    # Bar gaps down below stop: open=92, low=88, stop=95 → fill=min(95,92)*0.999
    set_bar(state, open_=92.0, high=93.0, low=88.0)
    book.register_order("BTC/USDT", "sell", TYPE_STOP, size=1.0, stop_price=95.0)

    book.check_pending_orders()

    assert state.position          is None
    assert state.trade_log[0]["exit_price"] == pytest.approx(92.0 * 0.999)


# buy limit triggers when low <= limit_price; fills at limit_price (maker)
def test_check_pending_orders_buy_limit_triggers():
    state   = make_state()
    _, book = make_engines(state)
    set_bar(state, open_=100.0, high=101.0, low=95.0)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=97.0)

    book.check_pending_orders()

    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(97.0)


# sell limit triggers when high >= limit_price; fills at limit_price (maker)
def test_check_pending_orders_sell_limit_triggers():
    state   = make_state()
    _, book = make_engines(state)
    inject_long_position(state)
    set_bar(state, open_=100.0, high=115.0, low=99.0)
    book.register_order("BTC/USDT", "sell", TYPE_LIMIT, size=1.0, price=110.0)

    book.check_pending_orders()

    assert state.position              is None
    assert state.trade_log[0]["exit_price"] == pytest.approx(110.0)


# limit order skipped when bar_volume * partial_fill_fraction < order_size
def test_check_pending_orders_partial_fill_guard_skips():
    state   = make_state(partial_fill_fraction=0.5)
    _, book = make_engines(state)
    # volume=150, fraction=0.5 → available=75 < order_size=100 → skip
    set_bar(state, open_=100.0, high=115.0, low=95.0, volume=150.0)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=100.0, price=97.0)

    book.check_pending_orders()

    assert state.position             is None   # order not filled
    assert len(state.open_orders)     == 1      # still registered


# ── check_pending_orders() — limit fill policy ────────────────────────────────

# through policy: buy limit does NOT fill when low exactly equals limit_price (touches, not through)
def test_check_pending_orders_buy_limit_through_no_fill_on_touch():
    state   = make_state(limit_fill_policy="through", tick_size=1.0)
    _, book = make_engines(state)
    # low == limit exactly → low < limit - tick (97 - 1 = 96) is False → no fill
    set_bar(state, open_=100.0, high=101.0, low=97.0)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=97.0)

    book.check_pending_orders()

    assert state.position    is None
    assert len(state.open_orders) == 1  # order still resting


# through policy: buy limit fills when low trades THROUGH limit by more than one tick
def test_check_pending_orders_buy_limit_through_fills_on_breach():
    state   = make_state(limit_fill_policy="through", tick_size=1.0)
    _, book = make_engines(state)
    # low=95.5 < limit - tick = 97 - 1 = 96 → fills at limit_price
    set_bar(state, open_=100.0, high=101.0, low=95.5)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=97.0)

    book.check_pending_orders()

    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(97.0)


# through policy: sell limit does NOT fill when high exactly equals limit_price (touches, not through)
def test_check_pending_orders_sell_limit_through_no_fill_on_touch():
    state   = make_state(limit_fill_policy="through", tick_size=1.0)
    _, book = make_engines(state)
    inject_long_position(state)
    # high == limit exactly → high > limit + tick (110 + 1 = 111) is False → no fill
    set_bar(state, open_=100.0, high=110.0, low=99.0)
    book.register_order("BTC/USDT", "sell", TYPE_LIMIT, size=1.0, price=110.0)

    book.check_pending_orders()

    assert state.position is not None  # position still open
    assert len(state.open_orders) == 1


# through policy: sell limit fills when high trades THROUGH limit by more than one tick
def test_check_pending_orders_sell_limit_through_fills_on_breach():
    state   = make_state(limit_fill_policy="through", tick_size=1.0)
    _, book = make_engines(state)
    inject_long_position(state)
    # high=111.5 > limit + tick = 110 + 1 = 111 → fills at limit_price
    set_bar(state, open_=100.0, high=111.5, low=99.0)
    book.register_order("BTC/USDT", "sell", TYPE_LIMIT, size=1.0, price=110.0)

    book.check_pending_orders()

    assert state.position is None
    assert state.trade_log[0]["exit_price"] == pytest.approx(110.0)


# touch policy (legacy): buy limit fills when low == limit_price (price touches)
def test_check_pending_orders_buy_limit_touch_fills_on_exact_touch():
    state   = make_state(limit_fill_policy="touch", tick_size=1.0)
    _, book = make_engines(state)
    # low == limit exactly → low <= limit is True → fills
    set_bar(state, open_=100.0, high=101.0, low=97.0)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=97.0)

    book.check_pending_orders()

    assert state.position                is not None
    assert state.position["entry_price"] == pytest.approx(97.0)


# touch policy (legacy): sell limit fills when high == limit_price (price touches)
def test_check_pending_orders_sell_limit_touch_fills_on_exact_touch():
    state   = make_state(limit_fill_policy="touch", tick_size=1.0)
    _, book = make_engines(state)
    inject_long_position(state)
    # high == limit exactly → high >= limit is True → fills
    set_bar(state, open_=100.0, high=110.0, low=99.0)
    book.register_order("BTC/USDT", "sell", TYPE_LIMIT, size=1.0, price=110.0)

    book.check_pending_orders()

    assert state.position is None
    assert state.trade_log[0]["exit_price"] == pytest.approx(110.0)


# stops processed before limits: when both trigger in the same bar, stop fires first
def test_check_pending_orders_stops_before_limits():
    state   = make_state()
    _, book = make_engines(state)
    inject_long_position(state, entry_price=100.0)

    # Both are exit orders (is_entry=False since position is open)
    # Stop sell at 95 (SL) — triggers when low<=95
    # Limit sell at 110 (TP) — triggers when high>=110
    # Bar: low=90 and high=115 → both would trigger
    set_bar(state, open_=100.0, high=115.0, low=90.0)
    book.register_order("BTC/USDT", "sell", TYPE_STOP,  size=1.0, stop_price=95.0)
    book.register_order("BTC/USDT", "sell", TYPE_LIMIT, size=1.0, price=110.0)

    book.check_pending_orders()

    # Stop closed position → limit became an orphaned exit → discarded silently
    # Only 1 trade log entry proves stop fired and limit was discarded (not both)
    assert len(state.trade_log) == 1

# ── cancel_order() / cancel_all_orders() / fetch_open_orders() ────────────────────────────────────────────────────────

# cancel_order removes from both open_orders and queued_market_orders
def test_cancel_order_removes_from_both():
    state         = make_state()
    _, book       = make_engines(state)
    resting_order = book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=95.0)
    queued_order  = book.queue_market_order("BTC/USDT", "buy", 1.0)

    book.cancel_order(resting_order["id"])
    book.cancel_order(queued_order["id"])

    assert len(state.open_orders)          == 0
    assert len(state.queued_market_orders) == 0


# cancel_all_orders clears both collections
def test_cancel_all_orders_clears_both():
    state   = make_state()
    _, book = make_engines(state)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=95.0)
    book.queue_market_order("BTC/USDT", "buy", 1.0)

    book.cancel_all_orders()

    assert len(state.open_orders)          == 0
    assert len(state.queued_market_orders) == 0


# fetch_open_orders strips is_entry from resting and fill_at_bar from queued orders
def test_fetch_open_orders_strips_internal_fields():
    state   = make_state()
    _, book = make_engines(state)
    book.register_order("BTC/USDT", "buy", TYPE_LIMIT, size=1.0, price=95.0)
    book.queue_market_order("BTC/USDT", "buy", 1.0)

    orders = book.fetch_open_orders()

    for order in orders:
        assert "is_entry"   not in order
        assert "fill_at_bar" not in order
