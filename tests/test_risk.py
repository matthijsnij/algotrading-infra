"""
================================================================================
UNIT TESTS FOR RISK.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_risk.py -v

To run a specific test function:
    pytest tests/test_risk.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
from lighthouse.domain.enums import Side
from lighthouse.domain.risk import (
    calc_pct_stop_loss,
    calc_pct_take_profit,
    calc_atr_stop_loss,
    calc_atr_take_profit,
    calc_rr_take_profit,
    calc_level_stop_loss,
    calc_level_take_profit,
    calc_size_fixedfractional,
    kelly_risk,
    calc_size_kelly,
)

################### TESTS ##########################

# ── calc_pct_stop_loss ────────────────────────────────────────────────────────

# normal case; long
def test_calc_pct_stop_loss_long():
    result = calc_pct_stop_loss(entry_price=100.0, side=Side.LONG, sl_pct=0.02)
    assert result == 98.0

# normal case; short
def test_calc_pct_stop_loss_short():
    result = calc_pct_stop_loss(entry_price=100.0, side=Side.SHORT, sl_pct=0.02)
    assert result == 102.0

# invalid entry price; must be positive
def test_calc_pct_stop_loss_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_pct_stop_loss(entry_price=0.0, side=Side.LONG, sl_pct=0.02)

# invalid SL percentage; must be in (0, 1)
def test_calc_pct_stop_loss_invalid_sl_pct():
    with pytest.raises(ValueError):
        calc_pct_stop_loss(entry_price=100.0, side=Side.LONG, sl_pct=0.0)
    with pytest.raises(ValueError):
        calc_pct_stop_loss(entry_price=100.0, side=Side.LONG, sl_pct=1.0)


# ── calc_pct_take_profit ────────────────────────────────────────────────────────

# normal case; long
def test_calc_pct_take_profit_long():
    result = calc_pct_take_profit(entry_price=100.0, side=Side.LONG, tp_pct=0.05)
    assert result == 105.0

# normal case; short
def test_calc_pct_take_profit_short():
    result = calc_pct_take_profit(entry_price=100.0, side=Side.SHORT, tp_pct=0.05)
    assert result == 95.0

# invalid entry price; must be positive
def test_calc_pct_take_profit_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_pct_take_profit(entry_price=0.0, side=Side.LONG, tp_pct=0.05)

# invalid TP percentage; must be in (0, 1)
def test_calc_pct_take_profit_invalid_tp_pct():
    with pytest.raises(ValueError):
        calc_pct_take_profit(entry_price=100.0, side=Side.LONG, tp_pct=0.0)
    with pytest.raises(ValueError):
        calc_pct_take_profit(entry_price=100.0, side=Side.LONG, tp_pct=1.0)


# ── calc_atr_stop_loss ────────────────────────────────────────────────────────

# normal case; long
def test_calc_atr_stop_loss_long():
    result = calc_atr_stop_loss(entry_price=100.0, side=Side.LONG, atr=3.0, atr_mult=2.0)
    assert result == 94.0

# normal case; short
def test_calc_atr_stop_loss_short():
    result = calc_atr_stop_loss(entry_price=100.0, side=Side.SHORT, atr=3.0, atr_mult=2.0)
    assert result == 106.0

# invalid entry price; must be positive
def test_calc_atr_stop_loss_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_atr_stop_loss(entry_price=0.0, side=Side.LONG, atr=3.0, atr_mult=2.0)

# invalid ATR value; must be positive
def test_calc_atr_stop_loss_invalid_atr_value():
    with pytest.raises(ValueError):
        calc_atr_stop_loss(entry_price=100.0, side=Side.LONG, atr=0.0, atr_mult=2.0)

# invalid ATR multiplier; must be positive
def test_calc_atr_stop_loss_invalid_atr_multiplier():
    with pytest.raises(ValueError):
        calc_atr_stop_loss(entry_price=100.0, side=Side.LONG, atr=3.0, atr_mult=0.0)


# ── calc_atr_take_profit ────────────────────────────────────────────────────────

# normal case; long
def test_calc_atr_take_profit_long():
    result = calc_atr_take_profit(entry_price=100.0, side=Side.LONG, atr=3.0, atr_mult=2.0)
    assert result == 106.0

# normal case; short
def test_calc_atr_take_profit_short():
    result = calc_atr_take_profit(entry_price=100.0, side=Side.SHORT, atr=3.0, atr_mult=2.0)
    assert result == 94.0

# invalid entry price; must be positive
def test_calc_atr_take_profit_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_atr_take_profit(entry_price=0.0, side=Side.LONG, atr=3.0, atr_mult=2.0)

# invalid ATR value; must be positive
def test_calc_atr_take_profit_invalid_atr_value():
    with pytest.raises(ValueError):
        calc_atr_take_profit(entry_price=100.0, side=Side.LONG, atr=0.0, atr_mult=2.0)

# invalid ATR multiplier; must be positive
def test_calc_atr_take_profit_invalid_atr_multiplier():
    with pytest.raises(ValueError):
        calc_atr_take_profit(entry_price=100.0, side=Side.LONG, atr=3.0, atr_mult=0.0)


# ── calc_rr_take_profit ────────────────────────────────────────────────────────

# normal case; long
def test_calc_rr_take_profit_long():
    result = calc_rr_take_profit(entry_price=100.0, side=Side.LONG, sl_price=95.0, rr_ratio=2.0)
    assert result == 110.0

# normal case; short
def test_calc_rr_take_profit_short():
    result = calc_rr_take_profit(entry_price=100.0, side=Side.SHORT, sl_price=105.0, rr_ratio=2.0)
    assert result == 90.0

# invalid entry price; must be positive
def test_calc_rr_take_profit_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_rr_take_profit(entry_price=0.0, side=Side.LONG, sl_price=95.0, rr_ratio=2.0)

# invalid SL price; must be positive
def test_calc_rr_take_profit_invalid_sl_price():
    with pytest.raises(ValueError):
        calc_rr_take_profit(entry_price=100.0, side=Side.LONG, sl_price=0.0, rr_ratio=2.0)

# invalid RR ratio; must be positive
def test_calc_rr_take_profit_invalid_rr_ratio():
    with pytest.raises(ValueError):
        calc_rr_take_profit(entry_price=100.0, side=Side.LONG, sl_price=95.0, rr_ratio=0.0)

# sl price equal to entry price
def test_calc_rr_take_profit_sl_price_equal_entry_price():
    with pytest.raises(ValueError):
        calc_rr_take_profit(entry_price=100.0, side=Side.LONG, sl_price=100.0, rr_ratio=2.0)


# ── calc_level_stop_loss ────────────────────────────────────────────────────────

# normal case; long
def test_calc_level_stop_loss_long():
    result = calc_level_stop_loss(entry_price=100.0, side=Side.LONG, level_price=95.0, buffer = 4.0)
    assert result == 91.0

# normal case; short
def test_calc_level_stop_loss_short():
    result = calc_level_stop_loss(entry_price=100.0, side=Side.SHORT, level_price=105.0, buffer = 4.0)
    assert result == 109.0

# invalid entry price; must be positive
def test_calc_level_stop_loss_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_level_stop_loss(entry_price=0.0, side=Side.LONG, level_price=95.0, buffer = 4.0)

# invalid level price; must be positive
def test_calc_level_stop_loss_invalid_level_price():
    with pytest.raises(ValueError):
        calc_level_stop_loss(entry_price=100.0, side=Side.LONG, level_price=0.0, buffer = 4.0)

# invalid buffer; must be non-negative
def test_calc_level_stop_loss_invalid_buffer():
    with pytest.raises(ValueError):
        calc_level_stop_loss(entry_price=100.0, side=Side.LONG, level_price=95.0, buffer = -0.01)

# invalid prices; short, level price must be above entry price
def test_calc_level_stop_loss_invalid_prices_short():
    with pytest.raises(ValueError):
        calc_level_stop_loss(entry_price=100.0, side=Side.SHORT, level_price=100.0, buffer = 4.0)

# invalid prices; long, level price must be below entry price
def test_calc_level_stop_loss_invalid_prices_long():
    with pytest.raises(ValueError):
        calc_level_stop_loss(entry_price=100.0, side=Side.LONG, level_price=100.0, buffer = 4.0)


# ── calc_level_take_profit ────────────────────────────────────────────────────────

# normal case; long
def test_calc_level_take_profit_long():
    result = calc_level_take_profit(entry_price=100.0, side=Side.LONG, level_price=110.0, buffer = 4.0)
    assert result == 106.0

# normal case; short
def test_calc_level_take_profit_short():
    result = calc_level_take_profit(entry_price=100.0, side=Side.SHORT, level_price=90.0, buffer = 4.0)
    assert result == 94.0

# invalid entry price; must be positive
def test_calc_level_take_profit_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_level_take_profit(entry_price=0.0, side=Side.LONG, level_price=110.0, buffer = 4.0)

# invalid level price; must be positive
def test_calc_level_take_profit_invalid_level_price():
    with pytest.raises(ValueError):
        calc_level_take_profit(entry_price=100.0, side=Side.LONG, level_price=0.0, buffer = 4.0)

# invalid buffer; must be non-negative
def test_calc_level_take_profit_invalid_buffer():
    with pytest.raises(ValueError):
        calc_level_take_profit(entry_price=100.0, side=Side.LONG, level_price=110.0, buffer = -0.01) 

# invalid prices; short, level price must be below entry price
def test_calc_level_take_profit_invalid_prices_short():
    with pytest.raises(ValueError):
        calc_level_take_profit(entry_price=100.0, side=Side.SHORT, level_price=100.0, buffer = 4.0)

# invalid prices; long, level price must be above entry price
def test_calc_level_take_profit_invalid_prices_long():
    with pytest.raises(ValueError):
        calc_level_take_profit(entry_price=100.0, side=Side.LONG, level_price=100.0, buffer = 4.0)


# ── calc_size_fixedfractional ────────────────────────────────────────────────────────

# normal case
def test_calc_size_fixedfractional():
    result = calc_size_fixedfractional(account_balance=10000.0, risk_pct=0.01, entry_price=100.0, sl_price=95.0)
    assert result == 20.0

# invalid account balance; must be positive
def test_calc_size_fixedfractional_invalid_account_balance():
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=0.0, risk_pct=0.01, entry_price=100.0, sl_price=95.0)

# invalid risk percentage; must be in (0, 1)
def test_calc_size_fixedfractional_invalid_risk_pct():
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=10000.0, risk_pct=0.0, entry_price=100.0, sl_price=95.0)
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=10000.0, risk_pct=1.0, entry_price=100.0, sl_price=95.0)

# invalid entry price; must be positive
def test_calc_size_fixedfractional_invalid_entry_price():
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=10000.0, risk_pct=0.01, entry_price=0.0, sl_price=95.0)

# invalid SL price; must be positive and not equal to entry price
def test_calc_size_fixedfractional_invalid_sl_price():
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=10000.0, risk_pct=0.01, entry_price=100.0, sl_price=0.0)
    with pytest.raises(ValueError):
        calc_size_fixedfractional(account_balance=10000.0, risk_pct=0.01, entry_price=100.0, sl_price=100.0)


# ── kelly_risk ────────────────────────────────────────────────────────

# normal case
def test_kelly_risk():
    result = kelly_risk(win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1)
    assert result == 0.25

# normal case with Kelly fraction
def test_kelly_risk_with_fraction():
    result = kelly_risk(win_rate=0.5, payoff_ratio=2.0, kelly_fraction=0.5)
    assert result == 0.125

# clamping - negative Kelly output
def test_kelly_risk_clamping_negative():
    result = kelly_risk(win_rate=0.1, payoff_ratio=2.0, kelly_fraction=1)
    assert result == 0.0

# invalid win rate; must be in (0, 1)
def test_kelly_risk_invalid_win_rate():
    with pytest.raises(ValueError):
        kelly_risk(win_rate=0.0, payoff_ratio=2.0, kelly_fraction=0.5)
    with pytest.raises(ValueError):
        kelly_risk(win_rate=1.0, payoff_ratio=2.0, kelly_fraction=0.5)

# invalid payoff ratio; must be positive
def test_kelly_risk_invalid_rr_ratio():
    with pytest.raises(ValueError):
        kelly_risk(win_rate=0.5, payoff_ratio=0.0, kelly_fraction=0.5)

# invalid Kelly fraction; must be in (0, 1]
def test_kelly_risk_invalid_kelly_fraction():
    with pytest.raises(ValueError):
        kelly_risk(win_rate=0.5, payoff_ratio=2.0, kelly_fraction=0.0)
    with pytest.raises(ValueError):
        kelly_risk(win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1.01)


# ── calc_size_kelly ────────────────────────────────────────────────────────

# normal case
def test_calc_size_kelly():
    result = calc_size_kelly(account_balance=10000.0, win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1, entry_price=100.0, sl_price=95.0, risk_cap=0.25)
    assert result == 500.0

# invalid risk cap; must be in (0, 1)
def test_calc_size_kelly_invalid_risk_cap():
    with pytest.raises(ValueError):
        calc_size_kelly(account_balance=10000.0, win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1, entry_price=100.0, sl_price=95.0, risk_cap=0.0)
    with pytest.raises(ValueError):
        calc_size_kelly(account_balance=10000.0, win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1, entry_price=100.0, sl_price=95.0, risk_cap=1.0)

# risk cap works as ceiling
def test_calc_size_kelly_risk_cap():
    result = calc_size_kelly(account_balance=10000.0, win_rate=0.5, payoff_ratio=2.0, kelly_fraction=1, entry_price=100.0, sl_price=95.0, risk_cap=0.10)
    assert result == 200.0

