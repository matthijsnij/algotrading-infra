"""
================================================================================
RISK MODULE
================================================================================

Stateless risk management and position sizing functions.

Functions:
    calc_pct_stop_loss(): Calculate stop loss price using a percentage-based method.
    calc_pct_take_profit(): Calculate take profit price using a percentage-based method.
    calc_atr_stop_loss(): Calculate stop loss price using an ATR-based method.
    calc_atr_take_profit(): Calculate take profit price using an ATR-based method.
    calc_rr_take_profit(): Calculate take profit price using a fixed reward-to-risk ratio.
    calc_level_stop_loss(): Calculate stop loss price relative to a structural price level.
    calc_level_take_profit(): Calculate take profit price relative to a structural price level.
    calc_size_fixedfractional(): Calculate position size using the fixed-fractional method.
    kelly_risk(): Calculate the Kelly criterion fraction of capital to risk.
    calc_size_kelly(): Calculate position size using the Kelly criterion.
================================================================================
"""

################# IMPORTS ##################
from lighthouse.utils.logging import get_logger
from lighthouse.domain.enums import Side

logger = get_logger(__name__)

################# FUNCTIONS ##################

# ── Stop loss / take profit ──────────────────────────────────────────────────
def calc_pct_stop_loss(entry_price: float, side: Side, sl_pct: float) -> float:
    """
    Calculate the stop loss price using a simple percentage-based method.

    Args:
        entry_price: position entry price
        side:        Side.LONG or Side.SHORT
        sl_pct:      stop loss percentage as a decimal fraction, e.g. 0.02 for 2%

    Returns:
        The stop loss price as a float.
    
    Raises:
        ValueError: If entry_price is not positive, or if sl_pct is not in (0, 1).
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if sl_pct <= 0.0 or sl_pct >= 1.0:
        raise ValueError(f"sl_pct must be in (0, 1), got {sl_pct}")

    if side == Side.LONG:
        price = entry_price * (1 - sl_pct)
    elif side == Side.SHORT:
        price = entry_price * (1 + sl_pct)
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


def calc_pct_take_profit(entry_price: float, side: Side, tp_pct: float) -> float:
    """
    Calculate the take profit price using a simple percentage-based method.

    Args:
        entry_price: position entry price
        side:        Side.LONG or Side.SHORT
        tp_pct:      take profit percentage as a decimal fraction, e.g. 0.04 for 4 %

    Returns:
        The take profit price as a float.

    Raises:
        ValueError: If entry_price is not positive, or if tp_pct is not in (0, 1).
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if tp_pct <= 0.0 or tp_pct >= 1.0:
        raise ValueError(f"tp_pct must be in (0, 1), got {tp_pct}")

    if side == Side.LONG:
        price = entry_price * (1 + tp_pct)
    elif side == Side.SHORT:
        price = entry_price * (1 - tp_pct)
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


def calc_atr_stop_loss(entry_price: float, side: Side, atr: float, atr_mult: float) -> float:
    """
    Calculate the stop loss price using an ATR-based method.

    SL is placed atr_mult * atr away from entry, adapting the distance to
    current market volatility.

    Args:
        entry_price: position entry price
        side:        Side.LONG or Side.SHORT
        atr:         current ATR value in price units
        atr_mult:    multiplier applied to ATR

    Returns:
        The stop loss price as a float.
    
    Raises:
        ValueError: If entry_price, atr, or atr_mult are not positive.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if atr <= 0.0:
        raise ValueError(f"atr must be positive, got {atr}")
    if atr_mult <= 0.0:
        raise ValueError(f"atr_mult must be positive, got {atr_mult}")

    distance = atr_mult * atr

    if side == Side.LONG:
        price = entry_price - distance
    elif side == Side.SHORT:
        price = entry_price + distance
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


def calc_atr_take_profit(entry_price: float, side: Side, atr: float, atr_mult: float) -> float:
    """
    Calculate the take profit price using an ATR-based method.

    TP is placed atr_mult * atr away from entry in the direction of the trade.

    Args:
        entry_price: position entry price
        side:        Side.LONG or Side.SHORT
        atr:         current ATR value in price units
        atr_mult:    multiplier applied to ATR

    Returns:
        The take profit price as a float.

    Raises:
        ValueError: If entry_price, atr, or atr_mult are not positive.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if atr <= 0.0:
        raise ValueError(f"atr must be positive, got {atr}")
    if atr_mult <= 0.0:
        raise ValueError(f"atr_mult must be positive, got {atr_mult}")

    distance = atr_mult * atr

    if side == Side.LONG:
        price = entry_price + distance
    elif side == Side.SHORT:
        price = entry_price - distance
    else:
        raise ValueError(f"Unrecognized side '{side.value}' — expected Side.LONG or Side.SHORT")

    return price


def calc_rr_take_profit(entry_price: float, side: Side, sl_price: float, rr_ratio: float) -> float:
    """
    Calculate the take profit price using a fixed reward-to-risk ratio.

    TP is placed rr_ratio * SL distance away from entry in the direction of
    the trade, guaranteeing a fixed R:R regardless of how the SL was calculated.

    Args:
        entry_price: position entry price
        sl_price:    stop loss price
        side:        Side.LONG or Side.SHORT
        rr_ratio:    reward-to-risk multiplier

    Returns:
        The take profit price as a float.
    
    Raises:
        ValueError: If entry_price, sl_price, or rr_ratio are not positive, or if sl_price equals entry_price.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if sl_price <= 0.0:
        raise ValueError(f"sl_price must be positive, got {sl_price}")
    if rr_ratio <= 0.0:
        raise ValueError(f"rr_ratio must be positive, got {rr_ratio}")

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0.0:
        raise ValueError("sl_price equals entry_price - SL distance is zero")

    if side == Side.LONG:
        price = entry_price + rr_ratio * sl_distance
    elif side == Side.SHORT:
        price = entry_price - rr_ratio * sl_distance
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


def calc_level_stop_loss(entry_price: float, side: Side, level_price: float, buffer: float = 0.0) -> float:
    """
    Calculate the stop loss price relative to a structural price level.

    SL is placed on the far side of level_price, including a buffer.
    
    Determining the appropriate level_price is the
    responsibility of the caller.

    Args:
        entry_price:  position entry price
        side:         Side.LONG or Side.SHORT
        level_price:  price level to target
        buffer:       additional buffer beyond the level (default 0.0)

    Returns:
        The stop loss price as a float.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if level_price <= 0.0:
        raise ValueError(f"level_price must be positive, got {level_price}")
    if buffer < 0.0:
        raise ValueError(f"buffer must be non-negative, got {buffer}")
    if side == Side.LONG and level_price >= entry_price:
        raise ValueError(
            f"For a LONG, SL level_price must be below entry_price, got level={level_price} entry={entry_price}"
        )
    if side == Side.SHORT and level_price <= entry_price:
        raise ValueError(
            f"For a SHORT, SL level_price must be above entry_price, got level={level_price} entry={entry_price}"
        )

    if side == Side.LONG:
        price = level_price - buffer
    elif side == Side.SHORT:
        price = level_price + buffer
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


def calc_level_take_profit(entry_price: float, side: Side, level_price: float, buffer: float = 0.0) -> float:
    """
    Calculate the take profit price relative to a structural price level (e.g. swing high/low).

    TP is placed just before level_price, offset inward by buffer price units.
    The buffer accounts for the possibility that price may not fully reach the level,
    improving the probability of the order being filled.
    Determining the appropriate level_price (e.g. swing high for a long) is the
    responsibility of the caller.

    Args:
        entry_price:  position entry price
        side:         Side.LONG or Side.SHORT
        level_price:  price level to target
        buffer:       additional buffer beyond the level (default 0.0)

    Returns:
        The take profit price as a float.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if level_price <= 0.0:
        raise ValueError(f"level_price must be positive, got {level_price}")
    if buffer < 0.0:
        raise ValueError(f"buffer must be non-negative, got {buffer}")
    if side == Side.LONG and level_price <= entry_price:
        raise ValueError(
            f"For a LONG, TP level_price must be above entry_price, got level={level_price} entry={entry_price}"
        )
    if side == Side.SHORT and level_price >= entry_price:
        raise ValueError(
            f"For a SHORT, TP level_price must be below entry_price, got level={level_price} entry={entry_price}"
        )

    if side == Side.LONG:
        price = level_price - buffer
    elif side == Side.SHORT:
        price = level_price + buffer
    else:
        raise ValueError(f"Unrecognized side '{side.value}' - expected Side.LONG or Side.SHORT")

    return price


# ── Position sizing ──────────────────────────────────────────────────────────
def calc_size_fixedfractional(account_balance: float, risk_pct: float, entry_price: float, sl_price: float) -> float:
    """
    Calculates position size in base currency units, using the fixed-fractional method:

    risk_amount   = account_balance * risk_pct
    sl_distance   = abs(entry_price - sl_price)
    position_size = risk_amount / sl_distance

    Args:
        account_balance: total account balance in quote currency 
        risk_pct:        fraction of balance to risk per trade, e.g. 0.01 for 1%
        entry_price:     expected entry price
        sl_price:        stop loss price

    Returns:
        Position size in base currency units, or 0.0 if sl_price equals entry_price.
        Raises ValueError on invalid inputs.
    """
    if account_balance <= 0.0:
        raise ValueError(f"account_balance must be positive, got {account_balance}")
    if risk_pct <= 0.0 or risk_pct >= 1.0:
        raise ValueError(f"risk_pct must be in (0, 1), got {risk_pct}")
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if sl_price <= 0.0:
        raise ValueError(f"sl_price must be positive, got {sl_price}")

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0.0:
        raise ValueError("sl_price equals entry_price — SL distance is zero")

    risk_amount   = account_balance * risk_pct
    position_size = risk_amount / sl_distance

    logger.debug(
        "Position size calculated | balance=%.2f risk_pct=%.4f risk_amount=%.2f "
        "entry=%.6f sl=%.6f sl_distance=%.6f size=%.6f",
        account_balance, risk_pct, risk_amount,
        entry_price, sl_price, sl_distance, position_size,
    )
    return position_size


def kelly_risk(win_rate: float, payoff_ratio: float, kelly_fraction: float = 1.0) -> float:
    """
    Calculate the Kelly criterion fraction of capital to risk.

    Full Kelly:      f = W - (1 - W) / R
    Fractional Kelly: multiply by `kelly_fraction` to scale down and reduce volatility.

    Args:
        win_rate:         (estimated) historical win rate as a decimal, e.g. 0.55 for 55%
        payoff_ratio:     (estimated) average gain % for winning trades divided by average loss % for losing trades
        kelly_fraction:   scaling factor applied to full Kelly, e.g. 0.5 for half-Kelly
                          (default = 1.0, full Kelly)

    Returns:
        The suggested fraction of capital to risk as a float, clamped to [0, 1].
    
    Raises:
        ValueError: If win_rate is not in (0, 1), payoff_ratio is not positive, or kelly_fraction is not in (0, 1].
    """
    if win_rate <= 0.0 or win_rate >= 1.0:
        raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    if kelly_fraction <= 0.0 or kelly_fraction > 1.0:
        raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")

    full_kelly = win_rate - (1 - win_rate) / payoff_ratio
    result = max(0.0, min(1.0, full_kelly * kelly_fraction)) # clamp to [0, 1]

    logger.debug(
        "Kelly fraction | win_rate=%.4f payoff_ratio=%.4f kelly_fraction=%.4f full_kelly=%.4f result=%.4f",
        win_rate, payoff_ratio, kelly_fraction, full_kelly, result,
    )
    return result


def calc_size_kelly(account_balance: float, win_rate: float, payoff_ratio: float, kelly_fraction: float, entry_price: float, sl_price: float, risk_cap: float = 0.25) -> float:
    """
    Calculates position size in base currency units using the Kelly criterion.

    Derives a risk fraction via kelly_risk(), then delegates to calc_size_fixedfractional().

    Args:
        account_balance:   total account balance in quote currency
        win_rate:          historical win rate as a decimal, e.g. 0.55 for 55%
        payoff_ratio:      (estimated) average gain % for winning trades divided by average loss % for losing trades
        kelly_fraction:    scaling factor applied to full Kelly, e.g. 0.5 for half-Kelly
        entry_price:       expected entry price
        sl_price:          stop loss price
        risk_cap:          maximum fraction of balance allowed to risk (default 0.25), acts as a hard ceiling regardless of Kelly output

    Returns:
        Position size in base currency units. 
    Raises:
        ValueError: If risk cap is not in (0, 1), or if any other inputs are invalid.
    """
    if risk_cap <= 0.0 or risk_cap >= 1.0:
        raise ValueError(f"risk_cap must be in (0, 1), got {risk_cap}")

    risk_pct = kelly_risk(win_rate, payoff_ratio, kelly_fraction)
    risk_pct = min(risk_pct, risk_cap)

    logger.debug(
        "Kelly position size | win_rate=%.4f payoff_ratio=%.4f kelly_risk_pct=%.4f cap=%.4f",
        win_rate, payoff_ratio, risk_pct, risk_cap,
    )
    return calc_size_fixedfractional(account_balance, risk_pct, entry_price, sl_price)
