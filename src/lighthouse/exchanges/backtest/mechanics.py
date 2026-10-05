"""
================================================================================
INSTRUMENT MECHANICS
================================================================================

This file contains two engines:
  - FundingEngine: applies funding charges/credits, delegates rate computation to pluggable funding.
  - LiquidationEngine: updates mark price and checks if position is liquidated.

Functions:
    calc_liquidation_price(): compute the liquidation price for a newly opened position
================================================================================
"""

from __future__ import annotations

############# IMPORTS ############
import uuid
from typing import Callable
from .state import BacktestState

############# PURE FUNCTION ############
def calc_liquidation_price(
    entry_price: float,
    side: str,
    leverage: float,
    maintenance_margin: float,
) -> float | None:
    """
    Compute the liquidation price for a newly opened position.

    Args:
        entry_price        : position entry price
        side               : "long" or "short"
        leverage           : position leverage
        maintenance_margin : minimum margin fraction before liquidation

    Returns:
        The liquidation price as a float.
    """
    if side == "long":
        return entry_price * (1.0 - 1.0 / leverage + maintenance_margin)
    else:
        return entry_price * (1.0 + 1.0 / leverage - maintenance_margin)

################## CLASSES ################
class FundingEngine:
    """
    Generic funding engine which applies funding charges/credits each settlement period.

    Delegates funding rate computation entirely to the BaseFundingModel subclass stored in
    BacktestConfig.funding_model. 

    In case config.funding_model is None, apply_funding() returns immediately (no funding).

    Methods:
        apply_funding(): charge or credit the funding rate every funding_interval_bars base bars
    """

    def __init__(self, state: BacktestState) -> None:
        """
        Constructor. Initialize the FundingEngine with a reference to the shared BacktestState.
        
        Args:
            state : shared BacktestState instance
        """
        self._s = state

    def apply_funding(self) -> None:
        """
        Charge or credit the funding rate every funding_interval_bars base bars.

        Calls model.compute_rate() on every base bar so internal model state evolves
        continuously, even during bars that are not settlement periods.

        Positive effective rate : long pays, short receives.
        Negative effective rate : short pays, long receives.
        Cost / credit = size * mark_price * abs(effective_rate).
        """
        s = self._s

        # Skip if no funding model configured
        if s.config.funding_model is None:
            return
        if not s.instrument.has_funding:
            return

        # Compute effective funding rate for this bar using the configured model
        close = float(s.current_bar["close"])
        effective_rate = s.config.funding_model.compute_rate(
            bar_idx=s.cursor,
            close_price=close,
            base_funding_rate=s.instrument.base_funding_rate,
        )

        s.bars_since_funding += 1
        if s.bars_since_funding < s.instrument.funding_interval_bars:
            # Model state updated above; not yet time to settle
            return
        s.bars_since_funding = 0

        if s.position is None:
            return

        mark         = s.position["mark_price"]
        size         = s.position["size"]
        pos_side     = s.position["side"]
        funding_cost = size * mark * abs(effective_rate)

        pays_funding = (
            (pos_side == "long"  and effective_rate >= 0) or
            (pos_side == "short" and effective_rate <  0)
        )

        if pays_funding:
            s.balance["free"]    -= funding_cost
            s.balance["total"]   -= funding_cost
            s.total_funding_paid += funding_cost
            if s.open_trade is not None:
                s.open_trade["funding_paid"] = s.open_trade.get("funding_paid", 0.0) + funding_cost
        else:
            s.balance["free"]    += funding_cost
            s.balance["total"]   += funding_cost
            s.total_funding_paid -= funding_cost
            if s.open_trade is not None:
                s.open_trade["funding_paid"] = s.open_trade.get("funding_paid", 0.0) - funding_cost


# ── LiquidationEngine ─────────────────────────────────────────────────────────

class LiquidationEngine:
    """
    Handles per-base-bar liquidation checks:
      - Update mark price to current base bar close (step 3 of advance())
      - Check if position breached liquidation price and force-close (step 5)

    Receives a close_position_fn callback (FillEngine.close_position) at
    construction to avoid a circular import between mechanics and fill_engine.

    Methods:
        update_mark_price() : update the open position's mark price to the current bar's close
        check_liquidation()  : force-close the position if mark price has breached the liquidation level
    """

    def __init__(self, state: BacktestState, close_position_fn: Callable) -> None:
        """
        Constructor. Initialize the LiquidationEngine with a reference to the shared BacktestState and a callback to FillEngine.close_position.
        
        Args:
            state             : shared BacktestState instance
            close_position_fn : callback to FillEngine.close_position
        """
        self._s              = state
        self._close_position = close_position_fn

    def update_mark_price(self) -> None:
        """
        Update the open position's mark_price to the current bar's close.
        Called after all order fills (step 3 of advance()).
        """
        s = self._s
        if s.position is None:
            return
        s.position["mark_price"] = float(s.current_bar["close"])

    # ── Liquidation ───────────────────────────────────────────────────────────

    def check_liquidation(self) -> None:
        """
        Force-close the position if mark price has breached the liquidation level.

        Calls the injected close_position_fn (FillEngine.close_position) at the
        liquidation price with taker fee. Cancels all open and queued orders after.
        """
        s = self._s
        if s.position is None or s.liquidation_price is None:
            return

        mark      = s.position["mark_price"]
        pos_side  = s.position["side"]
        triggered = (
            (pos_side == "long"  and mark <= s.liquidation_price) or
            (pos_side == "short" and mark >= s.liquidation_price)
        )

        # Force-close path
        if triggered:
            force_id = str(uuid.uuid4())
            self._close_position(
                s.liquidation_price,
                order_id=force_id,
                is_maker=False,
                close_reason="liquidation",
            )
            s.open_orders.clear()
            s.queued_market_orders.clear()
