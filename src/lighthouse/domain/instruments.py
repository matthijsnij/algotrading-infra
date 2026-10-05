"""
================================================================================
INSTRUMENT SPECIFICATION
================================================================================

Defines InstrumentSpec, a frozen dataclass that encodes the trading rules and
mechanics of a specific instrument type.

Used by BacktestExchange to configure simulation behaviour without any 
instrument-specific branching inside the engine itself.  

────────────────────────────────────────────────────────────────────────────────
Presets
────────────────────────────────────────────────────────────────────────────────
Ready-made specs are provided for the three most common instrument types (SPOT, FUTURES, PERP_FUTURES).
Use them directly or as a base for customization via dataclasses.replace():

    from dataclasses import replace
    from lighthouse.domain.instruments import PERP_FUTURES

    my_spec = replace(PERP_FUTURES, leverage=10.0, quote_currency="USD")
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import dataclasses

############ DATACLASS ############

@dataclasses.dataclass(frozen=True)
class InstrumentSpec:
    """
    Immutable specification of an instrument's mechanics.

    Attributes:
        quote_currency        : currency the account balance is denominated in
        can_short             : whether a naked sell opens a short position
        leverage              : position-size multiplier applied to margin
        has_liquidation       : whether the exchange enforces forced liquidation when mark price breaches the liquidation level
        maintenance_margin    : fraction of notional held as maintenance margin
        base_funding_rate     : baseline periodic carry rate per funding interval, expressed as a fraction of notional.
        funding_interval_bars : number of bars between funding payments
        settlement            : p&l settlement convention, either "linear" (quote currency) or "inverse" (base currency)
        has_funding           : whether the instrument charges/pays funding 
        tick_size             : minimum price increment for the instrument (used for conservative limit-fill logic)
        calendar              : name of the TradingCalendar which governs session
                                gaps and live trading hours
    """

    quote_currency:        str
    can_short:             bool
    leverage:              float
    has_liquidation:       bool
    maintenance_margin:    float
    base_funding_rate:     float
    funding_interval_bars: int
    settlement:            str
    has_funding:           bool = False
    tick_size:             float = 0.01
    calendar:              str = "24/7"


############ PRESETS ############

SPOT = InstrumentSpec(
    quote_currency        = "USDT",
    can_short             = False,
    leverage              = 1.0,
    has_liquidation       = False,
    maintenance_margin    = 0.005,
    base_funding_rate     = 0.0,
    funding_interval_bars = 8,
    settlement            = "linear",
    has_funding           = False,
    tick_size             = 0.01,
    calendar              = "24/7",
)
"""
Spot market preset.

No leverage, no short selling, no liquidation, no funding.
Suitable for crypto spot, equities, and any cash-settled long-only instrument.
"""

FUTURES = InstrumentSpec(
    quote_currency        = "USDT",
    can_short             = True,
    leverage              = 1.0,
    has_liquidation       = True,
    maintenance_margin    = 0.005,
    base_funding_rate     = 0.0,
    funding_interval_bars = 8,
    settlement            = "linear",
    has_funding           = False,
    tick_size             = 0.01,
    calendar              = "24/7",
)
"""
Dated futures preset.

Short selling and liquidation enabled; no funding (use basis cost externally
if needed).  Leverage defaults to 1.0, override as needed.
"""

PERP_FUTURES = InstrumentSpec(
    quote_currency        = "USDT",
    can_short             = True,
    leverage              = 1.0,
    has_liquidation       = True,
    maintenance_margin    = 0.005,
    base_funding_rate     = 0.0001,
    funding_interval_bars = 8,
    settlement            = "linear",
    has_funding           = True,
    tick_size             = 0.01,
    calendar              = "24/7",
)
"""
Perpetual futures preset.

Short selling, liquidation, and funding all enabled.
base_funding_rate = 0.0001 (0.01 %) per 8-bar interval matches the typical
crypto 8-hour funding cycle on 1-hour bars.
Override leverage and base_funding_rate as needed.
"""
