"""
================================================================================
SIGNALS - TRIGGERS
================================================================================

Combinators that turn a level-triggered signal into an edge-triggered one.

Every boolean in domain/signals/ is level-triggered by default: it reports a
condition that persists for as long as it holds (see CONTEXT.md, "Signal").
became_true() reports the moment that condition starts holding instead, by
comparing the predicate against the frame one bar earlier. This only shifts
any predicate back exactly one bar because every domain function reads the
last row of the frame it is handed (docs/adr/0001, issue #10).

No became_false() mirror yet: deliberately deferred until a caller needs one
(issue #12), rather than adding it speculatively.

Functions:
    became_true() : True if predicate holds on the last bar of df but did not
                    one bar earlier
================================================================================
"""

from __future__ import annotations

################# IMPORTS ##################

from typing import Callable
import pandas as pd

################# TRIGGERS ##################

def became_true(predicate: Callable[[pd.DataFrame], bool], df: pd.DataFrame) -> bool:
    """
    Edge-trigger a level-triggered signal.

    Args:
        predicate: A signal function taking an OHLCV DataFrame and returning
                   a bool evaluated on its last row (e.g. is_long_breakout).
        df:        OHLCV DataFrame; predicate is evaluated on df and on
                    df.iloc[:-1] (one bar earlier).

    Returns:
        True if predicate(df) is True and predicate(df.iloc[:-1]) is False.

    Raises:
        ValueError: Propagated from predicate if df.iloc[:-1] is too short for
                    the indicator period predicate relies on.
    """
    return bool(predicate(df) and not predicate(df.iloc[:-1]))
