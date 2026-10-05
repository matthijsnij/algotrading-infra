"""
================================================================================
ENUMERATION CLASSES
================================================================================

Shared enumeration types used across both the core layer and the bot layer.

Functions:
    normalize_side() : normalizes exchange side strings to Side.LONG or Side.SHORT
================================================================================
"""

########## IMPORTS ##########

from enum import Enum

########## CLASSES ##########

class Side(Enum):
    """
    Directional side of a trade or position.

    Used whenever the current or intended trade direction is needed or represented, without relying on raw strings.
    """
    FLAT  = "flat"   
    LONG  = "long"    
    SHORT = "short"  

########## FUNCTIONS ##########

# ── Helpers ─────────────────────────────────────────────────────
_SIDE_LONG  = {"long", "buy"}
_SIDE_SHORT = {"short", "sell"}

def normalize_side(side: str) -> Side:
    """
    Normalize exchange side strings to Side.LONG or Side.SHORT.

    Args:
        side: raw side string from exchange position dict

    Returns:
        Side.LONG or Side.SHORT.

    Raises ValueError if the side string is not a recognized value.
    """
    s = side.lower()
    if s in _SIDE_LONG:
        return Side.LONG
    if s in _SIDE_SHORT:
        return Side.SHORT
    raise ValueError(
        f"Unrecognised side string '{side}' — expected one of {_SIDE_LONG | _SIDE_SHORT}"
    )
