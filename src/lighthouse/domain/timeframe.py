"""
================================================================================
TIMEFRAME VALUE OBJECT
================================================================================

The fixed duration of a single price bar, written as a count and a unit
(e.g. "5m", "4h", "1d", "2w"). See CONTEXT.md for the full definition and how
Timeframe relates to Base Timeframe / Metrics Timeframe / Scan Interval.

Open format, not a closed enum: any positive amount of minutes, hours, days or
weeks parses, matching what exchange SDKs accept (ccxt allows "6h", "12h";
Alpaca accepts any amount + unit). Calendar months are rejected because they
have no fixed duration in seconds.

Classes:
    TimeframeUnit : minute/hour/day/week unit of a Timeframe
    Timeframe     : frozen value object; amount + unit, comparable by .seconds
================================================================================
"""

from __future__ import annotations

########## IMPORTS ##########

import re
from dataclasses import dataclass
from enum import Enum

########## CONSTANTS ##########

_PATTERN = re.compile(r"^(\d+)([mhdw])$")
_CALENDAR_MONTH_PATTERN = re.compile(r"^\d+M$")

########## CLASSES ##########

class TimeframeUnit(Enum):
    """Unit of a Timeframe's amount."""
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"
    WEEK = "w"

_UNIT_SECONDS: dict[TimeframeUnit, int] = {
    TimeframeUnit.MINUTE: 60,
    TimeframeUnit.HOUR: 3_600,
    TimeframeUnit.DAY: 86_400,
    TimeframeUnit.WEEK: 604_800,
}

_UNIT_PANDAS_ALIAS: dict[TimeframeUnit, str] = {
    TimeframeUnit.MINUTE: "min",
    TimeframeUnit.HOUR: "h",
    TimeframeUnit.DAY: "D",
    TimeframeUnit.WEEK: "W",
}


@dataclass(frozen=True, eq=False)
class Timeframe:
    """
    The fixed duration of a single price bar.

    `eq`, `hash` and ordering all delegate to `.seconds`, so `Timeframe.parse("60m")`
    equals `Timeframe.parse("1h")`. Explicit dunders are used instead of the dataclass
    defaults because field-order comparison would make `4h < 1d` false, and eq-by-fields
    would not agree with that ordering.

    Attributes:
        amount : positive count, as parsed (e.g. 4 in "4h")
        unit   : TimeframeUnit, as parsed (e.g. TimeframeUnit.HOUR in "4h")

    Methods:
        parse()          : parse a "<amount><unit>" string into a Timeframe
        is_multiple_of() : True if this timeframe's duration is an exact multiple of another's
    """
    amount: int
    unit: TimeframeUnit

    @classmethod
    def parse(cls, raw: str) -> Timeframe:
        """
        Parse a "<amount><unit>" string into a Timeframe.

        Args:
            raw: string matching ^(\\d+)([mhdw])$, lowercase only, e.g. "5m", "4h", "1d", "2w"

        Returns:
            Parsed Timeframe.

        Raises:
            ValueError: `raw` is calendar-month notation (e.g. "1M"), or does not match
                the expected "<amount><unit>" pattern.
        """
        if _CALENDAR_MONTH_PATTERN.match(raw):
            raise ValueError(
                f"Timeframe.parse: '{raw}' is calendar-month notation, which is not "
                "supported -- calendar months have no fixed duration in seconds. "
                "Use days or weeks instead."
            )

        match = _PATTERN.match(raw)
        if not match:
            raise ValueError(
                f"Timeframe.parse: '{raw}' is not a valid timeframe. Expected an integer "
                "amount followed by one of m/h/d/w (lowercase), e.g. '5m', '4h', '1d', '2w'."
            )

        amount, unit_char = match.groups()
        return cls(int(amount), TimeframeUnit(unit_char))

    @property
    def seconds(self) -> int:
        """Duration of this timeframe in seconds."""
        return self.amount * _UNIT_SECONDS[self.unit]

    @property
    def pandas_freq(self) -> str:
        """Pandas resample/frequency alias for this timeframe, e.g. "4h", "1D", "2W"."""
        return f"{self.amount}{_UNIT_PANDAS_ALIAS[self.unit]}"

    def is_multiple_of(self, other: Timeframe) -> bool:
        """
        Check whether this timeframe's duration is an exact multiple of another's.

        Args:
            other: Timeframe to divide by.

        Returns:
            True if `self.seconds` is an exact multiple of `other.seconds`.
        """
        return self.seconds % other.seconds == 0

    def __str__(self) -> str:
        """Preserves the parsed form, e.g. Timeframe.parse("60m") stringifies to "60m"."""
        return f"{self.amount}{self.unit.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.seconds == other.seconds

    def __hash__(self) -> int:
        return hash(self.seconds)

    def __lt__(self, other: Timeframe) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.seconds < other.seconds
