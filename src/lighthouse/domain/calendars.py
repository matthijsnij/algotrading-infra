"""
================================================================================
TRADING CALENDARS
================================================================================

Defines TradingCalendar, a frozen dataclass encoding when a market trades and
which gaps between consecutive bars are structurally expected (weekends,
holidays, early closes).

Structural calendars (no `market` set) do not contain holiday lists: a gap is legal
purely on structure; it must span a calendar-day boundary in the exchange timezone
and land on a trading weekday. A hole inside a single trading day is always regarded
as a data error. This covers weekends and early closes for free, at the cost of a
wholly-missing trading day passing silently — bounded by `max_gap_days`, which caps
how many missing calendar days a single gap may span regardless of structural
legality. Calendars with `market` set delegate to pandas_market_calendars instead,
so holidays and half-days are recognized exactly.

Functions:
    create_calendar()    : factory to look up a calendar by name
    register_calendar()  : register a new calendar in the registry
================================================================================
"""

from __future__ import annotations

############ IMPORTS ############

import dataclasses
import functools
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

############ DATACLASS ############

@dataclasses.dataclass(frozen=True)
class TradingCalendar:
    """
    Immutable specification of when a market trades.

    Attributes:
        name         : calendar identifier; used in error messages and the registry
        timezone     : IANA timezone name the session is defined in ("UTC" for 24/7)
        weekdays     : local weekdays the market trades on (0=Mon..6=Sun); all seven for 24/7
        open_time    : "HH:MM" local session open; live gate only, unused by the backtester
        close_time   : "HH:MM" local session close; live gate only, unused by the backtester
        max_gap_days : cap on how many missing calendar days a single structurally-legal
                       gap may span (safety net; None means unbounded)
        market       : optional pandas_market_calendars calendar name (e.g. "NYSE"); when
                       set, is_open()/allows_gap()/next_open() use its holiday-accurate
                       schedule instead of the weekdays/open_time/close_time fields

    Methods:
        allows_gap()  : whether a gap between two consecutive bar timestamps is legal
        session_id()  : group key so bars in the same trading session aggregate together
        is_open()     : whether the market is open at a given UTC timestamp (live gate)
        next_open()   : next UTC timestamp at or after which the market is open (live sleep target)
    """

    name:         str
    timezone:     str
    weekdays:     tuple[int, ...]
    open_time:    str | None = None
    close_time:   str | None = None
    max_gap_days: int | None = None
    market:       str | None = None

    def __post_init__(self) -> None:
        """
        Validate calendar configuration invariants.

        Raises:
            ValueError: if timezone is invalid, weekdays are out of range, open/close
                        times are malformed, or `market` names an unknown
                        pandas_market_calendars calendar
        """

        # Validate timezone
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ValueError(f"TradingCalendar '{self.name}': invalid timezone '{self.timezone}': {exc}")

        # Validate weekdays
        for weekday in self.weekdays:
            if weekday not in range(7):
                raise ValueError(
                    f"TradingCalendar '{self.name}': weekdays must be in 0..6 (Mon..Sun), got {weekday}"
                )

        for label, value in (("open_time", self.open_time), ("close_time", self.close_time)):
            # Skip if None, otherwise validate HH:MM format.
            if value is None:
                continue
            _parse_hhmm(value, f"TradingCalendar '{self.name}': {label}")

        if self.max_gap_days is not None and self.max_gap_days < 1:
            raise ValueError(
                f"TradingCalendar '{self.name}': max_gap_days must be >= 1, got {self.max_gap_days}"
            )

        if self.market is not None:
            try:
                _get_market_calendar(self.market)
            except Exception as exc:
                raise ValueError(f"TradingCalendar '{self.name}': invalid market '{self.market}': {exc}")

    # ── Internal helpers ───────────────────────────────────────────────────

    @property
    def _always_open(self) -> bool:
        """True for calendars with no weekday restriction at all (e.g. 24/7)."""
        return set(self.weekdays) == set(range(7)) and self.open_time is None and self.close_time is None

    # ── Gap validation ─────────────────────────────────────────────────────

    def allows_gap(self, prev_ts: pd.Timestamp, next_ts: pd.Timestamp, base_delta: pd.Timedelta) -> bool:
        """
        Return True if the gap between two consecutive bar timestamps is legal.

        Args:
            prev_ts    : timestamp of the earlier bar (UTC)
            next_ts    : timestamp of the later bar (UTC)
            base_delta : the expected (modal) interval between bars

        Returns:
            True when the delta matches base_delta exactly. Otherwise, for `market`
            calendars: True iff no scheduled trading session between prev_ts and
            next_ts was skipped (per the pandas_market_calendars schedule), still
            capped by `max_gap_days`. For structural calendars: True when the gap
            spans a calendar-day boundary in `timezone`, next_ts's local weekday is a
            trading weekday, and the gap does not exceed `max_gap_days` (if set).
        """
        # If the gap is exactly the base_delta, it's always legal; subsequent bars in real time
        if next_ts - prev_ts == base_delta:
            return True

        tz         = ZoneInfo(self.timezone)
        prev_local = pd.Timestamp(prev_ts).tz_convert(tz)
        next_local = pd.Timestamp(next_ts).tz_convert(tz)

        if self.market is not None:
            return self._allows_gap_market(prev_local, next_local)

        # If the market is always open, any gap is illegal
        if self._always_open:
            return False

        # Gap found, check if it spans a calendar-day boundary in the exchange timezone and lands on a trading weekday
        spans_day_boundary = prev_local.date() != next_local.date() # Resolves to True if the gap spans a calendar-day boundary in the exchange timezone
        if not (spans_day_boundary and next_local.weekday() in self.weekdays):
            return False

        # Structurally legal so far; enforce the safety-net bound on missing calendar days, if any
        if self.max_gap_days is not None:
            gap_days = (next_local.date() - prev_local.date()).days
            if gap_days > self.max_gap_days:
                return False

        return True

    def _allows_gap_market(self, prev_local: pd.Timestamp, next_local: pd.Timestamp) -> bool:
        """Holiday-accurate gap check backed by pandas_market_calendars (see allows_gap())."""
        if prev_local.date() == next_local.date():
            return False  # no day boundary spanned -> in-session hole, illegal

        cal = _get_market_calendar(self.market)
        valid_dates = {d.date() for d in cal.valid_days(start_date=prev_local.date(), end_date=next_local.date())}
        if next_local.date() not in valid_dates:
            return False
        if any(prev_local.date() < d < next_local.date() for d in valid_dates):
            return False  # a scheduled session in between was skipped

        if self.max_gap_days is not None:
            gap_days = (next_local.date() - prev_local.date()).days
            if gap_days > self.max_gap_days:
                return False

        return True

    def session_id(self, ts_utc: pd.Timestamp) -> str:
        """
        Return a group key so bars belonging to the same trading session aggregate together.

        For 24/7 markets, there is a single global session, so the session_id is always set to "24/7".
        For non-24/7 markets, assumes one session per calendar day.

        Args:
            ts_utc: UTC timestamp of the bar

        Returns:
            "24/7" for 24/7 markets, otherwise the ISO date string of ts_utc in its local timezone
        """
        if self._always_open:
            return "24/7"
        tz = ZoneInfo(self.timezone)
        return pd.Timestamp(ts_utc).tz_convert(tz).date().isoformat()

    def is_open(self, ts_utc: pd.Timestamp) -> bool:
        """
        Return True if the market is open at ts_utc (live gate).

        Args:
            ts_utc: UTC timestamp to check

        Returns:
            True if the market is open at ts_utc, False otherwise
        """
        if self.market is not None:
            return self._is_open_market(pd.Timestamp(ts_utc))

        if self._always_open:
            return True

        # Convert the UTC timestamp to the market's local timezone and check if it's a trading weekday and within the open/close times
        tz    = ZoneInfo(self.timezone)
        local = pd.Timestamp(ts_utc).tz_convert(tz)
        if local.weekday() not in self.weekdays:
            return False
        if self.open_time is None or self.close_time is None:
            return True

        open_dt  = _at_local_time(local, self.open_time)
        close_dt = _at_local_time(local, self.close_time)
        return open_dt <= local < close_dt

    def _is_open_market(self, ts_utc: pd.Timestamp) -> bool:
        """Holiday/half-day-accurate open check backed by pandas_market_calendars (see is_open())."""
        cal   = _get_market_calendar(self.market)
        day   = ts_utc.normalize()
        sched = cal.schedule(start_date=day - pd.Timedelta(days=1), end_date=day + pd.Timedelta(days=1))
        if sched.empty:
            return False
        in_session = (sched["market_open"] <= ts_utc) & (ts_utc < sched["market_close"])
        return bool(in_session.any())

    def next_open(self, ts_utc: pd.Timestamp) -> datetime:
        """
        Return the next UTC timestamp at or after which the market is open.

        For 24/7 markets, returns ts_utc itself (already open). For `market` calendars,
        returns the next scheduled session open per pandas_market_calendars (holiday and
        half-day aware). Otherwise walks forward day by day (at most one week) to find
        the next trading weekday's open_time.

        Args:
            ts_utc: UTC timestamp to start searching from

        Returns:
            The next UTC timestamp at or after which the market is open
        
        Raises:
            RuntimeError: if no trading session/weekday is found within the search window
        """
        ts_utc = pd.Timestamp(ts_utc)

        if self.market is not None:
            return self._next_open_market(ts_utc)

        # Already open, return ts_utc itself
        if self._always_open or self.is_open(ts_utc):
            return ts_utc.to_pydatetime()

        tz          = ZoneInfo(self.timezone)
        local       = ts_utc.tz_convert(tz)
        open_time   = self.open_time or "00:00"
        candidate   = _at_local_time(local, open_time)

        # If candidate is still in the past relative to local, move to the next day
        if candidate <= local:
            candidate = candidate + pd.Timedelta(days=1)

        for _ in range(8):  # at most one full week forward
            if candidate.weekday() in self.weekdays:
                return candidate.tz_convert("UTC").to_pydatetime()
            candidate = candidate + pd.Timedelta(days=1)

        raise RuntimeError(f"TradingCalendar '{self.name}': no trading weekday found within a week of {ts_utc}")

    def _next_open_market(self, ts_utc: pd.Timestamp) -> datetime:
        """Holiday-accurate next-open search backed by pandas_market_calendars (see next_open())."""
        if self._is_open_market(ts_utc):
            return ts_utc.to_pydatetime()

        cal   = _get_market_calendar(self.market)
        sched = cal.schedule(start_date=ts_utc.normalize(), end_date=ts_utc.normalize() + pd.Timedelta(days=14))
        future_opens = sched["market_open"][sched["market_open"] > ts_utc]
        if future_opens.empty:
            raise RuntimeError(
                f"TradingCalendar '{self.name}': no scheduled session found within two weeks of {ts_utc}"
            )
        return future_opens.iloc[0].to_pydatetime()


############ HELPERS ############

def _parse_hhmm(value: str, context: str) -> tuple[int, int]:
    """Parse and validate an "HH:MM" string, raising ValueError with `context` on failure."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"{context} must be 'HH:MM', got '{value}'")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"{context} must be 'HH:MM', got '{value}'")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{context} must be 'HH:MM' with HH in 0..23 and MM in 0..59, got '{value}'")
    return hour, minute


def _at_local_time(local_ts: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    """Return local_ts's calendar day combined with the given "HH:MM" local time."""
    hour, minute = _parse_hhmm(hhmm, "time")
    return local_ts.replace(hour=hour, minute=minute, second=0, microsecond=0)


@functools.lru_cache(maxsize=None)
def _get_market_calendar(market: str):
    """Return (and cache) a pandas_market_calendars calendar instance for `market`.

    Lazy import so structural calendars (market=None) never touch the dependency.
    """
    import pandas_market_calendars as mcal
    return mcal.get_calendar(market)


############ PRESETS ############

CALENDAR_24_7 = TradingCalendar(
    name       = "24/7",
    timezone   = "UTC",
    weekdays   = (0, 1, 2, 3, 4, 5, 6),
    open_time  = None,
    close_time = None,
)
"""Always-open calendar for crypto markets. Identity case: guarantees backward compatibility."""

CALENDAR_WEEKDAY = TradingCalendar(
    name         = "weekday",
    timezone     = "America/New_York",
    weekdays     = (0, 1, 2, 3, 4),
    open_time    = "09:30",
    close_time   = "16:00",
    max_gap_days = 4,
)
"""Mon-Fri US equities session, 09:30-16:00 America/New_York (DST-aware via zoneinfo)."""

CALENDAR_NYSE = TradingCalendar(
    name         = "nyse",
    timezone     = "America/New_York",
    weekdays     = (0, 1, 2, 3, 4),
    max_gap_days = 4,
    market       = "NYSE",
)
"""
Holiday-accurate NYSE calendar via pandas_market_calendars: recognizes federal
holidays and early-close (half) days exactly, instead of only weekends.
"""


############ REGISTRY ############

_CALENDAR_REGISTRY: dict[str, TradingCalendar] = {
    "24/7":    CALENDAR_24_7,
    "weekday": CALENDAR_WEEKDAY,
    "nyse":    CALENDAR_NYSE,
}

def create_calendar(name: str) -> TradingCalendar:
    """
    Factory function to look up a TradingCalendar by name.

    Args:
        name: the calendar name as in the registry.

    Returns:
        The registered TradingCalendar instance.
    """
    if name not in _CALENDAR_REGISTRY:
        raise ValueError(
            f"Calendar '{name}' not supported. Valid: {list(_CALENDAR_REGISTRY.keys())}"
        )
    return _CALENDAR_REGISTRY[name]

def register_calendar(name: str, calendar: TradingCalendar) -> None:
    """
    Register a new calendar (or override an existing one) in the registry.

    Args:
        name:     the calendar name to register it under
        calendar: the TradingCalendar instance
    """
    _CALENDAR_REGISTRY[name] = calendar
