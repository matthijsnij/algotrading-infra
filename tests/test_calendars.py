"""
================================================================================
UNIT TESTS FOR domain/calendars.py

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_calendars.py -v

To run a specific test function:
    pytest tests/test_calendars.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import pytest
import pandas as pd
from lighthouse.domain.calendars import (
    TradingCalendar,
    CALENDAR_24_7,
    CALENDAR_WEEKDAY,
    CALENDAR_NYSE,
    create_calendar,
    register_calendar,
)

HOUR = pd.Timedelta(hours=1)

################### TESTS ##########################

# ── __post_init__ validation ──────────────────────────────────────────────────

# invalid IANA timezone name raises ValueError
def test_post_init_raises_invalid_timezone():
    with pytest.raises(ValueError, match="timezone"):
        TradingCalendar(name="bad", timezone="Not/A_Zone", weekdays=(0, 1, 2, 3, 4))


# weekday outside 0..6 raises ValueError
def test_post_init_raises_invalid_weekday():
    with pytest.raises(ValueError, match="weekdays"):
        TradingCalendar(name="bad", timezone="UTC", weekdays=(0, 7))


# open_time not in "HH:MM" format raises ValueError
def test_post_init_raises_invalid_time_format():
    with pytest.raises(ValueError, match="HH:MM"):
        TradingCalendar(name="bad", timezone="UTC", weekdays=(0,), open_time="9:30am")


# max_gap_days below 1 raises ValueError
def test_post_init_raises_invalid_max_gap_days():
    with pytest.raises(ValueError, match="max_gap_days"):
        TradingCalendar(name="bad", timezone="UTC", weekdays=(0, 1, 2, 3, 4), max_gap_days=0)


# ── allows_gap() ───────────────────────────────────────────────────────────────

# delta exactly matching the base interval is always legal, for any calendar
def test_allows_gap_one_interval():
    prev_ts = pd.Timestamp("2024-01-08 14:30", tz="UTC")
    next_ts = prev_ts + HOUR
    assert CALENDAR_WEEKDAY.allows_gap(prev_ts, next_ts, HOUR) is True


# a Friday-close -> Monday-open gap spans a day boundary onto a trading weekday
def test_allows_gap_weekend():
    prev_ts = pd.Timestamp("2024-01-12 20:30", tz="UTC")  # Friday
    next_ts = pd.Timestamp("2024-01-15 14:30", tz="UTC")  # Monday
    assert CALENDAR_WEEKDAY.allows_gap(prev_ts, next_ts, HOUR) is True


# a wholly-missing trading day (Tuesday) still spans a day boundary onto a
# trading weekday (Wednesday) -> passes silently, as documented (no holiday list)
def test_allows_gap_holiday_shaped_whole_day_skip():
    prev_ts = pd.Timestamp("2024-01-08 20:30", tz="UTC")  # Monday
    next_ts = pd.Timestamp("2024-01-10 14:30", tz="UTC")  # Wednesday (Tuesday missing)
    assert CALENDAR_WEEKDAY.allows_gap(prev_ts, next_ts, HOUR) is True


# a hole inside a single trading day does not span a day boundary -> rejected
def test_allows_gap_rejects_in_session_hole():
    prev_ts = pd.Timestamp("2024-01-08 14:30", tz="UTC")
    next_ts = pd.Timestamp("2024-01-08 16:30", tz="UTC")  # same day, missing a bar
    assert CALENDAR_WEEKDAY.allows_gap(prev_ts, next_ts, HOUR) is False


# next_ts landing on a non-trading weekday (Saturday) is rejected even though
# the gap spans a day boundary
def test_allows_gap_rejects_saturday_bar():
    prev_ts = pd.Timestamp("2024-01-12 20:30", tz="UTC")  # Friday
    next_ts = pd.Timestamp("2024-01-13 14:30", tz="UTC")  # Saturday
    assert CALENDAR_WEEKDAY.allows_gap(prev_ts, next_ts, HOUR) is False


# 24/7 calendar rejects every gap other than the exact base interval, even one
# that spans a UTC day boundary
def test_allows_gap_24_7_rejects_every_gap():
    prev_ts = pd.Timestamp("2024-01-08 23:00", tz="UTC")
    next_ts = pd.Timestamp("2024-01-09 02:00", tz="UTC")  # 3h gap across midnight
    assert CALENDAR_24_7.allows_gap(prev_ts, next_ts, HOUR) is False


# ── allows_gap() — max_gap_days bound ──────────────────────────────────────────

# a structurally legal gap within the max_gap_days bound is accepted
def test_allows_gap_within_max_gap_days_bound():
    bounded = TradingCalendar(name="bounded", timezone="UTC", weekdays=(0, 1, 2, 3, 4), max_gap_days=3)
    prev_ts = pd.Timestamp("2024-01-12 20:00", tz="UTC")  # Friday
    next_ts = pd.Timestamp("2024-01-15 14:00", tz="UTC")  # Monday (3 calendar days later)
    assert bounded.allows_gap(prev_ts, next_ts, HOUR) is True


# a structurally legal gap exceeding the max_gap_days bound is rejected
def test_allows_gap_exceeds_max_gap_days_bound():
    bounded = TradingCalendar(name="bounded", timezone="UTC", weekdays=(0, 1, 2, 3, 4), max_gap_days=3)
    prev_ts = pd.Timestamp("2024-01-10 20:00", tz="UTC")  # Wednesday
    next_ts = pd.Timestamp("2024-01-15 14:00", tz="UTC")  # Monday (5 calendar days later, Thu+Fri missing)
    assert bounded.allows_gap(prev_ts, next_ts, HOUR) is False


# max_gap_days=None (default) leaves structurally legal gaps unbounded
def test_allows_gap_max_gap_days_none_is_unbounded():
    unbounded = TradingCalendar(name="unbounded", timezone="UTC", weekdays=(0, 1, 2, 3, 4))
    prev_ts = pd.Timestamp("2024-01-10 20:00", tz="UTC")  # Wednesday
    next_ts = pd.Timestamp("2024-01-15 14:00", tz="UTC")  # Monday (5 calendar days later)
    assert unbounded.allows_gap(prev_ts, next_ts, HOUR) is True


# 24/7 calendars reject every non-exact gap regardless of max_gap_days (already illegal before the bound applies)
def test_allows_gap_24_7_unaffected_by_max_gap_days():
    bounded_24_7 = TradingCalendar(
        name="bounded_24_7", timezone="UTC", weekdays=(0, 1, 2, 3, 4, 5, 6), max_gap_days=30
    )
    prev_ts = pd.Timestamp("2024-01-08 23:00", tz="UTC")
    next_ts = pd.Timestamp("2024-01-09 02:00", tz="UTC")
    assert bounded_24_7.allows_gap(prev_ts, next_ts, HOUR) is False


# ── is_open() ──────────────────────────────────────────────────────────────────

# timestamp inside the session on a trading weekday is open
def test_is_open_inside_session():
    ts = pd.Timestamp("2024-01-08 15:00", tz="UTC")  # Monday, 10:00 EST
    assert CALENDAR_WEEKDAY.is_open(ts) is True


# timestamp before the session open on a trading weekday is closed
def test_is_open_outside_session():
    ts = pd.Timestamp("2024-01-08 13:00", tz="UTC")  # Monday, 08:00 EST
    assert CALENDAR_WEEKDAY.is_open(ts) is False


# timestamp during session hours but on a Saturday is closed
def test_is_open_weekend():
    ts = pd.Timestamp("2024-01-13 15:00", tz="UTC")  # Saturday, 10:00 EST
    assert CALENDAR_WEEKDAY.is_open(ts) is False


# 24/7 calendar is always open
def test_is_open_24_7_always_open():
    ts = pd.Timestamp("2024-01-13 15:00", tz="UTC")  # Saturday
    assert CALENDAR_24_7.is_open(ts) is True


# ── next_open() ────────────────────────────────────────────────────────────────

# after Friday's close, next_open lands on Monday's session open in UTC
def test_next_open_across_weekend():
    ts = pd.Timestamp("2024-01-12 22:00", tz="UTC")  # Friday, 17:00 EST (after close)
    expected = pd.Timestamp("2024-01-15 14:30", tz="UTC")  # Monday, 09:30 EST
    assert pd.Timestamp(CALENDAR_WEEKDAY.next_open(ts)) == expected


# 24/7 calendar's next_open is the timestamp itself (already open)
def test_next_open_24_7_is_identity():
    ts = pd.Timestamp("2024-01-13 15:00", tz="UTC")
    assert pd.Timestamp(CALENDAR_24_7.next_open(ts)) == ts


# ── session_id() ───────────────────────────────────────────────────────────────

# two timestamps in the same trading session share the same session_id
def test_session_id_consistent_within_session():
    ts1 = pd.Timestamp("2024-01-08 14:30", tz="UTC")
    ts2 = pd.Timestamp("2024-01-08 20:00", tz="UTC")
    assert CALENDAR_WEEKDAY.session_id(ts1) == CALENDAR_WEEKDAY.session_id(ts2)


# timestamps in adjacent trading days have different session_ids
def test_session_id_differs_across_sessions():
    ts1 = pd.Timestamp("2024-01-08 14:30", tz="UTC")
    ts2 = pd.Timestamp("2024-01-09 14:30", tz="UTC")
    assert CALENDAR_WEEKDAY.session_id(ts1) != CALENDAR_WEEKDAY.session_id(ts2)


# 24/7 calendar always returns the same session_id, regardless of timestamp
def test_session_id_24_7_constant():
    ts1 = pd.Timestamp("2024-01-08 00:00", tz="UTC")
    ts2 = pd.Timestamp("2024-06-15 12:00", tz="UTC")
    assert CALENDAR_24_7.session_id(ts1) == CALENDAR_24_7.session_id(ts2) == "24/7"


# ── DST ────────────────────────────────────────────────────────────────────────

# the same UTC clock time (13:30) is before the NY open in January (EST) but
# exactly at the open in July (EDT) -- the session maps to different UTC hours
def test_is_open_dst_shifts_utc_session_hours():
    jan_ts = pd.Timestamp("2024-01-08 13:30", tz="UTC")  # Monday, 08:30 EST
    jul_ts = pd.Timestamp("2024-07-08 13:30", tz="UTC")  # Monday, 09:30 EDT
    assert CALENDAR_WEEKDAY.is_open(jan_ts) is False
    assert CALENDAR_WEEKDAY.is_open(jul_ts) is True


# ── market calendars (pandas_market_calendars) ─────────────────────────────────

# unknown pandas_market_calendars name raises ValueError
def test_post_init_raises_unknown_market():
    with pytest.raises(ValueError, match="invalid market"):
        TradingCalendar(name="bad", timezone="America/New_York", weekdays=(0, 1, 2, 3, 4), market="NOT_A_MARKET")


# Independence Day 2024 (Thursday, a weekday) is a federal holiday -> closed
def test_is_open_market_holiday_is_closed():
    ts = pd.Timestamp("2024-07-04 15:00", tz="UTC")  # Thursday, would-be session hours
    assert CALENDAR_NYSE.is_open(ts) is False


# the day after the holiday (Friday) is a normal open session
def test_is_open_market_day_after_holiday_is_open():
    ts = pd.Timestamp("2024-07-05 15:00", tz="UTC")  # Friday, 11:00 EDT
    assert CALENDAR_NYSE.is_open(ts) is True


# July 3rd 2024 is a scheduled half day (early close at 13:00 EDT / 17:00 UTC);
# a timestamp after that early close is closed even though it's a weekday afternoon
def test_is_open_market_half_day_closes_early():
    ts = pd.Timestamp("2024-07-03 17:30", tz="UTC")  # 13:30 EDT, after the 13:00 EDT early close
    assert CALENDAR_NYSE.is_open(ts) is False


# a gap spanning a holiday weekday (Thu Jul 4) onto the next open session (Fri) is
# legal: no *scheduled* session was skipped, since Jul 4 was never a session
def test_allows_gap_market_across_holiday_is_legal():
    prev_ts = pd.Timestamp("2024-07-03 17:00", tz="UTC")  # Wed, half-day close
    next_ts = pd.Timestamp("2024-07-05 13:30", tz="UTC")  # Fri, session open
    assert CALENDAR_NYSE.allows_gap(prev_ts, next_ts, HOUR) is True


# a gap that skips an actual scheduled session (Fri) is illegal
def test_allows_gap_market_rejects_skipped_session():
    prev_ts = pd.Timestamp("2024-07-03 17:00", tz="UTC")  # Wed, half-day close
    next_ts = pd.Timestamp("2024-07-08 13:30", tz="UTC")  # Mon, skipping Friday's session
    assert CALENDAR_NYSE.allows_gap(prev_ts, next_ts, HOUR) is False


# next_open across a holiday weekday lands on the next scheduled session's open
def test_next_open_market_across_holiday():
    ts = pd.Timestamp("2024-07-04 15:00", tz="UTC")  # Thursday, holiday
    expected = pd.Timestamp("2024-07-05 13:30", tz="UTC")  # Friday session open
    assert pd.Timestamp(CALENDAR_NYSE.next_open(ts)) == expected


# ── Registry ───────────────────────────────────────────────────────────────────

# unknown calendar name raises ValueError listing the valid names
def test_create_calendar_unknown_name_raises():
    with pytest.raises(ValueError, match="not supported"):
        create_calendar("nonexistent_calendar")


# register_calendar makes a new calendar retrievable via create_calendar
def test_register_calendar_round_trip():
    custom = TradingCalendar(name="test_custom", timezone="UTC", weekdays=(0, 1, 2, 3, 4, 5, 6))
    try:
        register_calendar("test_custom", custom)
        assert create_calendar("test_custom") is custom
    finally:
        from lighthouse.domain.calendars import _CALENDAR_REGISTRY
        _CALENDAR_REGISTRY.pop("test_custom", None)
