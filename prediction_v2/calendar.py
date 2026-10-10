"""
prediction_v2/calendar.py — trading-day arithmetic for snapshot targeting.

Uses utils.market_session.is_trading_day (weekends + the administrator's
`market.holidays` list). If the holiday list is not configured, holidays are
treated as trading days; callers record `holiday_calendar_configured` in the
run configuration so that limitation is visible on every snapshot.
"""
from __future__ import annotations

import datetime as dt

from utils.market_session import is_trading_day

_MAX_SCAN = 15   # longer exchange closures do not occur on NSE


def next_trading_day(day: dt.date, holidays: list[str] | None = None) -> dt.date:
    """First trading day strictly after `day`."""
    d = day
    for _ in range(_MAX_SCAN):
        d += dt.timedelta(days=1)
        if is_trading_day(d, holidays):
            return d
    raise ValueError(f"no trading day within {_MAX_SCAN} days after {day}")


def previous_trading_day(day: dt.date, holidays: list[str] | None = None) -> dt.date:
    """Last trading day strictly before `day`."""
    d = day
    for _ in range(_MAX_SCAN):
        d -= dt.timedelta(days=1)
        if is_trading_day(d, holidays):
            return d
    raise ValueError(f"no trading day within {_MAX_SCAN} days before {day}")


def add_trading_days(day: dt.date, n: int, holidays: list[str] | None = None) -> dt.date:
    """`day` moved forward by n trading days (n >= 0; n == 0 returns `day`)."""
    d = day
    for _ in range(n):
        d = next_trading_day(d, holidays)
    return d
