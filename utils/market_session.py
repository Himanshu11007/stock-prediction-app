"""
utils/market_session.py — is a daily bar a finished NSE session?

yfinance returns the current session as the last daily bar while the market
is open; its Close is the latest traded price and its Volume is partial.
Production predicts from whatever the last bar is (intraday-aware), and
records whether that bar was complete so evaluation can tell end-of-day
predictions from intraday ones. See docs/PRODUCTION_TEMPORAL_INTEGRITY.md.

IST has no daylight saving, so a fixed +05:30 offset is exact.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NSE_CLOSE = dt.time(15, 30)
# Yahoo's daily bar for today is treated as final only after this time,
# leaving a margin after the 15:30 close for the closing price to settle.
DAILY_BAR_FINAL_AFTER = dt.time(16, 0)


def now_ist(now: dt.datetime | None = None) -> dt.datetime:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(IST)


def is_daily_bar_complete(bar_date, now: dt.datetime | None = None) -> bool:
    """
    True if the NSE daily bar dated `bar_date` was a finished session at `now`.
    Bars from earlier days are complete; today's bar is complete only after
    DAILY_BAR_FINAL_AFTER IST.
    """
    current = now_ist(now)
    bar_day = pd.Timestamp(bar_date).date()
    if bar_day < current.date():
        return True
    if bar_day > current.date():
        return False
    return current.time() >= DAILY_BAR_FINAL_AFTER


# ── NSE market status (Home screen, scheduled runs) ──────────────────────────
# Equity segment timings (IST): pre-open 09:00-09:15, normal market
# 09:15-15:30, post-close session 15:40-16:00. Weekends are closed. Trading
# holidays are NOT derived here: they come from the administrator-maintained
# list (app_config "market.holidays"); when that list is empty the status says
# so instead of guessing.
PRE_OPEN_START = dt.time(9, 0)
NSE_OPEN = dt.time(9, 15)
POST_CLOSE_START = dt.time(15, 40)
POST_CLOSE_END = dt.time(16, 0)

_LABELS = {
    "PRE_OPEN": "Pre-open session",
    "OPEN": "Market open",
    "CLOSING": "Closing - post-close session starts at 15:40 IST",
    "POST_CLOSE": "Post-close session",
    "CLOSED": "Market closed",
    "WEEKEND": "Closed - weekend",
    "HOLIDAY": "Closed - NSE trading holiday",
}


def is_trading_day(day: dt.date, holidays: list[str] | None = None) -> bool:
    return day.weekday() < 5 and day.isoformat() not in set(holidays or [])


def market_status(now: dt.datetime | None = None, holidays: list[str] | None = None) -> dict:
    """NSE equity market status at `now` (timezone-aware; default: now)."""
    current = now_ist(now)
    day, t = current.date(), current.time()
    if day.weekday() >= 5:
        status = "WEEKEND"
    elif day.isoformat() in set(holidays or []):
        status = "HOLIDAY"
    elif PRE_OPEN_START <= t < NSE_OPEN:
        status = "PRE_OPEN"
    elif NSE_OPEN <= t < NSE_CLOSE:
        status = "OPEN"
    elif NSE_CLOSE <= t < POST_CLOSE_START:
        status = "CLOSING"
    elif POST_CLOSE_START <= t < POST_CLOSE_END:
        status = "POST_CLOSE"
    else:
        status = "CLOSED"
    nxt = day if (status in ("PRE_OPEN", "CLOSED") and t < PRE_OPEN_START and is_trading_day(day, holidays)) \
        else day + dt.timedelta(days=1)
    while not is_trading_day(nxt, holidays):
        nxt += dt.timedelta(days=1)
    return {
        "status": status,
        "label": _LABELS[status],
        "is_open": status == "OPEN",
        "exchange": "NSE",
        "timezone": "Asia/Kolkata (IST, UTC+05:30)",
        "now_ist": current.isoformat(timespec="minutes"),
        "date": day.isoformat(),
        "session_hours": "Pre-open 09:00-09:15, normal 09:15-15:30, post-close 15:40-16:00 IST",
        "next_trading_day": None if status in ("PRE_OPEN", "OPEN") else nxt.isoformat(),
        "holiday_calendar_configured": bool(holidays),
        "note": None if holidays else
                "Exchange holiday calendar not configured; weekends are recognised, trading holidays are not.",
    }
