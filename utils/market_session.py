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
