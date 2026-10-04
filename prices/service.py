"""
prices/service.py — latest available market price per stock, kept separate
from the ranking reference price.

  ranking reference price  the close a ranking run used (market_snapshots /
                           ranking_snapshots); frozen with that run
  current price            this module (price_quotes): the newest daily bar
                           from the provider, refreshed by the scheduled
                           "prices" job and, when stale, on demand for the
                           symbols a client is displaying

Statuses (current_price_status):
  LAST_CLOSE        the day's bar is final (after 16:00 IST): the closing price
  DELAYED_INTRADAY  today's bar while the session is open; Yahoo Finance NSE
                    data is delayed, so this is never called "live"
  STALE             the newest bar is older than MARKET_DATA_STALE_DAYS
  NOT_AVAILABLE     no valid price has ever been received for the stock

Never invented: a failed or empty provider response keeps the previous price
(with its original timestamp) and records the error; a stock without any
valid price returns null.
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Callable, Iterable, Optional

from sqlmodel import Session, select

import masters.service as masters
from config import CURRENT_PRICE_ON_DEMAND, CURRENT_PRICE_TTL_MINUTES, MARKET_DATA_STALE_DAYS
from db.models.market import PriceQuote
from db.models.stock import Company
from utils.logger import get_logger
from utils.market_session import IST, NSE_CLOSE, is_daily_bar_complete, market_status

logger = get_logger(__name__)

SESSION_STATUSES = ("PRE_OPEN", "OPEN", "CLOSING", "POST_CLOSE")
RETRY_AFTER_ERROR = dt.timedelta(minutes=5)
CLOSED_MARKET_TTL = dt.timedelta(hours=6)
_refresh_lock = threading.Lock()


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def _default_fetch(symbols: list[str]) -> dict:
    from fundamentals.provider import fetch_price_history
    return fetch_price_history(symbols, period="5d")


# ── reading ──────────────────────────────────────────────────────────────────

def quote_payload(q: Optional[PriceQuote], now: Optional[dt.datetime] = None) -> dict:
    """API fields for one stock's current price."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if q is None or q.price is None or q.bar_date is None:
        return {"current_price": None, "current_price_as_of": None, "current_price_status": "NOT_AVAILABLE",
                "current_price_source": None, "current_price_date": None}
    status = q.status
    age = (now.astimezone(IST).date() - dt.date.fromisoformat(q.bar_date)).days
    if age > MARKET_DATA_STALE_DAYS:
        status = "STALE"
    return {"current_price": q.price, "current_price_as_of": _utc(q.as_of).isoformat() if q.as_of else None,
            "current_price_status": status, "current_price_source": q.source, "current_price_date": q.bar_date}


def get_quotes(session: Session, symbols: Iterable[str]) -> dict[str, PriceQuote]:
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return {}
    return {q.symbol: q for q in session.exec(select(PriceQuote).where(PriceQuote.symbol.in_(syms))).all()}


# ── refreshing ───────────────────────────────────────────────────────────────

def needs_refresh(q: Optional[PriceQuote], now: dt.datetime, holidays: list[str]) -> bool:
    if q is None:
        return True
    if q.last_error_at and now - _utc(q.last_error_at) < RETRY_AFTER_ERROR and \
            (q.fetched_at is None or _utc(q.last_error_at) > _utc(q.fetched_at)):
        return False                                         # don't hammer a failing provider
    age = now - _utc(q.fetched_at)
    if market_status(now, holidays)["status"] in SESSION_STATUSES:
        return age >= dt.timedelta(minutes=CURRENT_PRICE_TTL_MINUTES)
    if q.status == "DELAYED_INTRADAY":                       # session over: the final close is available
        return True
    return age >= CLOSED_MARKET_TTL


def refresh_quotes(session: Session, symbols: Iterable[str], now: Optional[dt.datetime] = None,
                   fetch: Optional[Callable[[list[str]], dict]] = None) -> dict[str, int]:
    """Fetch the newest daily bar for `symbols` and update price_quotes."""
    now = now or dt.datetime.now(dt.timezone.utc)
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return {"updated": 0, "failed": 0}
    fetch = fetch or _default_fetch
    try:
        data = fetch(syms) or {}
        error = None
    except Exception as e:                                   # provider outage: keep existing prices
        data, error = {}, f"{type(e).__name__}: {e}"[:300]
    existing = get_quotes(session, syms)
    updated = failed = 0
    for sym in syms:
        q = existing.get(sym) or PriceQuote(symbol=sym, fetched_at=now)
        df = data.get(sym)
        close = None
        if df is not None and not getattr(df, "empty", True) and "Volume" in df:
            # Yahoo adds zero-volume placeholder bars on exchange holidays;
            # they are not trading sessions and must not be shown as a close.
            traded = df[df["Volume"].fillna(0) > 0]
            df = traded if not traded.empty else df
        if df is not None and not getattr(df, "empty", True):
            try:
                value = float(df["Close"].iloc[-1])
                close = value if value > 0 and value == value else None
            except (KeyError, TypeError, ValueError):
                close = None
        if close is None:
            q.last_error = error or "provider returned no valid price"
            q.last_error_at = now
            failed += 1
        else:
            bar_day = df.index[-1].date()
            final = is_daily_bar_complete(bar_day, now)
            q.price, q.bar_date = round(close, 4), bar_day.isoformat()
            q.status = "LAST_CLOSE" if final else "DELAYED_INTRADAY"
            q.as_of = (dt.datetime.combine(bar_day, NSE_CLOSE, tzinfo=IST).astimezone(dt.timezone.utc)
                       if final else now)
            q.fetched_at, q.last_error, q.last_error_at = now, None, None
            updated += 1
        session.add(q)
    session.commit()
    return {"updated": updated, "failed": failed}


def ensure_fresh(session: Session, symbols: Iterable[str], now: Optional[dt.datetime] = None,
                 fetch: Optional[Callable[[list[str]], dict]] = None) -> dict[str, int]:
    """On-demand refresh of stale quotes for the symbols a client displays.
    Single-flight: if another refresh is running, the stored prices are used."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if not CURRENT_PRICE_ON_DEMAND:
        return {"updated": 0, "failed": 0}
    syms = list(dict.fromkeys(symbols))
    holidays = masters.get_config(session, "market.holidays")
    quotes = get_quotes(session, syms)
    stale = [s for s in syms if needs_refresh(quotes.get(s), now, holidays)]
    if not stale or not _refresh_lock.acquire(blocking=False):
        return {"updated": 0, "failed": 0}
    try:
        return refresh_quotes(session, stale, now, fetch)
    except Exception:
        session.rollback()
        logger.exception("CURRENT_PRICE_REFRESH_FAILED")
        return {"updated": 0, "failed": len(stale)}
    finally:
        _refresh_lock.release()


def universe_symbols(session: Session) -> list[str]:
    from db.models.stock import StockUniverseMember
    rows = session.exec(select(StockUniverseMember.symbol)
                        .join(Company, Company.symbol == StockUniverseMember.symbol)
                        .where(Company.active == True)).all()  # noqa: E712
    return sorted(set(rows))


def freshness_summary(session: Session, now: Optional[dt.datetime] = None) -> dict:
    """For the admin console: how fresh the current prices are."""
    now = now or dt.datetime.now(dt.timezone.utc)
    rows = session.exec(select(PriceQuote)).all()
    statuses: dict[str, int] = {}
    for q in rows:
        s = quote_payload(q, now)["current_price_status"]
        statuses[s] = statuses.get(s, 0) + 1
    fetched = [_utc(q.fetched_at) for q in rows if q.price is not None]
    return {"symbols_with_price": sum(1 for q in rows if q.price is not None), "by_status": statuses,
            "newest_fetch": max(fetched).isoformat() if fetched else None,
            "oldest_fetch": min(fetched).isoformat() if fetched else None,
            "with_recent_error": sum(1 for q in rows if q.last_error)}
