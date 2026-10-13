"""
ranking/watch_warnings.py — factual warnings for a user's watched stocks
(presentation only; the StockLens Score and the ranking are never changed).

A warning states what happened in the data or the news, with the numbers and
the date, so the user can decide for themselves. It is never "Sell" advice
and never a prediction:

  negative_news  a recent, credible, clear-cut, non-contradicted negative
                 report about the stock (same rules as ranking/news_labels.py)
  heavy_selling  the latest daily close fell HEAVY_DROP or more versus the
                 previous session's close, on volume of HEAVY_VOLUME times the
                 20-day average or more (price data must be current)
  downtrend      trend score at or below the existing WEAK threshold (30)
  score_drop     the StockLens Score fell SCORE_DROP points or more since the
                 previous full ranking run

Missing or out-of-date data gives no warning, never a guess.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.market import MarketSnapshot, StockAnalysisResult
from ranking import presenter
from ranking.news_labels import news_detail

HEAVY_DROP = 0.02          # close down 2 % or more ...
HEAVY_VOLUME = 1.5         # ... on 1.5x the 20-day average volume or more
MAX_PRICE_AGE_DAYS = 5     # latest daily bar older than this -> no price warning
MAX_SESSION_GAP_DAYS = 5   # previous bar must be the previous session (weekends / holidays allowed)
SCORE_DROP = 10.0

WARNINGS_NOTE = ("Warnings describe what happened in the stock's data and news. They are not sell advice "
                 "or predictions; check the details and decide for yourself.")


def _date(s: Optional[str]) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        return None


def heavy_selling(session: Session, symbol: str, today: dt.date) -> Optional[dict[str, Any]]:
    snaps = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == symbol, MarketSnapshot.status == "OK")
                         .order_by(MarketSnapshot.fetched_at.desc()).limit(20)).all()
    cur = next((x for x in snaps if _date(x.as_of_date)), None)
    if cur is None:
        return None
    cur_day = _date(cur.as_of_date)
    prev = next((x for x in snaps if _date(x.as_of_date) and _date(x.as_of_date) < cur_day), None)
    if prev is None or (today - cur_day).days > MAX_PRICE_AGE_DAYS \
            or (cur_day - _date(prev.as_of_date)).days > MAX_SESSION_GAP_DAYS:
        return None
    avg = (cur.technical or {}).get("avg_volume_20d")
    if not (cur.close and prev.close and cur.volume and avg) or prev.close <= 0 or avg <= 0:
        return None
    change, ratio = cur.close / prev.close - 1, cur.volume / avg
    if change > -HEAVY_DROP or ratio < HEAVY_VOLUME:
        return None
    return {"key": "heavy_selling", "label": "Heavy selling", "tone": "negative", "as_of": cur.as_of_date,
            "detail": f"Price fell {abs(change) * 100:.1f}% on {cur_day.strftime('%d %b %Y')} on "
                      f"{ratio:.1f}x its 20-day average volume"}


def warnings_for(session: Session, symbol: str, latest: Optional[StockAnalysisResult],
                 score_change: Optional[float], news: list[dict[str, Any]],
                 now: dt.datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    neg = next((n for n in news if n["tone"] == "negative"), None)
    if neg is not None:
        out.append({"key": "negative_news", "label": "Negative news", "tone": "negative",
                    "as_of": neg["published_at"][:10], "detail": news_detail(news, "negative"), "url": neg.get("url")})
    hs = heavy_selling(session, symbol, now.date())
    if hs is not None:
        out.append(hs)
    if latest is not None:
        trend = presenter._comp(latest, "technical_trend")
        if trend is not None and trend <= presenter.WEAK:
            out.append({"key": "downtrend", "label": "Downtrend", "tone": "negative",
                        "as_of": presenter._iso(latest.computed_at)[:10] if latest.computed_at else None,
                        "detail": f"Trend score {trend:.0f}/100 (30 or less)"})
    if score_change is not None and score_change <= -SCORE_DROP:
        out.append({"key": "score_drop", "label": "Score fell", "tone": "negative", "as_of": None,
                    "detail": f"StockLens Score fell {abs(score_change):.1f} points since the previous ranking"})
    return out
