"""
ranking/news_labels.py — factual news labels for ranked stocks (presentation
only; the StockLens Score and the ranking are never changed).

A stock gets "Positive news" / "Negative news" when, in the last
NEWS_WINDOW, a news event that was available to StockLens (point in time:
effective_available_at <= now) has the stock as its SUBJECT, was reported by
a credible source (PRIMARY / TIER1 / TIER2), is not contradicted by other
reports, and its text sentiment is clearly positive or negative
(|sentiment| >= SENTIMENT_MIN). Everything else gets no label: unclear,
contradicted, low-credibility or old news is not shown as a label.

The label states what was reported, never what to do: it is not a
prediction and not buy / sell advice. Each item carries the headline, the
source, the publication and availability times and the link, so a user can
read the report itself.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, Optional

from sqlmodel import Session, select

from catalysts.impact import _classification
from db.models.news import EventEntity, NewsArticle
from db.models.prediction import MarketEvent

NEWS_WINDOW = dt.timedelta(hours=72)
SENTIMENT_MIN = 0.2
CREDIBLE = ("PRIMARY", "TIER1", "TIER2")
MAX_ITEMS = 3


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def news_for(session: Session, symbols: Iterable[str], now: dt.datetime,
             window: dt.timedelta = NEWS_WINDOW) -> dict[str, list[dict[str, Any]]]:
    """symbol -> up to MAX_ITEMS recent, credible, clear-cut news items about it
    (newest first)."""
    now = _utc(now)
    since = now - window
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return {}
    links = session.exec(select(EventEntity).where(EventEntity.symbol.in_(syms),
                                                   EventEntity.relation == "SUBJECT")).all()
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for link in links:
        if (link.symbol, link.event_id) in seen or _utc(link.created_at) > now:
            continue
        seen.add((link.symbol, link.event_id))
        ev = session.get(MarketEvent, link.event_id)
        avail = _utc(ev.effective_available_at) if ev else None
        if ev is None or avail is None or not (since <= avail <= now):
            continue
        cls = _classification(session, ev.id, now)
        if cls is None or cls.contradicted or cls.sentiment is None or abs(cls.sentiment) < SENTIMENT_MIN:
            continue
        if cls.expires_at is not None and _utc(cls.expires_at) <= now:
            continue
        arts = [a for a in session.exec(select(NewsArticle).where(NewsArticle.event_id == ev.id)).all()
                if a.source_tier in CREDIBLE and _utc(a.effective_available_at) <= now]
        if not arts:
            continue
        first = min(arts, key=lambda a: _utc(a.published_at))
        out.setdefault(link.symbol, []).append({
            "tone": "positive" if cls.sentiment > 0 else "negative",
            "title": first.title, "source": first.source_domain or first.source_name, "url": first.url,
            "published_at": _utc(first.published_at).isoformat(), "available_at": avail.isoformat(),
            "category": cls.category, "sources": len({a.source_domain or a.source_name for a in arts}),
        })
    for sym in out:
        out[sym] = sorted(out[sym], key=lambda n: n["available_at"], reverse=True)[:MAX_ITEMS]
    return out


def news_labels(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Positive / Negative news labels for one stock's news items."""
    out = []
    for tone, key, text in (("positive", "news_positive", "Positive news"),
                            ("negative", "news_negative", "Negative news")):
        if any(n["tone"] == tone for n in items):
            out.append({"key": key, "label": text, "tone": tone})
    return out


def news_detail(items: list[dict[str, Any]], tone: str) -> Optional[str]:
    n = next((n for n in items if n["tone"] == tone), None)
    return None if n is None else f"{n['title']} ({n['source']}, reported {n['published_at'][:16].replace('T', ' ')} UTC)"
