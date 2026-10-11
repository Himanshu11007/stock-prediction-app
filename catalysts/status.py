"""
catalysts/status.py — what the news inputs behind a screen or snapshot are
actually based on, from the database only (no network calls).

  providers   every selected provider (NEWS_PROVIDERS) with: ENABLED or the
              refusal reason (research-only, credential missing), the last
              run's status / time / counts / error, the last SUCCESSFUL run
              and its age, and health:
                OK            succeeded within STALE_AFTER
                STALE         last success older than STALE_AFTER
                FAILING       the latest run failed / was rate-limited
                NOT_CONFIGURED / REFUSED   never queried
                NEVER_RUN     enabled but no run recorded
  coverage    universe stocks with linked news events available in the
              window, and the ones without (missing coverage)
  data_basis  REAL         at least one enabled provider is OK
              STALE        news exists but no provider succeeded recently
              UNAVAILABLE  no successful ingestion at all: any screen built
                           on it has no news input (and says so)
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.news import EventEntity, NewsArticle, NewsIngestionRun
from db.models.prediction import MarketEvent

STALE_AFTER = dt.timedelta(hours=3)          # hourly ingestion: three missed runs = stale
COVERAGE_WINDOW = dt.timedelta(hours=72)
OK_STATES = ("COMPLETED", "PARTIAL")


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def _iso(t: Optional[dt.datetime]) -> Optional[str]:
    return _utc(t).isoformat() if t else None


def news_status(session: Session, now: dt.datetime, selection: Optional[dict] = None,
                universe: Optional[list[str]] = None) -> dict[str, Any]:
    now = _utc(now)
    sel = selection or {"enabled": [], "refused": {}}
    runs = session.exec(select(NewsIngestionRun).order_by(NewsIngestionRun.started_at.desc())).all()
    names = list(dict.fromkeys(list(sel.get("enabled", [])) + list(sel.get("refused", {})) + [r.provider for r in runs]))
    providers = []
    for n in names:
        mine = [r for r in runs if r.provider == n]
        last = mine[0] if mine else None
        ok = next((r for r in mine if r.status in OK_STATES), None)
        age = (now - _utc(ok.finished_at or ok.started_at)) if ok else None
        refused = sel.get("refused", {}).get(n)
        if refused:
            health = "NOT_CONFIGURED" if "not set" in refused else "REFUSED"
        elif last is None:
            health = "NEVER_RUN" if n in sel.get("enabled", []) else "NOT_SELECTED"
        elif last.status not in OK_STATES:
            health = "NOT_CONFIGURED" if last.status == "NOT_CONFIGURED" else "FAILING"
        else:
            health = "OK" if age is not None and age <= STALE_AFTER else "STALE"
        providers.append({
            "provider": n, "selected": n in sel.get("enabled", []) or n in sel.get("refused", {}),
            "enabled": n in sel.get("enabled", []), "refused_reason": refused, "health": health,
            "last_run": None if last is None else {"status": last.status, "started_at": _iso(last.started_at),
                                                   "finished_at": _iso(last.finished_at), "fetched": last.fetched,
                                                   "inserted": last.inserted, "new_events": last.new_events,
                                                   "error": (last.error or "")[:300] or None},
            "last_success_at": _iso(ok.finished_at or ok.started_at) if ok else None,
            "last_success_age_hours": round(age.total_seconds() / 3600, 2) if age is not None else None,
        })
    newest = session.exec(select(NewsArticle).order_by(NewsArticle.ingested_at.desc())).first()
    since = now - COVERAGE_WINDOW
    covered: set[str] = set()
    if universe:
        events = {e.id for e in session.exec(select(MarketEvent).where(MarketEvent.effective_available_at >= since,
                                                                        MarketEvent.effective_available_at <= now)).all()}
        if events:
            for link in session.exec(select(EventEntity).where(EventEntity.event_id.in_(events),
                                                               EventEntity.symbol.is_not(None))).all():
                covered.add(link.symbol)
    ok_any = any(p["health"] == "OK" for p in providers)
    any_success = any(p["last_success_at"] for p in providers)
    basis = "REAL" if ok_any else "STALE" if any_success else "UNAVAILABLE"
    detail = {"REAL": "news ingested recently from at least one approved provider",
              "STALE": f"no provider has succeeded in the last {int(STALE_AFTER.total_seconds() // 3600)} h: "
                       "news input is out of date",
              "UNAVAILABLE": "no successful news ingestion recorded: results have no news input"}[basis]
    uni = sorted(universe or [])
    return {"as_of": now.isoformat(), "data_basis": basis, "detail": detail,
            "stale_after_hours": STALE_AFTER.total_seconds() / 3600, "providers": providers,
            "newest_article": None if newest is None else {"title": newest.title, "provider": newest.provider,
                                                           "published_at": _iso(newest.published_at),
                                                           "ingested_at": _iso(newest.ingested_at)},
            "coverage": {"window_hours": COVERAGE_WINDOW.total_seconds() / 3600, "universe": len(uni),
                         "with_news": len([s for s in uni if s in covered]),
                         "without_news": [s for s in uni if s not in covered][:200]}}
