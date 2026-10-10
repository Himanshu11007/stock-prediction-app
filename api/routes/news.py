"""
api/routes/news.py — news and catalyst intelligence (Prediction Engine v2,
SHADOW). Same access rule as the prediction API: administrators only unless
PREDICTION_V2_PUBLIC is true.

  GET  /news/events                      recent canonical events (filters: hours, category, symbol, scope)
  GET  /news/events/{event_id}           one event: articles (source, times), classification history,
                                         factual entities, inferred exposures (separate, with mechanism
                                         and validation status)
  GET  /news/market-summary              major macro / global catalysts in a window and the Indian sectors
                                         and stocks inferred to be exposed (grouped by sign)
  GET  /news/hypotheses                  the transmission hypotheses and their validation status
  GET  /admin/news/ingestion-runs        provider runs (admin)
  POST /admin/news/ingest                run the hourly ingestion job now (admin, audited; same slot lock)

Every response separates FACTS (what the source said, when it was published
and when StockLens ingested it) from INFERENCE (hypothesis-based exposures),
and states that no event is a trading signal on its own.
"""
from __future__ import annotations

import datetime as dt
import threading
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from admin.audit import log_action
from api.routes.predictions import prediction_reader
from api.schemas import success_envelope
from auth.dependencies import require_admin
from catalysts import CLASSIFIER_VERSION, transmission
from db.models.news import EventEntity, NewsArticle, NewsIngestionRun
from db.models.prediction import EventClassification, MarketEvent
from db.models.user import User
from db.session import engine, get_session
from ranking.presenter import to_jsonable

router = APIRouter(prefix="/news", dependencies=[Depends(prediction_reader)])
admin_router = APIRouter(prefix="/admin/news", dependencies=[Depends(require_admin)])

NOTICE = ("Events are reported facts with their source and timestamps. Exposures are hypotheses about how an event "
          "may transmit to Indian stocks; they are not predictions or trading signals.")
MACRO_CATEGORIES = ("MONETARY_POLICY", "TRADE_TARIFF", "GEOPOLITICAL", "FISCAL_POLICY", "ECONOMIC_DATA",
                    "FLOWS_CURRENCY", "COMMODITY", "REGULATORY")


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def _latest_classification(session: Session, event_id: int) -> Optional[EventClassification]:
    return session.exec(select(EventClassification).where(EventClassification.event_id == event_id,
                                                          EventClassification.classifier_version == CLASSIFIER_VERSION)
                        .order_by(EventClassification.id.desc())).first()


def event_payload(session: Session, ev: MarketEvent, detail: bool = False) -> dict:
    arts = session.exec(select(NewsArticle).where(NewsArticle.event_id == ev.id)
                        .order_by(NewsArticle.effective_available_at)).all()
    cls = _latest_classification(session, ev.id)
    ents = session.exec(select(EventEntity).where(EventEntity.event_id == ev.id)).all()
    facts = [{"type": e.entity_type, "key": e.entity_key, "symbol": e.symbol, "relation": e.relation}
             for e in ents if not e.inferred]
    inferred = [{"symbol": e.symbol, "hypothesis_id": e.hypothesis_id, "sign": e.sign, "mechanism": e.mechanism,
                 "weight": e.confidence} for e in ents if e.inferred]
    out = {
        "event_id": ev.id, "title": ev.title, "event_type": ev.event_type, "symbol": ev.symbol,
        "scope": (ev.raw or {}).get("scope"),
        "first_published_at": ev.published_at, "ingested_at": ev.ingested_at,
        "effective_available_at": ev.effective_available_at,
        "sources": [{"name": a.source_name, "domain": a.source_domain, "tier": a.source_tier,
                     "primary_source": a.is_primary_source, "url": a.url, "published_at": a.published_at,
                     "ingested_at": a.ingested_at, "corrected_at": a.corrected_at} for a in arts],
        "article_count": len(arts),
        "classification": None if cls is None else {
            "category": cls.category, "direction": cls.direction, "sentiment": cls.sentiment,
            "materiality": cls.materiality, "severity": cls.severity, "credibility": cls.credibility,
            "novelty": cls.novelty, "expectedness": cls.expectedness, "contradicted": cls.contradicted,
            "expires_at": cls.expires_at, "classifier_version": cls.classifier_version,
            "review_status": cls.review_status},
        "facts": {"entities": facts},
        "inference": {"exposures": inferred if detail else inferred[:10], "exposure_count": len(inferred),
                      "note": "hypothesis-based; see /news/hypotheses for validation status"},
    }
    if detail:
        out["classification_history"] = [c.model_dump() for c in session.exec(
            select(EventClassification).where(EventClassification.event_id == ev.id)
            .order_by(EventClassification.id)).all()]
        out["articles"] = [{"title": a.title, "excerpt": a.excerpt, "language": a.language, "provider": a.provider,
                            "url": a.url} for a in arts]
    return out


@router.get("/events")
def news_events(hours: int = Query(72, ge=1, le=24 * 30), category: Optional[str] = None,
                symbol: Optional[str] = None, scope: Optional[str] = Query(None, pattern="^(MACRO|SECTOR|COMPANY)$"),
                limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    stmt = select(MarketEvent).where(MarketEvent.source == "news", MarketEvent.effective_available_at >= since)
    if symbol:
        ids = {e.event_id for e in session.exec(select(EventEntity).where(EventEntity.symbol == symbol.upper())).all()}
        stmt = stmt.where(MarketEvent.id.in_(ids or {-1}))
    rows = session.exec(stmt.order_by(MarketEvent.effective_available_at.desc())).all()
    out = []
    for ev in rows:
        p = event_payload(session, ev)
        if category and (p["classification"] or {}).get("category") != category:
            continue
        if scope and p["scope"] != scope:
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return success_envelope(to_jsonable({"shadow": True, "notice": NOTICE, "since": since, "events": out}),
                            message=f"{len(out)} event(s)")


@router.get("/events/{event_id}")
def news_event(event_id: int, session: Session = Depends(get_session)):
    ev = session.get(MarketEvent, event_id)
    if ev is None or ev.source != "news":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
    return success_envelope(to_jsonable({"shadow": True, "notice": NOTICE, **event_payload(session, ev, detail=True)}),
                            message="Event")


@router.get("/market-summary")
def market_summary(hours: int = Query(24, ge=1, le=24 * 7), limit: int = Query(10, ge=1, le=50),
                   session: Session = Depends(get_session)):
    """Most material macro / global catalysts in the window, with the Indian
    stocks and sectors inferred to be exposed (positive / negative /
    ambiguous). Inference is labelled as such."""
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    rows = session.exec(select(MarketEvent).where(MarketEvent.source == "news",
                                                  MarketEvent.effective_available_at >= since)).all()
    items = []
    for ev in rows:
        cls = _latest_classification(session, ev.id)
        if cls is None or cls.category not in MACRO_CATEGORIES:
            continue
        p = event_payload(session, ev, detail=True)
        groups: dict[str, list[str]] = defaultdict(list)
        for x in p["inference"]["exposures"]:
            groups["positive" if (x["sign"] or 0) > 0 else "negative" if (x["sign"] or 0) < 0 else "ambiguous"].append(
                x["symbol"])
        items.append({"event_id": ev.id, "title": ev.title, "category": cls.category,
                      "materiality": cls.materiality, "contradicted": cls.contradicted,
                      "effective_available_at": ev.effective_available_at,
                      "sources": [{"name": s["name"], "tier": s["tier"], "url": s["url"]} for s in p["sources"]],
                      "exposed": {k: sorted(set(v)) for k, v in groups.items()},
                      "hypotheses": sorted({x["hypothesis_id"] for x in p["inference"]["exposures"]})})
    items.sort(key=lambda i: -(i["materiality"] or 0))
    last = session.exec(select(NewsIngestionRun).order_by(NewsIngestionRun.started_at.desc())).first()
    return success_envelope(to_jsonable({
        "shadow": True, "notice": NOTICE, "window_hours": hours, "catalysts": items[:limit],
        "coverage": {"latest_ingestion": None if last is None else {"provider": last.provider, "status": last.status,
                                                                   "finished_at": last.finished_at},
                     "available_sources": "official RBI, SEBI and US Federal Reserve feeds; corporate filings, "
                                          "broker ratings and licensed newswires are not connected"}}),
        message=f"{len(items[:limit])} catalyst(s)")


@router.get("/hypotheses")
def hypotheses():
    st = transmission.validation_status()
    return success_envelope({"transmission_version": transmission.TRANSMISSION_VERSION, "hypotheses": [
        {"id": h.id, "driver": h.driver, "categories": list(h.categories), "factor": h.factor,
         "status": st.get(h.id, {}).get("status", "UNVALIDATED"), "predictive": st.get(h.id, {}).get("predictive"),
         "targets": [{"sign": t.sign, "symbols": list(t.symbols), "industries": list(t.industries),
                      "sectors": list(t.sectors), "mechanism": t.mechanism} for t in h.targets]}
        for h in transmission.HYPOTHESES]}, message="Transmission hypotheses")


@admin_router.get("/ingestion-runs")
def ingestion_runs(limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    rows = session.exec(select(NewsIngestionRun).order_by(NewsIngestionRun.started_at.desc()).limit(limit)).all()
    return success_envelope(to_jsonable({"runs": [r.model_dump() for r in rows]}), message="News ingestion runs")


@admin_router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def admin_ingest(current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Recovery tool: the same hourly job as the scheduler (same slot lock)."""
    from scheduling import jobs
    threading.Thread(target=jobs.news_ingestion_job, args=(engine,), daemon=True, name="admin-news").start()
    log_action(session, current_admin, "NEWS_INGESTION_TRIGGERED", "news_ingestion", None, {})
    session.commit()
    return {"status": "STARTED", "detail": "See GET /admin/news/ingestion-runs for the result."}
