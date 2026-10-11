"""
api/routes/predictions.py — Prediction Engine v2 API (SHADOW MODE).

All routes are additive; no existing route or response changed.

Access: while PREDICTION_V2_PUBLIC is false (default) every route requires
the ADMIN role - v2 output is an unvalidated experiment, not a
recommendation. Responses always carry `shadow: true`, the run's cutoff and
versions, and a freshness status; a stale or missing snapshot is reported as
such, never presented as current.

Snapshot selection: a response is built from exactly ONE prediction run;
features, levels and outcomes shown with a prediction are the ones frozen
with it.
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select

import masters.service as masters
from admin.audit import log_action
from api.schemas import success_envelope
from auth.dependencies import get_current_user, require_admin
from db.models.market import MarketSnapshot
from db.models.prediction import (EventClassification, ExitState, ExitTransition, MarketEvent, Prediction,
                                  PredictionOutcome, PredictionRun, V2UniverseMember)
from db.models.stock import Company, StockUniverseMember
from db.models.tracker import WatchlistItem
from db.models.user import User
from db.session import engine, get_session
from prediction_v2 import calendar, performance as perf, universe
from ranking.presenter import to_jsonable
from utils.market_session import now_ist

SHADOW_NOTICE = ("Prediction Engine v2 is in shadow mode: experimental, unvalidated output for internal "
                 "evaluation. It is not investment advice and not a recommendation to trade.")
DIRECTIONS = ("UP", "DOWN", "NEUTRAL", "NO_CALL")
EVENT_PROVIDER_CONFIGURED = False      # no licensed filings provider; news feeds count once ingestion has run


def prediction_reader(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)) -> User:
    from auth.service import ADMIN_ROLE, get_user_roles
    from config import PREDICTION_V2_PUBLIC
    if not PREDICTION_V2_PUBLIC and ADMIN_ROLE not in get_user_roles(session, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Prediction Engine v2 is in shadow mode and available to administrators only.")
    return current_user


router = APIRouter(dependencies=[Depends(prediction_reader)])
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


# ── serialisation ────────────────────────────────────────────────────────────

def run_payload(r: PredictionRun) -> dict:
    return to_jsonable({
        "run_id": r.run_id, "run_type": r.run_type, "status": r.status, "engine_version": r.engine_version,
        "rule_version": r.rule_version, "feature_set_version": r.feature_set_version, "trading_date": r.trading_date,
        "target_session_date": r.target_session_date, "data_cutoff_at": r.data_cutoff_at, "started_at": r.started_at,
        "completed_at": r.completed_at, "universe_count": r.universe_count, "eligible_count": r.eligible_count,
        "prediction_count": r.prediction_count, "counts": r.counts, "failure_reason": r.failure_reason,
        "holiday_calendar_configured": (r.config or {}).get("holiday_calendar_configured"),
    })


def prediction_payload(p: Prediction, company: Optional[Company] = None, full: bool = False) -> dict:
    out = {"prediction_id": p.prediction_id, "run_id": p.run_id, "symbol": p.symbol,
           "name": company.name if company else None, "sector": company.sector if company else None,
           "direction": p.direction, "setup_type": p.setup_type, "horizon_sessions": p.horizon_sessions,
           "confidence": p.confidence, "calibration_version": p.calibration_version,
           "reference_price": p.reference_price, "reference_date": p.reference_date,
           "entry_condition": p.entry_condition, "stop_loss": p.stop_loss, "target": p.target,
           "trailing_stop_rule": p.trailing_stop_rule, "invalidation_condition": p.invalidation_condition,
           "reasons": p.reasons or [], "quality_flags": p.quality_flags or [], "event_ids": p.event_ids or [],
           "created_at": p.created_at}
    if full:
        out["features"] = p.features
    return to_jsonable(out)


def freshness(run: Optional[PredictionRun], session_kind: str, holidays: list[str]) -> dict:
    today = now_ist().date()
    if run is None:
        return {"status": "NONE", "detail": f"No completed {session_kind} snapshot exists yet."}
    target = dt.date.fromisoformat(run.target_session_date)
    if session_kind == "today":
        fresh = target == today
        detail = "Snapshot for today's session." if fresh else f"Latest snapshot is for {target}, not today."
    else:
        upcoming = calendar.next_trading_day(today, holidays) if target > today else None
        fresh = target > today and (upcoming is None or target <= upcoming)
        detail = (f"Forecast for the next trading session ({target})." if fresh
                  else f"Latest forecast was for {target}; no newer end-of-day snapshot exists.")
    return {"status": "FRESH" if fresh else "STALE", "target_session_date": run.target_session_date, "detail": detail}


def _select_run(session: Session, kind: str) -> Optional[PredictionRun]:
    types = ("TODAY_CONFIRMED", "TODAY_PREOPEN") if kind == "today" else ("TOMORROW_EOD",)
    today = now_ist().date().isoformat()
    stmt = select(PredictionRun).where(PredictionRun.status == "COMPLETED", PredictionRun.run_type.in_(types))
    if kind == "today":
        # today's confirmed snapshot wins over the pre-open one; else the newest
        todays = session.exec(stmt.where(PredictionRun.target_session_date == today)).all()
        for t in types:
            for r in sorted(todays, key=lambda r: r.completed_at, reverse=True):
                if r.run_type == t:
                    return r
    return session.exec(stmt.order_by(PredictionRun.target_session_date.desc(),
                                      PredictionRun.completed_at.desc())).first()


# ── user (shadow-gated) routes ───────────────────────────────────────────────

@router.get("/predictions")
def list_predictions(session_kind: str = Query("today", alias="session", pattern="^(today|tomorrow)$"),
                     direction: Optional[str] = Query(None), page: int = Query(1, ge=1),
                     page_size: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    if direction and direction not in DIRECTIONS:
        raise HTTPException(status_code=422, detail=f"direction must be one of {', '.join(DIRECTIONS)}")
    holidays = masters.get_config(session, "market.holidays") or []
    run = _select_run(session, session_kind)
    data: dict = {"shadow": True, "notice": SHADOW_NOTICE, "session": session_kind,
                  "freshness": freshness(run, session_kind, holidays), "run": run_payload(run) if run else None,
                  "predictions": [], "total": 0, "page": page, "page_size": page_size}
    if run is None:
        return success_envelope(data, message="No snapshot available")
    stmt = select(Prediction, Company).join(Company, Company.symbol == Prediction.symbol, isouter=True) \
        .where(Prediction.run_id == run.run_id)
    if direction:
        stmt = stmt.where(Prediction.direction == direction)
    rows = session.exec(stmt).all()
    order = {"UP": 0, "DOWN": 1, "NEUTRAL": 2, "NO_CALL": 3}
    rows.sort(key=lambda pc: (order.get(pc[0].direction, 9), pc[0].symbol))
    data["total"] = len(rows)
    data["predictions"] = [prediction_payload(p, c) for p, c in rows[(page - 1) * page_size: page * page_size]]
    return success_envelope(data, message="Prediction snapshot retrieved")


@router.get("/predictions/performance")
def prediction_performance(engine_version: Optional[str] = None, horizon: Optional[int] = Query(None, ge=1, le=5),
                           run_type: Optional[str] = None, session: Session = Depends(get_session)):
    data = perf.performance(session, engine_version, horizon, run_type)
    data.update(shadow=True, notice=SHADOW_NOTICE)
    return success_envelope(to_jsonable(data), message="Prediction performance")


@router.get("/predictions/runs")
def recent_runs(limit: int = Query(30, ge=1, le=200), session: Session = Depends(get_session)):
    runs = session.exec(select(PredictionRun).order_by(PredictionRun.started_at.desc()).limit(limit)).all()
    return success_envelope([run_payload(r) for r in runs], message="Prediction runs")


@router.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str, session: Session = Depends(get_session)):
    p = session.exec(select(Prediction).where(Prediction.prediction_id == prediction_id)).first()
    if p is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    run = session.exec(select(PredictionRun).where(PredictionRun.run_id == p.run_id)).one()
    outs = session.exec(select(PredictionOutcome).where(PredictionOutcome.prediction_id == prediction_id)
                        .order_by(PredictionOutcome.horizon_sessions)).all()
    exit_state = session.exec(select(ExitState).where(ExitState.prediction_id == prediction_id)).first()
    transitions = session.exec(select(ExitTransition).where(ExitTransition.prediction_id == prediction_id)
                               .order_by(ExitTransition.session_date)).all()
    events = session.exec(select(MarketEvent).where(MarketEvent.id.in_(p.event_ids or [-1]))).all()
    from prices.service import get_quotes, quote_payload
    current = quote_payload(get_quotes(session, [p.symbol]).get(p.symbol))
    return success_envelope(to_jsonable({
        "shadow": True, "notice": SHADOW_NOTICE, "run": run_payload(run),
        "prediction": prediction_payload(p, session.get(Company, p.symbol), full=True),
        # Today's market price, separate from the frozen reference price: the
        # prediction was NOT recalculated from it.
        "current_price": {**current, "note": "current market price; the prediction used the reference price at "
                                             "its cutoff and is not recalculated"},
        "outcomes": [o.model_dump() for o in outs],
        "exit": {"state": exit_state.model_dump() if exit_state else None,
                 "transitions": [t.model_dump() for t in transitions], "mode": "SHADOW (non-actionable)"},
        "events": [e.model_dump(exclude={"raw"}) for e in events],
    }), message="Prediction retrieved")


@router.get("/stocks/{symbol}/events")
def stock_events(symbol: str, limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    from db.models.news import EventEntity, NewsIngestionRun
    rows = session.exec(select(MarketEvent).where(MarketEvent.symbol == symbol.upper())
                        .order_by(MarketEvent.effective_available_at.desc()).limit(limit)).all()
    links = session.exec(select(EventEntity).where(EventEntity.symbol == symbol.upper())).all()
    linked = []
    for ln in links:
        ev = session.get(MarketEvent, ln.event_id)
        if ev is not None:
            linked.append({"event_id": ev.id, "title": ev.title, "event_type": ev.event_type,
                           "effective_available_at": ev.effective_available_at, "relation": ln.relation,
                           "inferred": ln.inferred, "hypothesis_id": ln.hypothesis_id or None, "sign": ln.sign,
                           "mechanism": ln.mechanism})
    linked.sort(key=lambda x: str(x["effective_available_at"]), reverse=True)
    ingested = session.exec(select(NewsIngestionRun.id).where(NewsIngestionRun.status.in_(("COMPLETED", "PARTIAL")))).first()
    return success_envelope(to_jsonable({
        "symbol": symbol.upper(), "events": [e.model_dump(exclude={"raw"}) for e in rows],
        "linked_news_events": linked[:limit],
        "provider_configured": EVENT_PROVIDER_CONFIGURED or ingested is not None,
        "note": "Times: published_at (source), ingested_at (StockLens), effective_available_at = max of both."}),
        message="Events retrieved")


@router.get("/watchlist/exit-signals")
def watchlist_exit_signals(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    symbols = {w.symbol for w in session.exec(select(WatchlistItem).where(WatchlistItem.user_id == current_user.id)).all()}
    out = []
    for sym in sorted(symbols):
        p = session.exec(select(Prediction).join(PredictionRun, PredictionRun.run_id == Prediction.run_id)
                         .where(Prediction.symbol == sym, Prediction.direction.in_(("UP", "DOWN")),
                                PredictionRun.status == "COMPLETED").order_by(Prediction.created_at.desc())).first()
        if p is None:
            continue
        st = session.exec(select(ExitState).where(ExitState.prediction_id == p.prediction_id)).first()
        out.append({"symbol": sym, "prediction_id": p.prediction_id, "direction": p.direction,
                    "setup_type": p.setup_type, "reference_price": p.reference_price, "stop_loss": p.stop_loss,
                    "target": p.target, "state": st.state if st else "HOLD", "reason": st.reason if st else None,
                    "rule_version": st.rule_version if st else None,
                    "last_session_date": st.last_session_date if st else None})
    return success_envelope(to_jsonable({"shadow": True, "actionable": False, "notice": SHADOW_NOTICE,
                                         "signals": out}), message="Shadow exit states")


# ── admin routes ─────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    run_type: str = Field(pattern="^(TODAY_PREOPEN|TODAY_CONFIRMED|TOMORROW_EOD)$")


@admin_router.get("/prediction-runs")
def admin_prediction_runs(limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    from db.models.market import ScheduledJobRun
    runs = session.exec(select(PredictionRun).order_by(PredictionRun.started_at.desc()).limit(limit)).all()
    jobs_ = session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job.in_(
        ("predict_preopen", "predict_confirmed", "predict_eod", "prediction_outcomes", "prediction_monitor")))
        .order_by(ScheduledJobRun.started_at.desc()).limit(limit)).all()
    return success_envelope(to_jsonable({"runs": [run_payload(r) for r in runs],
                                         "jobs": [j.model_dump() for j in jobs_]}), message="Prediction runs")


@admin_router.post("/prediction-runs", status_code=status.HTTP_202_ACCEPTED)
def admin_start_prediction_run(body: RunRequest, current_admin: User = Depends(require_admin),
                               session: Session = Depends(get_session)):
    """Recovery tool: runs the SAME calendar-aware job as the scheduler (same
    slot lock and idempotency), in the background."""
    from scheduling import jobs
    threading.Thread(target=jobs.prediction_job, args=(engine, body.run_type), daemon=True,
                     name=f"admin-{body.run_type}").start()
    log_action(session, current_admin, "PREDICTION_RUN_TRIGGERED", "prediction_run", body.run_type, {})
    session.commit()
    return {"status": "STARTED", "detail": "See GET /admin/prediction-runs for the result."}


@admin_router.get("/events")
def admin_events(symbol: Optional[str] = None, event_type: Optional[str] = None,
                 limit: int = Query(100, ge=1, le=1000), session: Session = Depends(get_session)):
    stmt = select(MarketEvent)
    if symbol:
        stmt = stmt.where(MarketEvent.symbol == symbol.upper())
    if event_type:
        stmt = stmt.where(MarketEvent.event_type == event_type)
    rows = session.exec(stmt.order_by(MarketEvent.effective_available_at.desc()).limit(limit)).all()
    cls = session.exec(select(EventClassification).where(EventClassification.event_id.in_([r.id for r in rows] or [-1]))
                       ).all()
    by_event: dict[int, list] = {}
    for c in cls:
        by_event.setdefault(c.event_id, []).append(c.model_dump())
    return success_envelope(to_jsonable([{**r.model_dump(), "classifications": by_event.get(r.id, [])} for r in rows]),
                            message="Events")


class ReviewRequest(BaseModel):
    status: str = Field(pattern="^(CONFIRMED|CORRECTED|REJECTED)$")
    corrected: Optional[dict] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


@admin_router.patch("/events/classifications/{classification_id}")
def admin_review_classification(classification_id: int, body: ReviewRequest,
                                current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    from events.ingest import review
    c = session.get(EventClassification, classification_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Classification not found")
    allowed = {"direction", "severity", "credibility", "expected_value", "actual_value"}
    if body.corrected and set(body.corrected) - allowed:
        raise HTTPException(status_code=422, detail=f"correctable fields: {', '.join(sorted(allowed))}")
    out = review(session, c, reviewer_id=current_admin.id, status=body.status, corrected=body.corrected,
                 notes=body.notes)
    payload = to_jsonable(out.model_dump())          # before the audit commit expires the instance
    log_action(session, current_admin, "EVENT_CLASSIFICATION_REVIEWED", "event_classification",
               str(classification_id), {"status": body.status, "new_id": payload["id"]})
    session.commit()
    return success_envelope(payload, message="Reviewed")


class SymbolsRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=1000)
    note: Optional[str] = Field(default=None, max_length=500)


@admin_router.get("/v2-universe")
def admin_v2_universe(session: Session = Depends(get_session)):
    rows = session.exec(select(V2UniverseMember).order_by(V2UniverseMember.symbol)).all()
    return success_envelope(to_jsonable({"members": [r.model_dump() for r in rows],
                                         "active": sum(1 for r in rows if r.active),
                                         "note": "Independent of the v1 ranking universe."}), message="v2 universe")


@admin_router.post("/v2-universe")
def admin_v2_universe_add(body: SymbolsRequest, current_admin: User = Depends(require_admin),
                          session: Session = Depends(get_session)):
    out = universe.add_symbols(session, body.symbols, added_by=current_admin.id, note=body.note)
    log_action(session, current_admin, "V2_UNIVERSE_ADDED", "v2_universe", None, out)
    session.commit()
    return success_envelope(out, message="v2 universe updated (v1 unchanged)")


@admin_router.post("/v2-universe/deactivate")
def admin_v2_universe_deactivate(body: SymbolsRequest, current_admin: User = Depends(require_admin),
                                 session: Session = Depends(get_session)):
    done = universe.deactivate(session, body.symbols)
    log_action(session, current_admin, "V2_UNIVERSE_DEACTIVATED", "v2_universe", None, {"symbols": done})
    session.commit()
    return success_envelope({"deactivated": done}, message="v2 universe updated (v1 unchanged)")


@admin_router.get("/universe-health")
def admin_universe_health(session: Session = Depends(get_session)):
    """Read-only: v1 members whose latest market data is unavailable (e.g.
    renamed or delisted symbols), and active companies that are not in the
    v1 ranking universe (created in Admin, therefore never ranked)."""
    v1 = sorted(set(session.exec(select(StockUniverseMember.symbol)).all()))
    dead = []
    for sym in v1:
        snap = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == sym)
                            .order_by(MarketSnapshot.fetched_at.desc())).first()
        if snap is None or snap.status == "UNAVAILABLE":
            dead.append({"symbol": sym, "latest_status": snap.status if snap else "NO_SNAPSHOT",
                         "error": snap.error if snap else None,
                         "fetched_at": snap.fetched_at.isoformat() if snap and snap.fetched_at else None})
    members = set(v1)
    outside = [c.symbol for c in session.exec(select(Company).where(Company.active == True)).all()  # noqa: E712
               if c.symbol not in members]
    v2 = set(session.exec(select(V2UniverseMember.symbol).where(V2UniverseMember.active == True)).all())  # noqa: E712
    return success_envelope({
        "v1_universe_size": len(v1), "v1_without_market_data": dead,
        "active_companies_outside_v1": len(outside),
        "outside_v1_but_in_v2": sorted(set(outside) & v2),
        "note": ("Companies outside the v1 universe are never ranked by v1 full runs; adding them to v1 changes the "
                 "frozen v1 universe and needs a separate approval. They can be added to the v2 universe."),
    }, message="Universe health")
