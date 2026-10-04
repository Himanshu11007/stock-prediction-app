"""
api/routes/admin_masters.py — ADMIN-only master data, configuration, engine
runs and health.

Complements api/routes/admin.py (users, roles, stock master, recommendations,
watchlist, audit log). Every mutation writes an admin_audit_logs row in the
same transaction (admin.audit.log_action). Read endpoints return stored data
only; nothing here fabricates or back-fills values.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

import engine_runs.service as runs
import masters.service as masters
from admin.audit import log_action
from api.schemas_admin import (ConfigUpdateRequest, EngineRunStartRequest, IndustryUpdateRequest,
                               SectorUpdateRequest)
from auth.dependencies import require_admin
from config import RANKING_ENGINE_STATUS, RANKING_ENGINE_VERSION
from data_health.service import api_health, data_health, engine_versions
from db.models.market import (EngineRun, FundamentalSnapshot, MarketRegimeSnapshot, MarketSnapshot,
                              RankingOutcome, RankingSnapshot, ScheduledJobRun, StockAnalysisResult)
from db.models.stock import Company
from db.models.user import User
from db.session import engine, get_session
from ranking import presenter, tracking

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


def _dump(rows) -> list[dict]:
    return [presenter.to_jsonable(r.model_dump()) for r in rows]


# ── Sector / Industry masters ────────────────────────────────────────────────

@router.get("/sectors")
def list_sectors(session: Session = Depends(get_session)):
    return presenter.to_jsonable(masters.list_sectors(session))


@router.patch("/sectors/{sector_id}")
def update_sector(sector_id: int, payload: SectorUpdateRequest,
                  current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Set or clear a sector's outlook (FQVF check 18) or its active flag."""
    sector = masters.update_sector(session, current_admin, sector_id, outlook=payload.outlook,
                                   outlook_notes=payload.outlook_notes, active=payload.active,
                                   clear_outlook=payload.clear_outlook)
    return presenter.to_jsonable(sector.model_dump())


@router.get("/industries")
def list_industries(session: Session = Depends(get_session)):
    return presenter.to_jsonable(masters.list_industries(session))


@router.patch("/industries/{industry_id}")
def update_industry(industry_id: int, payload: IndustryUpdateRequest,
                    current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    ind = masters.update_industry(session, current_admin, industry_id, active=payload.active,
                                  sector_id=payload.sector_id)
    return presenter.to_jsonable(ind.model_dump())


# ── Provider data (read-only) ────────────────────────────────────────────────

def _latest_rows(session: Session, model, symbol: Optional[str], limit: int, offset: int):
    if symbol:
        stmt = select(model).where(model.symbol == symbol.strip().upper()).order_by(model.fetched_at.desc())
        return session.exec(stmt.offset(offset).limit(limit)).all()
    # latest per symbol
    latest: dict[str, object] = {}
    for row in session.exec(select(model).order_by(model.fetched_at)).all():
        latest[row.symbol] = row
    return sorted(latest.values(), key=lambda r: r.symbol)[offset:offset + limit]


@router.get("/fundamentals")
def fundamentals(symbol: Optional[str] = None, limit: int = Query(50, ge=1, le=500), offset: int = 0,
                 session: Session = Depends(get_session)):
    """Latest fundamentals snapshot per stock, or one stock's fetch history."""
    return _dump(_latest_rows(session, FundamentalSnapshot, symbol, limit, offset))


@router.get("/market-data")
def market_data(symbol: Optional[str] = None, limit: int = Query(50, ge=1, le=500), offset: int = 0,
                session: Session = Depends(get_session)):
    """Latest market snapshot (price, technical metrics, issues) per stock."""
    rows = _latest_rows(session, MarketSnapshot, symbol, limit, offset)
    return [{"symbol": r.symbol, **presenter.market_payload(r)} for r in rows]


@router.get("/technical")
def technical(limit: int = Query(50, ge=1, le=500), offset: int = 0, session: Session = Depends(get_session)):
    """Technical / market signals per stock from the latest market snapshots."""
    rows = _latest_rows(session, MarketSnapshot, None, limit, offset)
    return [{"symbol": r.symbol, "as_of_date": r.as_of_date, **(presenter.market_payload(r)["technical"]),
             "ml_signal": (r.technical or {}).get("ml_signal")} for r in rows]


def _latest_run_results(session: Session):
    run = runs.latest_completed_run(session)
    if run is None:
        return None, []
    rows = session.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == run.run_id)
                        .order_by(StockAnalysisResult.symbol)).all()
    return run, rows


@router.get("/valuation")
def valuation(session: Session = Depends(get_session)):
    """Valuation checks (FQVF 6-11, 14) for every stock in the latest run."""
    run, rows = _latest_run_results(session)
    ids = (6, 7, 8, 9, 10, 11, 14)
    return {"run_id": run.run_id if run else None, "items": [
        {"symbol": r.symbol, "checks": [{k: c[k] for k in ("id", "name", "status", "grade", "value")}
                                        for c in (r.fqvf or {}).get("checks", []) if c["id"] in ids]}
        for r in rows]}


@router.get("/market-regime")
def market_regime_history(limit: int = Query(30, ge=1, le=365), session: Session = Depends(get_session)):
    rows = session.exec(select(MarketRegimeSnapshot).order_by(MarketRegimeSnapshot.computed_at.desc())
                        .limit(limit)).all()
    return [presenter.regime_payload(r) for r in rows]


# ── Framework reference and configuration ────────────────────────────────────

@router.get("/fqvf/reference")
def fqvf_reference():
    """The fixed 18-check framework and its thresholds (read-only by design)."""
    return presenter.fqvf_reference()


@router.get("/config")
def list_config(session: Session = Depends(get_session)):
    return presenter.to_jsonable(masters.list_config(session))


@router.put("/config/{key}")
def set_config(key: str, payload: ConfigUpdateRequest, current_admin: User = Depends(require_admin),
               session: Session = Depends(get_session)):
    """Set one configuration value (validated; e.g. ranking.weights, app.features)."""
    if key not in masters.CONFIG_KEYS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown configuration key: {key}")
    return presenter.to_jsonable(masters.set_config(session, current_admin, key, payload.value))


@router.get("/ranking/config")
def ranking_config(session: Session = Depends(get_session)):
    weights, rules = masters.ranking_config(session)
    return presenter.ranking_reference(weights, rules)


# ── Analysis results / Top Picks (admin view incl. ineligible) ──────────────

@router.get("/analysis-results")
def analysis_results(run_id: Optional[str] = None, eligible: Optional[bool] = None,
                     limit: int = Query(100, ge=1, le=1000), offset: int = 0,
                     session: Session = Depends(get_session)):
    """Every stock's score in a run (default: latest completed), including
    ineligible stocks and the reasons they were excluded from Top Picks."""
    run = (session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).first() if run_id
           else runs.latest_completed_run(session))
    if run is None:
        return {"run": None, "items": []}
    stmt = select(StockAnalysisResult).where(StockAnalysisResult.run_id == run.run_id)
    if eligible is not None:
        stmt = stmt.where(StockAnalysisResult.eligible == eligible)
    rows = session.exec(stmt.order_by(StockAnalysisResult.rank.is_(None), StockAnalysisResult.rank,
                                      StockAnalysisResult.stockai_score.desc()).offset(offset).limit(limit)).all()
    return {"run": presenter.to_jsonable(run.model_dump(exclude={"errors", "config"})), "items": [
        {"symbol": r.symbol, "rank": r.rank, "stockai_score": r.stockai_score, "score_coverage": r.score_coverage,
         "eligible": r.eligible, "ineligible_reasons": r.ineligible_reasons,
         "fqvf_score": (r.fqvf or {}).get("score"), "fqvf_summary": (r.fqvf or {}).get("summary")} for r in rows]}


# ── Engine runs ──────────────────────────────────────────────────────────────

@router.get("/engine-runs")
def list_engine_runs(limit: int = Query(30, ge=1, le=200), session: Session = Depends(get_session)):
    rows = session.exec(select(EngineRun).order_by(EngineRun.started_at.desc()).limit(limit)).all()
    return [presenter.to_jsonable({**r.model_dump(exclude={"errors", "config"}),
                                   "error_count": len(r.errors or [])}) for r in rows]


@router.get("/engine-runs/{run_id}")
def get_engine_run(run_id: str, session: Session = Depends(get_session)):
    run = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Engine run not found")
    return presenter.to_jsonable(run.model_dump())


@router.post("/engine-runs", status_code=status.HTTP_202_ACCEPTED)
def start_engine_run(payload: EngineRunStartRequest, current_admin: User = Depends(require_admin),
                     session: Session = Depends(get_session)):
    """Start a ranking run in the background (409 if one is already running)."""
    if payload.limit is not None and payload.limit < 1:
        raise ValueError("limit must be >= 1")
    try:
        run = runs.start_run_in_background(engine, triggered_by=current_admin.id, symbols=payload.symbols,
                                           limit=payload.limit, include_ml=payload.include_ml,
                                           refresh_fundamentals=payload.refresh_fundamentals)
    except runs.RunInProgressError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    log_action(session, current_admin, "ENGINE_RUN_STARTED", "engine_run", run.run_id,
               {"symbols": payload.symbols, "limit": payload.limit, "include_ml": payload.include_ml,
                "refresh_fundamentals": payload.refresh_fundamentals})
    session.commit()
    return {"run_id": run.run_id, "status": "RUNNING"}


# ── Scheduled jobs (daily ranking, prices) ───────────────────────────────────

@router.get("/scheduled-jobs")
def scheduled_jobs(limit: int = Query(30, ge=1, le=500), session: Session = Depends(get_session)):
    """Scheduler status: recent job executions (cron or admin-triggered), the
    latest ranking run with its ranking date and notification outcome, and
    the freshness of current prices."""
    from db.models.notifications import NotificationRun
    from notifications.detector import latest_full_run
    from prices.service import freshness_summary
    jobs_rows = session.exec(select(ScheduledJobRun).order_by(ScheduledJobRun.started_at.desc()).limit(limit)).all()
    run = latest_full_run(session)
    latest = None
    if run is not None:
        nrun = session.exec(select(NotificationRun).where(NotificationRun.kind == "RANKING_CHANGES",
                                                          NotificationRun.source_key == run.run_id)).first()
        latest = {"run_id": run.run_id, "status": run.status, "started_at": run.started_at,
                  "finished_at": run.finished_at, "ranking_date": presenter.ranking_date(session, run),
                  "processed": run.processed, "succeeded": run.succeeded, "skipped": run.skipped,
                  "failed": run.failed, "engine_version": run.engine_version,
                  "scheduled": bool((run.config or {}).get("scheduled")),
                  "notifications": nrun.model_dump() if nrun else None}
    return presenter.to_jsonable({"jobs": [r.model_dump() for r in jobs_rows], "latest_ranking": latest,
                                  "current_prices": freshness_summary(session)})


@router.post("/scheduled-jobs/ranking/run", status_code=status.HTTP_202_ACCEPTED)
def run_ranking_job_now(current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Run the calendar-aware daily ranking job once, in the background (the
    same code the cron job runs: trading-day and 16:00 IST checks, per-day
    lock, notifications, prices). For a ranking outside those rules use
    POST /admin/engine-runs."""
    import threading

    from scheduling import jobs
    threading.Thread(target=jobs.ranking_job, args=(engine,), name="admin-daily-ranking", daemon=True).start()
    log_action(session, current_admin, "SCHEDULED_RANKING_TRIGGERED", "scheduled_job", "ranking", {})
    session.commit()
    return {"status": "STARTED", "detail": "Daily ranking job started; see GET /admin/scheduled-jobs for the result."}


# ── Prospective ranking tracking (append-only) ───────────────────────────────

@router.get("/ranking-tracking/snapshots")
def ranking_snapshots(run_id: Optional[str] = None, limit: int = Query(100, ge=1, le=1000), offset: int = 0,
                      session: Session = Depends(get_session)):
    q = select(RankingSnapshot)
    if run_id:
        q = q.where(RankingSnapshot.run_id == run_id)
    rows = session.exec(q.order_by(RankingSnapshot.ranked_at.desc(), RankingSnapshot.rank)
                        .offset(offset).limit(limit)).all()
    return _dump(rows)


@router.get("/ranking-tracking/summary")
def ranking_tracking_summary(session: Session = Depends(get_session)):
    return {"engine_status": RANKING_ENGINE_STATUS, "engine_version": RANKING_ENGINE_VERSION,
            "snapshots": len(session.exec(select(RankingSnapshot.id)).all()),
            "outcomes": len(session.exec(select(RankingOutcome.id)).all()),
            "performance": tracking.summarise_outcomes(session)}


@router.post("/ranking-tracking/outcomes")
def record_ranking_outcomes(current_admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Insert realised outcomes for elapsed horizons (existing rows are never changed)."""
    result = tracking.record_outcomes(session)
    log_action(session, current_admin, "RANKING_OUTCOMES_RECORDED", "ranking_outcomes", None, result)
    session.commit()
    return result


# ── Health and versions ──────────────────────────────────────────────────────

@router.get("/data-health")
def get_data_health(universe_only: bool = True, session: Session = Depends(get_session)):
    return presenter.to_jsonable(data_health(session, universe_only=universe_only))


@router.get("/api-health")
def get_api_health(session: Session = Depends(get_session)):
    return presenter.to_jsonable(api_health(session))


@router.get("/engine-versions")
def get_engine_versions(session: Session = Depends(get_session)):
    counts = {}
    for version in session.exec(select(StockAnalysisResult.engine_version)).all():
        counts[version] = counts.get(version, 0) + 1
    rec_versions = {}
    from storage.tracker import _connect  # legacy tracker DB (recommendation_validation)
    try:
        con = _connect()
        rec_versions = {str(v): n for v, n in con.execute(
            "SELECT engine_version, COUNT(*) FROM recommendation_validation GROUP BY engine_version")}
        con.close()
    except Exception:
        rec_versions = {"unavailable": 0}
    return {"current": engine_versions(), "analysis_results_by_ranking_version": counts,
            "recommendations_by_engine_version": rec_versions,
            "notes": "NULL/v1.0 recommendation rows predate the Phase 11A temporal-integrity fix "
                     "(docs/PRODUCTION_TEMPORAL_INTEGRITY.md)."}


@router.get("/stock-master/summary")
def stock_master_summary(session: Session = Depends(get_session)):
    """Counts by data status for the Stock Master page."""
    out: dict[str, int] = {}
    for status_value in session.exec(select(Company.data_status)).all():
        out[str(status_value)] = out.get(str(status_value), 0) + 1
    return out
