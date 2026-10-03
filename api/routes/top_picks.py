"""
api/routes/top_picks.py — Background scan trigger, status, and result.

Requires an authenticated user (any role) - a normal mobile/web user's Top
Picks action, not an admin-only operation.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

import engine_runs.service as runs
import masters.service as masters
from api import services
from api.schemas import StartScanRequest, success_envelope
from auth.dependencies import get_current_user
from db.models.market import StockAnalysisResult
from db.models.stock import Company
from db.session import get_session
from notifications.detector import latest_full_run
from ranking import presenter

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/top-picks")
def top_investment_candidates(
    limit: int | None = Query(default=None, ge=1, le=100), session: Session = Depends(get_session)
):
    """
    Top Investment Candidates: the highest-ranked eligible stocks from the
    latest completed analysis-engine run (FQVF + StockLens Score), with
    positive factors, risks, data freshness and engine version.

    Rankings compare stocks on available data; they are not predictions or
    guarantees of returns. Returns an empty list (not an error) when no run
    has completed yet.
    """
    cap = masters.get_config(session, "top_picks.limit")
    limit = min(limit or cap, cap)
    # Prefer the latest full-universe run: an administrator's partial run
    # (explicit symbols or a limit) must not shrink the users' list.
    run = latest_full_run(session) or runs.latest_completed_run(session)
    data = {
        "items": [], "total_eligible": 0, "limit": limit, "run": None,
        "market_regime": presenter.regime_payload(runs.latest_market_regime(session)),
        "disclaimer": masters.get_config(session, "app.disclaimer"),
    }
    if run is None:
        return success_envelope(data, message="No completed analysis run yet")
    rows = session.exec(
        select(StockAnalysisResult, Company)
        .join(Company, Company.symbol == StockAnalysisResult.symbol)
        .where(StockAnalysisResult.run_id == run.run_id, StockAnalysisResult.eligible == True,  # noqa: E712
               Company.active == True)  # noqa: E712
        .order_by(StockAnalysisResult.rank)).all()
    data["total_eligible"] = len(rows)
    data["items"] = [presenter.candidate_payload(r, c, cap) for r, c in rows[:limit]]
    data["run"] = {"run_id": run.run_id, "status": run.status, "finished_at": presenter._iso(run.finished_at),
                   "stocks_analysed": run.processed, "engine_version": run.engine_version,
                   "fqvf_version": run.fqvf_version}
    return success_envelope(data, message=f"{len(data['items'])} candidate(s)")


@router.post("/top-picks/start")
def start_scan(payload: StartScanRequest):
    """
    Trigger a background scan and receive a scan_id to poll.

    Note: the underlying scanner (scanner/background.py) always scans all
    three categories (Large/Mid/Small Cap) in a single background run —
    that existing behaviour is unchanged. This endpoint hands back a
    scan_id bound to the category you asked about, so status/result calls
    know which cached results to report on. Returns immediately; does not
    block until the scan completes.
    """
    result = services.start_top_picks_scan(payload.category)
    return success_envelope(result, message="Scan started")


@router.get("/top-picks/status/{scan_id}")
def scan_status(scan_id: str):
    """Poll the status of a previously started scan."""
    result = services.get_scan_status(scan_id)
    return success_envelope(result, message="Scan status retrieved")


@router.get("/top-picks/result/{scan_id}")
def scan_result(scan_id: str):
    """
    Fetch the cached results for a previously started scan.

    Raises ValueError (→ 400) for an unknown scan_id — handled centrally.
    """
    result = services.get_scan_result(scan_id)
    return success_envelope(result, message="Scan result retrieved")