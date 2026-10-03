"""api/routes/stocks.py — stock discovery for normal authenticated users.

Distinct from api/routes/admin.py's /admin/stocks (ADMIN-only, exposes
active/analysis_enabled toggles and bookkeeping timestamps). This is the
endpoint the mobile app's stock search/selection is meant to use - any
authenticated user, active stocks only, minimal fields.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

import stocks.service as stocks_service
from api.schemas_stocks import StockSearchResultResponse
import engine_runs.service as runs
from api.schemas import success_envelope
from auth.dependencies import get_current_user
from db.models.user import User
from db.session import engine, get_session
from ranking import presenter

# A refresh request within this window returns the existing analysis instead
# of re-fetching provider data (protects the provider and the server).
ANALYSIS_REFRESH_MIN_AGE = timedelta(minutes=60)

router = APIRouter(prefix="/stocks", dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[StockSearchResultResponse])
def search_stocks(
    search: str | None = None, limit: int = 50, offset: int = 0, session: Session = Depends(get_session)
):
    return stocks_service.search_stocks(session, search=search, limit=limit, offset=offset)


@router.get("/{symbol}", response_model=StockSearchResultResponse)
def get_stock(symbol: str, session: Session = Depends(get_session)):
    company = stocks_service.get_stock(session, symbol)
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found")
    return company



def _active_company_or_404(session: Session, symbol: str):
    company = stocks_service.get_stock(session, symbol)
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found")
    return company


def _result_or_404(session: Session, symbol: str):
    result = runs.latest_result(session, symbol)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            # User-facing text (shown by clients); the refresh endpoint is
            # POST /api/v1/stocks/{symbol}/analysis/refresh.
            detail=f"{symbol.upper()} has not been analysed yet. Run an analysis now or wait "
                   f"for the next scheduled analysis run.")
    return result


@router.get("/{symbol}/analysis")
def get_stock_analysis(symbol: str, session: Session = Depends(get_session)):
    """Full investment analysis: StockAI Score with components and reasons,
    the 18 FQVF checks, market/technical data, ML signal (informational) and
    data freshness, from the latest completed engine run."""
    company = _active_company_or_404(session, symbol)
    result = _result_or_404(session, company.symbol)
    return success_envelope(presenter.analysis_payload(session, result, company), message="Analysis retrieved")


@router.get("/{symbol}/fqvf")
def get_stock_fqvf(symbol: str, session: Session = Depends(get_session)):
    """The 18 Fundamental Quality & Value Framework checks for this stock."""
    company = _active_company_or_404(session, symbol)
    result = _result_or_404(session, company.symbol)
    return success_envelope({"symbol": company.symbol, "name": company.name, **presenter.fqvf_payload(result)},
                            message="FQVF retrieved")


@router.get("/{symbol}/ranking")
def get_stock_ranking(symbol: str, session: Session = Depends(get_session)):
    """StockAI Score, components, positive factors and risks."""
    company = _active_company_or_404(session, symbol)
    result = _result_or_404(session, company.symbol)
    return success_envelope({"symbol": company.symbol, "name": company.name, **presenter.ranking_payload(result)},
                            message="Ranking retrieved")


@router.post("/{symbol}/analysis/refresh")
def refresh_stock_analysis(
    symbol: str, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)
):
    """Analyse one stock now (synchronous, ~10-20 s). Returns the existing
    analysis unchanged if it is less than 60 minutes old."""
    company = _active_company_or_404(session, symbol)
    existing = runs.latest_result(session, company.symbol)
    computed = existing.computed_at if existing else None
    if computed is not None and computed.tzinfo is None:
        computed = computed.replace(tzinfo=timezone.utc)
    if computed is not None and datetime.now(timezone.utc) - computed < ANALYSIS_REFRESH_MIN_AGE:
        return success_envelope(presenter.analysis_payload(session, existing, company),
                                message="Analysis is less than 60 minutes old; returning the existing result")
    try:
        runs.run_single_stock(engine, company.symbol, triggered_by=current_user.id)
    except runs.RunInProgressError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    session.expire_all()
    result = _result_or_404(session, company.symbol)
    return success_envelope(presenter.analysis_payload(session, result, company), message="Analysis refreshed")
