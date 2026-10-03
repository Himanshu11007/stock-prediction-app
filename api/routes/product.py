"""
api/routes/product.py — application configuration, market regime and the
FQVF / ranking reference, read by the mobile app.

/app/config is public (no secrets; the mobile app reads feature flags and the
disclaimer before sign-in). Everything else requires an authenticated user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

import engine_runs.service as runs
import masters.service as masters
from api.schemas import success_envelope
from auth.dependencies import get_current_user
from data_health.service import engine_versions
from db.session import get_session
from ranking import presenter

public_router = APIRouter()
router = APIRouter(dependencies=[Depends(get_current_user)])


@public_router.get("/app/config")
def app_config(session: Session = Depends(get_session)):
    """Backend-controlled client configuration: feature flags, disclaimer,
    announcement, Top Picks size and engine versions."""
    return success_envelope({
        "features": masters.get_config(session, "app.features"),
        "disclaimer": masters.get_config(session, "app.disclaimer"),
        "announcement": masters.get_config(session, "app.announcement"),
        "top_picks_limit": masters.get_config(session, "top_picks.limit"),
        "versions": engine_versions(),
    }, message="Configuration retrieved")


@router.get("/market/regime")
def market_regime(session: Session = Depends(get_session)):
    """Latest NIFTY 50 market regime computed by the analysis engine."""
    regime = presenter.regime_payload(runs.latest_market_regime(session))
    return success_envelope(regime, message="Market regime retrieved" if regime else "No market regime computed yet")


@router.get("/fqvf/reference")
def fqvf_reference():
    """Definitions, statuses and thresholds of the 18 FQVF checks."""
    return success_envelope(presenter.fqvf_reference(), message="FQVF reference")


@router.get("/ranking/reference")
def ranking_reference(session: Session = Depends(get_session)):
    """StockAI Score components, current weights and eligibility rules."""
    weights, rules = masters.ranking_config(session)
    return success_envelope(presenter.ranking_reference(weights, rules), message="Ranking reference")
