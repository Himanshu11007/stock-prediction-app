"""
api/routes/intelligence.py — Recommendation Intelligence Engine endpoint.

GET /api/v1/intelligence/report

Read-only analytics over historical validated recommendations.
Never modifies the database, weights, thresholds, or any configuration.

Requires an authenticated user (any role).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from analytics.intelligence_overview import overview
from db.session import get_session

from api import services
from api.schemas import success_envelope
from auth.dependencies import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/intelligence/overview")
def intelligence_overview(session: Session = Depends(get_session)):
    """What each kind of intelligence is and how much weight it carries:
    market regime, technical, fundamental (FQVF), ML signal (informational,
    no demonstrated skill), news (not point-in-time, not scored), legacy
    analytics (retired engine)."""
    return success_envelope(overview(session), message="Intelligence overview")


@router.get("/intelligence/report")
def intelligence_report():
    """
    Return the full Recommendation Intelligence Report.

    Analyses all validated recommendations and returns:
      - summary metrics
      - threshold analysis (0.50 → 0.70)
      - confidence band analysis
      - pillar correlation analysis
      - sector performance
      - regime performance
      - signal performance
      - deterministic developer recommendations

    Read-only: never modifies database, weights, or thresholds.
    """
    result = services.get_intelligence_report()
    n = result.get("meta", {}).get("records_analyzed", 0)
    return success_envelope(result, message=f"Intelligence report generated ({n} records analysed)")