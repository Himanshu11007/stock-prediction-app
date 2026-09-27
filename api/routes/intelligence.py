"""
api/routes/intelligence.py — Recommendation Intelligence Engine endpoint.

GET /api/v1/intelligence/report

Read-only analytics over historical validated recommendations.
Never modifies the database, weights, thresholds, or any configuration.

Requires an authenticated user (any role).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api import services
from api.schemas import success_envelope
from auth.dependencies import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


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