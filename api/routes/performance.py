"""
api/routes/performance.py — Read-only performance analytics endpoints.

Requires an authenticated user (any role) - a normal mobile/web user's
Performance screen, not an admin-only view.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from analytics.performance_overview import overview
from db.session import get_session

from api import services
from api.schemas import success_envelope
from auth.dependencies import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/performance/overview")
def performance_overview(session: Session = Depends(get_session)):
    """Performance by methodology, never pooled: live Ranking Engine v1.0
    tracking, its historical validation, and the legacy signal engine before
    and after the temporal-integrity fix."""
    return success_envelope(overview(session), message="Performance overview")


@router.get("/performance/summary")
def performance_summary():
    """Top-level KPIs across all validated recommendations."""
    result = services.get_performance_summary()
    return success_envelope(result, message="Performance summary retrieved")


@router.get("/performance/by-signal")
def performance_by_signal():
    """Performance grouped by signal type (BUY / SELL / HOLD / etc.)."""
    result = services.get_performance_by_signal()
    return success_envelope(result, message="Signal performance retrieved")


@router.get("/performance/by-confidence")
def performance_by_confidence():
    """Performance grouped by ML confidence band."""
    result = services.get_performance_by_confidence()
    return success_envelope(result, message="Confidence performance retrieved")


@router.get("/performance/by-confluence")
def performance_by_confluence():
    """Performance grouped by confluence-score band."""
    result = services.get_performance_by_confluence()
    return success_envelope(result, message="Confluence performance retrieved")