"""
api/routes/performance.py — Read-only performance analytics endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import success_envelope

router = APIRouter()


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