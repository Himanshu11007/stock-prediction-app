"""
api/routes/performance.py — Read-only performance analytics endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import PerformanceSummaryResponse
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)
router = APIRouter()


@router.get("/performance/summary", response_model=PerformanceSummaryResponse)
def performance_summary():
    """Top-level KPIs across all validated recommendations."""
    try:
        return services.get_performance_summary()
    except Exception as e:
        log_exception(logger, "performance_summary failed", e)
        raise HTTPException(status_code=500, detail="Failed to compute performance summary")


@router.get("/performance/by-signal")
def performance_by_signal():
    """Performance grouped by signal type (BUY / SELL / HOLD / etc.)."""
    try:
        return services.get_performance_by_signal()
    except Exception as e:
        log_exception(logger, "performance_by_signal failed", e)
        raise HTTPException(status_code=500, detail="Failed to compute signal performance")


@router.get("/performance/by-confidence")
def performance_by_confidence():
    """Performance grouped by ML confidence band."""
    try:
        return services.get_performance_by_confidence()
    except Exception as e:
        log_exception(logger, "performance_by_confidence failed", e)
        raise HTTPException(status_code=500, detail="Failed to compute confidence performance")


@router.get("/performance/by-confluence")
def performance_by_confluence():
    """Performance grouped by confluence-score band."""
    try:
        return services.get_performance_by_confluence()
    except Exception as e:
        log_exception(logger, "performance_by_confluence failed", e)
        raise HTTPException(status_code=500, detail="Failed to compute confluence performance")