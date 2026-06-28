"""
api/routes/tracker.py — Saved recommendations + manual save + validation trigger.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import (
    RecommendationSaveRequest, RecommendationSaveResponse,
    ValidateOldResponse,
)
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)
router = APIRouter()


@router.get("/tracker/recommendations")
def get_recommendations(limit: int = 50):
    """Return recently saved prediction signals."""
    try:
        return services.get_saved_recommendations(limit=limit)
    except Exception as e:
        log_exception(logger, "get_recommendations failed", e)
        raise HTTPException(status_code=500, detail="Failed to fetch recommendations")


@router.post("/tracker/save", response_model=RecommendationSaveResponse)
def save_recommendation(payload: RecommendationSaveRequest):
    """Manually save a recommendation row."""
    try:
        row_id = services.save_manual_recommendation(
            symbol=payload.symbol,
            stock=payload.stock,
            signal=payload.signal,
            cmp=payload.cmp,
            score=payload.score,
            confidence=payload.confidence,
            news_score=payload.news_score,
            accuracy=payload.accuracy or 0.0,
            target=payload.target,
            stop_loss=payload.stop_loss,
        )
        return {"success": True, "row_id": row_id}
    except Exception as e:
        log_exception(logger, "save_recommendation failed", e)
        raise HTTPException(status_code=500, detail="Failed to save recommendation")


@router.post("/tracker/validate-old", response_model=ValidateOldResponse)
def validate_old():
    """Run the existing 5-trading-day validation engine."""
    try:
        count = services.run_validation()
        return {"validated_count": count, "status": "success"}
    except Exception as e:
        log_exception(logger, "validate_old failed", e)
        raise HTTPException(status_code=500, detail="Validation run failed")