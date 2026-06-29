"""
api/routes/tracker.py — Saved recommendations + manual save + validation trigger.
"""
from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import RecommendationSaveRequest, success_envelope

router = APIRouter()


@router.get("/tracker/recommendations")
def get_recommendations(limit: int = 50):
    """Return recently saved prediction signals."""
    result = services.get_saved_recommendations(limit=limit)
    return success_envelope(result, message=f"Retrieved {len(result)} recommendation(s)")


@router.post("/tracker/save")
def save_recommendation(payload: RecommendationSaveRequest):
    """Manually save (or update, if symbol+date already exists) a recommendation row."""
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
    return success_envelope({"row_id": row_id}, message="Recommendation saved")


@router.post("/tracker/validate-old")
def validate_old():
    """Run the existing 5-trading-day validation engine."""
    count = services.run_validation()
    return success_envelope(
        {"validated_count": count},
        message=f"Validated {count} recommendation(s)",
    )