"""
api/routes/tracker.py — Saved recommendations + manual save + validation trigger.

GET /tracker/recommendations is a normal user's view of tracked
recommendations - any authenticated user. POST /tracker/save (writes
arbitrary signal/score/confidence values straight into the recommendation
data) and POST /tracker/validate-old (a global batch job that fetches
live prices for every pending recommendation) are operational/data-write
actions, not something a normal mobile user should be able to trigger -
both require ADMIN, matching how the admin panel already treats
recommendation data as read-only for normal users.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api import services
from api.schemas import RecommendationSaveRequest, success_envelope
from auth.dependencies import get_current_user, require_admin
from db.models.user import User

router = APIRouter()


@router.get("/tracker/recommendations")
def get_recommendations(limit: int = 50, current_user: User = Depends(get_current_user)):
    """Return recently saved prediction signals."""
    result = services.get_saved_recommendations(limit=limit)
    return success_envelope(result, message=f"Retrieved {len(result)} recommendation(s)")


@router.post("/tracker/save")
def save_recommendation(payload: RecommendationSaveRequest, admin_user: User = Depends(require_admin)):
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
def validate_old(admin_user: User = Depends(require_admin)):
    """Run the existing 5-trading-day validation engine."""
    count = services.run_validation()
    return success_envelope(
        {"validated_count": count},
        message=f"Validated {count} recommendation(s)",
    )