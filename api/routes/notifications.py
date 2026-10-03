"""
api/routes/notifications.py — the signed-in user's Notification Center,
notification preferences, push devices, and reports (feedback / account
deletion requests).

Every route is scoped to the authenticated user from the access token; no
route accepts a user id. Push tokens are returned masked and only to their
owner.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session

import feedback.service as feedback_service
import notifications.service as notify
from api.schemas import success_envelope
from auth.dependencies import get_current_user
from db.models.user import User
from db.session import get_session

router = APIRouter(dependencies=[Depends(get_current_user)])


# ── Notification Center ──────────────────────────────────────────────────────

@router.get("/notifications")
def list_notifications(unread_only: bool = False, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                       user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    rows, total, unread = notify.list_notifications(session, user, unread_only, limit, offset)
    return success_envelope({"items": [notify.notification_payload(n) for n in rows], "total": total,
                             "unread": unread, "limit": limit, "offset": offset},
                            message="You're all caught up." if not rows else f"{len(rows)} notification(s)")


@router.get("/notifications/unread-count")
def unread_count(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return success_envelope({"unread": notify.unread_count(session, user)}, message="Unread count")


@router.post("/notifications/read-all")
def read_all(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return success_envelope({"marked": notify.mark_all_read(session, user)}, message="All notifications marked as read")


@router.post("/notifications/{notification_id}/read")
def read_one(notification_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    n = notify.mark_read(session, user, notification_id)
    if n is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    return success_envelope(notify.notification_payload(n), message="Marked as read")


# ── Preferences ──────────────────────────────────────────────────────────────

class PreferencesUpdate(BaseModel):
    push_enabled: Optional[bool] = None
    new_top_candidate: Optional[bool] = None
    top_candidate_removed: Optional[bool] = None
    score_changes: Optional[bool] = None
    fqvf_changes: Optional[bool] = None
    watchlist_alerts: Optional[bool] = None
    daily_summary: Optional[bool] = None
    market_regime: Optional[bool] = None
    quiet_hours_enabled: Optional[bool] = None
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    daily_summary_time: Optional[str] = None


@router.get("/notifications/preferences")
def get_preferences(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return success_envelope(notify.preferences_payload(notify.get_preferences(session, user)),
                            message="Notification preferences")


@router.put("/notifications/preferences")
def put_preferences(payload: PreferencesUpdate, user: User = Depends(get_current_user),
                    session: Session = Depends(get_session)):
    changes = payload.model_dump(exclude_none=True)
    pref = notify.update_preferences(session, user, changes)   # ValueError -> 400
    return success_envelope(notify.preferences_payload(pref), message="Notification preferences saved")


# ── Devices ──────────────────────────────────────────────────────────────────

class DeviceRegistration(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=200)
    platform: Literal["android", "ios"]
    push_token: Optional[str] = Field(default=None, max_length=4096)
    app_version: Optional[str] = Field(default=None, max_length=40)
    permission: Literal["granted", "denied", "unknown"] = "unknown"
    provider: Optional[Literal["fcm", "apns"]] = None


@router.get("/devices")
def my_devices(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return success_envelope([notify.device_payload(d) for d in notify.list_devices(session, user)],
                            message="Registered devices")


@router.post("/devices")
def register_device(payload: DeviceRegistration, user: User = Depends(get_current_user),
                    session: Session = Depends(get_session)):
    d = notify.register_device(session, user, payload.device_id, payload.platform, payload.push_token,
                               payload.app_version, payload.permission, payload.provider)
    return success_envelope(notify.device_payload(d), message="Device registered")


@router.delete("/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_device(device_id: str, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    if not notify.unregister_device(session, user.id, device_id, reason="removed by user"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")


# ── Reports (feedback / data problems / account deletion) ────────────────────

class FeedbackRequest(BaseModel):
    category: Literal["INCORRECT_STOCK_DATA", "STALE_DATA", "INCORRECT_COMPANY_INFO", "RECOMMENDATION_ISSUE",
                      "APP_BUG", "OTHER"]
    message: str = Field(..., min_length=5, max_length=2000)
    symbol: Optional[str] = Field(default=None, max_length=30)
    app_version: Optional[str] = Field(default=None, max_length=40)
    platform: Optional[str] = Field(default=None, max_length=20)
    context: Optional[dict] = None


RECORDED = ("Report #{id} recorded. The StockAI Pro team reviews reports in the admin console; "
            "you can see its status under Account > My reports.")


def _submit(session, user, category, message, symbol=None, app_version=None, platform=None, context=None):
    try:
        return feedback_service.submit(session, user, category, message, symbol, app_version, platform, context)
    except feedback_service.RateLimitedError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: FeedbackRequest, user: User = Depends(get_current_user),
                    session: Session = Depends(get_session)):
    if payload.context is not None and len(str(payload.context)) > 2000:
        raise ValueError("context is too large")
    f = _submit(session, user, payload.category, payload.message, payload.symbol, payload.app_version,
                payload.platform, payload.context)
    return success_envelope(feedback_service.payload(f), message=RECORDED.format(id=f.id))


@router.get("/feedback")
def my_feedback(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return success_envelope([feedback_service.payload(f) for f in feedback_service.list_own(session, user)],
                            message="Your reports")


class DeletionRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1000)


@router.post("/account/deletion-request", status_code=status.HTTP_201_CREATED)
def request_account_deletion(payload: DeletionRequest, user: User = Depends(get_current_user),
                             session: Session = Depends(get_session)):
    """Records a deletion request for an administrator to process (account
    deletion is not automated). The account stays active until then."""
    f = _submit(session, user, "ACCOUNT_DELETION",
                "Account deletion requested." + (f" Reason: {payload.reason}" if payload.reason else ""))
    return success_envelope(feedback_service.payload(f),
                            message=f"Deletion request #{f.id} recorded. An administrator will process it; your "
                                    "account stays active until then.")
