"""
api/routes/admin_notifications.py — administrator notification controls and
user reports (ADMIN role).

Settings (global switch, event types, thresholds, rate limits, quiet-hour
defaults) and templates are app_config keys "notifications.settings" and
"notifications.templates", changed via PUT /admin/config/{key} (audited).
The actions below are audited too. Push tokens are never returned.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session, select

import feedback.service as feedback_service
import notifications.service as notify
from admin.audit import log_action
from auth.dependencies import require_admin
from db.models.notifications import Notification, NotificationRun
from db.models.user import User
from db.session import get_session
from notifications import detector
from ranking import presenter

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@router.get("/notifications/stats")
def notification_stats(session: Session = Depends(get_session)):
    return presenter.to_jsonable({**notify.admin_stats(session), "settings": notify.settings(session)})


@router.get("/notifications/runs")
def notification_runs(limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)):
    rows = session.exec(select(NotificationRun).order_by(NotificationRun.started_at.desc()).limit(limit)).all()
    return [presenter.to_jsonable(r.model_dump()) for r in rows]


@router.get("/notifications/recent")
def recent_notifications(limit: int = Query(100, ge=1, le=1000), type: Optional[str] = None,
                         session: Session = Depends(get_session)):
    q = select(Notification)
    if type:
        q = q.where(Notification.type == type)
    rows = session.exec(q.order_by(Notification.created_at.desc()).limit(limit)).all()
    return [presenter.to_jsonable({**notify.notification_payload(n), "user_id": n.user_id}) for n in rows]


class ProcessRunRequest(BaseModel):
    run_id: Optional[str] = None


@router.post("/notifications/process-run")
def process_run(payload: ProcessRunRequest, admin: User = Depends(require_admin),
                session: Session = Depends(get_session)):
    """Run the notification engine for a ranking run (default: the latest
    full run). Idempotent: an already-processed run returns its record."""
    run_id = payload.run_id
    if run_id is None:
        latest = detector.latest_full_run(session)
        if latest is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No completed full ranking run")
        run_id = latest.run_id
    nrun = notify.process_ranking_run(session, run_id)
    result = presenter.to_jsonable(nrun.model_dump())   # before the commit expires the instance
    log_action(session, admin, "NOTIFICATIONS_PROCESSED", "notification_run", nrun.id,
               {"ranking_run_id": run_id, "status": nrun.status})
    session.commit()
    return result


@router.post("/notifications/daily-summary")
def daily_summary(admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    result = notify.send_daily_summaries(session)
    log_action(session, admin, "DAILY_SUMMARY_TRIGGERED", "notification_run", None, result)
    session.commit()
    return result


@router.post("/notifications/dispatch")
def dispatch(admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Push notifications queued for quiet hours whose window has ended."""
    result = notify.dispatch_pending(session)
    log_action(session, admin, "NOTIFICATIONS_DISPATCHED", "notification", None, result)
    session.commit()
    return result


@router.post("/notifications/test")
def test_push(admin: User = Depends(require_admin), session: Session = Depends(get_session)):
    """Send a test notification to the calling administrator's own devices."""
    n = notify.send_test(session, admin)
    result = notify.notification_payload(n)
    log_action(session, admin, "TEST_NOTIFICATION_SENT", "notification", n.id, {"push_status": n.push_status})
    session.commit()
    return result


# ── User reports ─────────────────────────────────────────────────────────────

@router.get("/feedback")
def list_feedback(status_filter: Optional[str] = Query(None, alias="status"), limit: int = Query(200, ge=1, le=1000),
                  session: Session = Depends(get_session)):
    return [feedback_service.payload(f, admin=True) for f in feedback_service.list_all(session, status_filter, limit)]


class FeedbackUpdate(BaseModel):
    status: Optional[str] = None
    admin_note: Optional[str] = None


@router.patch("/feedback/{feedback_id}")
def update_feedback(feedback_id: int, payload: FeedbackUpdate, admin: User = Depends(require_admin),
                    session: Session = Depends(get_session)):
    f = feedback_service.update(session, admin, feedback_id, payload.status, payload.admin_note)
    if f is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return feedback_service.payload(f, admin=True)
