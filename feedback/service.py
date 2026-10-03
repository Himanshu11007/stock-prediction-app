"""
feedback/service.py — user reports reviewed in the admin console.

Reports are stored in user_feedback and listed at /admin/feedback; no email
or ticketing integration exists, so the API tells the user exactly that the
report was recorded for review (never that it was "sent" somewhere).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select

from admin.audit import log_action
from db.models.notifications import FEEDBACK_CATEGORIES, FEEDBACK_STATUSES, UserFeedback
from db.models.user import User

MAX_REPORTS_PER_DAY = 10


class RateLimitedError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def payload(f: UserFeedback, admin: bool = False) -> dict:
    out = {"id": f.id, "category": f.category, "symbol": f.symbol, "message": f.message, "status": f.status,
           "created_at": f.created_at.isoformat(), "updated_at": f.updated_at.isoformat(),
           "admin_note": f.admin_note}
    if admin:
        out.update({"user_id": f.user_id, "app_version": f.app_version, "platform": f.platform,
                    "context": f.context, "resolved_by": f.resolved_by})
    return out


def submit(session: Session, user: User, category: str, message: str, symbol: Optional[str] = None,
           app_version: Optional[str] = None, platform: Optional[str] = None,
           context: Optional[dict] = None) -> UserFeedback:
    if category not in FEEDBACK_CATEGORIES:
        raise ValueError(f"category must be one of {list(FEEDBACK_CATEGORIES)}")
    message = (message or "").strip()
    if not 5 <= len(message) <= 2000:
        raise ValueError("message must be between 5 and 2000 characters")
    since = _now() - timedelta(days=1)
    recent = session.exec(select(func.count()).select_from(UserFeedback).where(
        UserFeedback.user_id == user.id, UserFeedback.created_at >= since)).one()
    if recent >= MAX_REPORTS_PER_DAY:
        raise RateLimitedError("You have reached the limit of reports for today. Please try again tomorrow.")
    f = UserFeedback(user_id=user.id, category=category, message=message,
                     symbol=symbol.strip().upper()[:30] if symbol else None,
                     app_version=(app_version or None) and app_version[:40],
                     platform=(platform or None) and platform[:20], context=context)
    session.add(f)
    session.commit()
    session.refresh(f)
    return f


def list_own(session: Session, user: User) -> list[UserFeedback]:
    return list(session.exec(select(UserFeedback).where(UserFeedback.user_id == user.id)
                             .order_by(UserFeedback.created_at.desc())).all())


def list_all(session: Session, status: Optional[str] = None, limit: int = 200) -> list[UserFeedback]:
    q = select(UserFeedback)
    if status:
        q = q.where(UserFeedback.status == status)
    return list(session.exec(q.order_by(UserFeedback.created_at.desc()).limit(limit)).all())


def update(session: Session, admin: User, feedback_id: int, status: Optional[str],
           admin_note: Optional[str]) -> Optional[UserFeedback]:
    f = session.get(UserFeedback, feedback_id)
    if f is None:
        return None
    if status is not None and status not in FEEDBACK_STATUSES:
        raise ValueError(f"status must be one of {list(FEEDBACK_STATUSES)}")
    old = {"status": f.status, "admin_note": f.admin_note}
    if status is not None:
        f.status = status
        if status in ("RESOLVED", "REJECTED"):
            f.resolved_by = admin.id
    if admin_note is not None:
        f.admin_note = admin_note[:2000]
    f.updated_at = _now()
    session.add(f)
    log_action(session, admin, "FEEDBACK_UPDATED", "user_feedback", f.id,
               {"old": old, "new": {"status": f.status, "admin_note": f.admin_note}})
    session.commit()
    session.refresh(f)
    return f
