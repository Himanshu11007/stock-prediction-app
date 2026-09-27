"""Admin audit logging - the only place that writes to admin_audit_logs."""
from typing import Any, Optional

from sqlmodel import Session

from db.models.admin import AdminAuditLog
from db.models.user import User


def log_action(
    session: Session,
    admin_user: User,
    action: str,
    entity: str,
    entity_id: Optional[str] = None,
    extra_data: Optional[dict[str, Any]] = None,
) -> AdminAuditLog:
    entry = AdminAuditLog(
        admin_user_id=admin_user.id,
        action=action,
        entity=entity,
        entity_id=str(entity_id) if entity_id is not None else None,
        extra_data=extra_data,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry
