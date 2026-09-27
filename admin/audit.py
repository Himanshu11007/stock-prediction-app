"""Admin audit logging - the only place that writes to admin_audit_logs.

log_action() only adds/flushes the audit row - it deliberately does NOT
commit. The caller (a mutation function in admin/service.py) is responsible
for committing once, after both the entity change and the audit entry are
staged in the same session, so the two can never be split across separate
transactions: either both persist or (on any failure before that single
commit) neither does.
"""
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
    session.flush()  # assigns entry.id without committing the transaction
    return entry
