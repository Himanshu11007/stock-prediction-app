"""Admin action audit log.

One row per mutating admin action (user activate/deactivate, role change,
stock enable/disable, etc.) - who did what, to what, and when. Read-only
data otherwise: nothing in the application ever updates or deletes a row
once written.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AdminAuditLog(SQLModel, table=True):
    __tablename__ = "admin_audit_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    admin_user_id: int = Field(foreign_key="users.id", index=True)
    action: str = Field(index=True)
    entity: str = Field(index=True)
    entity_id: Optional[str] = Field(default=None, index=True)
    extra_data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
