"""Notifications, push devices, user preferences and user feedback.

  NotificationPreference   one row per user (absent = conservative defaults)
  PushDevice               a user's registered device + push token
  WatchlistAlertSetting    per-user, per-symbol watchlist alert switches
  NotificationRun          one notification-engine pass over a ranking run
                           (or a daily summary); unique per source, so the same
                           run is never processed twice
  Notification             a user's in-app notification (the Notification
                           Center). Unique (user_id, dedup_key): the same
                           event can never notify the same user twice
  NotificationDelivery     one push attempt to one device
  UserFeedback             a user's data-problem / app-issue report, reviewed
                           in the admin console

Push tokens are only ever returned to their owner (masked) and are never
logged. See docs/NOTIFICATIONS.md.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


NOTIFICATION_TYPES = (
    "NEW_TOP_CANDIDATE", "TOP_CANDIDATE_REMOVED", "SCORE_CHANGE", "FQVF_CHANGE",
    "WATCHLIST_ALERT", "DAILY_SUMMARY", "MARKET_REGIME_CHANGE", "TEST",
)

# Push outcome recorded on each Notification.
PUSH_STATUSES = (
    "PENDING", "QUEUED_QUIET_HOURS", "SENT", "PARTIAL", "FAILED", "NO_DEVICE", "PROVIDER_NOT_CONFIGURED",
    "SUPPRESSED_PREFERENCE", "SUPPRESSED_RATE_LIMIT", "SUPPRESSED_COOLDOWN", "DISABLED_GLOBALLY", "EXPIRED",
)


class NotificationPreference(SQLModel, table=True):
    __tablename__ = "notification_preferences"

    user_id: int = Field(foreign_key="users.id", primary_key=True)
    push_enabled: bool = Field(default=True)
    new_top_candidate: bool = Field(default=True)
    top_candidate_removed: bool = Field(default=False)
    score_changes: bool = Field(default=False)
    fqvf_changes: bool = Field(default=False)
    watchlist_alerts: bool = Field(default=True)
    daily_summary: bool = Field(default=False)
    market_regime: bool = Field(default=False)
    quiet_hours_enabled: bool = Field(default=True)
    quiet_hours_start: str = Field(default="22:00")   # HH:MM, Asia/Kolkata
    quiet_hours_end: str = Field(default="07:00")
    daily_summary_time: str = Field(default="08:30")  # HH:MM, Asia/Kolkata
    updated_at: datetime = Field(default_factory=utcnow)


class PushDevice(SQLModel, table=True):
    __tablename__ = "push_devices"
    __table_args__ = (UniqueConstraint("user_id", "device_id", name="uq_push_devices_user_device"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    device_id: str = Field(index=True)
    platform: str                                     # android / ios
    provider: str                                     # fcm / apns
    push_token: Optional[str] = Field(default=None, index=True)
    app_version: Optional[str] = Field(default=None)
    permission: str = Field(default="unknown")        # granted / denied / unknown
    active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    last_active_at: datetime = Field(default_factory=utcnow)
    invalidated_at: Optional[datetime] = Field(default=None)
    invalid_reason: Optional[str] = Field(default=None)


class WatchlistAlertSetting(SQLModel, table=True):
    __tablename__ = "watchlist_alert_settings"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_watchlist_alert_user_symbol"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    symbol: str = Field(index=True)
    score_changes: bool = Field(default=True)
    rank_changes: bool = Field(default=True)
    fqvf_changes: bool = Field(default=True)
    status_changes: bool = Field(default=True)
    muted: bool = Field(default=False)
    updated_at: datetime = Field(default_factory=utcnow)


class NotificationRun(SQLModel, table=True):
    __tablename__ = "notification_runs"
    __table_args__ = (UniqueConstraint("kind", "source_key", name="uq_notification_runs_kind_source"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str                                         # RANKING_CHANGES / DAILY_SUMMARY / TEST
    source_key: str                                   # ranking run_id, or summary date
    previous_run_id: Optional[str] = Field(default=None)
    engine_version: str
    status: str = Field(default="RUNNING")            # RUNNING / COMPLETED / SKIPPED / FAILED
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = Field(default=None)
    events_detected: int = Field(default=0)
    notifications_created: int = Field(default=0)
    pushes_sent: int = Field(default=0)
    pushes_suppressed: int = Field(default=0)
    detail: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class Notification(SQLModel, table=True):
    __tablename__ = "notifications"
    __table_args__ = (
        UniqueConstraint("user_id", "dedup_key", name="uq_notifications_user_dedup"),
        Index("ix_notifications_user_created", "user_id", "created_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    notification_run_id: Optional[int] = Field(default=None, foreign_key="notification_runs.id", index=True)
    type: str = Field(index=True)
    dedup_key: str
    title: str
    body: str
    symbol: Optional[str] = Field(default=None, index=True)
    route: str = Field(default="/notifications")      # in-app deep-link route
    data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    ranking_run_id: Optional[str] = Field(default=None)
    engine_version: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    read_at: Optional[datetime] = Field(default=None)
    push_status: str = Field(default="PENDING", index=True)
    push_after: Optional[datetime] = Field(default=None)
    pushed_at: Optional[datetime] = Field(default=None)


class NotificationDelivery(SQLModel, table=True):
    __tablename__ = "notification_deliveries"

    id: Optional[int] = Field(default=None, primary_key=True)
    notification_id: int = Field(foreign_key="notifications.id", index=True)
    device_id: int = Field(foreign_key="push_devices.id", index=True)
    provider: str
    status: str                                       # SENT / FAILED / INVALID_TOKEN / NOT_CONFIGURED
    provider_message_id: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    attempted_at: datetime = Field(default_factory=utcnow)


FEEDBACK_CATEGORIES = ("INCORRECT_STOCK_DATA", "STALE_DATA", "INCORRECT_COMPANY_INFO",
                       "RECOMMENDATION_ISSUE", "APP_BUG", "ACCOUNT_DELETION", "OTHER")
FEEDBACK_STATUSES = ("NEW", "IN_REVIEW", "RESOLVED", "REJECTED")


class UserFeedback(SQLModel, table=True):
    __tablename__ = "user_feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    category: str = Field(index=True)
    symbol: Optional[str] = Field(default=None)
    message: str
    app_version: Optional[str] = Field(default=None)
    platform: Optional[str] = Field(default=None)
    context: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    status: str = Field(default="NEW", index=True)
    admin_note: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)
    resolved_by: Optional[int] = Field(default=None, foreign_key="users.id")
