"""Database-backed rate limiting for authentication endpoints - the only
module that should touch the auth_throttle_events table.

Counted in the database (not in process memory) so the limit holds across
every API worker, container and restart. A limit is a soft bound: under
heavy concurrency a few requests over the limit can slip through between
the count and the insert, which is acceptable for throttling (unlike token
consumption, where exactly-once is enforced with atomic updates instead).

Keys (client IPs, emails) are stored only as SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, func
from sqlmodel import Session, select

from db.models.user import AuthThrottleEvent


class ThrottledError(Exception):
    """The caller exceeded a rate limit. retry_after_seconds is a safe
    upper bound for a Retry-After header."""

    def __init__(self, message: str, *, retry_after_seconds: int):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.strip().lower().encode("utf-8")).hexdigest()


def count_recent(session: Session, bucket: str, key: str, *, window_seconds: int) -> int:
    since = _now() - timedelta(seconds=window_seconds)
    return session.exec(
        select(func.count()).select_from(AuthThrottleEvent).where(
            AuthThrottleEvent.bucket == bucket,
            AuthThrottleEvent.key_hash == _key_hash(key),
            AuthThrottleEvent.created_at >= since,
        )
    ).one()


def record(session: Session, bucket: str, key: str) -> None:
    """Stages one event (the caller commits)."""
    session.add(AuthThrottleEvent(bucket=bucket, key_hash=_key_hash(key)))


def check_and_record(
    session: Session, bucket: str, key: Optional[str], *, limit: int, window_seconds: int, message: str
) -> None:
    """Raises ThrottledError if `key` already used `limit` requests in the
    window; otherwise records this request and commits. A missing key
    (no client address) is not throttled here - other limits still apply."""
    if not key or limit <= 0:
        return
    if count_recent(session, bucket, key, window_seconds=window_seconds) >= limit:
        raise ThrottledError(message, retry_after_seconds=window_seconds)
    record(session, bucket, key)
    _prune(session, bucket, window_seconds)
    session.commit()


def _prune(session: Session, bucket: str, window_seconds: int) -> None:
    # Keeps the table small: anything older than a day (and older than the
    # bucket's own window) can never count again.
    cutoff = _now() - timedelta(seconds=max(window_seconds, 86400))
    session.execute(delete(AuthThrottleEvent).where(
        AuthThrottleEvent.bucket == bucket, AuthThrottleEvent.created_at < cutoff))
