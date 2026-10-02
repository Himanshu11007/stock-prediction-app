"""Trusted-device and refresh-session management - the only module that
should touch the trusted_devices table directly (refresh_tokens itself is
still owned by auth/service.py; this module only queries/revokes rows there
by device_id/user_id, it never issues them).

A "trusted device" is purely a record that this user has completed full
(non-PIN) authentication on this device at least once, plus whether the
device has a local PIN configured (pin_enabled - informational only, see
db/models/user.py:TrustedDevice). It is NOT a credential and grants no
access by itself; the actual API authentication mechanism is unchanged
(JWT access tokens backed by refresh_tokens rows).
"""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select

from db.models.user import RefreshToken, TrustedDevice, User


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def upsert_trusted_device(
    session: Session, user: User, device_id: str, device_name: Optional[str] = None
) -> TrustedDevice:
    """Records that `user` has just completed full authentication on
    `device_id`. Idempotent: calling this again for the same (user, device)
    just bumps last_seen_at/un-revokes it rather than erroring or
    duplicating - every successful login naturally calls this."""
    device = session.exec(
        select(TrustedDevice).where(
            TrustedDevice.user_id == user.id, TrustedDevice.device_id == device_id
        )
    ).first()
    now = utcnow()
    if device is None:
        device = TrustedDevice(user_id=user.id, device_id=device_id, device_name=device_name)
        session.add(device)
    else:
        device.last_seen_at = now
        device.revoked_at = None  # a fresh full login un-revokes the device
        if device_name:
            device.device_name = device_name
        session.add(device)
    session.commit()
    session.refresh(device)
    return device


def set_pin_enabled(session: Session, user: User, device_id: str, enabled: bool) -> TrustedDevice:
    """Flips the purely-informational pin_enabled flag - called by the
    mobile client after it has set up (or cleared) its own local PIN. The
    backend never sees the PIN itself."""
    device = session.exec(
        select(TrustedDevice).where(
            TrustedDevice.user_id == user.id, TrustedDevice.device_id == device_id
        )
    ).first()
    if device is None:
        raise ValueError("Unknown device for this user - complete full authentication first")
    device.pin_enabled = enabled
    session.add(device)
    session.commit()
    session.refresh(device)
    return device


def list_sessions(session: Session, user: User) -> list[dict]:
    """One entry per currently-active (non-revoked, non-expired) refresh
    session, newest first, joined against trusted_devices for a friendly
    name where available. A session with no device_id (pre-Phase-8 token,
    or a non-device client) is still listed, just without a device name."""
    now = utcnow()
    rows = session.exec(
        select(RefreshToken)
        .where(
            RefreshToken.user_id == user.id,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > now,
        )
        .order_by(RefreshToken.created_at.desc())
    ).all()

    devices_by_id = {
        d.device_id: d
        for d in session.exec(
            select(TrustedDevice).where(TrustedDevice.user_id == user.id)
        ).all()
    }

    return [
        {
            "session_id": row.id,
            "device_id": row.device_id,
            "device_name": devices_by_id[row.device_id].device_name
            if row.device_id and row.device_id in devices_by_id
            else None,
            "created_at": row.created_at,
            "expires_at": row.expires_at,
        }
        for row in rows
    ]


def revoke_session(session: Session, user: User, session_id: int) -> bool:
    """Revokes one specific refresh session by its row id. Returns False
    (no-op, not an error) if it doesn't exist, isn't this user's, or is
    already revoked - matches revoke_refresh_token's silent-no-op style."""
    row = session.get(RefreshToken, session_id)
    if row is None or row.user_id != user.id or row.revoked_at is not None:
        return False
    row.revoked_at = utcnow()
    session.add(row)
    session.commit()
    return True


def revoke_device(session: Session, user: User, device_id: str) -> int:
    """Revokes every active refresh session tied to `device_id` for this
    user, and marks the trusted_devices row itself revoked (so a stale
    locally-cached PIN on that device can no longer unlock anything - see
    docs/AUTHENTICATION.md's PIN security model: PIN unlock always re-
    validates the stored refresh token server-side before it's trusted).
    Returns the number of refresh sessions revoked."""
    revoked_count = _revoke_matching(session, user, device_id=device_id)

    device = session.exec(
        select(TrustedDevice).where(
            TrustedDevice.user_id == user.id, TrustedDevice.device_id == device_id
        )
    ).first()
    if device is not None and device.revoked_at is None:
        device.revoked_at = utcnow()
        session.add(device)
        session.commit()

    return revoked_count


def revoke_all_sessions(session: Session, user: User, *, except_device_id: Optional[str] = None) -> int:
    """Revokes every active refresh session for this user - "sign out all
    devices" when except_device_id is None, or "sign out all OTHER devices"
    when the caller's own current device_id is passed. Returns the number
    of sessions revoked."""
    return _revoke_matching(session, user, exclude_device_id=except_device_id)


def _revoke_matching(
    session: Session,
    user: User,
    *,
    device_id: Optional[str] = None,
    exclude_device_id: Optional[str] = None,
) -> int:
    now = utcnow()
    stmt = select(RefreshToken).where(
        RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None)
    )
    if device_id is not None:
        stmt = stmt.where(RefreshToken.device_id == device_id)
    rows = session.exec(stmt).all()
    revoked_count = 0
    for row in rows:
        if exclude_device_id is not None and row.device_id == exclude_device_id:
            continue
        row.revoked_at = now
        session.add(row)
        revoked_count += 1
    if revoked_count:
        session.commit()
    return revoked_count
