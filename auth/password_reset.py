"""Forgot-password / reset-password - the only module that should touch the
password_reset_tokens table.

Security properties:
- request_password_reset() behaves identically (from the caller's point of
  view) whether or not the email has an account: the route always returns
  the same generic message, the per-account limit is silent, and the email
  is sent after the response (BackgroundTasks), so neither the body, the
  status code nor the response time reveals account existence. Only the
  per-IP limit answers 429, and it counts every request alike.
- the raw token (256 bits from `secrets`) exists only in the emailed link;
  the database stores its SHA-256 hash.
- tokens expire (PASSWORD_RESET_TOKEN_EXPIRY_MINUTES), are single-use, and
  are consumed with ONE conditional UPDATE
  (... WHERE token_hash = ? AND used_at IS NULL AND expires_at > now), so
  of any number of concurrent requests with the same token exactly one can
  succeed. A newer link supersedes older unused ones.
- a successful reset revokes every refresh token of the user and stamps
  users.token_valid_after, which makes get_current_user reject every access
  token issued before the reset - all existing sessions end immediately.
- never logged: the token, the reset URL, the password. Emails are masked.

Accounts without a password (created with Google or email OTP) may set one
through this flow. That does not weaken them: the link goes to the account's
own email address, and anyone who can read that mailbox can already sign in
with an email OTP. Deactivated accounts never get a link.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import update
from sqlmodel import Session, select

from auth import throttle
from auth.security import hash_password
from auth.security_events import fingerprint, log_security_event, mask_email
from auth.service import get_user_by_email
from config import (
    PASSWORD_RESET_MAX_ATTEMPTS_PER_IP,
    PASSWORD_RESET_MAX_EMAILS_PER_ACCOUNT,
    PASSWORD_RESET_MAX_REQUESTS_PER_IP,
    PASSWORD_RESET_TOKEN_EXPIRY_MINUTES,
    PASSWORD_RESET_WINDOW_SECONDS,
)
from db.models.user import PasswordResetToken, RefreshToken, User

GENERIC_FORGOT_PASSWORD_MESSAGE = (
    "If an account exists for this email address, a password reset link has been sent."
)

_REQUEST_BUCKET = "password_reset_request_ip"
_ATTEMPT_BUCKET = "password_reset_attempt_ip"


class InvalidResetTokenError(Exception):
    """Unknown, expired or already-used token - deliberately one error for
    all three, with one message."""


class WeakPasswordError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_reset_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)  # 256 bits


def request_password_reset(
    session: Session,
    email: str,
    *,
    requested_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    reset_url_configured: bool = True,
) -> Optional[tuple[str, str]]:
    """Returns (account_email, raw_token) when a link should be emailed,
    None otherwise - the caller must respond identically in both cases.
    Raises throttle.ThrottledError only for the per-IP limit."""
    throttle.check_and_record(
        session, _REQUEST_BUCKET, requested_ip,
        limit=PASSWORD_RESET_MAX_REQUESTS_PER_IP, window_seconds=PASSWORD_RESET_WINDOW_SECONDS,
        message="Too many password reset requests. Please try again later.",
    )

    normalized = email.strip().lower()
    user = get_user_by_email(session, normalized)
    if user is None or not user.is_active or not user.email:
        log_security_event("PASSWORD_RESET_REQUESTED", outcome="no_eligible_account",
                           email=mask_email(normalized), ip=requested_ip)
        return None

    now = _now()
    window_start = now - timedelta(seconds=PASSWORD_RESET_WINDOW_SECONDS)
    recent = session.exec(
        select(PasswordResetToken.id).where(
            PasswordResetToken.user_id == user.id, PasswordResetToken.created_at >= window_start)
    ).all()
    if len(recent) >= PASSWORD_RESET_MAX_EMAILS_PER_ACCOUNT:
        log_security_event("PASSWORD_RESET_REQUESTED", outcome="account_limit_reached",
                           user_id=user.id, ip=requested_ip)
        return None

    if not reset_url_configured:
        log_security_event("PASSWORD_RESET_REQUESTED", outcome="reset_url_not_configured", user_id=user.id)
        return None

    # A new link supersedes any older unused one.
    session.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None))
        .values(used_at=now)
    )
    raw_token = generate_reset_token()
    session.add(PasswordResetToken(
        user_id=user.id,
        token_hash=hash_reset_token(raw_token),
        created_at=now,
        expires_at=now + timedelta(minutes=PASSWORD_RESET_TOKEN_EXPIRY_MINUTES),
        requested_ip=(requested_ip or None) and requested_ip[:64],
        user_agent=(user_agent or None) and user_agent[:256],
    ))
    session.commit()
    log_security_event("PASSWORD_RESET_REQUESTED", outcome="link_issued", user_id=user.id, ip=requested_ip)
    return user.email, raw_token


def reset_password(
    session: Session, raw_token: str, new_password: str, *, requested_ip: Optional[str] = None
) -> User:
    """Consumes the token and sets the new password. Raises
    throttle.ThrottledError, WeakPasswordError (token NOT consumed - the user
    can retry with a better password) or InvalidResetTokenError."""
    throttle.check_and_record(
        session, _ATTEMPT_BUCKET, requested_ip,
        limit=PASSWORD_RESET_MAX_ATTEMPTS_PER_IP, window_seconds=PASSWORD_RESET_WINDOW_SECONDS,
        message="Too many password reset attempts. Please try again later.",
    )

    # Hash first: a password the policy rejects must not burn the token.
    try:
        new_hash = hash_password(new_password)
    except ValueError as exc:
        raise WeakPasswordError(str(exc)) from exc

    token_hash = hash_reset_token(raw_token)
    now = _now()
    consumed = session.execute(
        update(PasswordResetToken)
        .where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > now,
        )
        .values(used_at=now)
    )
    if consumed.rowcount != 1:
        session.rollback()
        row = session.exec(select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)).first()
        reason = "unknown" if row is None else ("reused" if row.used_at is not None else "expired")
        log_security_event("PASSWORD_RESET_FAILED", reason=reason, ip=requested_ip,
                           user_id=row.user_id if row else None, token_fp=fingerprint(token_hash))
        raise InvalidResetTokenError("This password reset link is invalid or has expired.")

    row = session.exec(select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)).one()
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        session.commit()  # keep the token consumed
        log_security_event("PASSWORD_RESET_FAILED", reason="account_unavailable", user_id=row.user_id)
        raise InvalidResetTokenError("This password reset link is invalid or has expired.")

    user.hashed_password = new_hash
    user.updated_at = now
    user.token_valid_after = now
    session.add(user)
    # Every other outstanding link for this account dies too ...
    session.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None))
        .values(used_at=now)
    )
    # ... and so does every session (refresh token) on every device.
    revoked = session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    ).rowcount
    session.commit()
    session.refresh(user)
    log_security_event("PASSWORD_RESET_COMPLETED", user_id=user.id, sessions_revoked=revoked, ip=requested_ip)
    return user
