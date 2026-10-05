"""OTP (one-time passcode) challenge issuance and verification - the only
module that should touch the otp_challenges table directly.

Security properties:
- the code itself is never stored - only a per-row-salted hash of it (see
  db/models/user.py:OtpChallenge's docstring for why the salt matters)
- one-time-use: a successfully verified challenge is marked consumed_at and
  can never be verified again
- bounded attempts: OTP_MAX_ATTEMPTS wrong codes against a single challenge
  permanently fails that challenge (the caller must request a new one)
- bounded requests: a resend cooldown AND a rolling-window cap per
  destination defend against OTP spam; the same rolling window also applies
  per requesting IP so one attacker can't dodge the per-destination limit
  by rotating destinations
- the raw code is never logged, never put in a URL, and never included in
  an exception message - verify_otp()'s errors are generic by design
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import update
from sqlmodel import Session, select

from auth.otp_delivery import IOtpDeliveryService
from auth.security_events import log_security_event
from config import (
    OTP_CODE_LENGTH,
    OTP_EXPIRE_SECONDS,
    OTP_MAX_ATTEMPTS,
    OTP_MAX_REQUESTS_PER_WINDOW,
    OTP_REQUEST_WINDOW_SECONDS,
    OTP_RESEND_COOLDOWN_SECONDS,
)
from db.models.user import OtpChallenge


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OtpError(Exception):
    """Base for all OTP failures - every message here is already safe to
    show directly to the end user (never a stack trace/internal detail)."""


class OtpRateLimitedError(OtpError):
    pass


class OtpResendCooldownError(OtpError):
    pass


class OtpInvalidError(OtpError):
    pass


class OtpExpiredError(OtpError):
    pass


class OtpAlreadyUsedError(OtpError):
    pass


class OtpMaxAttemptsError(OtpError):
    pass


def normalize_destination(destination: str) -> str:
    destination = destination.strip()
    return destination.lower() if "@" in destination else destination


def _generate_code(length: int = OTP_CODE_LENGTH) -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


def _hash_code(code: str, salt: str) -> str:
    return hashlib.sha256((salt + code).encode("utf-8")).hexdigest()


def request_otp(
    session: Session,
    delivery: IOtpDeliveryService,
    destination: str,
    *,
    purpose: str = "login",
    requested_ip: Optional[str] = None,
) -> None:
    """Generates, stores (hashed), and delivers a new OTP.

    Raises OtpResendCooldownError if the last code for this destination was
    issued too recently, or OtpRateLimitedError if too many codes have
    already been issued for this destination OR this IP within the rolling
    window. Always raises or succeeds the same way regardless of whether
    `destination` has an existing StockLens account - account lookup only
    happens at verify time - so this endpoint can't be used to enumerate
    registered users.
    """
    destination = normalize_destination(destination)
    now = utcnow()

    most_recent = session.exec(
        select(OtpChallenge)
        .where(OtpChallenge.destination == destination, OtpChallenge.purpose == purpose)
        .order_by(OtpChallenge.created_at.desc())
    ).first()
    if most_recent is not None and (now - most_recent.created_at) < timedelta(
        seconds=OTP_RESEND_COOLDOWN_SECONDS
    ):
        raise OtpResendCooldownError("Please wait before requesting another code")

    window_start = now - timedelta(seconds=OTP_REQUEST_WINDOW_SECONDS)
    destination_recent = session.exec(
        select(OtpChallenge).where(
            OtpChallenge.destination == destination,
            OtpChallenge.purpose == purpose,
            OtpChallenge.created_at >= window_start,
        )
    ).all()
    if len(destination_recent) >= OTP_MAX_REQUESTS_PER_WINDOW:
        raise OtpRateLimitedError("Too many codes requested for this destination. Try again later.")

    if requested_ip:
        ip_recent = session.exec(
            select(OtpChallenge).where(
                OtpChallenge.requested_ip == requested_ip,
                OtpChallenge.created_at >= window_start,
            )
        ).all()
        if len(ip_recent) >= OTP_MAX_REQUESTS_PER_WINDOW:
            raise OtpRateLimitedError("Too many codes requested from this network. Try again later.")

    code = _generate_code()
    salt = secrets.token_hex(16)
    session.add(
        OtpChallenge(
            destination=destination,
            purpose=purpose,
            code_salt=salt,
            code_hash=_hash_code(code, salt),
            expires_at=now + timedelta(seconds=OTP_EXPIRE_SECONDS),
            requested_ip=requested_ip,
        )
    )
    session.commit()

    delivery.send(destination, code)


def verify_otp(session: Session, destination: str, code: str, *, purpose: str = "login") -> None:
    """Validates `code` against the most recently issued challenge for
    `destination`. Raises OtpInvalidError / OtpExpiredError /
    OtpAlreadyUsedError / OtpMaxAttemptsError on failure - all with generic,
    user-safe messages (never reveals which specific check failed beyond
    what the user needs to decide their next action).

    On success the challenge is consumed so it can never be verified again.
    This function only validates the code itself - the caller is
    responsible for finding/creating the StockLens user for `destination`.

    Exactly-once, enforced by the database rather than by a read-then-write:
    1. the attempt is counted with one conditional UPDATE
       (attempt_count = attempt_count + 1 WHERE not consumed AND not expired
       AND attempt_count < OTP_MAX_ATTEMPTS) - concurrent guesses can no
       longer overwrite each other's increments and exceed the limit;
    2. a matching code consumes the challenge with one conditional UPDATE
       (consumed_at = now WHERE consumed_at IS NULL) - of any number of
       simultaneous requests with the same correct code, exactly one
       matches a row; the rest get OtpAlreadyUsedError.
    """
    destination = normalize_destination(destination)
    challenge = session.exec(
        select(OtpChallenge)
        .where(OtpChallenge.destination == destination, OtpChallenge.purpose == purpose)
        .order_by(OtpChallenge.created_at.desc(), OtpChallenge.id.desc())
    ).first()
    if challenge is None:
        raise OtpInvalidError("Invalid or expired code")
    challenge_id = challenge.id

    now = utcnow()
    counted = session.execute(
        update(OtpChallenge)
        .where(
            OtpChallenge.id == challenge_id,
            OtpChallenge.consumed_at.is_(None),
            OtpChallenge.expires_at > now,
            OtpChallenge.attempt_count < OTP_MAX_ATTEMPTS,
        )
        .values(attempt_count=OtpChallenge.attempt_count + 1)
    )
    if counted.rowcount != 1:
        session.rollback()
        _raise_for_unusable_challenge(session, challenge_id, now)
    session.commit()

    if not hmac.compare_digest(_hash_code(code, challenge.code_salt), challenge.code_hash):
        raise OtpInvalidError("Invalid or expired code")

    consumed = session.execute(
        update(OtpChallenge)
        .where(OtpChallenge.id == challenge_id, OtpChallenge.consumed_at.is_(None))
        .values(consumed_at=now)
    )
    if consumed.rowcount != 1:
        session.rollback()
        log_security_event("OTP_REUSE", level=logging.WARNING, challenge_id=challenge_id)
        raise OtpAlreadyUsedError("This code has already been used")
    session.commit()


def _raise_for_unusable_challenge(session: Session, challenge_id: int, now: datetime) -> None:
    challenge = session.get(OtpChallenge, challenge_id)
    session.refresh(challenge)
    if challenge.consumed_at is not None:
        log_security_event("OTP_REUSE", level=logging.WARNING, challenge_id=challenge_id)
        raise OtpAlreadyUsedError("This code has already been used")
    if challenge.attempt_count >= OTP_MAX_ATTEMPTS:
        log_security_event("OTP_MAX_ATTEMPTS", level=logging.WARNING, challenge_id=challenge_id)
        raise OtpMaxAttemptsError("Too many incorrect attempts - request a new code")
    raise OtpExpiredError("This code has expired")
