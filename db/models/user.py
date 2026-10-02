"""Users, roles, and refresh tokens.

Roles are modeled as a many-to-many (users <-> roles via user_roles) rather
than a single enum column on User, so additional roles can be added later
without a schema change (per the architecture spec).
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    """email/hashed_password are nullable because Phase 8 introduces accounts
    that never set a password (Google/Apple SSO, phone-only OTP) - identity
    for those accounts lives in ExternalIdentity/phone, not a password hash.
    A user with neither hashed_password nor any ExternalIdentity row would be
    unable to authenticate at all; application code is responsible for never
    producing that state (every creation path sets up at least one)."""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: Optional[str] = Field(default=None, unique=True, index=True)
    phone: Optional[str] = Field(default=None, unique=True, index=True)
    hashed_password: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Role(SQLModel, table=True):
    __tablename__ = "roles"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)  # "ADMIN" | "USER" | future roles


class UserRoleLink(SQLModel, table=True):
    """Many-to-many association between users and roles."""

    __tablename__ = "user_roles"

    user_id: int = Field(foreign_key="users.id", primary_key=True)
    role_id: int = Field(foreign_key="roles.id", primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)


class RefreshToken(SQLModel, table=True):
    """Refresh tokens are never stored in plaintext - only a SHA-256 hash of
    the token the client holds. Rotated (old row revoked, new row issued) on
    every use so a stolen-and-replayed old token is detectable."""

    __tablename__ = "refresh_tokens"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime
    revoked_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    # Opaque client-generated installation id (see TrustedDevice) - lets the
    # backend tell sessions on different devices apart for session-management
    # ("list my devices" / "sign out this device" / "sign out all devices").
    # Nullable: tokens issued before this feature, or by any non-mobile
    # client that never sends a device id, simply aren't attributable to a
    # specific device and only show up in "revoke all".
    device_id: Optional[str] = Field(default=None, index=True)


class ExternalIdentity(SQLModel, table=True):
    """Links a StockAI user to an external identity provider (Google, Apple).

    provider_subject is the provider's own stable, durable user identifier
    (Google's `sub` claim / Apple's `sub` claim) - never the email, which can
    change, be absent (Apple private relay / subsequent logins), or be
    reused across different provider accounts. The (provider, provider_subject)
    pair is the actual identity key; email here is only a point-in-time
    snapshot for display/support purposes and must never be used to look up
    or merge accounts.
    """

    __tablename__ = "external_identities"
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject", name="uq_external_identity_subject"),
        UniqueConstraint("user_id", "provider", name="uq_external_identity_user_provider"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    provider: str = Field(index=True)  # "google" | "apple"
    provider_subject: str = Field(index=True)
    email: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class OtpChallenge(SQLModel, table=True):
    """A single OTP code issued for one login/verification attempt.

    The code itself is never stored - only a salted hash of it (salt is
    per-row, so the 10^6-entry keyspace of a 6-digit code can't be
    pre-computed once and reused against every row). expires_at/attempt_count
    enforce one-time-use, expiry, and brute-force limits; consumed_at marks a
    challenge that has already been successfully used (or is being verified
    right now - see auth/otp_service.py for the atomic consume step).
    """

    __tablename__ = "otp_challenges"

    id: Optional[int] = Field(default=None, primary_key=True)
    destination: str = Field(index=True)  # normalized email or E.164 phone
    purpose: str = Field(default="login", index=True)
    code_salt: str
    code_hash: str
    attempt_count: int = Field(default=0)
    expires_at: datetime
    consumed_at: Optional[datetime] = Field(default=None)
    requested_ip: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)


class TrustedDevice(SQLModel, table=True):
    """A device the user has completed full (non-PIN) authentication on at
    least once. pin_enabled is purely informational - the backend never
    stores or verifies the PIN itself (see auth/PIN security model in
    docs/AUTHENTICATION.md); PIN verification happens entirely on-device,
    and only ever unlocks whatever refresh session is already stored
    locally. Revoking a device (revoked_at) is independent of - but should
    normally be paired with - revoking that device's refresh_tokens rows.
    """

    __tablename__ = "trusted_devices"
    __table_args__ = (
        UniqueConstraint("user_id", "device_id", name="uq_trusted_device_user_device"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    device_id: str = Field(index=True)
    device_name: Optional[str] = Field(default=None)
    pin_enabled: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)
    revoked_at: Optional[datetime] = Field(default=None)
