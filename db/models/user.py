"""Users, roles, and refresh tokens.

Roles are modeled as a many-to-many (users <-> roles via user_roles) rather
than a single enum column on User, so additional roles can be added later
without a schema change (per the architecture spec).
"""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
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
