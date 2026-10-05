"""User/role/password/refresh-token business logic - the core module for the
users/roles/user_roles/refresh_tokens tables. Phase 8 added three more
auth-related tables, each owned by its own sibling module instead of being
piled into this file: external_identities -> auth/external_identity.py,
otp_challenges -> auth/otp_service.py, trusted_devices -> auth/device_service.py.
All four modules may still read/write `users` directly (e.g. to create a new
user), since user creation is fundamentally shared."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import update
from sqlmodel import Session, select

from auth.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from auth.security_events import fingerprint, log_security_event
from db.models.user import RefreshToken, Role, User, UserRoleLink

ADMIN_ROLE = "ADMIN"
USER_ROLE = "USER"
ALL_ROLES = (ADMIN_ROLE, USER_ROLE)


class DuplicateEmailError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InactiveUserError(Exception):
    pass


def ensure_roles_exist(session: Session) -> dict[str, Role]:
    """Idempotent: creates ADMIN/USER roles if missing, returns both either way."""
    roles: dict[str, Role] = {}
    for name in ALL_ROLES:
        role = session.exec(select(Role).where(Role.name == name)).first()
        if role is None:
            role = Role(name=name)
            session.add(role)
            session.commit()
            session.refresh(role)
        roles[name] = role
    return roles


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(User.email == email.strip().lower())).first()


def get_user_roles(session: Session, user: User) -> list[str]:
    stmt = (
        select(Role.name)
        .join(UserRoleLink, UserRoleLink.role_id == Role.id)
        .where(UserRoleLink.user_id == user.id)
    )
    return list(session.exec(stmt).all())


def assign_role(session: Session, user: User, role_name: str, *, commit: bool = True) -> bool:
    """Returns True if a role assignment was newly added, False if the user
    already had that role (no-op, not an error). commit=False lets a caller
    (e.g. admin/service.py) stage this as part of a larger single transaction
    instead of committing it in isolation."""
    role = session.exec(select(Role).where(Role.name == role_name)).first()
    if role is None:
        raise ValueError(f"Unknown role: {role_name!r}. Call ensure_roles_exist() first.")
    existing = session.exec(
        select(UserRoleLink).where(
            UserRoleLink.user_id == user.id, UserRoleLink.role_id == role.id
        )
    ).first()
    if existing is not None:
        return False
    session.add(UserRoleLink(user_id=user.id, role_id=role.id))
    if commit:
        session.commit()
    return True


def remove_role(session: Session, user: User, role_name: str, *, commit: bool = True) -> bool:
    """Returns True if a role assignment was removed, False if the user
    didn't have that role (no-op, not an error). commit=False lets a caller
    stage this as part of a larger single transaction."""
    role = session.exec(select(Role).where(Role.name == role_name)).first()
    if role is None:
        raise ValueError(f"Unknown role: {role_name!r}. Call ensure_roles_exist() first.")
    existing = session.exec(
        select(UserRoleLink).where(
            UserRoleLink.user_id == user.id, UserRoleLink.role_id == role.id
        )
    ).first()
    if existing is None:
        return False
    session.delete(existing)
    if commit:
        session.commit()
    return True


def create_user(
    session: Session, email: str, password: str, roles: Optional[list[str]] = None
) -> User:
    """Raises DuplicateEmailError if the email is already registered."""
    normalized_email = email.strip().lower()
    if get_user_by_email(session, normalized_email) is not None:
        raise DuplicateEmailError(f"A user with email {normalized_email!r} already exists")

    user = User(email=normalized_email, hashed_password=hash_password(password))
    session.add(user)
    session.commit()
    session.refresh(user)

    if roles is None:
        roles = [USER_ROLE]
    for role_name in roles:
        assign_role(session, user, role_name)
    return user


def authenticate_user(session: Session, email: str, password: str) -> User:
    """Raises InvalidCredentialsError on wrong email/password, InactiveUserError
    if the account exists but is deactivated."""
    user = get_user_by_email(session, email)
    # hashed_password is None for accounts that only ever authenticated via
    # Google/Apple/OTP (see db/models/user.py:User) - such an account simply
    # has no password to check, which must behave exactly like a wrong
    # password rather than raising.
    if user is None or user.hashed_password is None or not verify_password(password, user.hashed_password):
        raise InvalidCredentialsError("Invalid email or password")
    if not user.is_active:
        raise InactiveUserError("This account has been deactivated")
    return user


def change_password(session: Session, user: User, current_password: str, new_password: str) -> None:
    """Raises InvalidCredentialsError if current_password is wrong, or if the
    account has no password set yet (sign-in-only via Google/Apple/OTP) -
    "change" implies one already exists; setting an initial password is a
    deliberately separate, not-yet-implemented flow."""
    if user.hashed_password is None or not verify_password(current_password, user.hashed_password):
        raise InvalidCredentialsError("Current password is incorrect")
    user.hashed_password = hash_password(new_password)
    user.updated_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()


def create_external_user(
    session: Session, *, email: Optional[str] = None, phone: Optional[str] = None, commit: bool = True
) -> User:
    """Creates a new StockLens user with no password - used the first time an
    external identity (Google/Apple/OTP) is seen with no existing account to
    attach to. Assigns the default USER role same as create_user(). Does NOT
    check for an existing user with this email/phone first - callers
    (auth/external_identity.py, auth/otp_service.py) are responsible for that
    lookup, since the right thing to do when one already exists is provider-
    specific (link vs. reject), not a blind "reuse it" here.

    commit=False only flushes (the user gets its id) so a caller can create
    the user together with its first sign-in method in ONE transaction -
    see auth/external_identity.py:find_or_create_user_for_identity.
    """
    user = User(
        email=email.strip().lower() if email else None,
        phone=phone,
        hashed_password=None,
    )
    session.add(user)
    if not commit:
        session.flush()
        assign_role(session, user, USER_ROLE, commit=False)
        return user
    session.commit()
    session.refresh(user)

    assign_role(session, user, USER_ROLE)
    return user


def issue_refresh_token(
    session: Session, user: User, *, device_id: Optional[str] = None, commit: bool = True
) -> str:
    raw_token = generate_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(raw_token),
            expires_at=expires_at,
            device_id=device_id,
        )
    )
    if commit:
        session.commit()
    return raw_token


def rotate_refresh_token(session: Session, raw_token: str) -> tuple[User, str]:
    """Validates + revokes the presented refresh token and issues a new one
    (rotation). Raises InvalidCredentialsError if the token is unknown,
    expired, or already used/revoked. The new token keeps the same
    device_id as the one being rotated - rotation continues the same
    device's session, it doesn't start a new one.

    Exactly-once: the old token is consumed by ONE conditional UPDATE
    (... WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > now).
    The database decides which of several simultaneous requests carrying the
    same token wins - on PostgreSQL the losers block on the row lock and then
    match zero rows; on SQLite writes are serialized - so only one request
    can ever rotate a given token. The revocation and the replacement token
    are committed in the same transaction: either both happen or neither.
    (The previous SELECT-then-UPDATE let two concurrent requests both see the
    token as valid and both receive a new token.)"""
    token_hash = hash_refresh_token(raw_token)
    now = datetime.now(timezone.utc)
    consumed = session.execute(
        update(RefreshToken)
        .where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > now,
        )
        .values(revoked_at=now)
    )
    if consumed.rowcount != 1:
        session.rollback()
        row = find_refresh_token(session, raw_token)
        if row is not None and row.revoked_at is not None:
            # A rotated/revoked token was presented again: either a client
            # race or a stolen token being replayed.
            log_security_event("REFRESH_TOKEN_REUSE", level=logging.WARNING, user_id=row.user_id,
                               device_id=row.device_id, token_fp=fingerprint(token_hash))
        raise InvalidCredentialsError("Invalid or expired refresh token")

    row = session.exec(select(RefreshToken).where(RefreshToken.token_hash == token_hash)).one()
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        session.commit()  # the presented token stays revoked
        raise InvalidCredentialsError("Invalid or expired refresh token")

    new_raw_token = issue_refresh_token(session, user, device_id=row.device_id, commit=False)
    session.commit()
    return user, new_raw_token


def find_refresh_token(session: Session, raw_token: str) -> Optional[RefreshToken]:
    return session.exec(select(RefreshToken).where(
        RefreshToken.token_hash == hash_refresh_token(raw_token))).first()


def revoke_refresh_token(session: Session, raw_token: str) -> None:
    """Used for logout. Silently no-ops if the token is already unknown/revoked."""
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(raw_token), RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    session.commit()
