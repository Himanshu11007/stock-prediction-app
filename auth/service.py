"""User/role/authentication business logic - the only module that should
touch the users/roles/user_roles/refresh_tokens tables directly."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlmodel import Session, select

from auth.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
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


def assign_role(session: Session, user: User, role_name: str) -> None:
    role = session.exec(select(Role).where(Role.name == role_name)).first()
    if role is None:
        raise ValueError(f"Unknown role: {role_name!r}. Call ensure_roles_exist() first.")
    existing = session.exec(
        select(UserRoleLink).where(
            UserRoleLink.user_id == user.id, UserRoleLink.role_id == role.id
        )
    ).first()
    if existing is None:
        session.add(UserRoleLink(user_id=user.id, role_id=role.id))
        session.commit()


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
    if user is None or not verify_password(password, user.hashed_password):
        raise InvalidCredentialsError("Invalid email or password")
    if not user.is_active:
        raise InactiveUserError("This account has been deactivated")
    return user


def change_password(session: Session, user: User, current_password: str, new_password: str) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise InvalidCredentialsError("Current password is incorrect")
    user.hashed_password = hash_password(new_password)
    user.updated_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()


def issue_refresh_token(session: Session, user: User) -> str:
    raw_token = generate_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    session.add(
        RefreshToken(user_id=user.id, token_hash=hash_refresh_token(raw_token), expires_at=expires_at)
    )
    session.commit()
    return raw_token


def rotate_refresh_token(session: Session, raw_token: str) -> tuple[User, str]:
    """Validates + revokes the presented refresh token and issues a new one
    (rotation). Raises InvalidCredentialsError if the token is unknown,
    expired, or already used/revoked."""
    token_hash = hash_refresh_token(raw_token)
    row = session.exec(select(RefreshToken).where(RefreshToken.token_hash == token_hash)).first()
    now = datetime.now(timezone.utc)
    if row is None or row.revoked_at is not None or row.expires_at < now:
        raise InvalidCredentialsError("Invalid or expired refresh token")

    row.revoked_at = now
    session.add(row)
    session.commit()

    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        raise InvalidCredentialsError("Invalid or expired refresh token")

    new_raw_token = issue_refresh_token(session, user)
    return user, new_raw_token


def revoke_refresh_token(session: Session, raw_token: str) -> None:
    """Used for logout. Silently no-ops if the token is already unknown/revoked."""
    token_hash = hash_refresh_token(raw_token)
    row = session.exec(select(RefreshToken).where(RefreshToken.token_hash == token_hash)).first()
    if row is not None and row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)
        session.add(row)
        session.commit()
