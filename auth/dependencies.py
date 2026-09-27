"""FastAPI dependencies enforcing authentication/authorization.

Authorization is entirely backend-controlled: every protected route depends
on get_current_user (validates the JWT against the DB-backed user) and,
where relevant, require_role(...) (checks DB-backed role assignment). No
client-supplied claim about identity or entitlement is trusted on its own -
the JWT only carries a user id; roles are the token's cached copy of a
DB fact, re-checked here against nothing client-controlled.
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from auth.security import decode_access_token
from auth.service import get_user_by_email
from db.models.user import User
from db.session import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise credentials_error

    email = payload.get("sub")
    if not email:
        raise credentials_error

    user = get_user_by_email(session, email)
    if user is None or not user.is_active:
        raise credentials_error
    return user


def require_role(role_name: str):
    """Dependency factory: require_role("ADMIN") -> a dependency that 403s
    unless the authenticated user has that role in the DB right now."""

    def _dependency(
        current_user: User = Depends(get_current_user),
        session: Session = Depends(get_session),
    ) -> User:
        from auth.service import get_user_roles  # local import avoids a cycle

        if role_name not in get_user_roles(session, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {role_name} role",
            )
        return current_user

    return _dependency


require_admin = require_role("ADMIN")
