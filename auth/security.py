"""Password hashing and JWT access/refresh-token primitives.

No plaintext password or raw refresh token is ever persisted - passwords are
bcrypt-hashed, refresh tokens are SHA-256 hashed before being stored (see
db/models/user.py:RefreshToken).
"""
import hashlib
import os
import secrets
import warnings
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30

_BCRYPT_MAX_PASSWORD_BYTES = 72

_env_secret = os.environ.get("JWT_SECRET_KEY")
if _env_secret:
    JWT_SECRET_KEY = _env_secret
else:
    # Dev-only fallback so the app can run without extra setup. Ephemeral:
    # regenerated every process start, which invalidates all outstanding
    # tokens on restart. Production MUST set JWT_SECRET_KEY explicitly.
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    warnings.warn(
        "JWT_SECRET_KEY is not set in the environment - using an ephemeral "
        "per-process secret (all issued tokens are invalidated on restart). "
        "Set JWT_SECRET_KEY for any persistent or production deployment.",
        RuntimeWarning,
        stacklevel=2,
    )


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > _BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {_BCRYPT_MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > _BCRYPT_MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(*, subject: str, roles: list[str]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "roles": roles,
        "type": "access",
        # Fractional seconds (RFC 7519 NumericDate allows them): lets
        # get_current_user tell a token issued just before a password reset
        # from one issued just after it, even within the same second.
        "iat": now.timestamp(),
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.InvalidTokenError (or a subclass) on any failure."""
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Not an access token")
    return payload


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
