"""tests/test_auth.py — users, roles, password hashing, authentication,
and authorization (admin vs normal user).

Uses an isolated in-memory SQLite engine per test (never storage/app.db) so
these tests never touch real data and can run in any order/CI environment.
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from auth.dependencies import get_current_user, require_role
from auth.security import hash_password, verify_password
from db.models.user import User  # noqa: F401 (registers table on SQLModel.metadata)
from db.session import get_session


@pytest.fixture()
def engine():
    # StaticPool: a single shared in-memory connection, so every Session()
    # (including ones opened from a different thread by TestClient) sees the
    # same DB rather than each getting its own empty :memory: database.
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        yield s


# ══════════════════════════════════════════════════════════════════════════
# Password hashing/verification
# ══════════════════════════════════════════════════════════════════════════

def test_password_hash_is_not_plaintext():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert hashed.startswith("$2b$")  # bcrypt hash prefix


def test_password_verify_correct_and_incorrect():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed) is True
    assert verify_password("wrong-password", hashed) is False


# ══════════════════════════════════════════════════════════════════════════
# User creation
# ══════════════════════════════════════════════════════════════════════════

def test_create_user_stores_hashed_password_and_default_role(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "New.User@Example.com", "S3curePass!")

    assert user.id is not None
    assert user.email == "new.user@example.com"  # normalized (stripped/lowered)
    assert user.hashed_password != "S3curePass!"
    assert verify_password("S3curePass!", user.hashed_password)
    assert auth_service.get_user_roles(session, user) == [auth_service.USER_ROLE]


def test_duplicate_email_prevention(session):
    auth_service.ensure_roles_exist(session)
    auth_service.create_user(session, "dup@example.com", "Password1!")

    with pytest.raises(auth_service.DuplicateEmailError):
        auth_service.create_user(session, "dup@example.com", "AnotherPass1!")

    # case-insensitive duplicate should also be rejected
    with pytest.raises(auth_service.DuplicateEmailError):
        auth_service.create_user(session, "DUP@EXAMPLE.COM", "AnotherPass1!")


# ══════════════════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════════════════

def test_authenticate_user_success(session):
    auth_service.ensure_roles_exist(session)
    auth_service.create_user(session, "auth@example.com", "CorrectPass1!")

    user = auth_service.authenticate_user(session, "auth@example.com", "CorrectPass1!")
    assert user.email == "auth@example.com"


def test_authenticate_user_wrong_password(session):
    auth_service.ensure_roles_exist(session)
    auth_service.create_user(session, "auth2@example.com", "CorrectPass1!")

    with pytest.raises(auth_service.InvalidCredentialsError):
        auth_service.authenticate_user(session, "auth2@example.com", "WrongPass1!")


def test_authenticate_user_unknown_email(session):
    auth_service.ensure_roles_exist(session)
    with pytest.raises(auth_service.InvalidCredentialsError):
        auth_service.authenticate_user(session, "nobody@example.com", "whatever")


def test_authenticate_inactive_user_rejected(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "inactive@example.com", "CorrectPass1!")
    user.is_active = False
    session.add(user)
    session.commit()

    with pytest.raises(auth_service.InactiveUserError):
        auth_service.authenticate_user(session, "inactive@example.com", "CorrectPass1!")


# ══════════════════════════════════════════════════════════════════════════
# Role assignment
# ══════════════════════════════════════════════════════════════════════════

def test_role_assignment(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "roles@example.com", "Password1!", roles=[])
    assert auth_service.get_user_roles(session, user) == []

    auth_service.assign_role(session, user, auth_service.ADMIN_ROLE)
    assert auth_service.get_user_roles(session, user) == [auth_service.ADMIN_ROLE]

    # assigning the same role twice must not duplicate it
    auth_service.assign_role(session, user, auth_service.ADMIN_ROLE)
    assert auth_service.get_user_roles(session, user) == [auth_service.ADMIN_ROLE]

    auth_service.assign_role(session, user, auth_service.USER_ROLE)
    assert set(auth_service.get_user_roles(session, user)) == {
        auth_service.ADMIN_ROLE,
        auth_service.USER_ROLE,
    }


def test_assign_unknown_role_rejected(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "badrole@example.com", "Password1!")
    with pytest.raises(ValueError):
        auth_service.assign_role(session, user, "SUPERUSER")


# ══════════════════════════════════════════════════════════════════════════
# Authorization (admin vs normal user), exercised through the real FastAPI
# dependencies (get_current_user / require_role) against a minimal test app.
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def client(engine):
    app = FastAPI()

    @app.get("/whoami")
    def whoami(current_user: User = Depends(get_current_user)):
        return {"email": current_user.email}

    @app.get("/admin-only")
    def admin_only(current_user: User = Depends(require_role(auth_service.ADMIN_ROLE))):
        return {"email": current_user.email}

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session

    with Session(engine) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "normal@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])

    return TestClient(app)


def test_normal_user_can_access_authenticated_route(client):
    from auth.security import create_access_token

    token = create_access_token(subject="normal@example.com", roles=[auth_service.USER_ROLE])
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "normal@example.com"


def test_unauthenticated_request_rejected(client):
    resp = client.get("/whoami")
    assert resp.status_code == 401


def test_normal_user_denied_admin_route(client):
    from auth.security import create_access_token

    token = create_access_token(subject="normal@example.com", roles=[auth_service.USER_ROLE])
    resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_admin_user_allowed_admin_route(client):
    from auth.security import create_access_token

    token = create_access_token(subject="admin@example.com", roles=[auth_service.ADMIN_ROLE])
    resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "admin@example.com"
