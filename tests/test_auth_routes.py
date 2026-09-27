"""tests/test_auth_routes.py — end-to-end test of /api/v1/auth/* against the
real router (api/routes/auth.py), with only the DB session swapped for an
isolated in-memory SQLite engine.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from api.routes import auth as auth_routes
from db.session import get_session


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


def test_register_then_me(client):
    resp = client.post(
        "/api/v1/auth/register", json={"email": "routes@example.com", "password": "RoutesPass1!"}
    )
    assert resp.status_code == 200, resp.text
    tokens = resp.json()
    assert "access_token" in tokens and "refresh_token" in tokens

    me = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "routes@example.com"
    assert body["roles"] == ["USER"]


def test_register_duplicate_email_returns_409(client):
    client.post("/api/v1/auth/register", json={"email": "dup2@example.com", "password": "Pass1234!"})
    resp = client.post("/api/v1/auth/register", json={"email": "dup2@example.com", "password": "Other123!"})
    assert resp.status_code == 409


def test_login_with_form_data(client):
    client.post("/api/v1/auth/register", json={"email": "login@example.com", "password": "LoginPass1!"})
    resp = client.post(
        "/api/v1/auth/login",
        data={"username": "login@example.com", "password": "LoginPass1!"},
    )
    assert resp.status_code == 200, resp.text
    assert "access_token" in resp.json()


def test_login_wrong_password_returns_401(client):
    client.post("/api/v1/auth/register", json={"email": "wrongpw@example.com", "password": "RightPass1!"})
    resp = client.post(
        "/api/v1/auth/login",
        data={"username": "wrongpw@example.com", "password": "WrongPass1!"},
    )
    assert resp.status_code == 401


def test_refresh_rotates_token(client):
    reg = client.post(
        "/api/v1/auth/register", json={"email": "refresh@example.com", "password": "RefreshPass1!"}
    )
    old_refresh = reg.json()["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200
    new_tokens = resp.json()
    assert new_tokens["refresh_token"] != old_refresh

    # old refresh token must now be rejected (rotation)
    reuse = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert reuse.status_code == 401


def test_logout_revokes_refresh_token(client):
    reg = client.post(
        "/api/v1/auth/register", json={"email": "logout@example.com", "password": "LogoutPass1!"}
    )
    refresh_token = reg.json()["refresh_token"]

    logout_resp = client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout_resp.status_code == 204

    reuse = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert reuse.status_code == 401


def test_change_password_then_login_with_new_password(client):
    reg = client.post(
        "/api/v1/auth/register", json={"email": "changepw@example.com", "password": "OldPass123!"}
    )
    access_token = reg.json()["access_token"]

    change = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": "OldPass123!", "new_password": "NewPass456!"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert change.status_code == 204

    login = client.post(
        "/api/v1/auth/login",
        data={"username": "changepw@example.com", "password": "NewPass456!"},
    )
    assert login.status_code == 200

    old_login = client.post(
        "/api/v1/auth/login",
        data={"username": "changepw@example.com", "password": "OldPass123!"},
    )
    assert old_login.status_code == 401
