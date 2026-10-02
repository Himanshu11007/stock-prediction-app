"""tests/test_device_sessions.py — end-to-end test of /api/v1/auth/sessions,
/sessions/revoke, /sessions/revoke-all, and /auth/devices/pin-enabled
against the real routers, with an isolated in-memory SQLite engine.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from api.routes import auth as auth_routes
from api.routes import auth_devices as auth_devices_routes
from db.session import get_session


@pytest.fixture()
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(auth_devices_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


def _register(client, email, device_id, device_name="Test device"):
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123!", "device_id": device_id, "device_name": device_name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _auth_header(tokens):
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def test_list_sessions_shows_each_device_once(client):
    tokens_a = _register(client, "multi@example.com", "device-A", "Phone A")
    # A second login from the SAME account, different device.
    login_b = client.post(
        "/api/v1/auth/login",
        data={"username": "multi@example.com", "password": "Password123!", "device_id": "device-B", "device_name": "Tablet B"},
    )
    assert login_b.status_code == 200

    sessions = client.get("/api/v1/auth/sessions", headers=_auth_header(tokens_a)).json()

    device_ids = {s["device_id"] for s in sessions}
    assert device_ids == {"device-A", "device-B"}
    names_by_device = {s["device_id"]: s["device_name"] for s in sessions}
    assert names_by_device["device-A"] == "Phone A"
    assert names_by_device["device-B"] == "Tablet B"


def test_revoke_one_session_only_affects_that_device(client):
    tokens_a = _register(client, "revoke1@example.com", "device-A")
    login_b = client.post(
        "/api/v1/auth/login",
        data={"username": "revoke1@example.com", "password": "Password123!", "device_id": "device-B"},
    ).json()

    sessions = client.get("/api/v1/auth/sessions", headers=_auth_header(tokens_a)).json()
    session_a_id = next(s["session_id"] for s in sessions if s["device_id"] == "device-A")

    revoke = client.post(
        "/api/v1/auth/sessions/revoke", json={"session_id": session_a_id}, headers=_auth_header(tokens_a)
    )
    assert revoke.status_code == 204

    # Device A's refresh token must now be rejected...
    refresh_a = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_a["refresh_token"]})
    assert refresh_a.status_code == 401

    # ...but device B's must still work.
    refresh_b = client.post("/api/v1/auth/refresh", json={"refresh_token": login_b["refresh_token"]})
    assert refresh_b.status_code == 200


def test_revoke_all_except_current_keeps_only_the_calling_device(client):
    tokens_a = _register(client, "exceptcurrent@example.com", "device-A")
    tokens_b = client.post(
        "/api/v1/auth/login",
        data={"username": "exceptcurrent@example.com", "password": "Password123!", "device_id": "device-B"},
    ).json()
    tokens_c = client.post(
        "/api/v1/auth/login",
        data={"username": "exceptcurrent@example.com", "password": "Password123!", "device_id": "device-C"},
    ).json()

    resp = client.post(
        "/api/v1/auth/sessions/revoke-all",
        json={"except_current": True, "current_device_id": "device-A"},
        headers=_auth_header(tokens_a),
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["revoked_count"] == 2

    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_a["refresh_token"]}).status_code == 200
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_b["refresh_token"]}).status_code == 401
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_c["refresh_token"]}).status_code == 401


def test_sign_out_all_devices_revokes_every_session_including_current(client):
    tokens_a = _register(client, "signoutall@example.com", "device-A")
    tokens_b = client.post(
        "/api/v1/auth/login",
        data={"username": "signoutall@example.com", "password": "Password123!", "device_id": "device-B"},
    ).json()

    resp = client.post(
        "/api/v1/auth/sessions/revoke-all", json={}, headers=_auth_header(tokens_a)
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["revoked_count"] == 2

    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_a["refresh_token"]}).status_code == 401
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_b["refresh_token"]}).status_code == 401


def test_revoking_one_users_sessions_does_not_touch_another_users(client):
    tokens_a = _register(client, "isolateda@example.com", "device-A")
    tokens_other = _register(client, "isolatedb@example.com", "device-X")

    client.post("/api/v1/auth/sessions/revoke-all", json={}, headers=_auth_header(tokens_a))

    # User A's session is gone, but the completely unrelated user is untouched.
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_a["refresh_token"]}).status_code == 401
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_other["refresh_token"]}).status_code == 200


def test_pin_enabled_flag_requires_a_known_device(client):
    tokens_a = _register(client, "pinflag@example.com", "device-A")

    ok = client.post(
        "/api/v1/auth/devices/pin-enabled",
        json={"device_id": "device-A", "enabled": True},
        headers=_auth_header(tokens_a),
    )
    assert ok.status_code == 204

    unknown = client.post(
        "/api/v1/auth/devices/pin-enabled",
        json={"device_id": "never-registered-device", "enabled": True},
        headers=_auth_header(tokens_a),
    )
    assert unknown.status_code == 404


def test_revoked_refresh_session_cannot_be_restored_by_anything_client_side(client):
    """Simulates the PIN-unlock boundary condition from the security brief:
    once the server has revoked a session, nothing the client does locally
    (including a correct PIN) can bring it back - the client must fall
    through to full re-authentication. This test operates purely at the
    HTTP/refresh-token layer, which is exactly the boundary the mobile
    PIN-unlock flow calls into (see StockAIPro.Mobile.Core's TryRestoreSessionAsync/refresh pipeline)."""
    tokens_a = _register(client, "pinboundary@example.com", "device-A")

    client.post("/api/v1/auth/sessions/revoke-all", json={}, headers=_auth_header(tokens_a))

    refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens_a["refresh_token"]})
    assert refresh.status_code == 401
