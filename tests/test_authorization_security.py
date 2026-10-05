"""tests/test_authorization_security.py — authorization is decided by the
database on every request, never by what an (old or forged) JWT claims.

Covers: admin role revoked/granted while an old JWT is still unexpired,
deactivated and deleted users, forged role claims, wrong signing key,
alg=none, a refresh token used as an access token, expired tokens, and
horizontal privilege escalation (another user's watchlist item / session).
"""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
from api.routes import admin as admin_routes
from api.routes import auth as auth_routes
from api.routes import auth_devices as auth_devices_routes
from api.routes import watchlist as watchlist_routes
from auth.security import JWT_ALGORITHM, JWT_SECRET_KEY, create_access_token
from db.models.stock import Company
from db.models.tracker import WatchlistItem
from db.models.user import RefreshToken, User, UserRoleLink
from db.session import get_session


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=["ADMIN", "USER"])
        auth_service.create_user(s, "alice@example.com", "AlicePass1!")
        auth_service.create_user(s, "bob@example.com", "BobPass12!")
        s.add(Company(symbol="TCS.NS", name="Tata Consultancy Services Ltd.", active=True))
        s.commit()
    return eng


@pytest.fixture()
def client(engine):
    app = FastAPI()
    for r in (auth_routes, admin_routes, auth_devices_routes):
        app.include_router(r.router, prefix="/api/v1")
    app.include_router(watchlist_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


def login(client, email, password):
    resp = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _user(engine, email):
    with Session(engine) as s:
        return s.exec(select(User).where(User.email == email)).one()


def test_admin_role_removed_takes_effect_on_an_old_jwt(client, engine):
    tokens = login(client, "admin@example.com", "AdminPass1!")
    assert client.get("/api/v1/admin/users", headers=bearer(tokens["access_token"])).status_code == 200

    with Session(engine) as s:
        auth_service.remove_role(s, s.exec(select(User).where(User.email == "admin@example.com")).one(), "ADMIN")

    # The JWT still says roles=["ADMIN", ...] and is unexpired - irrelevant.
    payload = jwt.decode(tokens["access_token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert "ADMIN" in payload["roles"]
    assert client.get("/api/v1/admin/users", headers=bearer(tokens["access_token"])).status_code == 403


def test_admin_role_granted_takes_effect_without_a_new_token(client, engine):
    tokens = login(client, "alice@example.com", "AlicePass1!")
    assert client.get("/api/v1/admin/users", headers=bearer(tokens["access_token"])).status_code == 403
    with Session(engine) as s:
        auth_service.assign_role(s, s.exec(select(User).where(User.email == "alice@example.com")).one(), "ADMIN")
    assert client.get("/api/v1/admin/users", headers=bearer(tokens["access_token"])).status_code == 200


def test_forged_admin_role_claim_is_ignored(client, engine):
    alice = _user(engine, "alice@example.com")
    forged = create_access_token(subject=str(alice.id), roles=["ADMIN"])  # validly signed, lying claim
    assert client.get("/api/v1/admin/users", headers=bearer(forged)).status_code == 403


def test_deactivated_user_is_locked_out_with_an_old_jwt_and_refresh_token(client, engine):
    tokens = login(client, "alice@example.com", "AlicePass1!")
    admin = login(client, "admin@example.com", "AdminPass1!")
    alice = _user(engine, "alice@example.com")
    resp = client.post(f"/api/v1/admin/users/{alice.id}/deactivate", headers=bearer(admin["access_token"]))
    assert resp.status_code == 200, resp.text

    assert client.get("/api/v1/auth/me", headers=bearer(tokens["access_token"])).status_code == 401
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401
    assert client.post("/api/v1/auth/login",
                       data={"username": "alice@example.com", "password": "AlicePass1!"}).status_code == 401


def test_deactivated_admin_loses_admin_access_immediately(client, engine):
    tokens = login(client, "admin@example.com", "AdminPass1!")
    with Session(engine) as s:
        u = s.exec(select(User).where(User.email == "admin@example.com")).one()
        u.is_active = False
        s.add(u)
        s.commit()
    assert client.get("/api/v1/admin/users", headers=bearer(tokens["access_token"])).status_code == 401


def test_deleted_user_token_is_rejected(client, engine):
    tokens = login(client, "bob@example.com", "BobPass12!")
    with Session(engine) as s:
        bob = s.exec(select(User).where(User.email == "bob@example.com")).one()
        for link in s.exec(select(UserRoleLink).where(UserRoleLink.user_id == bob.id)).all():
            s.delete(link)
        for rt in s.exec(select(RefreshToken).where(RefreshToken.user_id == bob.id)).all():
            s.delete(rt)
        s.delete(bob)
        s.commit()
    assert client.get("/api/v1/auth/me", headers=bearer(tokens["access_token"])).status_code == 401


@pytest.mark.parametrize("make", [
    lambda uid: jwt.encode({"sub": uid, "roles": ["ADMIN"], "type": "access",
                            "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                           "not-the-server-secret", algorithm="HS256"),
    lambda uid: jwt.encode({"sub": uid, "roles": ["ADMIN"], "type": "access",
                            "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, None, algorithm="none"),
    lambda uid: jwt.encode({"sub": uid, "roles": [], "type": "refresh",
                            "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                           JWT_SECRET_KEY, algorithm=JWT_ALGORITHM),
    lambda uid: jwt.encode({"sub": uid, "roles": [], "type": "access",
                            "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
                           JWT_SECRET_KEY, algorithm=JWT_ALGORITHM),
], ids=["wrong-secret", "alg-none", "wrong-token-type", "expired"])
def test_malicious_or_stale_access_tokens_are_rejected(client, engine, make):
    admin = _user(engine, "admin@example.com")
    assert client.get("/api/v1/admin/users", headers=bearer(make(str(admin.id)))).status_code == 401


def test_opaque_refresh_token_is_not_an_access_token(client):
    tokens = login(client, "alice@example.com", "AlicePass1!")
    assert client.get("/api/v1/auth/me", headers=bearer(tokens["refresh_token"])).status_code == 401


def test_user_cannot_modify_another_users_watchlist_item(client, engine):
    alice = _user(engine, "alice@example.com")
    with Session(engine) as s:
        item = WatchlistItem(user_id=alice.id, symbol="TCS.NS", stock_name="TCS")
        s.add(item)
        s.commit()
        item_id = item.id
    bob = login(client, "bob@example.com", "BobPass12!")
    resp = client.put(f"/api/v1/watchlist/{item_id}/alerts", json={"muted": True}, headers=bearer(bob["access_token"]))
    assert resp.status_code == 404


def test_user_cannot_revoke_another_users_session(client, engine):
    alice = login(client, "alice@example.com", "AlicePass1!")
    bob = login(client, "bob@example.com", "BobPass12!")
    with Session(engine) as s:
        alice_session_id = s.exec(select(RefreshToken).where(
            RefreshToken.user_id == _user(engine, "alice@example.com").id)).one().id
    client.post("/api/v1/auth/sessions/revoke", json={"session_id": alice_session_id},
                headers=bearer(bob["access_token"]))
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": alice["refresh_token"]}).status_code == 200


def test_non_admin_cannot_reach_any_admin_route(client):
    user = login(client, "alice@example.com", "AlicePass1!")
    for path in ("/api/v1/admin/users", "/api/v1/admin/dashboard"):
        assert client.get(path, headers=bearer(user["access_token"])).status_code == 403
        assert client.get(path).status_code == 401
