"""tests/test_auth_sso_routes.py — end-to-end test of /api/v1/auth/google,
/apple, /link/*, /identities against the real routers, with the identity
verifiers swapped for deterministic fakes (no network calls) and the DB
session swapped for an isolated in-memory SQLite engine - exactly the
pattern tests/test_auth_routes.py already uses for get_session.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from api.routes import auth as auth_routes
from api.routes import auth_sso as auth_sso_routes
from auth.external_identity import FakeAppleIdentityVerifier, FakeGoogleIdentityVerifier
from db.session import get_session


@pytest.fixture()
def google_fake():
    return FakeGoogleIdentityVerifier()


@pytest.fixture()
def apple_fake():
    return FakeAppleIdentityVerifier()


@pytest.fixture()
def client(google_fake, apple_fake):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(auth_sso_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    app.dependency_overrides[auth_sso_routes.get_google_identity_verifier] = lambda: google_fake
    app.dependency_overrides[auth_sso_routes.get_apple_identity_verifier] = lambda: apple_fake
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════
# Google
# ══════════════════════════════════════════════════════════════════════════

def test_google_new_user_creates_account_and_issues_tokens(client, google_fake):
    google_fake.register("good-token", subject="g-sub-1", email="newgoogle@example.com")

    resp = client.post("/api/v1/auth/google", json={"id_token": "good-token"})

    assert resp.status_code == 200, resp.text
    tokens = resp.json()
    assert "access_token" in tokens and "refresh_token" in tokens

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == "newgoogle@example.com"


def test_google_existing_identity_logs_in_the_same_user(client, google_fake):
    google_fake.register("tok-1", subject="g-sub-2", email="returning@example.com")
    first = client.post("/api/v1/auth/google", json={"id_token": "tok-1"})
    first_user_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"}
    ).json()["id"]

    second = client.post("/api/v1/auth/google", json={"id_token": "tok-1"})
    second_user_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {second.json()['access_token']}"}
    ).json()["id"]

    assert first_user_id == second_user_id


def test_google_invalid_token_is_rejected(client):
    resp = client.post("/api/v1/auth/google", json={"id_token": "not-a-real-token"})
    assert resp.status_code == 401


def test_google_email_collision_with_password_account_requires_linking(client, google_fake):
    client.post("/api/v1/auth/register", json={"email": "collide@example.com", "password": "Password123!"})
    google_fake.register("collide-token", subject="g-collide", email="collide@example.com")

    resp = client.post("/api/v1/auth/google", json={"id_token": "collide-token"})

    assert resp.status_code == 409


def test_google_login_accepts_device_id_and_name(client, google_fake):
    # Device/session listing itself is covered end-to-end in
    # tests/test_device_sessions.py against the full app (auth_devices
    # router) - this just confirms the /auth/google payload shape works.
    google_fake.register("dev-token", subject="g-dev", email="dev@example.com")

    resp = client.post(
        "/api/v1/auth/google", json={"id_token": "dev-token", "device_id": "device-123", "device_name": "Pixel 7"}
    )

    assert resp.status_code == 200, resp.text


# ══════════════════════════════════════════════════════════════════════════
# Apple
# ══════════════════════════════════════════════════════════════════════════

def test_apple_new_user_with_email_creates_account(client, apple_fake):
    apple_fake.register("apple-tok-1", subject="a-sub-1", email="applefirst@example.com")

    resp = client.post("/api/v1/auth/apple", json={"identity_token": "apple-tok-1"})

    assert resp.status_code == 200, resp.text
    me = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {resp.json()['access_token']}"}
    )
    assert me.json()["email"] == "applefirst@example.com"


def test_apple_subsequent_login_without_email_still_resolves_same_account(client, apple_fake):
    apple_fake.register("apple-tok-2", subject="a-sub-2", email="applesecond@example.com")
    first = client.post("/api/v1/auth/apple", json={"identity_token": "apple-tok-2"})
    first_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"}
    ).json()["id"]

    # Private relay / later logins: Apple sends no email this time.
    apple_fake.register("apple-tok-2b", subject="a-sub-2", email=None)
    second = client.post("/api/v1/auth/apple", json={"identity_token": "apple-tok-2b"})
    second_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {second.json()['access_token']}"}
    ).json()["id"]

    assert first_id == second_id


def test_apple_invalid_token_is_rejected(client):
    resp = client.post("/api/v1/auth/apple", json={"identity_token": "garbage"})
    assert resp.status_code == 401


def test_apple_duplicate_external_identity_across_accounts_is_not_possible(client, apple_fake):
    # Same provider_subject can never belong to two different users - the
    # DB unique constraint plus find_or_create's "match -> same user" path
    # guarantees the second login resolves back to the FIRST account, it
    # never creates a second one.
    apple_fake.register("dup-tok-a", subject="a-dup", email="dupfirst@example.com")
    first = client.post("/api/v1/auth/apple", json={"identity_token": "dup-tok-a"})
    first_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"}
    ).json()["id"]

    apple_fake.register("dup-tok-b", subject="a-dup", email="dupfirst@example.com")
    second = client.post("/api/v1/auth/apple", json={"identity_token": "dup-tok-b"})
    second_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {second.json()['access_token']}"}
    ).json()["id"]

    assert first_id == second_id


# ══════════════════════════════════════════════════════════════════════════
# Account linking
# ══════════════════════════════════════════════════════════════════════════

def test_link_google_to_authenticated_password_account(client, google_fake):
    reg = client.post("/api/v1/auth/register", json={"email": "linkme@example.com", "password": "Password123!"})
    access_token = reg.json()["access_token"]
    google_fake.register("link-tok", subject="g-link", email="linkme-google@example.com")

    resp = client.post(
        "/api/v1/auth/link/google",
        json={"id_token": "link-tok"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 200, resp.text

    identities = client.get(
        "/api/v1/auth/identities", headers={"Authorization": f"Bearer {access_token}"}
    ).json()
    assert len(identities) == 1
    assert identities[0]["provider"] == "google"


def test_link_identity_already_used_by_another_account_is_rejected(client, google_fake):
    user_a = client.post("/api/v1/auth/register", json={"email": "a2@example.com", "password": "Password123!"})
    user_b = client.post("/api/v1/auth/register", json={"email": "b2@example.com", "password": "Password123!"})
    google_fake.register("shared-tok", subject="g-shared", email="shared-google@example.com")

    link_a = client.post(
        "/api/v1/auth/link/google",
        json={"id_token": "shared-tok"},
        headers={"Authorization": f"Bearer {user_a.json()['access_token']}"},
    )
    assert link_a.status_code == 200

    link_b = client.post(
        "/api/v1/auth/link/google",
        json={"id_token": "shared-tok"},
        headers={"Authorization": f"Bearer {user_b.json()['access_token']}"},
    )
    assert link_b.status_code == 409


def test_unauthorized_linking_is_rejected(client):
    resp = client.post("/api/v1/auth/link/google", json={"id_token": "whatever"})
    assert resp.status_code == 401


def test_unlink_last_auth_method_is_rejected(client, google_fake):
    # Sign up via Google only - no password.
    google_fake.register("onlymethod-tok", subject="g-onlymethod", email="onlymethod@example.com")
    resp = client.post("/api/v1/auth/google", json={"id_token": "onlymethod-tok"})
    access_token = resp.json()["access_token"]

    delete = client.delete(
        "/api/v1/auth/identities/google", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert delete.status_code == 400
