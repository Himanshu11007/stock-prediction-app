"""tests/test_password_reset.py — POST /api/v1/auth/forgot-password and
/reset-password end-to-end against the real routers (in-memory SQLite, fake
email delivery), plus the email content and the never-log guarantees.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.password_reset as password_reset
import auth.service as auth_service
from api.routes import auth as auth_routes
from api.routes import auth_otp as auth_otp_routes
from api.routes import auth_password as auth_password_routes
from api.routes import auth_sso as auth_sso_routes
from auth.external_identity import FakeGoogleIdentityVerifier
from auth.otp_delivery import FakeOtpDeliveryService, get_otp_delivery_service
from auth.password_reset_delivery import (
    FakePasswordResetDeliveryService,
    SmtpPasswordResetDeliveryService,
    build_reset_email,
    build_reset_url,
    get_password_reset_delivery_service,
)
from db.models.user import PasswordResetToken, RefreshToken, User
from db.session import get_session

RESET_URL = "https://web.example.test/reset-password"
GENERIC = "If an account exists for this email address, a password reset link has been sent."


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def mailer():
    return FakePasswordResetDeliveryService()


@pytest.fixture()
def google_fake():
    return FakeGoogleIdentityVerifier()


@pytest.fixture()
def otp_delivery():
    return FakeOtpDeliveryService()


@pytest.fixture()
def client(engine, mailer, google_fake, otp_delivery):
    app = FastAPI()
    for r in (auth_routes, auth_password_routes, auth_sso_routes, auth_otp_routes):
        app.include_router(r.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    app.dependency_overrides[get_password_reset_delivery_service] = lambda: mailer
    app.dependency_overrides[auth_password_routes.get_password_reset_url] = lambda: RESET_URL
    app.dependency_overrides[auth_sso_routes.get_google_identity_verifier] = lambda: google_fake
    app.dependency_overrides[get_otp_delivery_service] = lambda: otp_delivery
    return TestClient(app)


def register(client, email="reset@example.com", password="OldPassword1!", device_id="phone-1"):
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password, "device_id": device_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


def forgot(client, email="reset@example.com"):
    return client.post("/api/v1/auth/forgot-password", json={"email": email})


def reset(client, token, new_password="NewPassword1!"):
    return client.post("/api/v1/auth/reset-password", json={"token": token, "new_password": new_password})


def login(client, email="reset@example.com", password="NewPassword1!"):
    return client.post("/api/v1/auth/login", data={"username": email, "password": password})


# ── forgot-password: no account enumeration ──────────────────────────────────

def test_forgot_password_for_existing_account_sends_one_link(client, mailer):
    register(client)
    resp = forgot(client)
    assert resp.status_code == 202
    assert resp.json() == {"message": GENERIC}
    assert len(mailer.sent) == 1
    email, url = mailer.sent[0]
    assert email == "reset@example.com"
    assert url.startswith(RESET_URL + "?token=")


def test_forgot_password_for_unknown_email_is_indistinguishable(client, mailer):
    register(client)
    existing = forgot(client, "reset@example.com")
    missing = forgot(client, "nobody@example.com")
    assert (missing.status_code, missing.json()) == (existing.status_code, existing.json())
    assert missing.headers.get("content-length") == existing.headers.get("content-length")
    assert len(mailer.sent) == 1  # only the real account got an email


def test_email_lookup_is_case_insensitive(client, mailer):
    register(client)
    forgot(client, "  RESET@Example.COM ")
    assert mailer.sent[0][0] == "reset@example.com"


def test_deactivated_account_gets_the_generic_answer_and_no_email(client, engine, mailer):
    register(client)
    with Session(engine) as s:
        u = s.exec(select(User)).one()
        u.is_active = False
        s.add(u)
        s.commit()
    assert forgot(client).json() == {"message": GENERIC}
    assert mailer.sent == []


def test_invalid_email_is_a_validation_error_for_everyone(client):
    assert forgot(client, "not-an-email").status_code == 422


def test_missing_reset_url_configuration_still_answers_generically(client, mailer, engine):
    register(client)
    client.app.dependency_overrides[auth_password_routes.get_password_reset_url] = lambda: ""
    assert forgot(client).json() == {"message": GENERIC}
    assert mailer.sent == []
    with Session(engine) as s:
        assert s.exec(select(PasswordResetToken)).all() == []


# ── tokens: random, hashed at rest, expiring, single use ────────────────────

def test_token_is_random_and_only_its_hash_is_stored(client, engine, mailer):
    register(client)
    forgot(client)
    raw = mailer.last_token
    assert len(raw) >= 43  # 32 bytes, base64url
    with Session(engine) as s:
        row = s.exec(select(PasswordResetToken)).one()
        assert row.token_hash == hashlib.sha256(raw.encode()).hexdigest()
        assert raw not in (row.token_hash, row.requested_ip, row.user_agent)
        expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=timezone.utc)
        created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
        assert timedelta(minutes=29) < expires - created <= timedelta(minutes=30)
        assert row.used_at is None


def test_tokens_are_unique_per_request():
    assert len({password_reset.generate_reset_token() for _ in range(200)}) == 200


def test_successful_reset_changes_the_password(client, mailer):
    register(client)
    forgot(client)
    resp = reset(client, mailer.last_token)
    assert resp.status_code == 200, resp.text
    assert login(client).status_code == 200
    assert login(client, password="OldPassword1!").status_code == 401


def test_token_cannot_be_reused(client, mailer):
    register(client)
    forgot(client)
    token = mailer.last_token
    assert reset(client, token).status_code == 200
    second = reset(client, token, "AnotherPassword1!")
    assert second.status_code == 400
    assert login(client).status_code == 200  # the first reset stands


def test_expired_token_is_rejected(client, engine, mailer):
    register(client)
    forgot(client)
    with Session(engine) as s:
        row = s.exec(select(PasswordResetToken)).one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        s.add(row)
        s.commit()
    resp = reset(client, mailer.last_token)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "This password reset link is invalid or has expired."


@pytest.mark.parametrize("token", ["x" * 43, "definitely-not-a-real-reset-token"])
def test_unknown_token_is_rejected_with_the_same_message(client, token):
    resp = reset(client, token)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "This password reset link is invalid or has expired."


def test_newer_link_supersedes_older_unused_link(client, mailer):
    register(client)
    forgot(client)
    old = mailer.last_token
    forgot(client)
    new = mailer.last_token
    assert reset(client, old).status_code == 400
    assert reset(client, new).status_code == 200


# ── password policy (same as registration) ───────────────────────────────────

@pytest.mark.parametrize("bad", ["short1!", "x" * 73, "é" * 40])  # too short / too long / >72 bytes
def test_weak_password_is_rejected_and_does_not_burn_the_token(client, mailer, bad):
    register(client)
    forgot(client)
    token = mailer.last_token
    assert reset(client, token, bad).status_code == 422
    assert reset(client, token).status_code == 200


# ── sessions end on reset ────────────────────────────────────────────────────

def test_reset_revokes_every_session_and_existing_access_tokens(client, engine, mailer):
    phone = register(client, device_id="phone-1")
    tablet = login(client, password="OldPassword1!").json()
    headers = {"Authorization": f"Bearer {phone['access_token']}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200

    forgot(client)
    assert reset(client, mailer.last_token).status_code == 200

    # Refresh tokens are dead ...
    for tokens in (phone, tablet):
        assert client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401
    # ... and so are the still-unexpired access tokens issued before the reset.
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
    with Session(engine) as s:
        assert all(r.revoked_at is not None for r in s.exec(select(RefreshToken)).all())

    fresh = login(client).json()
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {fresh['access_token']}"}).status_code == 200


# ── rate limiting ────────────────────────────────────────────────────────────

def test_per_account_limit_is_silent(client, mailer, monkeypatch):
    monkeypatch.setattr(password_reset, "PASSWORD_RESET_MAX_EMAILS_PER_ACCOUNT", 2)
    register(client)
    responses = [forgot(client) for _ in range(4)]
    assert {r.status_code for r in responses} == {202}  # no 429 that would reveal the account
    assert len(mailer.sent) == 2


def test_per_ip_limit_applies_to_existing_and_unknown_emails_alike(client, monkeypatch):
    monkeypatch.setattr(password_reset, "PASSWORD_RESET_MAX_REQUESTS_PER_IP", 3)
    register(client)
    codes = [forgot(client, e).status_code for e in
             ("reset@example.com", "nobody1@example.com", "nobody2@example.com", "nobody3@example.com")]
    assert codes == [202, 202, 202, 429]
    blocked = forgot(client, "reset@example.com")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_reset_attempts_are_limited_per_ip(client, mailer, monkeypatch):
    monkeypatch.setattr(password_reset, "PASSWORD_RESET_MAX_ATTEMPTS_PER_IP", 3)
    register(client)
    forgot(client)
    codes = [reset(client, f"guess-{i}-aaaaaaaaaaaaaaaa").status_code for i in range(3)]
    assert codes == [400, 400, 400]
    assert reset(client, mailer.last_token).status_code == 429  # brute force stopped, even for a valid token


# ── accounts without a password (Google / OTP) ───────────────────────────────

def test_google_created_account_can_establish_a_password_via_its_verified_email(client, google_fake, mailer):
    google_fake.register("g-token", subject="g-sub", email="gmailuser@example.com")
    google_tokens = client.post("/api/v1/auth/google", json={"id_token": "g-token"}).json()
    assert login(client, "gmailuser@example.com", "NewPassword1!").status_code == 401  # no password yet

    forgot(client, "gmailuser@example.com")
    assert reset(client, mailer.last_token).status_code == 200
    assert login(client, "gmailuser@example.com", "NewPassword1!").status_code == 200
    # Google sign-in keeps working for the same account; old sessions ended.
    assert client.post("/api/v1/auth/google", json={"id_token": "g-token"}).status_code == 200
    assert client.post("/api/v1/auth/refresh",
                       json={"refresh_token": google_tokens["refresh_token"]}).status_code == 401


def test_otp_created_account_can_establish_a_password(client, otp_delivery, mailer):
    client.post("/api/v1/auth/otp/request", json={"destination": "otpuser@example.com"})
    assert client.post("/api/v1/auth/otp/verify",
                       json={"destination": "otpuser@example.com", "code": otp_delivery.last_code}).status_code == 200
    forgot(client, "otpuser@example.com")
    assert reset(client, mailer.last_token).status_code == 200
    assert login(client, "otpuser@example.com", "NewPassword1!").status_code == 200


# ── email + logging ──────────────────────────────────────────────────────────

def test_reset_email_contents():
    msg = build_reset_email("a@example.com", "no-reply@example.com",
                            build_reset_url(RESET_URL, "tok/en+="), 30)
    body = msg.get_content()
    assert msg["To"] == "a@example.com"
    assert "reset" in msg["Subject"].lower()
    assert "https://web.example.test/reset-password?token=tok%2Fen%2B%3D" in body
    assert "30 minutes" in body
    assert "If you did not request this" in body


def test_reset_url_respects_an_existing_query_string():
    assert build_reset_url("https://x.test/r?lang=en", "abc") == "https://x.test/r?lang=en&token=abc"


@patch("smtplib.SMTP")
def test_smtp_reset_delivery_uses_the_shared_smtp_settings(mock_smtp_class):
    smtp = mock_smtp_class.return_value.__enter__.return_value
    SmtpPasswordResetDeliveryService(host="smtp.test", port=587, username="u", password="p",
                                     from_address="no-reply@x.test").send("a@example.com", RESET_URL + "?token=t", 30)
    mock_smtp_class.assert_called_once_with("smtp.test", 587, timeout=10)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("u", "p")
    assert smtp.send_message.call_args[0][0]["To"] == "a@example.com"


def test_delivery_failure_does_not_change_the_response(client, mailer, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("smtp down " + mailer_url_holder[0])
    mailer_url_holder = ["https://leak.test/?token=secret"]
    register(client)
    mailer.send = boom
    with caplog.at_level("DEBUG"):
        resp = forgot(client)
    assert resp.status_code == 202 and resp.json() == {"message": GENERIC}
    assert "token=" not in caplog.text


def test_nothing_secret_is_logged(client, mailer, caplog):
    register(client)
    with caplog.at_level("DEBUG"):
        forgot(client)
        token = mailer.last_token
        reset(client, token)
        reset(client, token)  # reuse
    assert token not in caplog.text
    assert "NewPassword1!" not in caplog.text
    assert "reset@example.com" not in caplog.text
    for event in ("PASSWORD_RESET_REQUESTED", "PASSWORD_RESET_COMPLETED", "PASSWORD_RESET_FAILED"):
        assert event in caplog.text
