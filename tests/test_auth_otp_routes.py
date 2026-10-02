"""tests/test_auth_otp_routes.py — end-to-end test of /api/v1/auth/otp/*
against the real router, with the delivery service swapped for
FakeOtpDeliveryService (no real SMS/email ever sent) and the DB session
swapped for an isolated in-memory SQLite engine.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.otp_service as otp_service
from api.routes import auth as auth_routes
from api.routes import auth_otp as auth_otp_routes
from auth.otp_delivery import FakeOtpDeliveryService, get_otp_delivery_service
from db.models.user import OtpChallenge
from db.session import get_session


@pytest.fixture()
def delivery():
    return FakeOtpDeliveryService()


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def client(engine, delivery):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(auth_otp_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    app.dependency_overrides[get_otp_delivery_service] = lambda: delivery
    return TestClient(app)


def test_request_then_verify_creates_a_new_user_and_issues_tokens(client, delivery):
    req = client.post("/api/v1/auth/otp/request", json={"destination": "otpuser@example.com"})
    assert req.status_code == 204

    code = delivery.last_code
    verify = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "otpuser@example.com", "code": code}
    )

    assert verify.status_code == 200, verify.text
    tokens = verify.json()
    assert "access_token" in tokens and "refresh_token" in tokens


def test_request_response_never_reveals_the_code(client, delivery):
    req = client.post("/api/v1/auth/otp/request", json={"destination": "silent@example.com"})
    assert req.status_code == 204
    assert req.text == "" or delivery.last_code not in req.text


def test_verify_with_wrong_code_is_rejected(client, delivery):
    client.post("/api/v1/auth/otp/request", json={"destination": "wrongcode@example.com"})

    resp = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "wrongcode@example.com", "code": "000000"}
    )
    assert resp.status_code == 401


def test_verify_without_requesting_first_is_rejected(client):
    resp = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "never-requested@example.com", "code": "123456"}
    )
    assert resp.status_code == 401


def test_second_verify_with_same_code_is_rejected(client, delivery):
    client.post("/api/v1/auth/otp/request", json={"destination": "reuse@example.com"})
    code = delivery.last_code

    first = client.post("/api/v1/auth/otp/verify", json={"destination": "reuse@example.com", "code": code})
    assert first.status_code == 200

    second = client.post("/api/v1/auth/otp/verify", json={"destination": "reuse@example.com", "code": code})
    assert second.status_code == 401


def test_immediate_resend_is_rate_limited(client):
    first = client.post("/api/v1/auth/otp/request", json={"destination": "resend@example.com"})
    assert first.status_code == 204

    second = client.post("/api/v1/auth/otp/request", json={"destination": "resend@example.com"})
    assert second.status_code == 429


def test_otp_login_with_phone_destination_creates_phone_only_user(client, delivery):
    req = client.post("/api/v1/auth/otp/request", json={"destination": "+919876543210"})
    assert req.status_code == 204

    verify = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "+919876543210", "code": delivery.last_code}
    )
    assert verify.status_code == 200, verify.text


def test_invalid_destination_format_is_rejected_before_any_otp_is_generated(client, delivery):
    resp = client.post("/api/v1/auth/otp/request", json={"destination": "not-an-email-or-phone"})
    assert resp.status_code == 422
    assert delivery.sent == []


def test_returning_user_logs_in_with_otp_without_creating_a_duplicate_account(client, delivery, engine):
    client.post("/api/v1/auth/otp/request", json={"destination": "returning-otp@example.com"})
    first = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "returning-otp@example.com", "code": delivery.last_code}
    )
    first_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"}
    ).json()["id"]

    # Back-date the first challenge directly in the DB so the resend
    # cooldown has "elapsed" without an actual wall-clock sleep.
    with Session(engine) as session:
        challenge = session.exec(select(OtpChallenge)).first()
        challenge.created_at = datetime.now(timezone.utc) - timedelta(
            seconds=otp_service.OTP_RESEND_COOLDOWN_SECONDS + 1
        )
        session.add(challenge)
        session.commit()

    client.post("/api/v1/auth/otp/request", json={"destination": "returning-otp@example.com"})
    second = client.post(
        "/api/v1/auth/otp/verify", json={"destination": "returning-otp@example.com", "code": delivery.last_code}
    )
    second_id = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {second.json()['access_token']}"}
    ).json()["id"]

    assert first_id == second_id
