"""tests/test_otp_service.py — OTP generation/storage/verification at the
service layer (auth/otp_service.py), independent of the HTTP routes.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.otp_service as otp_service
from auth.otp_delivery import FakeOtpDeliveryService
from db.models.user import OtpChallenge  # noqa: F401


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture()
def delivery():
    return FakeOtpDeliveryService()


# ══════════════════════════════════════════════════════════════════════════
# request_otp
# ══════════════════════════════════════════════════════════════════════════

def test_request_otp_delivers_a_code_of_the_configured_length(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")

    assert len(delivery.sent) == 1
    destination, code = delivery.sent[0]
    assert destination == "user@example.com"
    assert code.isdigit()
    assert len(code) == otp_service.OTP_CODE_LENGTH


def test_otp_code_is_never_stored_in_plaintext(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")

    challenge = session.exec(select(OtpChallenge)).first()
    assert challenge is not None
    assert delivery.last_code not in challenge.code_hash
    assert challenge.code_hash != delivery.last_code


def test_destination_is_normalized_case_insensitively_for_email(session, delivery):
    otp_service.request_otp(session, delivery, "User@Example.com")
    challenge = session.exec(select(OtpChallenge)).first()
    assert challenge.destination == "user@example.com"


def test_resend_before_cooldown_is_rejected(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")
    with pytest.raises(otp_service.OtpResendCooldownError):
        otp_service.request_otp(session, delivery, "user@example.com")


def test_resend_after_cooldown_elapses_is_allowed(session, delivery, monkeypatch):
    otp_service.request_otp(session, delivery, "user@example.com")

    # Simulate the cooldown having elapsed by backdating the stored
    # challenge rather than sleeping - deterministic, no wall-clock wait.
    challenge = session.exec(select(OtpChallenge)).first()
    challenge.created_at = datetime.now(timezone.utc) - timedelta(
        seconds=otp_service.OTP_RESEND_COOLDOWN_SECONDS + 1
    )
    session.add(challenge)
    session.commit()

    otp_service.request_otp(session, delivery, "user@example.com")  # must not raise
    assert len(delivery.sent) == 2


def test_too_many_requests_for_one_destination_are_rate_limited(session, delivery):
    now = datetime.now(timezone.utc)
    # Seed OTP_MAX_REQUESTS_PER_WINDOW challenges already issued, each far
    # enough apart to clear the resend cooldown but still inside the
    # rate-limit window.
    for i in range(otp_service.OTP_MAX_REQUESTS_PER_WINDOW):
        session.add(
            OtpChallenge(
                destination="spammed@example.com",
                purpose="login",
                code_salt="salt",
                code_hash="hash",
                expires_at=now + timedelta(minutes=5),
                created_at=now - timedelta(seconds=otp_service.OTP_RESEND_COOLDOWN_SECONDS * (i + 1)),
            )
        )
    session.commit()

    with pytest.raises(otp_service.OtpRateLimitedError):
        otp_service.request_otp(session, delivery, "spammed@example.com")


def test_too_many_requests_from_one_ip_across_destinations_are_rate_limited(session, delivery):
    now = datetime.now(timezone.utc)
    for i in range(otp_service.OTP_MAX_REQUESTS_PER_WINDOW):
        session.add(
            OtpChallenge(
                destination=f"victim{i}@example.com",
                purpose="login",
                code_salt="salt",
                code_hash="hash",
                expires_at=now + timedelta(minutes=5),
                requested_ip="1.2.3.4",
                created_at=now - timedelta(seconds=otp_service.OTP_RESEND_COOLDOWN_SECONDS * (i + 1)),
            )
        )
    session.commit()

    with pytest.raises(otp_service.OtpRateLimitedError):
        otp_service.request_otp(session, delivery, "newvictim@example.com", requested_ip="1.2.3.4")


# ══════════════════════════════════════════════════════════════════════════
# verify_otp
# ══════════════════════════════════════════════════════════════════════════

def test_verify_with_correct_code_succeeds(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")
    code = delivery.last_code

    otp_service.verify_otp(session, "user@example.com", code)  # must not raise


def test_verify_with_wrong_code_fails(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")

    with pytest.raises(otp_service.OtpInvalidError):
        otp_service.verify_otp(session, "user@example.com", "000000")


def test_verify_with_no_challenge_at_all_fails(session):
    with pytest.raises(otp_service.OtpInvalidError):
        otp_service.verify_otp(session, "nobody@example.com", "123456")


def test_verify_after_expiry_fails(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")
    code = delivery.last_code

    challenge = session.exec(select(OtpChallenge)).first()
    challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.add(challenge)
    session.commit()

    with pytest.raises(otp_service.OtpExpiredError):
        otp_service.verify_otp(session, "user@example.com", code)


def test_verify_twice_with_same_code_fails_the_second_time(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")
    code = delivery.last_code

    otp_service.verify_otp(session, "user@example.com", code)
    with pytest.raises(otp_service.OtpAlreadyUsedError):
        otp_service.verify_otp(session, "user@example.com", code)


def test_verify_stops_after_max_attempts(session, delivery):
    otp_service.request_otp(session, delivery, "user@example.com")

    for _ in range(otp_service.OTP_MAX_ATTEMPTS):
        with pytest.raises(otp_service.OtpInvalidError):
            otp_service.verify_otp(session, "user@example.com", "000000")

    # Even the CORRECT code must now be rejected - the challenge is burned.
    with pytest.raises(otp_service.OtpMaxAttemptsError):
        otp_service.verify_otp(session, "user@example.com", delivery.last_code)


def test_verify_only_checks_the_most_recently_requested_code(session, delivery, monkeypatch):
    otp_service.request_otp(session, delivery, "user@example.com")
    first_code = delivery.last_code

    # Clear the cooldown so a second request is allowed.
    challenge = session.exec(select(OtpChallenge)).first()
    challenge.created_at = datetime.now(timezone.utc) - timedelta(
        seconds=otp_service.OTP_RESEND_COOLDOWN_SECONDS + 1
    )
    session.add(challenge)
    session.commit()

    # Force a deterministically different second code so this test can't
    # flake on a 1-in-a-million random collision with the first one.
    second_code = "000000" if first_code != "000000" else "111111"
    monkeypatch.setattr(otp_service, "_generate_code", lambda length=otp_service.OTP_CODE_LENGTH: second_code)
    otp_service.request_otp(session, delivery, "user@example.com")
    assert delivery.last_code == second_code

    # The OLD code must no longer work once a newer one has been issued.
    with pytest.raises(otp_service.OtpInvalidError):
        otp_service.verify_otp(session, "user@example.com", first_code)

    otp_service.verify_otp(session, "user@example.com", second_code)  # the latest one still works
