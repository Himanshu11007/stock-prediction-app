"""tests/test_auth_concurrency.py — exactly-once guarantees under real
concurrency: N threads, each with its own database connection, released at
the same instant (conftest.run_concurrently), against a file-backed SQLite
database and - when TEST_DATABASE_URL is set - PostgreSQL.

Invariants:
- a refresh token can be rotated by exactly one request;
- an OTP challenge can be consumed by exactly one request, and concurrent
  wrong guesses can't exceed OTP_MAX_ATTEMPTS;
- a password-reset token can be used by exactly one request;
- at most one RANKING engine run can be RUNNING.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

import auth.otp_service as otp_service
import auth.password_reset as password_reset
import auth.service as auth_service
import engine_runs.service as runs
from auth.otp_delivery import FakeOtpDeliveryService
from db.models.market import EngineRun
from db.models.user import OtpChallenge, RefreshToken, User

N = 10


@pytest.fixture()
def user_id(concurrent_engine):
    with Session(concurrent_engine) as s:
        auth_service.ensure_roles_exist(s)
        return auth_service.create_user(s, "race@example.com", "Password123!").id


def _ok(results):
    return [r for r, e in results if e is None]


def _errors(results):
    return [e for _, e in results if e is not None]


# ── Refresh-token rotation ────────────────────────────────────────────────────

def test_simultaneous_refresh_with_the_same_token_succeeds_exactly_once(concurrent_engine, concurrently, user_id):
    with Session(concurrent_engine) as s:
        raw = auth_service.issue_refresh_token(s, s.get(User, user_id), device_id="dev-1")

    def rotate(_):
        with Session(concurrent_engine) as s:
            return auth_service.rotate_refresh_token(s, raw)[1]

    results = concurrently(rotate, N)
    assert len(_ok(results)) == 1
    assert len(_errors(results)) == N - 1
    assert all(isinstance(e, auth_service.InvalidCredentialsError) for e in _errors(results))

    with Session(concurrent_engine) as s:
        rows = s.exec(select(RefreshToken).where(RefreshToken.user_id == user_id)).all()
        active = [r for r in rows if r.revoked_at is None]
        assert len(rows) == 2 and len(active) == 1  # old revoked + exactly one replacement
        assert active[0].device_id == "dev-1"

    new_token = _ok(results)[0]
    with Session(concurrent_engine) as s:  # the replacement works, the old one never again
        assert auth_service.rotate_refresh_token(s, new_token)[0].id == user_id
        with pytest.raises(auth_service.InvalidCredentialsError):
            auth_service.rotate_refresh_token(s, raw)


def test_refresh_token_reuse_is_logged_as_a_security_event(concurrent_engine, user_id, caplog):
    with Session(concurrent_engine) as s:
        raw = auth_service.issue_refresh_token(s, s.get(User, user_id))
        auth_service.rotate_refresh_token(s, raw)
        with caplog.at_level("WARNING"), pytest.raises(auth_service.InvalidCredentialsError):
            auth_service.rotate_refresh_token(s, raw)
    assert "REFRESH_TOKEN_REUSE" in caplog.text
    assert raw not in caplog.text


def test_expired_refresh_token_is_rejected(concurrent_engine, user_id):
    with Session(concurrent_engine) as s:
        raw = auth_service.issue_refresh_token(s, s.get(User, user_id))
        row = s.exec(select(RefreshToken)).one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        s.add(row)
        s.commit()
        with pytest.raises(auth_service.InvalidCredentialsError):
            auth_service.rotate_refresh_token(s, raw)


def test_refresh_for_a_deactivated_user_fails_and_burns_the_token(concurrent_engine, user_id):
    with Session(concurrent_engine) as s:
        raw = auth_service.issue_refresh_token(s, s.get(User, user_id))
        user = s.get(User, user_id)
        user.is_active = False
        s.add(user)
        s.commit()
        with pytest.raises(auth_service.InvalidCredentialsError):
            auth_service.rotate_refresh_token(s, raw)
        assert s.exec(select(RefreshToken)).one().revoked_at is not None


# ── OTP ───────────────────────────────────────────────────────────────────────

def _issue_otp(engine, destination="otp-race@example.com"):
    delivery = FakeOtpDeliveryService()
    with Session(engine) as s:
        otp_service.request_otp(s, delivery, destination)
    return destination, delivery.last_code


def test_simultaneous_verification_of_the_same_otp_succeeds_exactly_once(concurrent_engine, concurrently):
    destination, code = _issue_otp(concurrent_engine)

    def verify(_):
        with Session(concurrent_engine) as s:
            otp_service.verify_otp(s, destination, code)
            return True

    results = concurrently(verify, N)
    assert len(_ok(results)) == 1
    assert all(isinstance(e, otp_service.OtpError) for e in _errors(results))
    with Session(concurrent_engine) as s:
        assert s.exec(select(OtpChallenge)).one().consumed_at is not None
        with pytest.raises(otp_service.OtpAlreadyUsedError):
            otp_service.verify_otp(s, destination, code)


def test_concurrent_wrong_guesses_cannot_exceed_the_attempt_limit(concurrent_engine, concurrently):
    destination, code = _issue_otp(concurrent_engine)
    wrong = "000000" if code != "000000" else "111111"

    def guess(_):
        with Session(concurrent_engine) as s:
            otp_service.verify_otp(s, destination, wrong)

    results = concurrently(guess, 3 * otp_service.OTP_MAX_ATTEMPTS)
    invalid = [e for e in _errors(results) if type(e) is otp_service.OtpInvalidError]
    assert len(invalid) == otp_service.OTP_MAX_ATTEMPTS  # only these were actually compared
    with Session(concurrent_engine) as s:
        assert s.exec(select(OtpChallenge)).one().attempt_count == otp_service.OTP_MAX_ATTEMPTS
        with pytest.raises(otp_service.OtpMaxAttemptsError):
            otp_service.verify_otp(s, destination, code)  # even the right code is refused now


def test_expired_otp_is_rejected(concurrent_engine):
    destination, code = _issue_otp(concurrent_engine)
    with Session(concurrent_engine) as s:
        ch = s.exec(select(OtpChallenge)).one()
        ch.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        s.add(ch)
        s.commit()
        with pytest.raises(otp_service.OtpExpiredError):
            otp_service.verify_otp(s, destination, code)


def test_otp_reuse_is_rejected_and_logged(concurrent_engine, caplog):
    destination, code = _issue_otp(concurrent_engine)
    with Session(concurrent_engine) as s:
        otp_service.verify_otp(s, destination, code)
        with caplog.at_level("WARNING"), pytest.raises(otp_service.OtpAlreadyUsedError):
            otp_service.verify_otp(s, destination, code)
    assert "OTP_REUSE" in caplog.text
    assert code not in caplog.text


# ── Password reset ───────────────────────────────────────────────────────────

def test_simultaneous_resets_with_the_same_token_succeed_exactly_once(concurrent_engine, concurrently, user_id):
    with Session(concurrent_engine) as s:
        _email, raw = password_reset.request_password_reset(s, "race@example.com")

    def reset(i):
        with Session(concurrent_engine) as s:
            password_reset.reset_password(s, raw, f"NewPassword{i}!")
            return i

    results = concurrently(reset, N)
    assert len(_ok(results)) == 1
    assert all(isinstance(e, password_reset.InvalidResetTokenError) for e in _errors(results))
    winner = _ok(results)[0]
    with Session(concurrent_engine) as s:
        assert auth_service.authenticate_user(s, "race@example.com", f"NewPassword{winner}!").id == user_id


# ── Engine runs ──────────────────────────────────────────────────────────────

def test_simultaneous_ranking_runs_leave_exactly_one_running(concurrent_engine, concurrently):
    def start(_):
        with Session(concurrent_engine) as s:
            return runs.create_run(s, kind="RANKING", triggered_by=None, config={}).run_id

    results = concurrently(start, N)
    assert len(_ok(results)) == 1
    assert len(_errors(results)) == N - 1
    assert all(isinstance(e, runs.RunInProgressError) for e in _errors(results))
    with Session(concurrent_engine) as s:
        running = s.exec(select(EngineRun).where(EngineRun.status == "RUNNING")).all()
        assert [r.run_id for r in running] == _ok(results)


def test_a_new_ranking_run_can_start_once_the_previous_one_finished(concurrent_engine):
    with Session(concurrent_engine) as s:
        first = runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        first.status = "COMPLETED"
        s.add(first)
        s.commit()
        assert runs.create_run(s, kind="RANKING", triggered_by=None, config={}).status == "RUNNING"


def test_an_abandoned_running_run_does_not_block_forever(concurrent_engine):
    with Session(concurrent_engine) as s:
        stale = runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        stale.started_at = datetime.now(timezone.utc) - runs.RUN_STALE_AFTER - timedelta(minutes=1)
        s.add(stale)
        s.commit()
        fresh = runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        s.refresh(stale)
        assert stale.status == "FAILED" and fresh.status == "RUNNING"


def test_single_stock_runs_are_not_limited_by_the_ranking_guard(concurrent_engine, concurrently):
    with Session(concurrent_engine) as s:
        runs.create_run(s, kind="RANKING", triggered_by=None, config={})

    def single(_):
        with Session(concurrent_engine) as s:
            return runs.create_run(s, kind="SINGLE", triggered_by=None, config={}).run_id

    assert len(_ok(concurrently(single, 4))) == 4


def test_database_rejects_a_second_running_ranking_row_even_without_the_check(concurrent_engine):
    from sqlalchemy.exc import IntegrityError

    with Session(concurrent_engine) as s:
        for i in range(2):
            s.add(EngineRun(run_id=f"raw-{i}", kind="RANKING", status="RUNNING",
                            engine_version="v", fqvf_version="v"))
        with pytest.raises(IntegrityError):
            s.commit()
