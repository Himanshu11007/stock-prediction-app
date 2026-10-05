"""tests/test_google_id_token_security.py — the REAL GoogleIdentityVerifier
against real RS256 signatures, and /api/v1/auth/google end-to-end with it.

Google's JWKS endpoint is replaced by a locally generated RSA key set (the
real jwt.PyJWKClient is used, only its HTTP fetch is stubbed), so signature,
key id, audience, issuer, expiry and claim checks all run exactly as in
production - no network, no fakes for the verification itself.
"""
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
from api.routes import auth as auth_routes
from api.routes import auth_devices as auth_devices_routes
from api.routes import auth_sso as auth_sso_routes
from auth.external_identity import (
    ExternalIdentityError,
    GoogleIdentityVerifier,
    find_or_create_user_for_identity,
)
from db.models.user import ExternalIdentity, RefreshToken, User
from db.session import get_session

WEB_CLIENT_ID = "1111-web.apps.googleusercontent.com"
IOS_CLIENT_ID = "2222-ios.apps.googleusercontent.com"
KID = "test-key-1"


def _new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


GOOGLE_KEY = _new_key()
ATTACKER_KEY = _new_key()


class StubJwkClient(jwt.PyJWKClient):
    """The real PyJWKClient (key-id lookup, caching), serving a local JWKS."""

    def __init__(self, *keys):
        super().__init__("https://example.invalid/certs")
        self._jwks = {"keys": []}
        for kid, key in keys:
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
            jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
            self._jwks["keys"].append(jwk)

    def fetch_data(self):
        return self._jwks


def make_token(*, key=GOOGLE_KEY, kid=KID, alg="RS256", **overrides):
    now = int(time.time())
    claims = {
        "iss": "https://accounts.google.com",
        "aud": WEB_CLIENT_ID,
        "azp": WEB_CLIENT_ID,
        "sub": "109876543210",
        "email": "Jane.Doe@Gmail.com",
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    for k in [k for k, v in claims.items() if v is None]:
        del claims[k]
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid})


@pytest.fixture()
def verifier():
    return GoogleIdentityVerifier([WEB_CLIENT_ID, IOS_CLIENT_ID],
                                  jwk_client=StubJwkClient((KID, GOOGLE_KEY)), leeway=60)


# ══════════════════════════════════════════════════════════════════════════
# Verifier: signature / audience / issuer / expiry / claims
# ══════════════════════════════════════════════════════════════════════════

def test_valid_token_is_accepted_and_keyed_by_sub(verifier):
    identity = verifier.verify(make_token())
    assert identity.provider == "google"
    assert identity.subject == "109876543210"
    assert identity.email == "jane.doe@gmail.com"  # normalized for lookups
    assert identity.email_verified is True


def test_second_configured_audience_is_accepted(verifier):
    assert verifier.verify(make_token(aud=IOS_CLIENT_ID)).subject == "109876543210"


def test_wrong_audience_is_rejected(verifier):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(aud="9999-some-other-app.apps.googleusercontent.com"))


def test_missing_audience_is_rejected(verifier):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(aud=None))


@pytest.mark.parametrize("issuer", ["https://evil.example.com", "https://accounts.google.com.evil.com", None])
def test_wrong_or_missing_issuer_is_rejected(verifier, issuer):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(iss=issuer))


def test_legacy_google_issuer_without_scheme_is_accepted(verifier):
    assert verifier.verify(make_token(iss="accounts.google.com")).subject


def test_expired_token_is_rejected(verifier):
    now = int(time.time())
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(iat=now - 7200, exp=now - 3600))


def test_small_clock_skew_is_tolerated(verifier):
    now = int(time.time())
    # Phone clock 30 s ahead: iat slightly in the future, still accepted.
    assert verifier.verify(make_token(iat=now + 30, exp=now + 3630)).subject


def test_token_issued_far_in_the_future_is_rejected(verifier):
    now = int(time.time())
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(iat=now + 600, exp=now + 4200))


def test_signature_by_another_key_with_the_same_kid_is_rejected(verifier):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(key=ATTACKER_KEY))


def test_tampered_payload_is_rejected(verifier):
    header, payload, signature = make_token().split(".")
    forged = jwt.utils.base64url_encode(json.dumps(
        {"iss": "https://accounts.google.com", "aud": WEB_CLIENT_ID, "sub": "attacker",
         "email": "victim@gmail.com", "email_verified": True,
         "iat": int(time.time()), "exp": int(time.time()) + 3600}).encode()).decode()
    with pytest.raises(ExternalIdentityError):
        verifier.verify(f"{header}.{forged}.{signature}")


def test_unknown_key_id_is_rejected(verifier):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(kid="not-a-google-key"))


def test_alg_none_token_is_rejected(verifier):
    now = int(time.time())
    token = jwt.encode({"iss": "https://accounts.google.com", "aud": WEB_CLIENT_ID, "sub": "x",
                        "iat": now, "exp": now + 3600}, None, algorithm="none", headers={"kid": KID})
    with pytest.raises(ExternalIdentityError):
        verifier.verify(token)


def test_hs256_key_confusion_is_rejected(verifier):
    # Classic alg-confusion: sign with HS256 using the PUBLIC key as secret.
    from cryptography.hazmat.primitives import serialization
    public_pem = GOOGLE_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    now = int(time.time())
    header = jwt.utils.base64url_encode(json.dumps({"alg": "HS256", "kid": KID, "typ": "JWT"}).encode())
    payload = jwt.utils.base64url_encode(json.dumps(
        {"iss": "https://accounts.google.com", "aud": WEB_CLIENT_ID, "sub": "x",
         "iat": now, "exp": now + 3600}).encode())
    import hashlib
    import hmac
    sig = jwt.utils.base64url_encode(hmac.new(public_pem, header + b"." + payload, hashlib.sha256).digest())
    with pytest.raises(ExternalIdentityError):
        verifier.verify((header + b"." + payload + b"." + sig).decode())


@pytest.mark.parametrize("claim", ["sub", "exp", "iat"])
def test_missing_required_claim_is_rejected(verifier, claim):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(**{claim: None}))


def test_blank_sub_is_rejected(verifier):
    with pytest.raises(ExternalIdentityError):
        verifier.verify(make_token(sub="  "))


def test_string_email_verified_flags_are_understood(verifier):
    assert verifier.verify(make_token(email_verified="true")).email_verified is True
    assert verifier.verify(make_token(email_verified="false")).email_verified is False
    assert verifier.verify(make_token(email_verified=None)).email_verified is False


def test_no_configured_audience_fails_closed():
    v = GoogleIdentityVerifier([], jwk_client=StubJwkClient((KID, GOOGLE_KEY)))
    with pytest.raises(ExternalIdentityError):
        v.verify(make_token())


def test_jwks_client_is_reused_across_verifications(monkeypatch):
    import auth.external_identity as ext

    monkeypatch.setattr(ext, "_jwk_clients", {})
    assert ext._jwk_client_for(ext.GOOGLE_JWKS_URL) is ext._jwk_client_for(ext.GOOGLE_JWKS_URL)


def test_config_combines_all_google_client_ids(monkeypatch):
    import importlib

    import config

    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "a.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_IOS_CLIENT_ID", "b.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_ALLOWED_AUDIENCES", "c.apps.googleusercontent.com, a.apps.googleusercontent.com")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.GOOGLE_ALLOWED_AUDIENCES == [
            "a.apps.googleusercontent.com", "b.apps.googleusercontent.com", "c.apps.googleusercontent.com"]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# ══════════════════════════════════════════════════════════════════════════
# POST /api/v1/auth/google end-to-end with the real verifier
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def client(engine, verifier):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(auth_sso_routes.router, prefix="/api/v1")
    app.include_router(auth_devices_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    app.dependency_overrides[auth_sso_routes.get_google_identity_verifier] = lambda: verifier
    return TestClient(app)


def _google(client, token, **extra):
    return client.post("/api/v1/auth/google", json={"id_token": token, **extra})


def _me(client, access_token):
    return client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})


def test_google_login_issues_tokens_and_establishes_device_session(client):
    resp = _google(client, make_token(), device_id="device-abc", device_name="Pixel 9")
    assert resp.status_code == 200, resp.text
    tokens = resp.json()
    assert tokens["token_type"] == "bearer"

    me = _me(client, tokens["access_token"])
    assert me.status_code == 200
    assert me.json()["email"] == "jane.doe@gmail.com"
    assert me.json()["roles"] == ["USER"]

    sessions = client.get("/api/v1/auth/sessions",
                          headers={"Authorization": f"Bearer {tokens['access_token']}"}).json()
    assert [(s["device_id"], s["device_name"]) for s in sessions] == [("device-abc", "Pixel 9")]

    refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    assert _me(client, refreshed.json()["access_token"]).json()["id"] == me.json()["id"]

    assert client.post("/api/v1/auth/logout", json={"refresh_token": refreshed.json()["refresh_token"]}).status_code == 204
    assert client.post("/api/v1/auth/refresh",
                       json={"refresh_token": refreshed.json()["refresh_token"]}).status_code == 401


def test_returning_google_user_gets_the_same_account_even_if_email_changed(client):
    first = _google(client, make_token())
    second = _google(client, make_token(email="renamed@gmail.com"))
    assert _me(client, first.json()["access_token"]).json()["id"] == \
        _me(client, second.json()["access_token"]).json()["id"]


def test_email_case_differences_never_create_a_second_account(client, engine):
    _google(client, make_token(sub="sub-a", email="Case@Example.com"))
    # A different Google account claiming the same address in another case
    # hits the existing account (409), it does not create a duplicate.
    resp = _google(client, make_token(sub="sub-b", email="CASE@example.COM"))
    assert resp.status_code == 409
    with Session(engine) as s:
        assert len(s.exec(select(User)).all()) == 1


@pytest.mark.parametrize("case,status", [
    ("invalid_signature", 401), ("wrong_audience", 401), ("wrong_issuer", 401),
    ("expired", 401), ("garbage", 401), ("unverified_email", 403),
])
def test_rejected_google_tokens(client, engine, case, status):
    now = int(time.time())
    token = {
        "invalid_signature": lambda: make_token(key=ATTACKER_KEY),
        "wrong_audience": lambda: make_token(aud="other.apps.googleusercontent.com"),
        "wrong_issuer": lambda: make_token(iss="https://login.example.com"),
        "expired": lambda: make_token(iat=now - 7200, exp=now - 3600),
        "garbage": lambda: "definitely.not.ajwt",
        "unverified_email": lambda: make_token(email_verified=False),
    }[case]()
    resp = _google(client, token)
    assert resp.status_code == status, resp.text
    assert "access_token" not in resp.text
    with Session(engine) as s:
        assert s.exec(select(User)).all() == []  # nothing created
        assert s.exec(select(RefreshToken)).all() == []


def test_existing_password_account_with_same_email_requires_explicit_linking(client, engine):
    client.post("/api/v1/auth/register", json={"email": "jane.doe@gmail.com", "password": "Password123!"})
    resp = _google(client, make_token())
    assert resp.status_code == 409
    assert "link" in resp.json()["detail"].lower()

    # The owner signs in with the password and links Google explicitly ...
    login = client.post("/api/v1/auth/login", data={"username": "jane.doe@gmail.com", "password": "Password123!"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.post("/api/v1/auth/link/google", json={"id_token": make_token()}, headers=headers).status_code == 200
    # ... after which Google sign-in reaches that same account.
    google = _google(client, make_token())
    assert google.status_code == 200
    assert _me(client, google.json()["access_token"]).json()["email"] == "jane.doe@gmail.com"
    with Session(engine) as s:
        assert len(s.exec(select(User)).all()) == 1


def test_google_identity_already_linked_to_another_user_cannot_be_linked_again(client):
    _google(client, make_token(sub="owned-sub", email="owner@gmail.com"))
    client.post("/api/v1/auth/register", json={"email": "other@example.com", "password": "Password123!"})
    login = client.post("/api/v1/auth/login", data={"username": "other@example.com", "password": "Password123!"})
    resp = client.post("/api/v1/auth/link/google", json={"id_token": make_token(sub="owned-sub", email="owner@gmail.com")},
                       headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    assert resp.status_code == 409


def test_deactivated_google_user_cannot_sign_in(client, engine):
    _google(client, make_token())
    with Session(engine) as s:
        user = s.exec(select(User)).one()
        user.is_active = False
        s.add(user)
        s.commit()
    assert _google(client, make_token()).status_code == 401


def test_unlinking_the_only_sign_in_method_is_refused(client):
    tokens = _google(client, make_token()).json()
    resp = client.delete("/api/v1/auth/identities/google",
                         headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert resp.status_code == 400


def test_google_login_never_logs_the_id_token(client, caplog):
    token = make_token()
    with caplog.at_level("DEBUG"):
        _google(client, token)
        _google(client, make_token(key=ATTACKER_KEY))
    assert token.split(".")[2] not in caplog.text
    assert "jane.doe@gmail.com" not in caplog.text
    assert "GOOGLE_LOGIN_SUCCEEDED" in caplog.text and "GOOGLE_LOGIN_FAILED" in caplog.text


# ══════════════════════════════════════════════════════════════════════════
# Concurrency: simultaneous first sign-ins with the same Google account
# ══════════════════════════════════════════════════════════════════════════

def test_simultaneous_first_google_logins_create_exactly_one_user(concurrent_engine, concurrently, verifier):
    with Session(concurrent_engine) as s:
        auth_service.ensure_roles_exist(s)
    token = make_token(sub="race-sub", email="race@gmail.com")

    def attempt(_):
        with Session(concurrent_engine) as s:
            user, _new = find_or_create_user_for_identity(s, verifier.verify(token))
            return user.id

    results = concurrently(attempt, 8)
    errors = [e for _, e in results if e is not None]
    assert errors == [], errors
    assert len({uid for uid, _ in results}) == 1
    with Session(concurrent_engine) as s:
        assert len(s.exec(select(User)).all()) == 1
        assert len(s.exec(select(ExternalIdentity)).all()) == 1
