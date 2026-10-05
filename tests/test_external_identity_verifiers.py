"""tests/test_external_identity_verifiers.py — the REAL GoogleIdentityVerifier/
AppleIdentityVerifier classes (not the fakes used elsewhere), covering the
failure paths that are reachable without any network call: a malformed
token fails JWT header parsing locally before PyJWKClient would ever fetch
a provider's JWKS, and an unconfigured (empty) client id is rejected before
even attempting to parse the token. These are the only Google/Apple
verifier behaviors this test suite can exercise without reaching the real
network - see docs/AUTHENTICATION.md for what remains untested as a result
(live provider verification).
"""
import pytest

from auth.external_identity import AppleIdentityVerifier, ExternalIdentityError, GoogleIdentityVerifier


def test_google_verifier_rejects_malformed_token_without_a_network_call():
    verifier = GoogleIdentityVerifier(client_id="test-client-id")
    with pytest.raises(ExternalIdentityError):
        verifier.verify("not-a-real-jwt")


def test_google_verifier_with_no_client_id_fails_closed():
    verifier = GoogleIdentityVerifier(client_id=None)
    with pytest.raises(ExternalIdentityError):
        verifier.verify("irrelevant-since-it-should-fail-before-parsing")


def test_google_verifier_with_empty_client_id_fails_closed():
    verifier = GoogleIdentityVerifier(client_id="")
    with pytest.raises(ExternalIdentityError):
        verifier.verify("irrelevant-since-it-should-fail-before-parsing")


def test_apple_verifier_rejects_malformed_token_without_a_network_call():
    verifier = AppleIdentityVerifier(client_id="test-services-id")
    with pytest.raises(ExternalIdentityError):
        verifier.verify("not-a-real-jwt")


def test_apple_verifier_with_no_client_id_fails_closed():
    verifier = AppleIdentityVerifier(client_id=None)
    with pytest.raises(ExternalIdentityError):
        verifier.verify("irrelevant-since-it-should-fail-before-parsing")


def test_google_verifier_defaults_to_config_client_id_when_not_overridden(monkeypatch):
    import config

    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_ID", None)
    monkeypatch.setattr(config, "GOOGLE_ALLOWED_AUDIENCES", [])
    verifier = GoogleIdentityVerifier()  # no explicit client_id -> reads config
    with pytest.raises(ExternalIdentityError):
        verifier.verify("irrelevant-since-it-should-fail-before-parsing")


def test_apple_verifier_defaults_to_config_client_id_when_not_overridden(monkeypatch):
    import config

    monkeypatch.setattr(config, "APPLE_SERVICES_ID", None)
    verifier = AppleIdentityVerifier()  # no explicit client_id -> reads config
    with pytest.raises(ExternalIdentityError):
        verifier.verify("irrelevant-since-it-should-fail-before-parsing")
