"""Google/Apple external-identity verification and account resolution - the
only module that should touch the external_identities table directly.

Both Google and Apple issue the same kind of credential to a signed-in
client: an OIDC id_token - a JWT signed with the provider's own rotating
RSA keys (published as a JWKS), carrying `sub` (the provider's stable,
durable user id), `aud` (must match our registered client id), `iss` (the
provider's issuer), and `exp`. Verifying it is "fetch the provider's
current public keys, check the signature, check aud/iss/exp" - the same
shape of check for both providers, so both verifiers share the
_verify_oidc_id_token() helper below instead of duplicating it.

CRITICAL: the client-supplied email is NEVER trusted as proof of identity.
Only `sub`, once verified against the provider's own signature, is used to
find or create a StockLens account - see find_or_create_user_for_identity().
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Union

import jwt
from sqlalchemy.exc import IntegrityError

from auth.security_events import log_security_event, mask_email
from auth.service import create_external_user
from db.models.user import ExternalIdentity, User
from sqlmodel import Session, select

GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"


class ExternalIdentityError(Exception):
    """Raised when a Google/Apple identity token fails verification (bad
    signature, wrong audience/issuer, expired, malformed, or the provider
    isn't configured on this server)."""


class ExternalEmailNotVerifiedError(ExternalIdentityError):
    """The token is genuine, but the provider says its email address is not
    verified. Such an email is never attached to a new StockLens account:
    anyone can create a Google account claiming someone else's address, and
    letting that claim create (and reserve) the account would let an
    attacker pre-register a victim's email before the victim signs up."""


class AccountLinkingRequiredError(Exception):
    """Raised instead of silently merging accounts when a brand-new
    external identity's email matches an existing StockLens account. An
    email match alone is never proof of ownership (a password account's
    email could be unverified, a typo, or stale) - the user must sign in to
    the existing account and explicitly link the provider from there (see
    link_identity())."""

    def __init__(self, message: str, *, existing_email: str):
        super().__init__(message)
        self.existing_email = existing_email


class DuplicateExternalIdentityError(Exception):
    """Raised when linking an identity that's already linked to a
    *different* StockLens account, or when the calling account already has a
    linked identity for this provider."""


class LastAuthMethodError(Exception):
    """Raised when unlinking would leave the account with no way to sign
    in at all."""


@dataclass(frozen=True)
class VerifiedExternalIdentity:
    provider: str  # "google" | "apple"
    subject: str  # the provider's stable `sub` claim - the actual identity key
    email: Optional[str]
    email_verified: bool


class IGoogleIdentityVerifier(Protocol):
    def verify(self, id_token: str) -> VerifiedExternalIdentity: ...


class IAppleIdentityVerifier(Protocol):
    def verify(self, identity_token: str) -> VerifiedExternalIdentity: ...


class _JwkClient(Protocol):
    def get_signing_key_from_jwt(self, token: str): ...


# One PyJWKClient per JWKS URL for the whole process: it caches the
# provider's key set (and individual keys), so a login no longer re-downloads
# Google's certificates every time - previously each request built a fresh
# client, adding a network round trip to every sign-in and turning any
# transient googleapis.com hiccup into a failed login. A token signed with a
# key id the cache doesn't know yet triggers one refetch (PyJWKClient's own
# behaviour), so key rotation still works.
_jwk_clients: dict[str, jwt.PyJWKClient] = {}
_jwk_clients_lock = threading.Lock()


def _jwk_client_for(jwks_url: str) -> jwt.PyJWKClient:
    with _jwk_clients_lock:
        client = _jwk_clients.get(jwks_url)
        if client is None:
            client = jwt.PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=3600, timeout=10)
            _jwk_clients[jwks_url] = client
        return client


def _normalize_audiences(audience: Union[None, str, Sequence[str]]) -> list[str]:
    if audience is None:
        return []
    if isinstance(audience, str):
        audience = [audience]
    return [a.strip() for a in audience if a and a.strip()]


def _verify_oidc_id_token(
    token: str,
    *,
    jwks_url: str,
    issuers: tuple[str, ...],
    audience: Union[None, str, Sequence[str]],
    jwk_client: Optional[_JwkClient] = None,
    leeway: Optional[int] = None,
) -> dict:
    """Verifies signature (provider JWKS, RS256 only), `aud` (one of the
    configured client ids), `iss`, `exp`/`iat` (with a small clock-skew
    leeway) and the presence of `sub`. Raises ExternalIdentityError with a
    generic message on any failure - the precise reason goes to the caller
    via the exception chain, never to the client."""
    audiences = _normalize_audiences(audience)
    if not audiences:
        raise ExternalIdentityError(
            "This sign-in method is not configured on the server (missing client id)"
        )
    if leeway is None:
        from config import EXTERNAL_ID_TOKEN_LEEWAY_SECONDS

        leeway = EXTERNAL_ID_TOKEN_LEEWAY_SECONDS
    try:
        signing_key = (jwk_client or _jwk_client_for(jwks_url)).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audiences,
            leeway=leeway,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise ExternalIdentityError("Invalid identity token") from exc
    except Exception as exc:  # network/JWKS-fetch failures etc.
        raise ExternalIdentityError("Could not verify identity token") from exc

    if claims.get("iss") not in issuers:
        raise ExternalIdentityError("Invalid identity token issuer")
    if not isinstance(claims.get("sub"), str) or not claims["sub"].strip():
        raise ExternalIdentityError("Invalid identity token")
    return claims


def _claim_is_true(value) -> bool:
    # Google/Apple have both sent email_verified as a JSON bool and, in some
    # token flavours, as the string "true"/"false".
    return value in (True, "true", "True", "1", 1)


class GoogleIdentityVerifier:
    """Production verifier - validates against Google's own rotating public
    keys over the network. The token's `aud` must be one of
    config.GOOGLE_ALLOWED_AUDIENCES (GOOGLE_OAUTH_CLIENT_ID plus any extra
    client ids of this same Google Cloud project) - otherwise any Google
    user's token for a completely different app would be accepted here.

    client_id / jwk_client are injectable for tests (a locally generated RSA
    key set instead of Google's), production uses neither."""

    def __init__(
        self,
        client_id: Union[None, str, Sequence[str]] = None,
        *,
        jwk_client: Optional[_JwkClient] = None,
        leeway: Optional[int] = None,
    ):
        if client_id is None:
            from config import GOOGLE_ALLOWED_AUDIENCES

            client_id = GOOGLE_ALLOWED_AUDIENCES
        self._audiences = _normalize_audiences(client_id)
        self._jwk_client = jwk_client
        self._leeway = leeway

    def verify(self, id_token: str) -> VerifiedExternalIdentity:
        claims = _verify_oidc_id_token(
            id_token, jwks_url=GOOGLE_JWKS_URL, issuers=GOOGLE_ISSUERS, audience=self._audiences,
            jwk_client=self._jwk_client, leeway=self._leeway,
        )
        email = claims.get("email")
        return VerifiedExternalIdentity(
            provider="google",
            subject=claims["sub"],
            email=email.strip().lower() if isinstance(email, str) and email.strip() else None,
            email_verified=_claim_is_true(claims.get("email_verified", False)),
        )


class AppleIdentityVerifier:
    """Production verifier - validates against Apple's own rotating public
    keys over the network. Requires APPLE_SERVICES_ID (config.py) to be set
    to this app's registered Sign in with Apple Services ID, which must
    match the token's `aud` claim exactly."""

    def __init__(self, client_id: Optional[str] = None):
        if client_id is None:
            from config import APPLE_SERVICES_ID

            client_id = APPLE_SERVICES_ID
        self._client_id = client_id

    def verify(self, identity_token: str) -> VerifiedExternalIdentity:
        claims = _verify_oidc_id_token(
            identity_token, jwks_url=APPLE_JWKS_URL, issuers=(APPLE_ISSUER,), audience=self._client_id
        )
        # Apple's email_verified has been a bool and, in older/web tokens, a
        # string "true"/"false" - accept both rather than silently treating
        # a verified email as unverified.
        return VerifiedExternalIdentity(
            provider="apple",
            subject=claims["sub"],
            # Apple only includes `email` on the FIRST authorization (or
            # when using their private relay) - a None here on later logins
            # is expected and must not be treated as the identity changing;
            # callers must key on `subject`, never on email, for exactly
            # this reason.
            email=claims.get("email"),
            email_verified=_claim_is_true(claims.get("email_verified", False)),
        )


class FakeGoogleIdentityVerifier:
    """Deterministic test double - no network calls. Register tokens up
    front with .register(); verify() raises ExternalIdentityError for any
    unrecognized token, exactly like a real invalid/forged/expired token."""

    def __init__(self):
        self._identities: dict[str, VerifiedExternalIdentity] = {}

    def register(
        self, token: str, *, subject: str, email: Optional[str] = None, email_verified: bool = True
    ) -> None:
        self._identities[token] = VerifiedExternalIdentity("google", subject, email, email_verified)

    def verify(self, id_token: str) -> VerifiedExternalIdentity:
        identity = self._identities.get(id_token)
        if identity is None:
            raise ExternalIdentityError("Invalid identity token")
        return identity


class FakeAppleIdentityVerifier:
    """Deterministic test double - no network calls. See FakeGoogleIdentityVerifier."""

    def __init__(self):
        self._identities: dict[str, VerifiedExternalIdentity] = {}

    def register(
        self, token: str, *, subject: str, email: Optional[str] = None, email_verified: bool = True
    ) -> None:
        self._identities[token] = VerifiedExternalIdentity("apple", subject, email, email_verified)

    def verify(self, identity_token: str) -> VerifiedExternalIdentity:
        identity = self._identities.get(identity_token)
        if identity is None:
            raise ExternalIdentityError("Invalid identity token")
        return identity


def _find_link(session: Session, provider: str, subject: str) -> Optional[ExternalIdentity]:
    return session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.provider == provider,
            ExternalIdentity.provider_subject == subject,
        )
    ).first()


def _user_for_link(session: Session, link: ExternalIdentity) -> User:
    user = session.get(User, link.user_id)
    if user is None or not user.is_active:
        raise ExternalIdentityError("This account is unavailable")
    return user


def _raise_if_email_taken(session: Session, identity: VerifiedExternalIdentity) -> None:
    if not identity.email:
        return
    normalized_email = identity.email.strip().lower()
    colliding_user = session.exec(select(User).where(User.email == normalized_email)).first()
    if colliding_user is not None:
        raise AccountLinkingRequiredError(
            "An account with this email already exists. Sign in to that account and link "
            f"{identity.provider.title()} from account settings.",
            existing_email=normalized_email,
        )


def find_or_create_user_for_identity(
    session: Session, identity: VerifiedExternalIdentity
) -> tuple[User, bool]:
    """Returns (user, is_new_user). The identity key is ALWAYS the
    provider's stable (provider, sub) pair - never the email. Resolution:

    1. An external_identities row already matches (provider, subject) ->
       that user (returning user - the common case on every login after
       the first). The token's current email is irrelevant here: a Google
       account whose address changed still signs in to the same account.
    2. No match, and the provider does not vouch for the email
       (email_verified false) -> ExternalEmailNotVerifiedError. A Google
       token must also carry an email at all (Google always includes it for
       the `email` scope Credential Manager / Google Identity Services use).
    3. No match, but an existing StockLens account already has this exact
       (normalized) email -> AccountLinkingRequiredError rather than
       silently attaching this identity to it (see that class's docstring).
    4. No match anywhere -> create the user AND its identity link in one
       transaction. Two simultaneous first sign-ins with the same Google
       account race on uq_external_identity_subject: the loser rolls back
       (no orphan password-less user is left behind) and signs in to the
       winner's account instead of getting a 500.
    """
    existing_link = _find_link(session, identity.provider, identity.subject)
    if existing_link is not None:
        return _user_for_link(session, existing_link), False

    if identity.provider == "google" and not identity.email:
        raise ExternalEmailNotVerifiedError("Your Google account has no email address to sign in with")
    if identity.email and not identity.email_verified:
        raise ExternalEmailNotVerifiedError(
            f"Your {identity.provider.title()} account's email address is not verified"
        )

    try:
        _raise_if_email_taken(session, identity)
    except AccountLinkingRequiredError:
        # The email may belong to the account a concurrent first sign-in
        # with this very identity just created (it committed between our
        # link lookup above and the email check) - that is the same person.
        existing_link = _find_link(session, identity.provider, identity.subject)
        if existing_link is not None:
            return _user_for_link(session, existing_link), False
        raise

    try:
        user = create_external_user(session, email=identity.email, commit=False)
        session.add(
            ExternalIdentity(
                user_id=user.id,
                provider=identity.provider,
                provider_subject=identity.subject,
                email=identity.email,
            )
        )
        session.commit()
    except IntegrityError:
        session.rollback()
        # Lost a race: either the same identity was linked concurrently
        # (-> sign in to that account) or the email was registered
        # concurrently (-> same answer as step 3).
        existing_link = _find_link(session, identity.provider, identity.subject)
        if existing_link is not None:
            return _user_for_link(session, existing_link), False
        _raise_if_email_taken(session, identity)
        raise ExternalIdentityError("Could not complete sign-in, please try again")
    session.refresh(user)
    log_security_event("EXTERNAL_ACCOUNT_CREATED", provider=identity.provider, user_id=user.id,
                       email=mask_email(identity.email))
    return user, True


def link_identity(session: Session, user: User, identity: VerifiedExternalIdentity) -> ExternalIdentity:
    """Explicit account-linking, initiated by an already-authenticated
    user (never automatic). Idempotent if this exact identity is already
    linked to this same account."""
    existing_link = session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.provider == identity.provider,
            ExternalIdentity.provider_subject == identity.subject,
        )
    ).first()
    if existing_link is not None:
        if existing_link.user_id == user.id:
            return existing_link
        raise DuplicateExternalIdentityError(
            f"This {identity.provider.title()} account is already linked to a different StockLens account"
        )

    already_has_provider = session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.user_id == user.id, ExternalIdentity.provider == identity.provider
        )
    ).first()
    if already_has_provider is not None:
        raise DuplicateExternalIdentityError(
            f"Your account already has a linked {identity.provider.title()} identity"
        )

    link = ExternalIdentity(
        user_id=user.id,
        provider=identity.provider,
        provider_subject=identity.subject,
        email=identity.email,
    )
    session.add(link)
    try:
        session.commit()
    except IntegrityError:
        # The same identity (or this provider for this user) was linked by a
        # concurrent request - the unique constraints decided, not us.
        session.rollback()
        raise DuplicateExternalIdentityError(
            f"This {identity.provider.title()} account is already linked to a StockLens account"
        )
    session.refresh(link)
    log_security_event("EXTERNAL_IDENTITY_LINKED", provider=identity.provider, user_id=user.id)
    return link


def unlink_identity(session: Session, user: User, provider: str) -> None:
    """No-op (not an error) if this provider isn't linked. Raises
    LastAuthMethodError if removing it would leave the account with no way
    to sign in at all (no password, no phone/OTP, and no other linked
    provider)."""
    link = session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.user_id == user.id, ExternalIdentity.provider == provider
        )
    ).first()
    if link is None:
        return

    has_other_identity = session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.user_id == user.id, ExternalIdentity.id != link.id
        )
    ).first() is not None
    has_other_login_method = has_other_identity or user.hashed_password is not None or user.phone is not None
    if not has_other_login_method:
        raise LastAuthMethodError(
            "Can't remove your only sign-in method. Set a password or link another provider first."
        )

    session.delete(link)
    session.commit()
    log_security_event("EXTERNAL_IDENTITY_UNLINKED", provider=provider, user_id=user.id)


def list_identities(session: Session, user: User) -> list[ExternalIdentity]:
    return list(
        session.exec(select(ExternalIdentity).where(ExternalIdentity.user_id == user.id)).all()
    )
