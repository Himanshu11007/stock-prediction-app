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

from dataclasses import dataclass
from typing import Optional, Protocol

import jwt

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


def _verify_oidc_id_token(
    token: str, *, jwks_url: str, issuers: tuple[str, ...], audience: Optional[str]
) -> dict:
    if not audience:
        raise ExternalIdentityError(
            "This sign-in method is not configured on the server (missing client id)"
        )
    try:
        jwk_client = jwt.PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise ExternalIdentityError("Invalid identity token") from exc
    except Exception as exc:  # network/JWKS-fetch failures etc.
        raise ExternalIdentityError("Could not verify identity token") from exc

    if claims.get("iss") not in issuers:
        raise ExternalIdentityError("Invalid identity token issuer")
    return claims


class GoogleIdentityVerifier:
    """Production verifier - validates against Google's own rotating public
    keys over the network. Requires GOOGLE_OAUTH_CLIENT_ID (config.py) to be
    set to this app's registered OAuth client id, which must match the
    token's `aud` claim exactly - otherwise any Google user's token for a
    completely different app would be accepted here."""

    def __init__(self, client_id: Optional[str] = None):
        if client_id is None:
            from config import GOOGLE_OAUTH_CLIENT_ID

            client_id = GOOGLE_OAUTH_CLIENT_ID
        self._client_id = client_id

    def verify(self, id_token: str) -> VerifiedExternalIdentity:
        claims = _verify_oidc_id_token(
            id_token, jwks_url=GOOGLE_JWKS_URL, issuers=GOOGLE_ISSUERS, audience=self._client_id
        )
        return VerifiedExternalIdentity(
            provider="google",
            subject=claims["sub"],
            email=claims.get("email"),
            email_verified=bool(claims.get("email_verified", False)),
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
        raw_verified = claims.get("email_verified", False)
        return VerifiedExternalIdentity(
            provider="apple",
            subject=claims["sub"],
            # Apple only includes `email` on the FIRST authorization (or
            # when using their private relay) - a None here on later logins
            # is expected and must not be treated as the identity changing;
            # callers must key on `subject`, never on email, for exactly
            # this reason.
            email=claims.get("email"),
            email_verified=raw_verified in (True, "true", "1", 1),
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


def find_or_create_user_for_identity(
    session: Session, identity: VerifiedExternalIdentity
) -> tuple[User, bool]:
    """Returns (user, is_new_user). Resolution order:

    1. An external_identities row already matches (provider, subject) ->
       that user (returning user - the common case on every login after
       the first).
    2. No match, but an existing StockLens account already has this exact
       email -> raise AccountLinkingRequiredError rather than silently
       attaching this identity to it (see that class's docstring for why).
    3. No match anywhere -> create a brand-new user from this identity.
    """
    existing_link = session.exec(
        select(ExternalIdentity).where(
            ExternalIdentity.provider == identity.provider,
            ExternalIdentity.provider_subject == identity.subject,
        )
    ).first()
    if existing_link is not None:
        user = session.get(User, existing_link.user_id)
        if user is None or not user.is_active:
            raise ExternalIdentityError("This account is unavailable")
        return user, False

    if identity.email:
        normalized_email = identity.email.strip().lower()
        colliding_user = session.exec(select(User).where(User.email == normalized_email)).first()
        if colliding_user is not None:
            raise AccountLinkingRequiredError(
                "An account with this email already exists. Sign in to that account and link "
                f"{identity.provider.title()} from account settings.",
                existing_email=normalized_email,
            )

    user = create_external_user(session, email=identity.email)
    session.add(
        ExternalIdentity(
            user_id=user.id,
            provider=identity.provider,
            provider_subject=identity.subject,
            email=identity.email,
        )
    )
    session.commit()
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
    session.commit()
    session.refresh(link)
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


def list_identities(session: Session, user: User) -> list[ExternalIdentity]:
    return list(
        session.exec(select(ExternalIdentity).where(ExternalIdentity.user_id == user.id)).all()
    )
