"""api/routes/auth_sso.py — Google / Apple sign-in and account linking.

Shares the /auth prefix with api/routes/auth.py but lives in its own file
(like every other route module in this package) so the SSO-specific
request/response shapes and error mapping don't bloat the core
register/login/refresh/logout/me file.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

import auth.service as auth_service
from api.schemas_auth import (
    AppleAuthRequest,
    GoogleAuthRequest,
    LinkAppleRequest,
    LinkedIdentityResponse,
    LinkGoogleRequest,
    TokenResponse,
)
from auth.dependencies import get_current_user
from auth.external_identity import (
    AccountLinkingRequiredError,
    AppleIdentityVerifier,
    DuplicateExternalIdentityError,
    ExternalIdentityError,
    GoogleIdentityVerifier,
    IAppleIdentityVerifier,
    IGoogleIdentityVerifier,
    LastAuthMethodError,
    find_or_create_user_for_identity,
    link_identity,
    list_identities,
    unlink_identity,
)
from auth.token_issuance import issue_token_pair
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/auth")


# Dependency indirection (not just module-level singletons) so tests can
# swap in FakeGoogleIdentityVerifier/FakeAppleIdentityVerifier via FastAPI's
# dependency_overrides without any network call ever happening in a test.
def get_google_identity_verifier() -> IGoogleIdentityVerifier:
    return GoogleIdentityVerifier()


def get_apple_identity_verifier() -> IAppleIdentityVerifier:
    return AppleIdentityVerifier()


@router.post("/google", response_model=TokenResponse)
def login_with_google(
    payload: GoogleAuthRequest,
    verifier: IGoogleIdentityVerifier = Depends(get_google_identity_verifier),
    session: Session = Depends(get_session),
):
    auth_service.ensure_roles_exist(session)
    try:
        identity = verifier.verify(payload.id_token)
        user, _is_new = find_or_create_user_for_identity(session, identity)
    except ExternalIdentityError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google identity token")
    except AccountLinkingRequiredError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="This account has been deactivated")

    access_token, refresh_token = issue_token_pair(
        session, user, device_id=payload.device_id, device_name=payload.device_name
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/apple", response_model=TokenResponse)
def login_with_apple(
    payload: AppleAuthRequest,
    verifier: IAppleIdentityVerifier = Depends(get_apple_identity_verifier),
    session: Session = Depends(get_session),
):
    auth_service.ensure_roles_exist(session)
    try:
        identity = verifier.verify(payload.identity_token)
        user, _is_new = find_or_create_user_for_identity(session, identity)
    except ExternalIdentityError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Apple identity token")
    except AccountLinkingRequiredError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="This account has been deactivated")

    access_token, refresh_token = issue_token_pair(
        session, user, device_id=payload.device_id, device_name=payload.device_name
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/link/google", response_model=LinkedIdentityResponse)
def link_google(
    payload: LinkGoogleRequest,
    verifier: IGoogleIdentityVerifier = Depends(get_google_identity_verifier),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        identity = verifier.verify(payload.id_token)
        link = link_identity(session, current_user, identity)
    except ExternalIdentityError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google identity token")
    except DuplicateExternalIdentityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return LinkedIdentityResponse(provider=link.provider, email=link.email, created_at=link.created_at)


@router.post("/link/apple", response_model=LinkedIdentityResponse)
def link_apple(
    payload: LinkAppleRequest,
    verifier: IAppleIdentityVerifier = Depends(get_apple_identity_verifier),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        identity = verifier.verify(payload.identity_token)
        link = link_identity(session, current_user, identity)
    except ExternalIdentityError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Apple identity token")
    except DuplicateExternalIdentityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return LinkedIdentityResponse(provider=link.provider, email=link.email, created_at=link.created_at)


@router.get("/identities", response_model=list[LinkedIdentityResponse])
def get_linked_identities(
    current_user: User = Depends(get_current_user), session: Session = Depends(get_session)
):
    return [
        LinkedIdentityResponse(provider=i.provider, email=i.email, created_at=i.created_at)
        for i in list_identities(session, current_user)
    ]


@router.delete("/identities/{provider}", status_code=status.HTTP_204_NO_CONTENT)
def unlink(
    provider: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        unlink_identity(session, current_user, provider)
    except LastAuthMethodError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
