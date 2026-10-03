"""api/routes/auth.py — registration, login, token refresh, logout, profile.

Authentication is entirely backend-controlled: passwords are verified and
JWTs issued/validated here against the DB, never trusted from the client.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from typing import Optional

import auth.service as auth_service
from api.schemas_auth import (
    ChangePasswordRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)
from auth.dependencies import get_current_user
from auth.security import create_access_token
from auth.token_issuance import issue_token_pair
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/auth")


def _issue_token_pair(
    session: Session, user: User, device_id: Optional[str] = None, device_name: Optional[str] = None
) -> TokenResponse:
    access_token, refresh_token = issue_token_pair(
        session, user, device_id=device_id, device_name=device_name
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest, session: Session = Depends(get_session)):
    auth_service.ensure_roles_exist(session)
    try:
        user = auth_service.create_user(session, payload.email, payload.password)
    except auth_service.DuplicateEmailError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return _issue_token_pair(session, user, payload.device_id, payload.device_name)


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    device_id: Optional[str] = Form(default=None),
    device_name: Optional[str] = Form(default=None),
    session: Session = Depends(get_session),
):
    """OAuth2 password flow: form fields are `username` (=email) and `password`,
    plus optional `device_id`/`device_name` for device/session management."""
    try:
        user = auth_service.authenticate_user(session, form_data.username, form_data.password)
    except (auth_service.InvalidCredentialsError, auth_service.InactiveUserError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _issue_token_pair(session, user, device_id, device_name)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, session: Session = Depends(get_session)):
    try:
        user, new_refresh_token = auth_service.rotate_refresh_token(session, payload.refresh_token)
    except auth_service.InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    roles = auth_service.get_user_roles(session, user)
    access_token = create_access_token(subject=str(user.id), roles=roles)
    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: LogoutRequest, session: Session = Depends(get_session)):
    row = auth_service.find_refresh_token(session, payload.refresh_token)
    auth_service.revoke_refresh_token(session, payload.refresh_token)
    # A signed-out device must stop receiving this user's push notifications.
    if row is not None and row.device_id:
        from notifications.service import unregister_device
        unregister_device(session, row.user_id, row.device_id)


@router.get("/me", response_model=UserProfileResponse)
def get_profile(
    current_user: User = Depends(get_current_user), session: Session = Depends(get_session)
):
    roles = auth_service.get_user_roles(session, current_user)
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        phone=current_user.phone,
        is_active=current_user.is_active,
        roles=roles,
        created_at=current_user.created_at,
    )


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        auth_service.change_password(
            session, current_user, payload.current_password, payload.new_password
        )
    except auth_service.InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )
