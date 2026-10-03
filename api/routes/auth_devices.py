"""api/routes/auth_devices.py — device/session listing and revocation,
including "sign out all devices".

Revoking a refresh session here only ever affects refresh capability: an
already-issued short-lived access token for that session keeps working
until its own expiry (see auth/security.py:ACCESS_TOKEN_EXPIRE_MINUTES).
Adding server-side access-token revocation (a blacklist) is explicitly out
of scope - access tokens already expire in minutes, and every authenticated
mobile request that gets a stale 401 naturally tries to refresh, which is
exactly where revocation takes effect.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from api.schemas_auth import (
    RevokeAllSessionsRequest,
    RevokeAllSessionsResponse,
    RevokeSessionRequest,
    SessionResponse,
    SetPinEnabledRequest,
)
from auth.dependencies import get_current_user
from auth.device_service import (
    list_sessions,
    revoke_all_sessions,
    revoke_session,
    set_pin_enabled,
)
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/auth")


@router.get("/sessions", response_model=list[SessionResponse])
def get_sessions(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    return [SessionResponse(**s) for s in list_sessions(session, current_user)]


@router.post("/sessions/revoke", status_code=status.HTTP_204_NO_CONTENT)
def revoke_one_session(
    payload: RevokeSessionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    from db.models.user import RefreshToken
    from notifications.service import unregister_device
    row = session.get(RefreshToken, payload.session_id)
    revoke_session(session, current_user, payload.session_id)
    if row is not None and row.user_id == current_user.id and row.device_id:
        unregister_device(session, current_user.id, row.device_id)


@router.post("/sessions/revoke-all", response_model=RevokeAllSessionsResponse)
def revoke_all(
    payload: RevokeAllSessionsRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Sign out all devices (except_current=False, the default) or sign out
    every OTHER device while keeping the caller's own session alive
    (except_current=True + current_device_id)."""
    except_device_id = payload.current_device_id if payload.except_current else None
    revoked_count = revoke_all_sessions(session, current_user, except_device_id=except_device_id)
    # Signed-out devices stop receiving push notifications.
    from notifications.service import list_devices, unregister_device
    for d in list_devices(session, current_user):
        if d.active and d.device_id != except_device_id:
            unregister_device(session, current_user.id, d.device_id)
    return RevokeAllSessionsResponse(revoked_count=revoked_count)


@router.post("/devices/pin-enabled", status_code=status.HTTP_204_NO_CONTENT)
def update_pin_enabled(
    payload: SetPinEnabledRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Purely informational flag the mobile client flips after setting up
    (or clearing) its own local PIN - the backend never sees the PIN."""
    try:
        set_pin_enabled(session, current_user, payload.device_id, payload.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
