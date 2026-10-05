"""api/routes/auth_password.py — forgot-password / reset-password.

POST /auth/forgot-password always answers 202 with the same generic message
(account or not, rate-limited per account or not); the email itself is sent
in the background after the response. Only the per-IP limit answers 429.
POST /auth/reset-password consumes the emailed token exactly once, sets the
new password and ends every existing session (see auth/password_reset.py).
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlmodel import Session

import auth.password_reset as password_reset
from api.schemas_auth import ForgotPasswordRequest, MessageResponse, ResetPasswordRequest
from auth.password_reset_delivery import (
    IPasswordResetDeliveryService,
    build_reset_url,
    get_password_reset_delivery_service,
)
from auth.security_events import log_security_event
from auth.throttle import ThrottledError
from db.session import get_session
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth")


def get_password_reset_url() -> str:
    """Dependency so tests can supply a URL without touching config."""
    from config import PASSWORD_RESET_URL

    return PASSWORD_RESET_URL


def _client_ip(request: Request):
    return request.client.host if request.client else None


def _too_many(exc: ThrottledError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc),
                         headers={"Retry-After": str(exc.retry_after_seconds)})


def _deliver(delivery: IPasswordResetDeliveryService, email: str, reset_url: str, expires_minutes: int) -> None:
    try:
        delivery.send(email, reset_url, expires_minutes)
    except Exception as exc:  # never log the URL / exception text (may echo it)
        logger.error("PASSWORD_RESET_EMAIL_FAILED | %s", type(exc).__name__)


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED, response_model=MessageResponse)
def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    background: BackgroundTasks,
    delivery: IPasswordResetDeliveryService = Depends(get_password_reset_delivery_service),
    reset_url: str = Depends(get_password_reset_url),
    session: Session = Depends(get_session),
):
    if not reset_url:
        logger.warning("PASSWORD_RESET_URL / FRONTEND_BASE_URL not configured - reset links cannot be sent")
    try:
        issued = password_reset.request_password_reset(
            session, payload.email, requested_ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"), reset_url_configured=bool(reset_url),
        )
    except ThrottledError as exc:
        raise _too_many(exc)
    if issued is not None:
        email, raw_token = issued
        from config import PASSWORD_RESET_TOKEN_EXPIRY_MINUTES

        background.add_task(_deliver, delivery, email, build_reset_url(reset_url, raw_token),
                            PASSWORD_RESET_TOKEN_EXPIRY_MINUTES)
    return MessageResponse(message=password_reset.GENERIC_FORGOT_PASSWORD_MESSAGE)


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, request: Request, session: Session = Depends(get_session)):
    try:
        user = password_reset.reset_password(
            session, payload.token, payload.new_password, requested_ip=_client_ip(request))
    except ThrottledError as exc:
        raise _too_many(exc)
    except password_reset.WeakPasswordError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except password_reset.InvalidResetTokenError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Signed-out devices stop receiving this user's push notifications, the
    # same as "sign out all devices" (api/routes/auth_devices.py).
    try:
        from notifications.service import list_devices, unregister_device

        for d in list_devices(session, user):
            if d.active:
                unregister_device(session, user.id, d.device_id)
    except Exception as exc:
        log_security_event("PASSWORD_RESET_PUSH_CLEANUP_FAILED", user_id=user.id, error=type(exc).__name__)
    return MessageResponse(message="Your password has been reset. Sign in with your new password.")
