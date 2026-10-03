"""api/routes/auth_otp.py — OTP (one-time passcode) login by email or phone.

Two-step flow: POST /auth/otp/request generates+delivers a code (always a
generic response, regardless of whether the destination has an account, so
this can't be used to enumerate registered users); POST /auth/otp/verify
checks it and issues a token pair, creating a new StockLens account on first
use of a destination exactly like Google/Apple do.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

import auth.otp_service as otp_service
import auth.service as auth_service
from api.schemas_auth import OtpRequestRequest, OtpVerifyRequest, TokenResponse
from auth.otp_delivery import IOtpDeliveryService, get_otp_delivery_service
from auth.service import create_external_user
from auth.token_issuance import issue_token_pair
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/auth/otp")


def _is_email(destination: str) -> bool:
    return "@" in destination


def _find_user_by_destination(session: Session, destination: str) -> User | None:
    destination = otp_service.normalize_destination(destination)
    if _is_email(destination):
        return session.exec(select(User).where(User.email == destination)).first()
    return session.exec(select(User).where(User.phone == destination)).first()


@router.post("/request", status_code=status.HTTP_204_NO_CONTENT)
def request_otp(
    payload: OtpRequestRequest,
    request: Request,
    delivery: IOtpDeliveryService = Depends(get_otp_delivery_service),
    session: Session = Depends(get_session),
):
    try:
        otp_service.request_otp(
            session,
            delivery,
            payload.destination,
            requested_ip=request.client.host if request.client else None,
        )
    except otp_service.OtpResendCooldownError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    except otp_service.OtpRateLimitedError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))


@router.post("/verify", response_model=TokenResponse)
def verify_otp(payload: OtpVerifyRequest, session: Session = Depends(get_session)):
    try:
        otp_service.verify_otp(session, payload.destination, payload.code)
    except (
        otp_service.OtpInvalidError,
        otp_service.OtpExpiredError,
        otp_service.OtpAlreadyUsedError,
        otp_service.OtpMaxAttemptsError,
    ) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    destination = otp_service.normalize_destination(payload.destination)
    user = _find_user_by_destination(session, destination)
    if user is None:
        auth_service.ensure_roles_exist(session)
        if _is_email(destination):
            user = create_external_user(session, email=destination)
        else:
            user = create_external_user(session, phone=destination)
    elif not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="This account has been deactivated")

    access_token, refresh_token = issue_token_pair(
        session, user, device_id=payload.device_id, device_name=payload.device_name
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)
