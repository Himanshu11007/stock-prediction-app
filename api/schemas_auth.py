"""Pydantic request/response models for /api/v1/auth/*."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[1-9]\d{7,14}$")  # loose E.164-ish: 8-15 digits


def _validate_otp_destination(destination: str) -> str:
    destination = destination.strip()
    if _EMAIL_RE.match(destination):
        return destination.lower()
    if _PHONE_RE.match(destination):
        return destination
    raise ValueError("Enter a valid email address or phone number")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    device_id: Optional[str] = Field(default=None, max_length=128)
    device_name: Optional[str] = Field(default=None, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    id: int
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    roles: list[str]
    created_at: datetime


# ── Google / Apple SSO ───────────────────────────────────────────────────────

class GoogleAuthRequest(BaseModel):
    id_token: str
    device_id: Optional[str] = Field(default=None, max_length=128)
    device_name: Optional[str] = Field(default=None, max_length=128)


class AppleAuthRequest(BaseModel):
    identity_token: str
    device_id: Optional[str] = Field(default=None, max_length=128)
    device_name: Optional[str] = Field(default=None, max_length=128)


class LinkGoogleRequest(BaseModel):
    id_token: str


class LinkAppleRequest(BaseModel):
    identity_token: str


class LinkedIdentityResponse(BaseModel):
    provider: str
    email: Optional[str] = None
    created_at: datetime


# ── OTP ───────────────────────────────────────────────────────────────────────

class OtpRequestRequest(BaseModel):
    destination: str = Field(..., min_length=3, max_length=320)

    @field_validator("destination")
    @classmethod
    def _valid_destination(cls, v: str) -> str:
        return _validate_otp_destination(v)


class OtpVerifyRequest(BaseModel):
    destination: str = Field(..., min_length=3, max_length=320)
    code: str = Field(..., min_length=4, max_length=8)
    device_id: Optional[str] = Field(default=None, max_length=128)
    device_name: Optional[str] = Field(default=None, max_length=128)

    @field_validator("destination")
    @classmethod
    def _valid_destination(cls, v: str) -> str:
        return _validate_otp_destination(v)


# ── Device / session management ─────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: int
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    created_at: datetime
    expires_at: datetime


class RevokeSessionRequest(BaseModel):
    session_id: int


class RevokeAllSessionsRequest(BaseModel):
    except_current: bool = False
    current_device_id: Optional[str] = Field(default=None, max_length=128)


class RevokeAllSessionsResponse(BaseModel):
    revoked_count: int


class SetPinEnabledRequest(BaseModel):
    device_id: str = Field(..., max_length=128)
    enabled: bool
