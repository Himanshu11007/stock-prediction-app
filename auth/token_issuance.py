"""Shared helper for turning an authenticated User into an issued token
pair. Every Phase 8 login path (password, Google, Apple, OTP) funnels
through this single function so they all produce tokens exactly the same
way and all tag the resulting session with the same device-association
logic - there is deliberately no separate "SSO token issuance" or "OTP
token issuance" path to drift out of sync with the original one.
"""
from __future__ import annotations

from typing import Optional

from sqlmodel import Session

import auth.service as auth_service
from auth.device_service import upsert_trusted_device
from auth.security import create_access_token
from db.models.user import User


def issue_token_pair(
    session: Session,
    user: User,
    *,
    device_id: Optional[str] = None,
    device_name: Optional[str] = None,
) -> tuple[str, str]:
    """Returns (access_token, refresh_token). If device_id is provided,
    also records/refreshes the trusted-device row for this (user, device) -
    see auth/device_service.py:upsert_trusted_device.

    The JWT subject is the user's numeric id (not email): some Phase 8
    accounts (phone-only OTP, Apple private-relay-only) have no email at
    all, so id is the only identifier guaranteed to exist. See
    auth/dependencies.py:get_current_user for the backward-compatible
    id-or-email lookup this pairs with.
    """
    if device_id:
        upsert_trusted_device(session, user, device_id, device_name)

    roles = auth_service.get_user_roles(session, user)
    access_token = create_access_token(subject=str(user.id), roles=roles)
    refresh_token = auth_service.issue_refresh_token(session, user, device_id=device_id)
    return access_token, refresh_token
