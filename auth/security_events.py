"""Structured security-event logging for authentication flows.

One line per event, in the same "EVENT | key=value ..." shape the rest of the
codebase logs in, under the dedicated "stocklens.security" logger so it can
be routed/alerted on separately.

NEVER pass a secret here: passwords, OTP codes, access/refresh/ID tokens,
password-reset tokens or URLs, Authorization headers. Emails are masked
(mask_email) and free-text values are truncated; a field whose name looks
like a secret is dropped outright as a last line of defence.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger("stocklens.security")

_SECRET_FIELD_MARKERS = ("password", "token", "secret", "code", "otp", "authorization", "credential")


def mask_email(email: Optional[str]) -> Optional[str]:
    """'jane.doe@example.com' -> 'j***@example.com' - enough to correlate a
    support request, not enough to harvest addresses from logs."""
    if not email:
        return None
    local, _, domain = email.strip().lower().partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def fingerprint(value: Optional[str]) -> Optional[str]:
    """Short, non-reversible correlation id for a value that must not be
    logged itself (e.g. which refresh token was replayed)."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def log_security_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        if any(marker in key.lower() for marker in _SECRET_FIELD_MARKERS) and not key.endswith("_fp"):
            continue  # defence in depth: never emit anything that looks like a secret
        text = str(value).replace("\n", " ").replace("|", "/")[:200]
        parts.append(f"{key}={text}")
    logger.log(level, "SECURITY_EVENT | %s | %s", event, " ".join(parts))
