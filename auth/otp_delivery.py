"""OTP delivery abstraction - how a generated code actually reaches the
user. Kept separate from auth/otp_service.py (which owns challenge
generation/storage/verification) so production can plug in a real SMS/email
provider without any challenge logic - or any route - needing to change.
"""
from __future__ import annotations

from typing import Protocol

from utils.logger import get_logger

logger = get_logger(__name__)


class IOtpDeliveryService(Protocol):
    def send(self, destination: str, code: str) -> None: ...


class FakeOtpDeliveryService:
    """Deterministic test double - records every (destination, code) pair
    instead of sending anything over the network. Use this in all automated
    tests; it must never be wired up outside of tests (see
    get_otp_delivery_service() below)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, destination: str, code: str) -> None:
        self.sent.append((destination, code))

    @property
    def last_code(self) -> str:
        return self.sent[-1][1]


class DevLoggingOtpDeliveryService:
    """Non-production fallback, only used when OTP_DEV_LOG_CODES is
    explicitly enabled (see config.py) and no real provider is configured -
    logs the code to the server's own log file so a developer can exercise
    the full OTP flow without a real SMS/email account.

    NEVER enable OTP_DEV_LOG_CODES in production: the whole point of
    hashing the stored code is that it isn't recoverable by anyone with
    read access to the server (including its logs) - this class exists
    specifically to defeat that for local development only.
    """

    def send(self, destination: str, code: str) -> None:
        logger.info(
            "DEV_OTP | destination=%s code=%s (dev-only - see OTP_DEV_LOG_CODES in config.py)",
            destination, code,
        )


class NotConfiguredOtpDeliveryService:
    """Default when no real provider is configured and dev logging is off -
    fails loudly instead of silently pretending to send an SMS/email that
    never arrives, which would otherwise look like a successful "code
    sent" response to the user forever."""

    def send(self, destination: str, code: str) -> None:
        raise RuntimeError(
            "No OTP delivery provider is configured. Set OTP_DEV_LOG_CODES=true for local "
            "development only, or wire a real SMS/email provider - see "
            "docs/AUTHENTICATION.md 'OTP provider setup'."
        )


def get_otp_delivery_service() -> IOtpDeliveryService:
    """Production wiring point: once a real SMS/email account is available,
    branch here on its configuration (e.g. an env var naming the provider)
    and return a real implementation. No real provider is wired up yet -
    see docs/AUTHENTICATION.md."""
    from config import OTP_DEV_LOG_CODES

    if OTP_DEV_LOG_CODES:
        return DevLoggingOtpDeliveryService()
    return NotConfiguredOtpDeliveryService()
