"""OTP delivery abstraction - how a generated code actually reaches the
user. Kept separate from auth/otp_service.py (which owns challenge
generation/storage/verification) so production can plug in a real SMS/email
provider without any challenge logic - or any route - needing to change.
"""
from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Protocol

from config import PRODUCT_NAME
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


class UnsupportedDestinationError(Exception):
    """Raised when a delivery implementation can't reach this kind of
    destination at all (e.g. an SMTP-only adapter asked to deliver to a
    phone number) - distinct from "not configured", since the adapter IS
    configured, it just fundamentally can't serve this request."""


class SmtpOtpDeliveryService:
    """Production email OTP delivery over SMTP - deliberately a PROTOCOL,
    not a vendor SDK: every major transactional-email provider (SendGrid,
    Mailgun, Amazon SES, Postmark, a corporate mail relay, ...) exposes a
    standard SMTP endpoint, so this one implementation works with whichever
    provider the deployment actually has credentials for, without this
    codebase hard-coding assumptions about any single vendor's proprietary
    API. Only handles email destinations - see UnsupportedDestinationError
    for phone/SMS, which has no equivalent universal protocol and is left
    for a future SMS-specific adapter once a provider is actually chosen
    (see docs/AUTHENTICATION.md "OTP provider setup").

    Uses Python's standard library smtplib/email - no third-party
    dependency, and nothing here is specific to any one email provider's
    SDK/account model.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        from_address: str,
        use_tls: bool = True,
    ):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_address = from_address
        self._use_tls = use_tls

    def send(self, destination: str, code: str) -> None:
        if "@" not in destination:
            raise UnsupportedDestinationError(
                "SmtpOtpDeliveryService can only deliver to email addresses, not phone numbers - "
                "no SMS provider is configured (see docs/AUTHENTICATION.md 'OTP provider setup')."
            )

        message = EmailMessage()
        message["Subject"] = f"Your {PRODUCT_NAME} verification code"
        message["From"] = self._from_address
        message["To"] = destination
        message.set_content(
            f"Your {PRODUCT_NAME} verification code is: {code}\n\n"
            "This code expires shortly and can only be used once. If you didn't request this, "
            "you can safely ignore this email."
        )

        with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
            if self._use_tls:
                smtp.starttls()
            if self._username:
                smtp.login(self._username, self._password)
            smtp.send_message(message)

        logger.info("OTP_SENT | destination=%s via=smtp", destination)


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
    """Production wiring point. Picks, in order:

    1. SmtpOtpDeliveryService, if OTP_SMTP_HOST is configured (real email
       delivery - see docs/AUTHENTICATION.md "OTP provider setup" for the
       exact environment variables required).
    2. DevLoggingOtpDeliveryService, if OTP_DEV_LOG_CODES is explicitly
       enabled (local development only - never in production).
    3. NotConfiguredOtpDeliveryService otherwise (fails loudly rather than
       silently dropping codes).

    No SMS provider is wired up - see docs/AUTHENTICATION.md for why
    (no universal SMS protocol the way SMTP is for email) and what's needed
    to add one once a specific provider is chosen.
    """
    from config import OTP_DEV_LOG_CODES, OTP_SMTP_FROM_ADDRESS, OTP_SMTP_HOST, OTP_SMTP_PASSWORD, OTP_SMTP_PORT, OTP_SMTP_USE_TLS, OTP_SMTP_USERNAME

    if OTP_SMTP_HOST:
        return SmtpOtpDeliveryService(
            host=OTP_SMTP_HOST,
            port=OTP_SMTP_PORT,
            username=OTP_SMTP_USERNAME,
            password=OTP_SMTP_PASSWORD,
            from_address=OTP_SMTP_FROM_ADDRESS,
            use_tls=OTP_SMTP_USE_TLS,
        )
    if OTP_DEV_LOG_CODES:
        return DevLoggingOtpDeliveryService()
    return NotConfiguredOtpDeliveryService()
