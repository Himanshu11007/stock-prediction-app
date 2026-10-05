"""Password-reset email delivery - how a reset link reaches the user. Same
shape as auth/otp_delivery.py (Protocol + fake + SMTP + not-configured) and
the SAME SMTP provider/settings (OTP_SMTP_*): no second email integration.

The reset URL contains a live credential, so it is never logged anywhere -
not even by the dev/not-configured fallback (there is deliberately no
"log the link instead" mode; run a local SMTP catcher such as MailHog or
`python -m aiosmtpd -n -l localhost:1025` to see emails in development).
"""
from __future__ import annotations

from email.message import EmailMessage
from typing import Protocol
from urllib.parse import quote

from auth.otp_delivery import send_smtp_message
from config import PRODUCT_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


def build_reset_url(base_url: str, raw_token: str) -> str:
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}token={quote(raw_token, safe='')}"


def build_reset_email(to_address: str, from_address: str, reset_url: str, expires_minutes: int) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = f"Reset your {PRODUCT_NAME} password"
    message["From"] = from_address
    message["To"] = to_address
    message.set_content(
        f"We received a request to reset the password for your {PRODUCT_NAME} account.\n\n"
        f"To choose a new password, open this link:\n\n{reset_url}\n\n"
        f"The link expires in {expires_minutes} minutes and can be used only once. "
        "Resetting your password signs you out on all devices.\n\n"
        "If you did not request this, ignore this email - your password will not change. "
        f"{PRODUCT_NAME} will never ask you for your password or this link by phone, chat or email; "
        "do not forward it to anyone.\n"
    )
    return message


class IPasswordResetDeliveryService(Protocol):
    def send(self, email: str, reset_url: str, expires_minutes: int) -> None: ...


class FakePasswordResetDeliveryService:
    """Test double - records (email, reset_url) instead of sending."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, email: str, reset_url: str, expires_minutes: int) -> None:
        self.sent.append((email, reset_url))

    @property
    def last_token(self) -> str:
        from urllib.parse import parse_qs, urlparse

        return parse_qs(urlparse(self.sent[-1][1]).query)["token"][0]


class SmtpPasswordResetDeliveryService:
    def __init__(self, *, host: str, port: int, username: str, password: str, from_address: str,
                 use_tls: bool = True):
        self._settings = dict(host=host, port=port, username=username, password=password, use_tls=use_tls)
        self._from_address = from_address

    def send(self, email: str, reset_url: str, expires_minutes: int) -> None:
        send_smtp_message(build_reset_email(email, self._from_address, reset_url, expires_minutes),
                          **self._settings)


class NotConfiguredPasswordResetDeliveryService:
    """No SMTP provider configured: the email cannot be sent. Logs a
    warning (without the link) instead of raising - the API response is
    generic either way, and this runs after the response was sent."""

    def send(self, email: str, reset_url: str, expires_minutes: int) -> None:
        logger.warning("PASSWORD_RESET_EMAIL_NOT_SENT | no email provider configured (set OTP_SMTP_HOST)")


def get_password_reset_delivery_service() -> IPasswordResetDeliveryService:
    from config import (OTP_SMTP_FROM_ADDRESS, OTP_SMTP_HOST, OTP_SMTP_PASSWORD, OTP_SMTP_PORT,
                        OTP_SMTP_USE_TLS, OTP_SMTP_USERNAME)

    if OTP_SMTP_HOST:
        return SmtpPasswordResetDeliveryService(
            host=OTP_SMTP_HOST, port=OTP_SMTP_PORT, username=OTP_SMTP_USERNAME,
            password=OTP_SMTP_PASSWORD, from_address=OTP_SMTP_FROM_ADDRESS, use_tls=OTP_SMTP_USE_TLS,
        )
    return NotConfiguredPasswordResetDeliveryService()
