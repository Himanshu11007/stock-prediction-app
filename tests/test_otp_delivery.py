"""tests/test_otp_delivery.py — IOtpDeliveryService implementations,
particularly the real SmtpOtpDeliveryService production adapter. The SMTP
connection itself is faked (no real network/mail server involved) so these
stay fast and deterministic; what's under test is this codebase's own
logic (message construction, destination validation, and - critically -
that the raw code is never logged), not a real mail provider.
"""
from unittest.mock import MagicMock, patch

import pytest

from auth.otp_delivery import (
    DevLoggingOtpDeliveryService,
    FakeOtpDeliveryService,
    NotConfiguredOtpDeliveryService,
    SmtpOtpDeliveryService,
    UnsupportedDestinationError,
    get_otp_delivery_service,
)


def make_smtp_service(**overrides):
    defaults = dict(
        host="smtp.example.com", port=587, username="apikey", password="secret",
        from_address="no-reply@stockaipro.app", use_tls=True,
    )
    defaults.update(overrides)
    return SmtpOtpDeliveryService(**defaults)


@patch("smtplib.SMTP")
def test_smtp_service_sends_to_an_email_destination(mock_smtp_class):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp
    service = make_smtp_service()

    service.send("user@example.com", "123456")

    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("apikey", "secret")
    mock_smtp.send_message.assert_called_once()


@patch("smtplib.SMTP")
def test_smtp_service_puts_the_code_in_the_message_body(mock_smtp_class):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp
    service = make_smtp_service()

    service.send("user@example.com", "654321")

    sent_message = mock_smtp.send_message.call_args[0][0]
    assert "654321" in sent_message.get_content()
    assert sent_message["To"] == "user@example.com"


@patch("smtplib.SMTP")
def test_smtp_service_without_tls_does_not_call_starttls(mock_smtp_class):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp
    service = make_smtp_service(use_tls=False)

    service.send("user@example.com", "123456")

    mock_smtp.starttls.assert_not_called()


@patch("smtplib.SMTP")
def test_smtp_service_without_username_skips_login(mock_smtp_class):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp
    service = make_smtp_service(username="", password="")

    service.send("user@example.com", "123456")

    mock_smtp.login.assert_not_called()


def test_smtp_service_rejects_a_phone_destination():
    service = make_smtp_service()
    with pytest.raises(UnsupportedDestinationError):
        service.send("+919876543210", "123456")


@patch("smtplib.SMTP")
def test_smtp_service_never_logs_the_raw_code(mock_smtp_class, caplog):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp
    service = make_smtp_service()

    with caplog.at_level("DEBUG"):
        service.send("user@example.com", "999999")

    assert "999999" not in caplog.text


def test_not_configured_service_raises_instead_of_silently_succeeding():
    with pytest.raises(RuntimeError):
        NotConfiguredOtpDeliveryService().send("user@example.com", "123456")


def test_dev_logging_service_logs_the_code_only_when_explicitly_used(caplog):
    with caplog.at_level("INFO"):
        DevLoggingOtpDeliveryService().send("user@example.com", "123456")
    assert "123456" in caplog.text  # by design - this class exists only for local dev


def test_fake_service_records_without_any_io():
    fake = FakeOtpDeliveryService()
    fake.send("user@example.com", "123456")
    assert fake.last_code == "123456"
    assert fake.sent == [("user@example.com", "123456")]


def test_factory_prefers_smtp_when_configured(monkeypatch):
    import config

    monkeypatch.setattr(config, "OTP_SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(config, "OTP_DEV_LOG_CODES", True)  # SMTP must still win over dev logging

    service = get_otp_delivery_service()

    assert isinstance(service, SmtpOtpDeliveryService)


def test_factory_falls_back_to_dev_logging_when_enabled_and_no_smtp(monkeypatch):
    import config

    monkeypatch.setattr(config, "OTP_SMTP_HOST", None)
    monkeypatch.setattr(config, "OTP_DEV_LOG_CODES", True)

    service = get_otp_delivery_service()

    assert isinstance(service, DevLoggingOtpDeliveryService)


def test_factory_fails_closed_with_no_provider_and_dev_logging_off(monkeypatch):
    import config

    monkeypatch.setattr(config, "OTP_SMTP_HOST", None)
    monkeypatch.setattr(config, "OTP_DEV_LOG_CODES", False)

    service = get_otp_delivery_service()

    assert isinstance(service, NotConfiguredOtpDeliveryService)
