"""
notifications/push.py — push delivery providers.

  FcmProvider    Android: Firebase Cloud Messaging HTTP v1. OAuth2 access
                 token from the service-account key (RS256 JWT exchanged at
                 Google's token endpoint); cached until shortly before expiry.
  ApnsProvider   iOS: Apple Push Notification service, token-based
                 authentication (ES256 JWT from the .p8 key), HTTP/2.
  Router         picks the provider by device.provider; a provider whose
                 credentials are not configured returns NOT_CONFIGURED,
                 never a fake success.

Credentials come only from the environment (config.py). Tokens and keys are
never logged. A provider response meaning "this token is no longer valid"
returns INVALID_TOKEN so the caller deactivates the device.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import config
from utils.logger import get_logger

logger = get_logger(__name__)

SENT, FAILED, INVALID_TOKEN, NOT_CONFIGURED = "SENT", "FAILED", "INVALID_TOKEN", "NOT_CONFIGURED"


@dataclass
class PushMessage:
    title: str
    body: str
    data: dict[str, str]          # route, notification_id, type, symbol (strings only)


@dataclass
class PushResult:
    status: str
    message_id: Optional[str] = None
    error: Optional[str] = None


class PushProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    def send(self, token: str, message: PushMessage) -> PushResult: ...


class FcmProvider:
    name = "fcm"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SCOPE = "https://www.googleapis.com/auth/firebase.messaging"

    def __init__(self, project_id: Optional[str] = None, service_account_file: Optional[str] = None,
                 http=None):
        self.project_id = project_id if project_id is not None else config.FCM_PROJECT_ID
        self.service_account_file = (service_account_file if service_account_file is not None
                                     else config.FCM_SERVICE_ACCOUNT_FILE)
        self._http = http
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at = 0.0

    def configured(self) -> bool:
        return bool(self.project_id and self.service_account_file)

    def _client(self):
        if self._http is None:
            import requests
            self._http = requests.Session()
        return self._http

    def _token(self) -> str:
        with self._lock:
            if self._access_token and time.time() < self._expires_at - 120:
                return self._access_token
            import jwt
            with open(self.service_account_file, encoding="utf-8") as f:
                sa = json.load(f)
            now = int(time.time())
            assertion = jwt.encode({"iss": sa["client_email"], "scope": self.SCOPE, "aud": self.TOKEN_URL,
                                    "iat": now, "exp": now + 3600}, sa["private_key"], algorithm="RS256")
            r = self._client().post(self.TOKEN_URL, data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion}, timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"FCM OAuth token request failed with HTTP {r.status_code}")
            body = r.json()
            self._access_token, self._expires_at = body["access_token"], now + int(body.get("expires_in", 3600))
            return self._access_token

    def send(self, token: str, message: PushMessage) -> PushResult:
        if not self.configured():
            return PushResult(NOT_CONFIGURED, error="FCM credentials are not configured")
        try:
            r = self._client().post(
                f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send",
                headers={"Authorization": f"Bearer {self._token()}"},
                json={"message": {"token": token,
                                  "notification": {"title": message.title, "body": message.body},
                                  "data": message.data,
                                  "android": {"priority": "high",
                                              "notification": {"channel_id": "stockai_alerts",
                                                               "icon": "ic_stat_stocklens",
                                                               "color": "#3B82F6"}}}},
                timeout=15)
        except Exception as e:
            return PushResult(FAILED, error=f"{type(e).__name__}"[:200])
        if r.status_code == 200:
            return PushResult(SENT, message_id=(r.json() or {}).get("name"))
        detail = ""
        try:
            err = r.json().get("error", {})
            detail = err.get("status", "") + " " + " ".join(
                d.get("errorCode", "") for d in err.get("details", []) if isinstance(d, dict))
        except Exception:
            pass
        if r.status_code == 404 or "UNREGISTERED" in detail or \
                (r.status_code == 400 and "INVALID_ARGUMENT" in detail):
            return PushResult(INVALID_TOKEN, error=f"HTTP {r.status_code} {detail.strip()}"[:200])
        return PushResult(FAILED, error=f"HTTP {r.status_code} {detail.strip()}"[:200])


class ApnsProvider:
    name = "apns"

    def __init__(self, key_file: Optional[str] = None, key_id: Optional[str] = None, team_id: Optional[str] = None,
                 bundle_id: Optional[str] = None, sandbox: Optional[bool] = None, http=None):
        self.key_file = key_file if key_file is not None else config.APNS_KEY_FILE
        self.key_id = key_id if key_id is not None else config.APNS_KEY_ID
        self.team_id = team_id if team_id is not None else config.APNS_TEAM_ID
        self.bundle_id = bundle_id if bundle_id is not None else config.APNS_BUNDLE_ID
        self.sandbox = config.APNS_USE_SANDBOX if sandbox is None else sandbox
        self._http = http
        self._lock = threading.Lock()
        self._jwt: Optional[str] = None
        self._jwt_at = 0.0

    def configured(self) -> bool:
        return bool(self.key_file and self.key_id and self.team_id and self.bundle_id)

    def _client(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(http2=True, timeout=15)
        return self._http

    def _auth(self) -> str:
        with self._lock:
            # Apple accepts a provider token for up to 60 minutes; refresh at 50.
            if self._jwt and time.time() - self._jwt_at < 50 * 60:
                return self._jwt
            import jwt
            with open(self.key_file, encoding="utf-8") as f:
                key = f.read()
            self._jwt_at = time.time()
            self._jwt = jwt.encode({"iss": self.team_id, "iat": int(self._jwt_at)}, key, algorithm="ES256",
                                   headers={"kid": self.key_id})
            return self._jwt

    def send(self, token: str, message: PushMessage) -> PushResult:
        if not self.configured():
            return PushResult(NOT_CONFIGURED, error="APNs credentials are not configured")
        host = "api.sandbox.push.apple.com" if self.sandbox else "api.push.apple.com"
        payload: dict[str, Any] = {"aps": {"alert": {"title": message.title, "body": message.body},
                                           "sound": "default"}, **message.data}
        try:
            r = self._client().post(f"https://{host}/3/device/{token}", json=payload, headers={
                "authorization": f"bearer {self._auth()}", "apns-topic": self.bundle_id,
                "apns-push-type": "alert", "apns-priority": "10"})
        except Exception as e:
            return PushResult(FAILED, error=f"{type(e).__name__}"[:200])
        if r.status_code == 200:
            return PushResult(SENT, message_id=r.headers.get("apns-id"))
        reason = ""
        try:
            reason = r.json().get("reason", "")
        except Exception:
            pass
        if r.status_code == 410 or reason in ("BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic"):
            return PushResult(INVALID_TOKEN, error=f"HTTP {r.status_code} {reason}"[:200])
        return PushResult(FAILED, error=f"HTTP {r.status_code} {reason}"[:200])


class PushRouter:
    def __init__(self, providers: Optional[dict[str, PushProvider]] = None):
        self.providers = providers if providers is not None else {"fcm": FcmProvider(), "apns": ApnsProvider()}

    def status(self) -> dict[str, bool]:
        return {name: p.configured() for name, p in self.providers.items()}

    def send(self, provider: str, token: str, message: PushMessage) -> PushResult:
        p = self.providers.get(provider)
        if p is None:
            return PushResult(NOT_CONFIGURED, error=f"unknown push provider {provider!r}")
        try:
            return p.send(token, message)
        except Exception as e:  # a provider bug must never break the notification run
            logger.warning("PUSH_PROVIDER_ERROR | %s | %s", provider, type(e).__name__)
            return PushResult(FAILED, error=type(e).__name__)


_router: Optional[PushRouter] = None


def get_router() -> PushRouter:
    global _router
    if _router is None:
        _router = PushRouter()
    return _router


def set_router(router: Optional[PushRouter]) -> None:
    """Tests and the E2E harness install a fake router here."""
    global _router
    _router = router
