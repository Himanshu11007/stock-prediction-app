"""
admin_console/client.py — the admin console's only path to data: the
StockLens REST API, authenticated with an ADMIN user's access token.

The console never touches the database directly, so every admin action goes
through the same authorization (require_admin) and audit logging as any
other API client. Tokens live only in memory (Streamlit session state).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_API_URL = os.environ.get("STOCKAI_API_URL", "http://127.0.0.1:8000/api/v1")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class AdminApiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

    # ── auth ──────────────────────────────────────────────────────────────
    def login(self, email: str, password: str) -> dict:
        r = requests.post(f"{self.base_url}/auth/login", data={"username": email, "password": password},
                          timeout=self.timeout)
        body = self._body(r)
        self.access_token, self.refresh_token = body["access_token"], body.get("refresh_token")
        me = self.request("GET", "/auth/me")
        if "ADMIN" not in me.get("roles", []):
            self.logout()
            raise ApiError(403, "This account does not have the ADMIN role")
        return me

    def logout(self) -> None:
        if self.refresh_token:
            try:
                requests.post(f"{self.base_url}/auth/logout", json={"refresh_token": self.refresh_token},
                              timeout=self.timeout)
            except requests.RequestException:
                pass
        self.access_token = self.refresh_token = None

    def _refresh(self) -> bool:
        if not self.refresh_token:
            return False
        r = requests.post(f"{self.base_url}/auth/refresh", json={"refresh_token": self.refresh_token},
                          timeout=self.timeout)
        if r.status_code != 200:
            return False
        body = r.json()
        self.access_token, self.refresh_token = body["access_token"], body.get("refresh_token", self.refresh_token)
        return True

    # ── requests ──────────────────────────────────────────────────────────
    def request(self, method: str, path: str, *, json: Any = None, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(2):
            headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
            try:
                r = requests.request(method, url, json=json, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                raise ApiError(0, f"Cannot reach the API at {self.base_url} ({type(e).__name__})") from e
            if r.status_code == 401 and attempt == 0 and self._refresh():
                continue
            return self._body(r)
        raise ApiError(401, "Session expired - sign in again")

    @staticmethod
    def _body(r: requests.Response) -> Any:
        if r.status_code == 204:
            return None
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 400:
            msg = None
            if isinstance(body, dict):
                msg = body.get("detail") or body.get("details") or body.get("error")
            raise ApiError(r.status_code, str(msg or r.reason))
        # Unwrap the {"success", "data", "message"} envelope where used.
        if isinstance(body, dict) and body.get("success") is True and "data" in body:
            return body["data"]
        return body

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, json: Any = None) -> Any:
        return self.request("POST", path, json=json)

    def patch(self, path: str, json: Any) -> Any:
        return self.request("PATCH", path, json=json)

    def put(self, path: str, json: Any) -> Any:
        return self.request("PUT", path, json=json)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)
