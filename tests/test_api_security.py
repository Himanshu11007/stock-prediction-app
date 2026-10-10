"""
tests/test_api_security.py — API documentation routes and security headers.

Production mode is tested in a subprocess (APP_ENV=production is read at
import time and enforces production settings), with throwaway test values
only; no real secret is used or printed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")
HEADERS = {
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
}

PROBE = r"""
import json, sys
from fastapi.testclient import TestClient
from api.main import app
c = TestClient(app)
out = {p: c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")}
h = c.get("/api/v1/health")
out["health"] = h.status_code
out["headers"] = {k: h.headers.get(k) for k in %r}
out["admin_anonymous"] = c.get("/api/v1/admin/users").status_code
out["cors_foreign"] = c.get("/api/v1/health", headers={"Origin": "https://evil.example"}).headers.get("access-control-allow-origin")
print("RESULT" + json.dumps(out))
""" % (list(HEADERS),)


def _run_production(tmp_path, **extra):
    env = {**os.environ, "APP_ENV": "production", "JWT_SECRET_KEY": "test-only-not-a-secret-" + "x" * 32,
           "CORS_ALLOWED_ORIGINS": "https://app.example", "DATABASE_URL": f"sqlite:///{(tmp_path / 'p.db').as_posix()}",
           **extra}
    env.pop("API_DOCS_ENABLED", None) if "API_DOCS_ENABLED" not in extra else None
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    assert line, r.stdout[-1500:] + r.stderr[-3000:]
    return json.loads(line[len("RESULT"):])


@pytest.fixture(scope="module")
def production(tmp_path_factory):
    return _run_production(tmp_path_factory.mktemp("prod"))


def test_docs_and_schema_are_not_served_in_production(production):
    for p in DOC_PATHS:
        assert production[p] == 404, p


def test_production_health_and_headers(production):
    assert production["health"] == 200
    assert production["headers"] == HEADERS


def test_production_admin_requires_auth_and_cors_is_not_open(production):
    assert production["admin_anonymous"] == 401
    assert production["cors_foreign"] is None


def test_docs_can_be_enabled_explicitly_in_production(tmp_path):
    out = _run_production(tmp_path, API_DOCS_ENABLED="true")
    for p in DOC_PATHS:
        assert out[p] == 200, p


def test_development_serves_docs_with_security_headers():
    from api.main import app
    c = TestClient(app)
    for p in DOC_PATHS:
        assert c.get(p).status_code == 200, p
    r = c.get("/api/v1/health")
    for k, v in HEADERS.items():
        assert r.headers.get(k) == v, k


def test_security_headers_on_errors_and_json_bodies_unchanged():
    from api.main import app
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/v1/top-picks")                        # 401 error response
    assert r.status_code == 401 and r.headers.get("x-content-type-options") == "nosniff"
    h = c.get("/api/v1/health").json()
    assert h == {"status": "ok", "app": "StockLens API", "version": "0.1.0"}   # contract unchanged
