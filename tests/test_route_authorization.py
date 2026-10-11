"""
tests/test_route_authorization.py — every API route, enumerated from the
application itself (so a new route is covered automatically):

  * anonymous requests are rejected (401) unless the route is in the explicit
    PUBLIC allowlist below; the scheduler job interface answers 404 while
    its token is not configured
  * an ordinary user (USER role) is refused (403) on every admin route and,
    while Prediction Engine v2 is in shadow mode, on every v2 / news route
"""
import re

import pytest

from api.main import app
from tests.test_prediction_api import H, client, db  # noqa: F401  (fixtures)

PUBLIC = {
    ("GET", "/api/v1/health"), ("GET", "/api/v1/app/config"),
    ("POST", "/api/v1/auth/login"), ("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/refresh"),
    ("POST", "/api/v1/auth/logout"), ("POST", "/api/v1/auth/google"), ("POST", "/api/v1/auth/apple"),
    ("POST", "/api/v1/auth/otp/request"), ("POST", "/api/v1/auth/otp/verify"),
    ("POST", "/api/v1/auth/forgot-password"), ("POST", "/api/v1/auth/reset-password"),
}
SCHEDULER = re.compile(r"^/api/v1/scheduler/")
SHADOW = re.compile(r"^/api/v1/(predictions|stocks/\{[^}]+\}/events|watchlist/exit-signals)")


def _routes():
    spec = app.openapi()
    return sorted((m.upper(), p) for p, ops in spec["paths"].items() for m in ops)


def _call(client, method, path, headers=None):
    return client.request(method, re.sub(r"\{[^}]+\}", "X1", path), json={}, headers=headers or {})


def test_route_inventory_is_complete():
    routes = _routes()
    assert len(routes) > 100
    assert PUBLIC <= set(routes), sorted(PUBLIC - set(routes))     # allowlist has no stale entries


@pytest.mark.parametrize("method,path", _routes())
def test_anonymous_requests_are_rejected_unless_public(client, method, path):
    r = _call(client, method, path)
    if (method, path) in PUBLIC:
        assert r.status_code != 401
    elif SCHEDULER.match(path):
        assert r.status_code == 404                # token-authenticated; hidden while unconfigured
    else:
        assert r.status_code == 401, f"{method} {path} answered {r.status_code} without credentials"


@pytest.mark.parametrize("method,path", [r for r in _routes() if "/admin/" in r[1] or SHADOW.match(r[1])])
def test_ordinary_users_cannot_reach_admin_or_shadow_routes(client, method, path):
    r = _call(client, method, path, H())
    assert r.status_code == 403, f"{method} {path} answered {r.status_code} to a USER token"
