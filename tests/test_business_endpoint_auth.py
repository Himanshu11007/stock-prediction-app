"""tests/test_business_endpoint_auth.py — authentication/authorization on
analyze-stock, top-picks, tracker, performance, and intelligence.

These routes call into the real ML/data-fetch pipeline (api/services.py),
which needs network access and trained models - not something a unit test
should invoke. Every services.* function is monkeypatched to a canned
return value so these tests verify ONLY the auth gate (who gets in), not
the business logic underneath it (already covered by this repo's existing
scanner/decision-engine/etc. test suites, untouched by this change).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from api import services
from api.routes import analysis, intelligence, performance, top_picks, tracker
from db.session import get_session


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def client(engine, monkeypatch):
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/v1")
    app.include_router(top_picks.router, prefix="/api/v1")
    app.include_router(tracker.router, prefix="/api/v1")
    app.include_router(performance.router, prefix="/api/v1")
    app.include_router(intelligence.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session

    # Stub out every underlying service call - see module docstring.
    monkeypatch.setattr(services, "analyze_stock", lambda symbol: {"symbol": symbol, "signal": "HOLD"})
    monkeypatch.setattr(
        services, "start_top_picks_scan", lambda category: {"scan_id": "x", "status": "started", "category": category}
    )
    monkeypatch.setattr(
        services, "get_scan_status", lambda scan_id: {"scan_id": scan_id, "status": "running", "progress": 0, "total": 0, "message": ""}
    )
    monkeypatch.setattr(services, "get_scan_result", lambda scan_id: {"scan_id": scan_id, "status": "completed", "results": []})
    monkeypatch.setattr(services, "get_saved_recommendations", lambda limit=50: [])
    monkeypatch.setattr(services, "save_manual_recommendation", lambda **kw: 1)
    monkeypatch.setattr(services, "run_validation", lambda: 0)
    monkeypatch.setattr(services, "get_performance_summary", lambda: {"total": 0})
    monkeypatch.setattr(services, "get_performance_by_signal", lambda: [])
    monkeypatch.setattr(services, "get_performance_by_confidence", lambda: [])
    monkeypatch.setattr(services, "get_performance_by_confluence", lambda: [])
    monkeypatch.setattr(services, "get_intelligence_report", lambda: {"meta": {"records_analyzed": 0}})

    return TestClient(app)


@pytest.fixture()
def seed(engine):
    with Session(engine) as session:
        auth_service.ensure_roles_exist(session)
        auth_service.create_user(session, "user@example.com", "Pass1234!", roles=[auth_service.USER_ROLE])
        auth_service.create_user(session, "admin@example.com", "Pass1234!", roles=[auth_service.ADMIN_ROLE])


def _headers(email: str, role: str) -> dict:
    from auth.security import create_access_token

    token = create_access_token(subject=email, roles=[role])
    return {"Authorization": f"Bearer {token}"}


def _user_headers() -> dict:
    return _headers("user@example.com", auth_service.USER_ROLE)


def _admin_headers() -> dict:
    return _headers("admin@example.com", auth_service.ADMIN_ROLE)


# Routes any authenticated user (USER or ADMIN) should be able to reach.
USER_ACCESSIBLE = [
    ("POST", "/api/v1/analyze-stock", {"symbol": "TCS.NS"}),
    ("POST", "/api/v1/top-picks/start", {"category": "Large Cap"}),
    ("GET", "/api/v1/top-picks/status/abc", None),
    ("GET", "/api/v1/top-picks/result/abc", None),
    ("GET", "/api/v1/tracker/recommendations", None),
    ("GET", "/api/v1/performance/summary", None),
    ("GET", "/api/v1/performance/by-signal", None),
    ("GET", "/api/v1/performance/by-confidence", None),
    ("GET", "/api/v1/performance/by-confluence", None),
    ("GET", "/api/v1/intelligence/report", None),
]

# Routes that write recommendation data directly / trigger a global batch
# job - ADMIN only, not a normal user's "view my data" action.
ADMIN_ONLY = [
    (
        "POST",
        "/api/v1/tracker/save",
        {
            "symbol": "TCS.NS", "stock": "Tata Consultancy Services Ltd.", "signal": "BUY",
            "cmp": 3500.0, "score": 0.7, "confidence": 80.0, "news_score": 0.1,
        },
    ),
    ("POST", "/api/v1/tracker/validate-old", None),
]


@pytest.mark.parametrize("method,path,body", USER_ACCESSIBLE)
def test_unauthenticated_rejected_on_user_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}"


@pytest.mark.parametrize("method,path,body", USER_ACCESSIBLE)
def test_normal_user_allowed_on_user_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body, headers=_user_headers())
    assert resp.status_code == 200, f"{method} {path} -> {resp.status_code}: {resp.text}"


@pytest.mark.parametrize("method,path,body", USER_ACCESSIBLE)
def test_admin_also_allowed_on_user_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body, headers=_admin_headers())
    assert resp.status_code == 200, f"{method} {path} -> {resp.status_code}: {resp.text}"


@pytest.mark.parametrize("method,path,body", ADMIN_ONLY)
def test_unauthenticated_rejected_on_admin_only_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path,body", ADMIN_ONLY)
def test_normal_user_denied_on_admin_only_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body, headers=_user_headers())
    assert resp.status_code == 403


@pytest.mark.parametrize("method,path,body", ADMIN_ONLY)
def test_admin_allowed_on_admin_only_routes(client, seed, method, path, body):
    resp = client.request(method, path, json=body, headers=_admin_headers())
    assert resp.status_code == 200, f"{method} {path} -> {resp.status_code}: {resp.text}"
