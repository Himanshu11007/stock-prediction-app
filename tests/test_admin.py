"""tests/test_admin.py — /api/v1/admin/* (Phase 5).

Runs against the real router (api/routes/admin.py) with only the DB session
swapped for an isolated in-memory SQLite engine, so authorization, audit
logging, and read/write behavior are all exercised through actual code, not
reimplemented in the test.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from api.routes import admin as admin_routes
from api.routes import auth as auth_routes
from db.models.admin import AdminAuditLog
from db.models.stock import Company
from db.models.tracker import Recommendation, WatchlistItem
from db.session import get_session


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def client(engine):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(admin_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


@pytest.fixture()
def seed(engine):
    """Admin user, normal user, two stocks, a recommendation, a watchlist
    item — enough fixture data for every read endpoint to return something
    real."""
    with Session(engine) as session:
        auth_service.ensure_roles_exist(session)
        admin_user = auth_service.create_user(
            session, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE]
        )
        normal_user = auth_service.create_user(
            session, "normal@example.com", "UserPass1!", roles=[auth_service.USER_ROLE]
        )
        session.add(Company(symbol="TCS.NS", name="Tata Consultancy Services Ltd.", active=True))
        session.add(Company(symbol="INFY.NS", name="Infosys Ltd.", active=True))
        session.add(
            Recommendation(
                legacy_id=1, is_legacy_migration=True, saved_date="2026-01-01",
                symbol="TCS.NS", stock="Tata Consultancy Services Ltd.", signal="BUY", cmp=3500.0,
            )
        )
        session.add(
            WatchlistItem(
                legacy_id=1, is_legacy_migration=True, user_id=admin_user.id,
                symbol="TCS.NS", stock_name="Tata Consultancy Services Ltd.",
                buy_price=3500.0, buy_date="2026-01-01",
            )
        )
        session.commit()
        return {"admin_id": admin_user.id, "normal_id": normal_user.id}


def _token(email: str, roles: list[str]) -> str:
    from auth.security import create_access_token

    return create_access_token(subject=email, roles=roles)


def _admin_headers() -> dict:
    return {"Authorization": f"Bearer {_token('admin@example.com', [auth_service.ADMIN_ROLE])}"}


def _normal_headers() -> dict:
    return {"Authorization": f"Bearer {_token('normal@example.com', [auth_service.USER_ROLE])}"}


# ══════════════════════════════════════════════════════════════════════════
# Authorization
# ══════════════════════════════════════════════════════════════════════════

def test_normal_user_denied_every_admin_route(client, seed):
    headers = _normal_headers()
    for method, path in [
        ("GET", "/api/v1/admin/dashboard"),
        ("GET", "/api/v1/admin/users"),
        ("GET", "/api/v1/admin/stocks"),
        ("GET", "/api/v1/admin/recommendations"),
        ("GET", "/api/v1/admin/watchlist"),
        ("GET", "/api/v1/admin/audit-logs"),
    ]:
        resp = client.request(method, path, headers=headers)
        assert resp.status_code == 403, f"{method} {path} -> {resp.status_code}"


def test_unauthenticated_denied(client, seed):
    resp = client.get("/api/v1/admin/dashboard")
    assert resp.status_code == 401


# ══════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════

def test_dashboard_reports_real_counts(client, seed):
    resp = client.get("/api/v1/admin/dashboard", headers=_admin_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_users"] == 2
    assert body["active_users"] == 2
    assert body["admin_users"] == 1
    assert body["total_stocks"] == 2
    assert body["total_recommendations"] == 1
    assert body["total_watchlist_items"] == 1
    assert "subscription" not in str(body).lower()
    assert "premium" not in str(body).lower()


# ══════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════

def test_list_and_get_user(client, seed):
    resp = client.get("/api/v1/admin/users", headers=_admin_headers())
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get(f"/api/v1/admin/users/{seed['normal_id']}", headers=_admin_headers())
    assert resp.status_code == 200
    assert resp.json()["email"] == "normal@example.com"
    assert resp.json()["roles"] == ["USER"]


def test_get_unknown_user_404(client, seed):
    resp = client.get("/api/v1/admin/users/9999", headers=_admin_headers())
    assert resp.status_code == 404


def test_deactivate_then_activate_user_writes_audit_log(client, seed, engine):
    uid = seed["normal_id"]
    resp = client.post(f"/api/v1/admin/users/{uid}/deactivate", headers=_admin_headers())
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    resp = client.post(f"/api/v1/admin/users/{uid}/activate", headers=_admin_headers())
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True

    with Session(engine) as session:
        from sqlmodel import select

        logs = session.exec(select(AdminAuditLog).where(AdminAuditLog.entity == "user")).all()
        actions = [l.action for l in logs]
        assert "deactivate_user" in actions
        assert "activate_user" in actions
        assert all(l.entity_id == str(uid) for l in logs)
        assert all(l.admin_user_id == seed["admin_id"] for l in logs)


def test_deactivated_user_cannot_authenticate(client, seed):
    uid = seed["normal_id"]
    client.post(f"/api/v1/admin/users/{uid}/deactivate", headers=_admin_headers())

    resp = client.post(
        "/api/v1/auth/login", data={"username": "normal@example.com", "password": "UserPass1!"}
    )
    assert resp.status_code == 401


def test_assign_and_remove_role_writes_audit_log(client, seed, engine):
    uid = seed["normal_id"]
    resp = client.post(
        f"/api/v1/admin/users/{uid}/roles", json={"role": "ADMIN"}, headers=_admin_headers()
    )
    assert resp.status_code == 200
    assert set(resp.json()["roles"]) == {"USER", "ADMIN"}

    resp = client.request(
        "DELETE", f"/api/v1/admin/users/{uid}/roles/ADMIN", headers=_admin_headers()
    )
    assert resp.status_code == 200
    assert resp.json()["roles"] == ["USER"]

    with Session(engine) as session:
        from sqlmodel import select

        logs = session.exec(
            select(AdminAuditLog).where(AdminAuditLog.action.in_(["assign_role", "remove_role"]))
        ).all()
        assert len(logs) == 2


def test_assign_unknown_role_returns_400(client, seed):
    uid = seed["normal_id"]
    resp = client.post(
        f"/api/v1/admin/users/{uid}/roles", json={"role": "SUPERUSER"}, headers=_admin_headers()
    )
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════
# Stock master
# ══════════════════════════════════════════════════════════════════════════

def test_list_and_search_stocks(client, seed):
    resp = client.get("/api/v1/admin/stocks", headers=_admin_headers())
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get("/api/v1/admin/stocks?search=TCS", headers=_admin_headers())
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["symbol"] == "TCS.NS"


def test_disable_stock_writes_audit_log_and_persists(client, seed, engine):
    resp = client.patch(
        "/api/v1/admin/stocks/TCS.NS", json={"active": False}, headers=_admin_headers()
    )
    assert resp.status_code == 200
    assert resp.json()["active"] is False
    assert resp.json()["analysis_enabled"] is True  # untouched field unchanged

    with Session(engine) as session:
        company = session.get(Company, "TCS.NS")
        assert company.active is False

        from sqlmodel import select

        logs = session.exec(select(AdminAuditLog).where(AdminAuditLog.entity == "company")).all()
        assert len(logs) == 1
        assert logs[0].entity_id == "TCS.NS"
        assert logs[0].extra_data == {"active": False}


def test_update_stock_with_no_change_writes_no_audit_log(client, seed, engine):
    # active already True — asking to set it True again is a no-op
    resp = client.patch(
        "/api/v1/admin/stocks/TCS.NS", json={"active": True}, headers=_admin_headers()
    )
    assert resp.status_code == 200

    with Session(engine) as session:
        from sqlmodel import select

        logs = session.exec(select(AdminAuditLog)).all()
        assert len(logs) == 0


def test_get_unknown_stock_404(client, seed):
    resp = client.get("/api/v1/admin/stocks/NOPE.NS", headers=_admin_headers())
    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# Recommendations / watchlist — read-only
# ══════════════════════════════════════════════════════════════════════════

def test_list_recommendations_read_only(client, seed):
    resp = client.get("/api/v1/admin/recommendations", headers=_admin_headers())
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["symbol"] == "TCS.NS"

    # no mutating route exists for recommendations - POST to the collection
    # path is 405 (path exists, method doesn't); no per-item path exists at
    # all, so that 404s rather than 405s.
    resp = client.post("/api/v1/admin/recommendations", json={}, headers=_admin_headers())
    assert resp.status_code == 405
    resp = client.patch("/api/v1/admin/recommendations/1", json={}, headers=_admin_headers())
    assert resp.status_code == 404


def test_list_watchlist_read_only(client, seed):
    resp = client.get("/api/v1/admin/watchlist", headers=_admin_headers())
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["symbol"] == "TCS.NS"

    resp = client.post("/api/v1/admin/watchlist", json={}, headers=_admin_headers())
    assert resp.status_code == 405


# ══════════════════════════════════════════════════════════════════════════
# Audit logs
# ══════════════════════════════════════════════════════════════════════════

def test_audit_log_listing_reflects_actions(client, seed):
    client.patch("/api/v1/admin/stocks/TCS.NS", json={"active": False}, headers=_admin_headers())
    client.post(
        f"/api/v1/admin/users/{seed['normal_id']}/deactivate", headers=_admin_headers()
    )

    resp = client.get("/api/v1/admin/audit-logs", headers=_admin_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    actions = {entry["action"] for entry in body}
    assert actions == {"update_stock", "deactivate_user"}
