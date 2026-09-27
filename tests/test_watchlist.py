"""tests/test_watchlist.py — /api/v1/watchlist/* (authenticated user's own
watchlist, distinct from the legacy storage/watchlist.py and from the
admin-only /admin/watchlist read view).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from api.routes import watchlist as watchlist_routes
from db.models.stock import Company
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
    app.include_router(watchlist_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


@pytest.fixture()
def seed(engine):
    with Session(engine) as session:
        auth_service.ensure_roles_exist(session)
        user_a = auth_service.create_user(session, "a@example.com", "PassA1234!", roles=[auth_service.USER_ROLE])
        user_b = auth_service.create_user(session, "b@example.com", "PassB1234!", roles=[auth_service.USER_ROLE])
        session.add(Company(symbol="TCS.NS", name="Tata Consultancy Services Ltd.", active=True))
        session.add(Company(symbol="INFY.NS", name="Infosys Ltd.", active=True))
        session.commit()
        return {"a_id": user_a.id, "b_id": user_b.id}


def _headers(email: str) -> dict:
    from auth.security import create_access_token

    token = create_access_token(subject=email, roles=[auth_service.USER_ROLE])
    return {"Authorization": f"Bearer {token}"}


def test_unauthenticated_rejected(client, seed):
    assert client.get("/api/v1/watchlist").status_code == 401
    assert client.post("/api/v1/watchlist", json={}).status_code == 401
    assert client.delete("/api/v1/watchlist/1").status_code == 401


def test_empty_watchlist(client, seed):
    resp = client.get("/api/v1/watchlist", headers=_headers("a@example.com"))
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_and_list(client, seed):
    resp = client.post(
        "/api/v1/watchlist",
        json={"symbol": "tcs.ns", "buy_price": 3500.0, "buy_date": "2026-01-01", "quantity": 2},
        headers=_headers("a@example.com"),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["symbol"] == "TCS.NS"  # normalized
    assert body["stock_name"] == "Tata Consultancy Services Ltd."  # from stock master, not client
    assert body["quantity"] == 2

    resp = client.get("/api/v1/watchlist", headers=_headers("a@example.com"))
    assert len(resp.json()) == 1


def test_add_unknown_symbol_returns_400(client, seed):
    resp = client.post(
        "/api/v1/watchlist",
        json={"symbol": "NOPE.NS", "buy_price": 100.0, "buy_date": "2026-01-01"},
        headers=_headers("a@example.com"),
    )
    assert resp.status_code == 400


def test_duplicate_symbol_returns_409(client, seed):
    payload = {"symbol": "TCS.NS", "buy_price": 3500.0, "buy_date": "2026-01-01"}
    first = client.post("/api/v1/watchlist", json=payload, headers=_headers("a@example.com"))
    assert first.status_code == 201
    second = client.post("/api/v1/watchlist", json=payload, headers=_headers("a@example.com"))
    assert second.status_code == 409


def test_remove_nonexistent_item_returns_404(client, seed):
    resp = client.delete("/api/v1/watchlist/9999", headers=_headers("a@example.com"))
    assert resp.status_code == 404


def test_add_then_remove(client, seed):
    add = client.post(
        "/api/v1/watchlist",
        json={"symbol": "INFY.NS", "buy_price": 1500.0, "buy_date": "2026-01-01"},
        headers=_headers("a@example.com"),
    )
    item_id = add.json()["id"]

    remove = client.delete(f"/api/v1/watchlist/{item_id}", headers=_headers("a@example.com"))
    assert remove.status_code == 204

    resp = client.get("/api/v1/watchlist", headers=_headers("a@example.com"))
    assert resp.json() == []


# ══════════════════════════════════════════════════════════════════════════
# Ownership isolation — the core security requirement
# ══════════════════════════════════════════════════════════════════════════

def test_user_cannot_see_another_users_watchlist(client, seed):
    client.post(
        "/api/v1/watchlist",
        json={"symbol": "TCS.NS", "buy_price": 3500.0, "buy_date": "2026-01-01"},
        headers=_headers("a@example.com"),
    )
    resp_b = client.get("/api/v1/watchlist", headers=_headers("b@example.com"))
    assert resp_b.status_code == 200
    assert resp_b.json() == []  # B sees nothing of A's


def test_user_cannot_delete_another_users_item(client, seed):
    add = client.post(
        "/api/v1/watchlist",
        json={"symbol": "TCS.NS", "buy_price": 3500.0, "buy_date": "2026-01-01"},
        headers=_headers("a@example.com"),
    )
    item_id = add.json()["id"]

    resp = client.delete(f"/api/v1/watchlist/{item_id}", headers=_headers("b@example.com"))
    assert resp.status_code == 404  # not 403 - existence isn't confirmed either

    # A's item must still be there, untouched by B's attempt
    still_there = client.get("/api/v1/watchlist", headers=_headers("a@example.com"))
    assert len(still_there.json()) == 1


def test_both_users_can_hold_the_same_symbol_independently(client, seed):
    payload = {"symbol": "TCS.NS", "buy_price": 3500.0, "buy_date": "2026-01-01"}
    resp_a = client.post("/api/v1/watchlist", json=payload, headers=_headers("a@example.com"))
    resp_b = client.post("/api/v1/watchlist", json=payload, headers=_headers("b@example.com"))
    assert resp_a.status_code == 201
    assert resp_b.status_code == 201  # per-user uniqueness, not global

    assert len(client.get("/api/v1/watchlist", headers=_headers("a@example.com")).json()) == 1
    assert len(client.get("/api/v1/watchlist", headers=_headers("b@example.com")).json()) == 1
