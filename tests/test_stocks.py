"""tests/test_stocks.py — /api/v1/stocks/* (normal-user stock discovery,
distinct from the ADMIN-only /admin/stocks)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from api.routes import stocks as stocks_routes
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
    app.include_router(stocks_routes.router, prefix="/api/v1")

    def _get_test_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _get_test_session
    return TestClient(app)


@pytest.fixture()
def seed(engine):
    with Session(engine) as session:
        auth_service.ensure_roles_exist(session)
        auth_service.create_user(session, "user@example.com", "Pass1234!", roles=[auth_service.USER_ROLE])
        session.add(Company(symbol="TCS.NS", name="Tata Consultancy Services Ltd.", sector="IT", active=True))
        session.add(Company(symbol="INFY.NS", name="Infosys Ltd.", sector="IT", active=True))
        session.add(Company(symbol="DEAD.NS", name="Delisted Corp Ltd.", active=False))
        session.commit()


def _headers() -> dict:
    from auth.security import create_access_token

    token = create_access_token(subject="user@example.com", roles=[auth_service.USER_ROLE])
    return {"Authorization": f"Bearer {token}"}


def test_unauthenticated_rejected(client, seed):
    assert client.get("/api/v1/stocks").status_code == 401
    assert client.get("/api/v1/stocks/TCS.NS").status_code == 401


def test_search_by_symbol(client, seed):
    resp = client.get("/api/v1/stocks?search=TCS", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "TCS.NS"
    assert body[0]["name"] == "Tata Consultancy Services Ltd."


def test_search_by_company_name(client, seed):
    resp = client.get("/api/v1/stocks?search=Infosys", headers=_headers())
    assert resp.status_code == 200
    assert resp.json()[0]["symbol"] == "INFY.NS"


def test_search_excludes_inactive_stocks(client, seed):
    resp = client.get("/api/v1/stocks?search=Delisted", headers=_headers())
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_all_active_stocks(client, seed):
    resp = client.get("/api/v1/stocks", headers=_headers())
    assert resp.status_code == 200
    symbols = {s["symbol"] for s in resp.json()}
    assert symbols == {"TCS.NS", "INFY.NS"}  # DEAD.NS excluded


def test_get_single_stock(client, seed):
    resp = client.get("/api/v1/stocks/TCS.NS", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "TCS.NS"
    assert body["sector"] == "IT"
    # admin-only bookkeeping fields must not leak
    assert "active" not in body
    assert "created_at" not in body
    assert "updated_at" not in body


def test_get_inactive_stock_returns_404(client, seed):
    resp = client.get("/api/v1/stocks/DEAD.NS", headers=_headers())
    assert resp.status_code == 404


def test_get_unknown_stock_returns_404(client, seed):
    resp = client.get("/api/v1/stocks/NOPE.NS", headers=_headers())
    assert resp.status_code == 404
