"""
tests/test_product_api.py — product API: Top Picks, stock analysis/FQVF/
ranking, app configuration, admin masters/engine/health routes, log routes,
authorization and error handling.

Runs the real application (api.main.app, real routers and exception
handlers) with only the DB session swapped for an isolated in-memory SQLite.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
import engine_runs.service as runs
from api.main import app
from auth.security import create_access_token
from db.models.admin import AdminAuditLog
from db.models.market import (EngineRun, FundamentalSnapshot, MarketRegimeSnapshot, MarketSnapshot, Sector,
                              StockAnalysisResult)
from db.models.stock import Company, StockUniverseMember
from db.session import get_session
from fqvf import FQVFInputs, evaluate


@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
    return eng


@pytest.fixture()
def client(db):
    def _session():
        with Session(db) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def H(email="user@example.com", roles=("USER",)):
    return {"Authorization": f"Bearer {create_access_token(subject=email, roles=list(roles))}"}


ADMIN = lambda: H("admin@example.com", ("ADMIN",))  # noqa: E731


def _seed_run(db, now=None):
    now = now or datetime.now(timezone.utc)
    fq = evaluate(FQVFInputs(price=100, price_age_days=1, sector="Energy", industry="Oil",
                             trailing_pe=10, price_to_book=0.8)).to_dict()
    with Session(db) as s:
        for sym, name, active in (("AAA.NS", "Alpha", True), ("BBB.NS", "Beta", True),
                                  ("CCC.NS", "Gamma", False), ("NEW.NS", "Never analysed", True)):
            s.add(Company(symbol=sym, name=name, active=active, sector="Energy"))
            s.add(StockUniverseMember(symbol=sym, category="Large Cap"))
        s.add(Sector(name="Energy"))
        s.add(EngineRun(run_id="RANKING-T", kind="RANKING", status="COMPLETED", started_at=now,
                        finished_at=now, engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0",
                        total=3, processed=3, succeeded=3, errors=[]))
        s.add(MarketRegimeSnapshot(run_id="RANKING-T", regime="Bullish", regime_score=0.5, as_of_date="2026-10-01"))
        s.add(MarketSnapshot(symbol="AAA.NS", fetched_at=now - timedelta(seconds=1), status="OK",
                             as_of_date=(now - timedelta(days=10)).date().isoformat(), close=100.0, technical={}))
        s.add(FundamentalSnapshot(symbol="AAA.NS", status="UNAVAILABLE", error="provider returned nothing",
                                  fetched_at=now))
        for sym, score, rank, eligible in (("AAA.NS", 70.0, 2, True), ("BBB.NS", 80.0, 1, True),
                                           ("CCC.NS", 90.0, None, False)):
            s.add(StockAnalysisResult(run_id="RANKING-T", symbol=sym, computed_at=now, stockai_score=score,
                                      score_coverage=0.9, eligible=eligible, rank=rank, components={},
                                      fqvf=fq, positives=["p"], risks=["r"], freshness={"market_data_as_of": "x"},
                                      ineligible_reasons=[] if eligible else ["stock is inactive"],
                                      engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0"))
        s.commit()


# ── public / authentication ──────────────────────────────────────────────────

def test_app_config_is_public_and_backend_controlled(client):
    r = client.get("/api/v1/app/config")
    assert r.status_code == 200 and r.json()["success"]
    data = r.json()["data"]
    assert data["features"]["top_picks"] is True and "not investment advice" in data["disclaimer"]
    assert data["versions"]["fqvf"].startswith("fqvf-")


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/top-picks"), ("GET", "/api/v1/stocks/AAA.NS/analysis"), ("GET", "/api/v1/stocks/AAA.NS/fqvf"),
    ("GET", "/api/v1/stocks/AAA.NS/ranking"), ("POST", "/api/v1/stocks/AAA.NS/analysis/refresh"),
    ("GET", "/api/v1/market/regime"), ("GET", "/api/v1/fqvf/reference"), ("GET", "/api/v1/ranking/reference"),
])
def test_business_routes_require_authentication(client, method, path):
    assert client.request(method, path).status_code == 401


ADMIN_GETS = ["/admin/sectors", "/admin/industries", "/admin/fundamentals", "/admin/market-data", "/admin/technical",
              "/admin/valuation", "/admin/market-regime", "/admin/fqvf/reference", "/admin/config",
              "/admin/ranking/config", "/admin/analysis-results", "/admin/engine-runs", "/admin/data-health",
              "/admin/api-health", "/admin/engine-versions", "/admin/stock-master/summary", "/admin/roles",
              "/logs/latest"]


@pytest.mark.parametrize("path", ADMIN_GETS)
def test_admin_routes_reject_normal_and_anonymous_users(client, path):
    assert client.get("/api/v1" + path).status_code == 401
    assert client.get("/api/v1" + path, headers=H()).status_code == 403


@pytest.mark.parametrize("path", ADMIN_GETS)
def test_admin_routes_work_for_admin(client, db, path):
    _seed_run(db)
    assert client.get("/api/v1" + path, headers=ADMIN()).status_code == 200


@pytest.mark.parametrize("method,path,body", [
    ("PATCH", "/api/v1/admin/sectors/1", {"outlook": "POSITIVE"}),
    ("PUT", "/api/v1/admin/config/top_picks.limit", {"value": 5}),
    ("POST", "/api/v1/admin/engine-runs", {}),
    ("POST", "/api/v1/admin/stocks", {"symbol": "ZZZ.NS", "name": "Zed"}),
    ("DELETE", "/api/v1/logs/clear", None),
])
def test_admin_mutations_forbidden_for_normal_users(client, method, path, body):
    assert client.request(method, path, json=body, headers=H()).status_code == 403


# ── Top Picks and analysis ───────────────────────────────────────────────────

def test_top_picks_empty_before_any_run(client):
    r = client.get("/api/v1/top-picks", headers=H())
    assert r.status_code == 200
    assert r.json()["data"]["items"] == [] and "No completed analysis run" in r.json()["message"]


def test_top_picks_ranked_eligible_active_only(client, db):
    _seed_run(db)
    data = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert [i["symbol"] for i in data["items"]] == ["BBB.NS", "AAA.NS"]
    item = data["items"][0]
    assert item["rank"] == 1 and item["stockai_score"] == 80.0 and item["positives"] and item["risks"]
    assert item["engine_version"] == "ranking-v1.0" and item["freshness"]
    assert data["run"]["run_id"] == "RANKING-T" and data["market_regime"]["regime"] == "Bullish"
    assert "not investment advice" in data["disclaimer"]
    assert len(client.get("/api/v1/top-picks?limit=1", headers=H()).json()["data"]["items"]) == 1


def test_top_picks_limit_capped_by_backend_config(client, db):
    _seed_run(db)
    assert client.put("/api/v1/admin/config/top_picks.limit", json={"value": 1}, headers=ADMIN()).status_code == 200
    assert len(client.get("/api/v1/top-picks?limit=50", headers=H()).json()["data"]["items"]) == 1


def test_stock_analysis_payload(client, db):
    _seed_run(db)
    r = client.get("/api/v1/stocks/AAA.NS/analysis", headers=H())
    assert r.status_code == 200
    d = r.json()["data"]
    assert len(d["fqvf"]["checks"]) == 18 and d["ranking"]["stockai_score"] == 70.0
    assert d["engine"]["ranking"] == "ranking-v1.0" and d["market"]["close"] == 100.0
    fq = client.get("/api/v1/stocks/aaa.ns/fqvf", headers=H()).json()["data"]
    assert fq["checks"][0]["name"] == "Stable Stock / Business Stability"
    rk = client.get("/api/v1/stocks/AAA.NS/ranking", headers=H()).json()["data"]
    assert rk["rank"] == 2 and rk["eligible_for_top_picks"] is True


def test_not_analysed_and_inactive_and_unknown_stocks_are_404(client, db):
    _seed_run(db)
    r = client.get("/api/v1/stocks/NEW.NS/analysis", headers=H())
    assert r.status_code == 404 and "not been analysed" in r.json()["detail"]
    assert client.get("/api/v1/stocks/CCC.NS/analysis", headers=H()).status_code == 404   # inactive
    assert client.get("/api/v1/stocks/NOPE.NS/fqvf", headers=H()).status_code == 404


def test_recent_analysis_is_not_recomputed(client, db, monkeypatch):
    _seed_run(db)
    monkeypatch.setattr(runs, "run_single_stock", lambda *a, **k: pytest.fail("should not re-run"))
    r = client.post("/api/v1/stocks/AAA.NS/analysis/refresh", headers=H())
    assert r.status_code == 200 and "less than 60 minutes" in r.json()["message"]


def test_refresh_reports_busy_engine_as_409(client, db, monkeypatch):
    _seed_run(db, now=datetime.now(timezone.utc) - timedelta(hours=5))

    def busy(*a, **k):
        raise runs.RunInProgressError("busy")
    monkeypatch.setattr(runs, "run_single_stock", busy)
    assert client.post("/api/v1/stocks/AAA.NS/analysis/refresh", headers=H()).status_code == 409


def test_reference_endpoints(client):
    ref = client.get("/api/v1/fqvf/reference", headers=H()).json()["data"]
    assert ref["name"] == "Fundamental Quality & Value Framework" and len(ref["checks"]) == 18
    rk = client.get("/api/v1/ranking/reference", headers=H()).json()["data"]
    assert {c["key"] for c in rk["components"]} >= {"quality", "valuation", "ml_signal"}


# ── admin mutations ──────────────────────────────────────────────────────────

def test_sector_outlook_update_is_validated_and_audited(client, db):
    _seed_run(db)
    sector_id = client.get("/api/v1/admin/sectors", headers=ADMIN()).json()[0]["id"]
    bad = client.patch(f"/api/v1/admin/sectors/{sector_id}", json={"outlook": "AMAZING"}, headers=ADMIN())
    assert bad.status_code == 400
    ok = client.patch(f"/api/v1/admin/sectors/{sector_id}", json={"outlook": "positive", "outlook_notes": "capex"},
                      headers=ADMIN())
    assert ok.status_code == 200 and ok.json()["outlook"] == "POSITIVE"
    with Session(db) as s:
        assert s.exec(select(AdminAuditLog).where(AdminAuditLog.action == "SECTOR_UPDATED")).first() is not None


def test_ranking_weights_config_validation(client):
    bad = client.put("/api/v1/admin/config/ranking.weights", json={"value": {"quality": -5}}, headers=ADMIN())
    assert bad.status_code == 400
    ok = client.put("/api/v1/admin/config/ranking.weights", json={"value": {"ml_signal": 5}}, headers=ADMIN())
    assert ok.status_code == 200 and ok.json()["value"]["ml_signal"] == 5.0 and ok.json()["value"]["quality"] == 25.0
    assert client.put("/api/v1/admin/config/nope", json={"value": 1}, headers=ADMIN()).status_code == 404


def test_create_stock_and_tradable_flag(client):
    r = client.post("/api/v1/admin/stocks", json={"symbol": "zzz.ns", "name": "Zed Ltd"}, headers=ADMIN())
    assert r.status_code == 201 and r.json()["symbol"] == "ZZZ.NS"
    assert client.post("/api/v1/admin/stocks", json={"symbol": "ZZZ.NS", "name": "Dup"},
                       headers=ADMIN()).status_code == 400
    assert client.post("/api/v1/admin/stocks", json={"symbol": "bad symbol", "name": "x"},
                       headers=ADMIN()).status_code == 400
    r = client.patch("/api/v1/admin/stocks/ZZZ.NS", json={"tradable": False}, headers=ADMIN())
    assert r.status_code == 200 and r.json()["tradable"] is False


def test_engine_run_start_and_conflict(client, monkeypatch):
    class FakeRun:
        run_id = "RANKING-FAKE"
    monkeypatch.setattr(runs, "start_run_in_background", lambda *a, **k: FakeRun())
    r = client.post("/api/v1/admin/engine-runs", json={"limit": 5, "include_ml": False}, headers=ADMIN())
    assert r.status_code == 202 and r.json()["run_id"] == "RANKING-FAKE"

    def busy(*a, **k):
        raise runs.RunInProgressError("busy")
    monkeypatch.setattr(runs, "start_run_in_background", busy)
    assert client.post("/api/v1/admin/engine-runs", json={}, headers=ADMIN()).status_code == 409


def test_data_health_reports_stale_and_missing_data(client, db):
    _seed_run(db)
    d = client.get("/api/v1/admin/data-health", headers=ADMIN()).json()
    assert any(f["symbol"] == "AAA.NS" for f in d["findings"]["stale_market_data"])
    assert any(f["symbol"] == "AAA.NS" for f in d["findings"]["missing_fundamentals"])
    assert any(f["symbol"] == "CCC.NS" for f in d["findings"]["inactive_stocks"])
    assert d["news_timestamps"]["status"] == "UNAVAILABLE"


# ── error handling ───────────────────────────────────────────────────────────

def test_unhandled_errors_do_not_leak_details(client, monkeypatch):
    import masters.service as masters

    def boom(*a, **k):
        raise RuntimeError("SECRET-internal-path /etc/db password=hunter2")
    monkeypatch.setattr(masters, "get_config", boom)
    r = client.get("/api/v1/app/config")
    assert r.status_code == 500
    body = r.text
    assert "SECRET" not in body and "hunter2" not in body and "reference" in r.json()["details"]


def test_production_settings_guard(monkeypatch):
    import api.main as main
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(main, "CORS_ALLOWED_ORIGINS", ["*"])
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        main._check_production_settings()
