"""
tests/test_prediction_api.py — Prediction v2 API (shadow): access control,
single-snapshot selection, freshness, pagination, detail, admin actions,
universe isolation and universe health.
"""
import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
from api.main import app
from auth.security import create_access_token
from db.models.market import MarketSnapshot
from db.models.prediction import (EventClassification, ExitState, MarketEvent, Prediction, PredictionOutcome,
                                  PredictionRun, V2UniverseMember)
from db.models.stock import Company, StockUniverseMember
from db.models.tracker import WatchlistItem
from db.session import get_session
from prediction_v2 import calendar
from utils.market_session import now_ist

UTC = dt.timezone.utc
TODAY = now_ist().date()


@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        admin = auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
        for sym in ("AAA.NS", "BBB.NS", "DEAD.NS", "OUT.NS"):
            s.add(Company(symbol=sym, name=sym[:3], sector="Energy"))
        for sym in ("AAA.NS", "BBB.NS", "DEAD.NS"):
            s.add(StockUniverseMember(symbol=sym, category="Large Cap"))
        s.add(MarketSnapshot(symbol="DEAD.NS", status="UNAVAILABLE", error="provider returned no price history"))
        s.add(MarketSnapshot(symbol="AAA.NS", status="OK", close=100.0))
        s.add(WatchlistItem(user_id=admin.id, symbol="AAA.NS", stock_name="AAA"))
        s.commit()
    return eng


def _run(s, run_id, run_type, target, completed_minutes_ago=10, status="COMPLETED"):
    t = dt.datetime.now(UTC) - dt.timedelta(minutes=completed_minutes_ago)
    s.add(PredictionRun(run_id=run_id, idempotency_key=f"{run_type}:{run_id}", engine_version="prediction-v2.0-shadow",
                        rule_version="baseline-v0.1", feature_set_version="v2-features-0.1", run_type=run_type,
                        trading_date=target.isoformat(), target_session_date=target.isoformat(), data_cutoff_at=t,
                        status=status, config_hash="h", started_at=t, completed_at=t, counts={"UP": 1}))
    for sym, d in (("AAA.NS", "UP"), ("BBB.NS", "NO_CALL")):
        s.add(Prediction(prediction_id=f"{run_id}-{sym}", run_id=run_id, symbol=sym, direction=d,
                         setup_type="MOMENTUM_CONTINUATION" if d == "UP" else "INSUFFICIENT_DATA",
                         reference_price=100.0, reference_date="2026-10-09", stop_loss=97.0 if d == "UP" else None,
                         target=104.0 if d == "UP" else None, features={"ret_5": 0.05, "cutoff_date": "2026-10-09"},
                         reasons=["r"], quality_flags=[] if d == "UP" else ["INSUFFICIENT_HISTORY_60"]))
    s.commit()


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

USER_ROUTES = ["/api/v1/predictions?session=today", "/api/v1/predictions?session=tomorrow",
               "/api/v1/predictions/performance", "/api/v1/predictions/runs", "/api/v1/predictions/x",
               "/api/v1/stocks/AAA.NS/events", "/api/v1/watchlist/exit-signals"]
ADMIN_ROUTES = ["/api/v1/admin/prediction-runs", "/api/v1/admin/events", "/api/v1/admin/v2-universe",
                "/api/v1/admin/universe-health"]


@pytest.mark.parametrize("path", USER_ROUTES + ADMIN_ROUTES)
def test_shadow_routes_are_admin_only_by_default(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers=H()).status_code == 403


def test_forbidden_message_explains_shadow_mode(client):
    r = client.get("/api/v1/predictions?session=today", headers=H())
    assert "shadow mode" in r.text


@pytest.mark.parametrize("path", USER_ROUTES[:4] + USER_ROUTES[5:] + ADMIN_ROUTES)
def test_admin_can_read(client, path):
    assert client.get(path, headers=ADMIN()).status_code == 200


def test_public_flag_opens_user_routes_but_not_admin_ones(client, monkeypatch):
    monkeypatch.setattr("config.PREDICTION_V2_PUBLIC", True)
    assert client.get("/api/v1/predictions?session=today", headers=H()).status_code == 200
    assert client.get("/api/v1/admin/prediction-runs", headers=H()).status_code == 403


def test_today_prefers_confirmed_and_returns_one_snapshot(client, db):
    with Session(db) as s:
        _run(s, "PRE", "TODAY_PREOPEN", TODAY, 120)
        _run(s, "CONF", "TODAY_CONFIRMED", TODAY, 30)
        _run(s, "OLDEOD", "TOMORROW_EOD", TODAY - dt.timedelta(days=7))
    d = client.get("/api/v1/predictions?session=today", headers=ADMIN()).json()["data"]
    assert d["shadow"] is True and "not investment advice" in d["notice"]
    assert d["run"]["run_id"] == "CONF" and d["freshness"]["status"] == "FRESH"
    assert {p["run_id"] for p in d["predictions"]} == {"CONF"}           # never mixes runs
    assert d["predictions"][0]["direction"] == "UP" and d["predictions"][0]["confidence"] is None
    t = client.get("/api/v1/predictions?session=tomorrow", headers=ADMIN()).json()["data"]
    assert t["run"]["run_id"] == "OLDEOD" and t["freshness"]["status"] == "STALE"


def test_no_snapshot_is_reported_not_invented(client):
    d = client.get("/api/v1/predictions?session=tomorrow", headers=ADMIN()).json()["data"]
    assert d["run"] is None and d["predictions"] == [] and d["freshness"]["status"] == "NONE"


def test_tomorrow_forecast_for_the_next_session_is_fresh(client, db):
    with Session(db) as s:
        _run(s, "EOD", "TOMORROW_EOD", calendar.next_trading_day(TODAY))
        _run(s, "FAILEDEOD", "TOMORROW_EOD", calendar.next_trading_day(TODAY), 1, status="FAILED")
    d = client.get("/api/v1/predictions?session=tomorrow", headers=ADMIN()).json()["data"]
    assert d["run"]["run_id"] == "EOD" and d["freshness"]["status"] == "FRESH"     # failed runs never shown


def test_filters_pagination_and_validation(client, db):
    with Session(db) as s:
        _run(s, "PRE", "TODAY_PREOPEN", TODAY)
    d = client.get("/api/v1/predictions?session=today&direction=NO_CALL", headers=ADMIN()).json()["data"]
    assert [p["symbol"] for p in d["predictions"]] == ["BBB.NS"]
    d = client.get("/api/v1/predictions?session=today&page=2&page_size=1", headers=ADMIN()).json()["data"]
    assert d["total"] == 2 and len(d["predictions"]) == 1
    assert client.get("/api/v1/predictions?session=today&direction=MAYBE", headers=ADMIN()).status_code == 422
    assert client.get("/api/v1/predictions?session=yesterday", headers=ADMIN()).status_code == 422


def test_detail_has_frozen_features_outcomes_and_shadow_exit(client, db):
    with Session(db) as s:
        _run(s, "PRE", "TODAY_PREOPEN", TODAY)
        s.add(PredictionOutcome(prediction_id="PRE-AAA.NS", horizon_sessions=1, evaluator_version="v",
                                outcome_status="EVALUATED", stock_return=0.01, hit=True))
        s.add(ExitState(prediction_id="PRE-AAA.NS", state="TIGHTEN", reason="DISTRIBUTION", rule_version="exit"))
        s.commit()
    d = client.get("/api/v1/predictions/PRE-AAA.NS", headers=ADMIN()).json()["data"]
    assert d["prediction"]["features"]["cutoff_date"] == "2026-10-09"
    assert d["outcomes"][0]["horizon_sessions"] == 1
    assert d["exit"]["state"]["state"] == "TIGHTEN" and "non-actionable" in d["exit"]["mode"]
    assert client.get("/api/v1/predictions/NOPE", headers=ADMIN()).status_code == 404


def test_watchlist_exit_signals_are_marked_non_actionable(client, db):
    with Session(db) as s:
        _run(s, "PRE", "TODAY_PREOPEN", TODAY)
    d = client.get("/api/v1/watchlist/exit-signals", headers=ADMIN()).json()["data"]
    assert d["actionable"] is False and d["signals"][0]["symbol"] == "AAA.NS" and d["signals"][0]["state"] == "HOLD"


def test_admin_trigger_uses_the_scheduled_job(client, monkeypatch):
    import scheduling.jobs as sj
    called = []
    monkeypatch.setattr(sj, "prediction_job", lambda engine, rt: called.append(rt))
    r = client.post("/api/v1/admin/prediction-runs", json={"run_type": "TOMORROW_EOD"}, headers=ADMIN())
    assert r.status_code == 202
    import time
    for _ in range(50):
        if called:
            break
        time.sleep(0.02)
    assert called == ["TOMORROW_EOD"]
    assert client.post("/api/v1/admin/prediction-runs", json={"run_type": "NOW"}, headers=ADMIN()).status_code == 422
    assert client.post("/api/v1/admin/prediction-runs", json={"run_type": "TOMORROW_EOD"}, headers=H()).status_code == 403


def test_admin_event_review_keeps_the_original(client, db):
    with Session(db) as s:
        e = MarketEvent(source="fixture", source_event_id="1", symbol="AAA.NS", event_type="RE_RATING", title="x",
                        published_at=dt.datetime(2026, 10, 5, tzinfo=UTC),
                        effective_available_at=dt.datetime(2026, 10, 5, tzinfo=UTC))
        s.add(e)
        s.commit()
        s.add(EventClassification(event_id=e.id, classifier_version="v0", direction="POSITIVE"))
        s.commit()
        cid = s.exec(select(EventClassification)).one().id
    r = client.patch(f"/api/v1/admin/events/classifications/{cid}", headers=ADMIN(),
                     json={"status": "CORRECTED", "corrected": {"direction": "NEGATIVE"}})
    assert r.status_code == 200 and r.json()["data"]["supersedes_id"] == cid
    bad = client.patch(f"/api/v1/admin/events/classifications/{cid}", headers=ADMIN(),
                       json={"status": "CORRECTED", "corrected": {"surprise_score": 9}})
    assert bad.status_code == 422
    events = client.get("/api/v1/admin/events?symbol=AAA.NS", headers=ADMIN()).json()["data"]
    assert len(events[0]["classifications"]) == 2
    with Session(db) as s:
        assert s.get(EventClassification, cid).direction == "POSITIVE"


def test_v2_universe_admin_changes_never_touch_v1(client, db):
    r = client.post("/api/v1/admin/v2-universe", headers=ADMIN(), json={"symbols": ["OUT.NS", "NOPE.NS"]})
    assert r.json()["data"]["added"] == ["OUT.NS"] and r.json()["data"]["unknown"] == ["NOPE.NS"]
    with Session(db) as s:
        assert "OUT.NS" not in s.exec(select(StockUniverseMember.symbol)).all()
        assert "OUT.NS" in s.exec(select(V2UniverseMember.symbol)).all()
    h = client.get("/api/v1/admin/universe-health", headers=ADMIN()).json()["data"]
    assert [d["symbol"] for d in h["v1_without_market_data"]] == ["BBB.NS", "DEAD.NS"]   # BBB: no snapshot at all
    assert h["outside_v1_but_in_v2"] == ["OUT.NS"]
