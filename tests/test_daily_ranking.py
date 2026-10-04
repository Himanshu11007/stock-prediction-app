"""
tests/test_daily_ranking.py — daily scheduled ranking, current prices kept
separate from ranking reference prices, and the notifications that follow a
new ranking run (scheduling/jobs.py, prices/service.py, GET /api/v1/top-picks).

The ranking methodology itself is covered by test_ranking*.py; here the engine
run is replaced by a fake that publishes a prepared ranking, so the tests can
control ranks and reference prices exactly (Day-1 / Day-2 scenario).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
import engine_runs.service as runs
import masters.service as masters
import notifications.service as notify
import prices.service as price_service
from api.main import app
from auth.security import create_access_token
from db.models.market import (EngineRun, MarketRegimeSnapshot, PriceQuote, RankingSnapshot, ScheduledJobRun,
                              StockAnalysisResult)
from db.models.notifications import Notification, NotificationRun
from db.models.stock import Company, StockUniverseMember
from db.session import get_session
from notifications.push import ApnsProvider, FcmProvider, PushRouter, set_router
from ranking import tracking
from scheduling import jobs
from utils.market_session import IST

UTC = dt.timezone.utc
SYMBOLS = ["RELIANCE.NS"] + [f"S{i:02d}.NS" for i in range(9)]


def _trading_days_before_today():
    """Two consecutive weekdays (Day 1, Day 2), Day 2 the latest weekday
    before today, so prices of Day 2 are recent (not STALE) when the test runs."""
    d = dt.datetime.now(IST).date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    d1 = d - dt.timedelta(days=1)
    while d1.weekday() >= 5:
        d1 -= dt.timedelta(days=1)
    return d1, d


DAY1, DAY2 = _trading_days_before_today()


def at(day, hh, mm=0):
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=IST)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
        for sym in SYMBOLS:
            name = "Reliance Industries" if sym == "RELIANCE.NS" else f"Stock {sym[1:3]}"
            s.add(Company(symbol=sym, name=name, sector="Energy", industry="Oil & Gas"))
            s.add(StockUniverseMember(symbol=sym, category="Large Cap"))
        s.commit()
        admin = auth_service.get_user_by_email(s, "admin@example.com")
        masters.set_config(s, admin, "top_picks.limit", 5)
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


def _fq(passes=12):
    return {"score": passes / 18 * 100, "coverage": 0.9, "counts": {"PASS": passes, "FAIL": 18 - passes},
            "checks": [], "summary": f"{passes} passed"}


def _comps(score):
    keys = ("quality", "valuation", "financial_health", "technical_trend", "momentum", "risk", "sector_outlook",
            "market_regime", "ml_signal")
    return {k: {"score": score, "weight": 0 if k == "ml_signal" else 10, "label": k, "basis": "test"} for k in keys}


def _publish(s, run_id, day, scores, ref_prices, status="COMPLETED", started=None):
    """Write a ranking run the way the engine does: results, regime and the
    frozen ranking snapshots with reference prices."""
    started = started or at(day, 16, 30).astimezone(UTC)
    run = s.exec(select(EngineRun).where(EngineRun.run_id == run_id)).first()
    if run is None:
        run = EngineRun(run_id=run_id, kind="RANKING", started_at=started, engine_version="ranking-v1.0",
                        fqvf_version="fqvf-v1.0", config={})
    ranked = sorted(scores, key=lambda x: -scores[x])
    for sym, sc in scores.items():
        s.add(StockAnalysisResult(
            run_id=run_id, symbol=sym, computed_at=run.started_at, stockai_score=sc, score_coverage=0.95,
            eligible=True, ineligible_reasons=[], rank=ranked.index(sym) + 1, components=_comps(sc), fqvf=_fq(),
            freshness={"market_data_as_of": day.isoformat()}, engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0"))
    s.add(MarketRegimeSnapshot(run_id=run_id, regime="Bullish", regime_score=0.6, computed_at=run.started_at,
                               as_of_date=day.isoformat()))
    s.flush()
    if status in ("COMPLETED", "COMPLETED_WITH_ERRORS"):
        tracking.record_run_snapshots(s, run_id, {k: (day.isoformat(), v) for k, v in ref_prices.items()},
                                      "Bullish", 25000.0)
    run.status, run.finished_at = status, (None if status == "RUNNING" else run.started_at)
    run.processed = run.succeeded = len(scores)
    s.add(run)
    s.commit()


def _scores(top3=None, reliance=80.0):
    sc = {f"S{i:02d}.NS": 95.0 - i for i in range(9)}
    sc["RELIANCE.NS"] = reliance
    return sc


DAY0_SCORES = _scores(reliance=60.0)                                  # Reliance not a Top Candidate
DAY1_SCORES = {**_scores(), "RELIANCE.NS": 93.5}                     # 95, 94, *93.5*, 93 ... -> rank 3
DAY2_SCORES = {**_scores(), "RELIANCE.NS": 96.0}                     # rank 1


def _frame(points):
    """points: [(date, close)] -> provider-shaped daily bars."""
    idx = pd.to_datetime([d for d, _ in points])
    close = [c for _, c in points]
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": [1_000_000.0] * len(points)}, index=idx)


class Market:
    """Fake price provider: the latest daily bar of each stock as of `day`."""

    def __init__(self):
        self.closes = {}          # day -> {symbol: close}
        self.day = None
        self.calls = 0

    def set(self, day, closes):
        self.closes[day] = closes
        self.day = day

    def fetch(self, symbols):
        self.calls += 1
        days = sorted(d for d in self.closes if d <= self.day)
        out = {}
        for sym in symbols:
            pts = [(d, self.closes[d][sym]) for d in days if sym in self.closes[d]]
            if pts:
                out[sym] = _frame(pts)
        return out

    def index(self):
        return _frame([(self.day, 25000.0)])


@pytest.fixture()
def market(monkeypatch):
    m = Market()
    monkeypatch.setattr(price_service, "_default_fetch", m.fetch)
    return m


@pytest.fixture()
def fake_engine(db, monkeypatch):
    """Replace the engine run with one that publishes the prepared ranking of
    the run's trading day (config.ranking_slot)."""
    plan = {}

    def execute(engine, run_id):
        with Session(engine) as s:
            run = s.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()
            day = dt.date.fromisoformat(run.config["ranking_slot"])
            scores, refs, status = plan[day]
            _publish(s, run_id, day, scores, refs, status=status)

    monkeypatch.setattr(runs, "execute_run", execute)
    return plan


def _top(client):
    r = client.get("/api/v1/top-picks", headers=H())
    assert r.status_code == 200, r.text
    return r.json()["data"]


def _item(data, sym):
    return next(i for i in data["items"] if i["symbol"] == sym)


# ── Top Picks reads the stored ranking; prices are separate ─────────────────

def test_top_picks_returns_latest_completed_ranking_without_recalculating(client, db, monkeypatch):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ranking must not be recalculated"))  # noqa: E731
    monkeypatch.setattr(runs, "execute_run", boom)
    monkeypatch.setattr(runs, "create_run", boom)
    monkeypatch.setattr("ranking.service.StockRankingService.rank", boom)
    first, second = _top(client), _top(client)
    assert first["run"]["run_id"] == second["run"]["run_id"] == "R1"
    assert [i["symbol"] for i in first["items"]] == [i["symbol"] for i in second["items"]]
    with Session(db) as s:
        assert len(s.exec(select(EngineRun)).all()) == 1
        assert len(s.exec(select(StockAnalysisResult)).all()) == len(SYMBOLS)


def test_candidate_contract_has_ranking_reference_and_current_price_fields(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {sym: 1000.0 + i for i, sym in enumerate(SYMBOLS)})
        s.add(PriceQuote(symbol="RELIANCE.NS", price=1234.5, bar_date=DAY2.isoformat(),
                         as_of=at(DAY2, 15, 30).astimezone(UTC), status="LAST_CLOSE",
                         fetched_at=dt.datetime.now(UTC)))
        s.commit()
    data = _top(client)
    item = _item(data, "RELIANCE.NS")
    required = {"rank", "symbol", "name", "sector", "industry", "stocklens_score", "stockai_score", "fqvf_score",
                "fqvf_summary", "positives", "risks", "labels", "components", "engine_version", "computed_at",
                "ranking_date", "current_price", "current_price_as_of", "current_price_status", "reference_price",
                "reference_price_as_of", "market_status", "freshness", "freshness_status"}
    assert required <= set(item)
    assert item["stocklens_score"] == item["stockai_score"] == 93.5
    assert item["ranking_date"] == data["ranking_date"] == data["run"]["ranking_date"] == DAY1.isoformat()
    # current price and its timestamp, separate from the frozen reference price
    assert item["current_price"] == 1234.5 and item["current_price_status"] == "LAST_CLOSE"
    assert item["current_price_as_of"] == at(DAY2, 15, 30).astimezone(UTC).isoformat()
    assert item["reference_price"] == 1000.0 and item["reference_price_as_of"] == DAY1.isoformat()
    assert item["market_status"] == data["market_status"]["status"]


def test_missing_current_price_is_not_available_never_the_reference_price(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    item = _item(_top(client), "RELIANCE.NS")     # provider returns nothing (conftest)
    assert item["current_price"] is None and item["current_price_as_of"] is None
    assert item["current_price_status"] == "NOT_AVAILABLE"
    assert item["reference_price"] == 1200.0
    other = _item(_top(client), "S00.NS")         # no reference either: null, never invented
    assert other["reference_price"] is None and other["current_price"] is None


def test_provider_failure_keeps_previous_price_with_its_timestamp(db):
    with Session(db) as s:
        good = lambda syms: {"RELIANCE.NS": _frame([(DAY1, 1200.0)])}  # noqa: E731
        price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY1, 17), good)
        boom = lambda syms: (_ for _ in ()).throw(ConnectionError("provider down"))  # noqa: E731
        res = price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 17), boom)
        q = s.get(PriceQuote, "RELIANCE.NS")
        assert res == {"updated": 0, "failed": 1}
        assert q.price == 1200.0 and q.bar_date == DAY1.isoformat() and "provider down" in q.last_error
        empty = price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 17, 10), lambda syms: {})
        assert empty["failed"] == 1 and s.get(PriceQuote, "RELIANCE.NS").price == 1200.0


def test_intraday_price_is_delayed_not_live_and_becomes_last_close_after_close(db):
    with Session(db) as s:
        bars = lambda syms: {"RELIANCE.NS": _frame([(DAY2, 1310.0)])}  # noqa: E731
        price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 11), bars)
        p = price_service.quote_payload(s.get(PriceQuote, "RELIANCE.NS"), at(DAY2, 11))
        assert p["current_price_status"] == "DELAYED_INTRADAY" and "LIVE" not in str(p).upper()
        assert p["current_price_as_of"] == at(DAY2, 11).astimezone(UTC).isoformat()
        assert price_service.needs_refresh(s.get(PriceQuote, "RELIANCE.NS"), at(DAY2, 16, 30), [])
        price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 16, 30), bars)
        assert s.get(PriceQuote, "RELIANCE.NS").status == "LAST_CLOSE"


def test_holiday_placeholder_bars_are_not_prices(db):
    with Session(db) as s:
        df = _frame([(DAY1, 1200.0), (DAY2, 1200.0)])
        df.loc[df.index[-1], "Volume"] = 0.0           # Yahoo's zero-volume holiday bar
        price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 17), lambda syms: {"RELIANCE.NS": df})
        assert s.get(PriceQuote, "RELIANCE.NS").bar_date == DAY1.isoformat()


def test_old_price_is_reported_stale():
    q = PriceQuote(symbol="X", price=10.0, bar_date="2020-01-01", status="LAST_CLOSE",
                   as_of=dt.datetime(2020, 1, 1, 10, tzinfo=UTC), fetched_at=dt.datetime(2020, 1, 1, tzinfo=UTC))
    assert price_service.quote_payload(q)["current_price_status"] == "STALE"


def test_reference_price_is_immutable(db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
        # re-recording the run's snapshots never overwrites them
        tracking.record_run_snapshots(s, "R1", {"RELIANCE.NS": (DAY2.isoformat(), 1400.0)}, "Bullish", 1.0)
        s.commit()
        # neither does a new current price
        price_service.refresh_quotes(s, ["RELIANCE.NS"], at(DAY2, 17),
                                     lambda syms: {"RELIANCE.NS": _frame([(DAY2, 1400.0)])})
        snap = s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == "R1",
                                                    RankingSnapshot.symbol == "RELIANCE.NS")).one()
        assert snap.reference_price == 1200.0 and snap.reference_date == DAY1.isoformat()
        result = s.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == "R1",
                                                          StockAnalysisResult.symbol == "RELIANCE.NS")).one()
        from ranking import presenter
        assert presenter.reference_prices(s, [result])["RELIANCE.NS"]["reference_price"] == 1200.0


def test_current_price_changes_while_ranking_stays_the_same(client, db, market):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    market.set(DAY1, {"RELIANCE.NS": 1200.0})
    jobs.prices_job(db, at(DAY1, 16, 30))
    before = _item(_top(client), "RELIANCE.NS")
    market.set(DAY2, {"RELIANCE.NS": 1250.0})
    jobs.prices_job(db, at(DAY2, 16, 30))
    data = _top(client)
    after = _item(data, "RELIANCE.NS")
    assert before["current_price"] == 1200.0 and after["current_price"] == 1250.0
    assert data["run"]["run_id"] == "R1" and after["rank"] == before["rank"] == 3
    assert after["stocklens_score"] == before["stocklens_score"] and after["reference_price"] == 1200.0


def test_stock_analysis_has_current_price_separate_from_reference(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
        s.add(PriceQuote(symbol="RELIANCE.NS", price=1400.0, bar_date=DAY2.isoformat(), status="LAST_CLOSE",
                         as_of=at(DAY2, 15, 30).astimezone(UTC), fetched_at=dt.datetime.now(UTC)))
        s.commit()
    data = client.get("/api/v1/stocks/RELIANCE.NS/analysis", headers=H()).json()["data"]
    assert data["current_price"]["current_price"] == 1400.0
    assert data["current_price"]["current_price_status"] == "LAST_CLOSE"
    assert data["ranking_date"] == DAY1.isoformat()


# ── publication: only completed runs, failed runs keep the previous ranking ──

def test_new_run_is_visible_only_after_it_completes(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
        _publish(s, "R2", DAY2, DAY2_SCORES, {"RELIANCE.NS": 1400.0}, status="RUNNING")
    data = _top(client)
    assert data["run"]["run_id"] == "R1" and _item(data, "RELIANCE.NS")["rank"] == 3
    with Session(db) as s:
        run = s.exec(select(EngineRun).where(EngineRun.run_id == "R2")).one()
        tracking.record_run_snapshots(s, "R2", {"RELIANCE.NS": (DAY2.isoformat(), 1400.0)}, "Bullish", 1.0)
        run.status, run.finished_at = "COMPLETED", run.started_at
        s.add(run)
        s.commit()
    data = _top(client)
    assert data["run"]["run_id"] == "R2" and _item(data, "RELIANCE.NS")["rank"] == 1


def _engine_with_universe(monkeypatch, failing):
    """Real engine run with a fake provider; `failing` stocks have no data."""
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    with Session(e) as s:
        for i in range(10):
            s.add(Company(symbol=f"T{i:02d}.NS", name=f"Test {i}"))
            s.add(StockUniverseMember(symbol=f"T{i:02d}.NS", category="Large Cap"))
        s.commit()

    def prices(symbols, period="2y"):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
        close = np.maximum(100 + np.cumsum(rng.normal(0, 1, 520)), 5)
        df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                           "Volume": np.full(520, 1_500_000.0)}, index=idx)
        return {sym: (None if sym in failing() else df) for sym in symbols}

    def fundamentals(sym):
        if sym in failing():
            raise RuntimeError("provider down")
        return {"status": "OK", "error": None, "fiscal_period_end": "2026-03-31",
                "data": {"sector": "S", "industry": "I", "trailing_pe": 12, "price_to_book": 1.5, "annual": [],
                         "issues": []},
                "fetched_at": dt.datetime.now(UTC)}

    monkeypatch.setattr(runs.provider, "fetch_price_history", prices)
    monkeypatch.setattr(runs.provider, "fetch_fundamentals", fundamentals)
    return e


def _engine_run(e):
    with Session(e) as s:
        run_id = runs.create_run(s, kind="RANKING", triggered_by=None, config={
            "symbols": None, "limit": None, "include_ml": False, "refresh_fundamentals": False,
            "weights": None, "rules": None}).run_id
    runs.execute_run(e, run_id)
    with Session(e) as s:
        return s.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()


def test_failed_run_keeps_the_previous_ranking(monkeypatch):
    from notifications import detector
    down = set()
    e = _engine_with_universe(monkeypatch, lambda: down)
    good = _engine_run(e)
    assert good.status in ("COMPLETED", "COMPLETED_WITH_ERRORS")
    down.update(f"T{i:02d}.NS" for i in range(8))                  # provider outage: 8 of 10 stocks
    bad = _engine_run(e)
    assert bad.status == "FAILED" and any(err["stage"] == "quality_gate" for err in bad.errors)
    with Session(e) as s:
        assert detector.latest_full_run(s).run_id == good.run_id
        assert runs.latest_result(s, "T00.NS").run_id == good.run_id
        assert not s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == bad.run_id)).all()
        assert len(s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == good.run_id)).all()) == 10
        # the previous run's results are untouched
        assert len(s.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == good.run_id)).all()) == 10


def test_failed_scheduled_run_sends_no_ranking_notifications(db, fake_engine, market):
    with Session(db) as s:
        _publish(s, "R0", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    fake_engine[DAY2] = (DAY2_SCORES, {"RELIANCE.NS": 1400.0}, "FAILED")
    market.set(DAY2, {"RELIANCE.NS": 1400.0})
    out = jobs.ranking_job(db, at(DAY2, 17), market.index)
    assert out["status"] == "FAILED"
    with Session(db) as s:
        slot = s.exec(select(ScheduledJobRun).where(ScheduledJobRun.slot == DAY2.isoformat())).one()
        assert slot.status == "FAILED" and slot.run_id == out["run_id"]
        assert not s.exec(select(NotificationRun)).all() and not s.exec(select(Notification)).all()
        from notifications import detector
        assert detector.latest_full_run(s).run_id == "R0"


# ── scheduler: calendar, duplicates ──────────────────────────────────────────

def test_non_trading_day_creates_no_ranking(db, fake_engine):
    saturday = DAY2 + dt.timedelta(days=(5 - DAY2.weekday()) % 7 or 7)
    no_call = lambda: (_ for _ in ()).throw(AssertionError("provider must not be called"))  # noqa: E731
    for _ in range(2):
        assert jobs.ranking_job(db, at(saturday, 17), no_call)["status"] == "SKIPPED"
    with Session(db) as s:
        masters.set_config(s, auth_service.get_user_by_email(s, "admin@example.com"), "market.holidays",
                           [DAY2.isoformat()])
    assert "not a trading day" in jobs.ranking_job(db, at(DAY2, 17), no_call)["reason"]
    with Session(db) as s:
        assert not s.exec(select(EngineRun)).all()
        rows = s.exec(select(ScheduledJobRun)).all()
        assert sorted((r.slot, r.status) for r in rows) == sorted([(saturday.isoformat(), "SKIPPED"),
                                                                    (DAY2.isoformat(), "SKIPPED")])
    assert jobs.prices_job(db, at(saturday, 11))["status"] == "SKIPPED"


def test_duplicate_scheduler_triggers_run_the_ranking_once(db, fake_engine, market):
    fake_engine[DAY1] = (DAY1_SCORES, {"RELIANCE.NS": 1200.0}, "COMPLETED")
    market.set(DAY1, {"RELIANCE.NS": 1200.0})
    first = jobs.ranking_job(db, at(DAY1, 17), market.index)
    second = jobs.ranking_job(db, at(DAY1, 17, 15), market.index)
    assert first["status"] == "COMPLETED" and second["status"] == "SKIPPED"
    with Session(db) as s:
        assert len(s.exec(select(EngineRun)).all()) == 1
        assert s.exec(select(ScheduledJobRun)).one().status == "COMPLETED"


def test_slot_lock_excludes_concurrent_triggers_and_recovers_abandoned_slots(db):
    t = at(DAY1, 17).astimezone(UTC)
    with Session(db) as s:
        assert jobs.acquire_slot(s, "ranking", "d", t) is not None
        assert jobs.acquire_slot(s, "ranking", "d", t) is None                       # running elsewhere
        assert jobs.acquire_slot(s, "ranking", "d", t + dt.timedelta(hours=4)) is not None   # abandoned
        row = s.exec(select(ScheduledJobRun)).one()
        jobs.finish_slot(s, row, "COMPLETED", {})
        assert jobs.acquire_slot(s, "ranking", "d", t + dt.timedelta(hours=9)) is None     # done
        jobs.acquire_slot(s, "ranking", "e", t)
        row = s.exec(select(ScheduledJobRun).where(ScheduledJobRun.slot == "e")).one()
        for _ in range(2):
            jobs.finish_slot(s, row, "FAILED", {})
            assert jobs.acquire_slot(s, "ranking", "e", t) is not None                 # retry allowed
        jobs.finish_slot(s, row, "FAILED", {})
        assert jobs.acquire_slot(s, "ranking", "e", t) is None                         # max attempts


def test_admin_scheduled_jobs_status_and_trigger(client, db, monkeypatch):
    import scheduling.jobs as sj
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
        s.add(ScheduledJobRun(job="ranking", slot=DAY1.isoformat(), status="COMPLETED", run_id="R1"))
        s.commit()
    assert client.get("/api/v1/admin/scheduled-jobs", headers=H()).status_code == 403
    data = client.get("/api/v1/admin/scheduled-jobs", headers=ADMIN()).json()
    assert data["latest_ranking"]["run_id"] == "R1" and data["latest_ranking"]["ranking_date"] == DAY1.isoformat()
    assert data["jobs"][0]["status"] == "COMPLETED" and "symbols_with_price" in data["current_prices"]
    called = []
    monkeypatch.setattr(sj, "ranking_job", lambda engine: called.append(engine))
    assert client.post("/api/v1/admin/scheduled-jobs/ranking/run", headers=H()).status_code == 403
    r = client.post("/api/v1/admin/scheduled-jobs/ranking/run", headers=ADMIN())
    assert r.status_code == 202
    import time
    for _ in range(50):
        if called:
            break
        time.sleep(0.02)
    assert len(called) == 1


# ── notifications after a new ranking run ────────────────────────────────────

@pytest.fixture()
def unconfigured_push():
    set_router(PushRouter({"fcm": FcmProvider(project_id="", service_account_file=""),
                           "apns": ApnsProvider(key_file="", key_id="", team_id="", bundle_id="")}))
    yield
    set_router(None)


def test_notifications_are_not_duplicated_and_report_missing_provider(db, fake_engine, market, unconfigured_push):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        uid = user.id
        notify.update_preferences(s, user, {"quiet_hours_enabled": False})
        notify.register_device(s, user, "dev-1", "android", "tok")
        _publish(s, "R0", DAY1 - dt.timedelta(days=1), DAY0_SCORES, {"RELIANCE.NS": 1150.0},
                 started=at(DAY1, 9).astimezone(UTC))
    fake_engine[DAY1] = (DAY1_SCORES, {"RELIANCE.NS": 1200.0}, "COMPLETED")
    market.set(DAY1, {"RELIANCE.NS": 1200.0})
    out = jobs.ranking_job(db, at(DAY1, 17), market.index)
    runs.notify_after_run(db, out["run_id"])                     # retried notification stage
    jobs.ranking_job(db, at(DAY1, 17, 30), market.index)         # duplicate trigger
    with Session(db) as s:
        assert len(s.exec(select(NotificationRun).where(NotificationRun.kind == "RANKING_CHANGES")).all()) == 1
        mine = s.exec(select(Notification).where(Notification.user_id == uid)).all()
        assert mine and len({n.dedup_key for n in mine}) == len(mine)
        assert all(n.push_status == "PROVIDER_NOT_CONFIGURED" for n in mine)


def test_daily_summary_only_for_a_new_ranking_run(db, unconfigured_push):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.update_preferences(s, user, {"daily_summary": True, "daily_summary_time": "08:30"})
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
        nxt = DAY1 + dt.timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(days=1)
        assert notify.send_daily_summaries(s, at(nxt, 9))["created"] == 1
        later = nxt + dt.timedelta(days=1)
        while later.weekday() >= 5:
            later += dt.timedelta(days=1)
        res = notify.send_daily_summaries(s, at(later, 9))       # no new ranking since
        assert res["created"] == 0 and "no new ranking" in res["reason"]
        n = s.exec(select(Notification).where(Notification.type == "DAILY_SUMMARY")).one()
        assert DAY1.isoformat() in n.body


# ── the Day-1 / Day-2 scenario ───────────────────────────────────────────────

def test_day1_day2_reliance_scenario(client, db, fake_engine, market, unconfigured_push):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        uid = user.id
        notify.update_preferences(s, user, {"quiet_hours_enabled": False})
        _publish(s, "R0", DAY1 - dt.timedelta(days=1), DAY0_SCORES, {"RELIANCE.NS": 1150.0},
                 started=at(DAY1, 9).astimezone(UTC))
    fake_engine[DAY1] = (DAY1_SCORES, {sym: (1200.0 if sym == "RELIANCE.NS" else 500.0) for sym in SYMBOLS},
                         "COMPLETED")
    fake_engine[DAY2] = (DAY2_SCORES, {sym: (1400.0 if sym == "RELIANCE.NS" else 510.0) for sym in SYMBOLS},
                         "COMPLETED")

    # DAY 1: ranking run after the close: Reliance #3 at a reference price of 1,200
    market.set(DAY1, {sym: (1200.0 if sym == "RELIANCE.NS" else 500.0) for sym in SYMBOLS})
    day1 = jobs.ranking_job(db, at(DAY1, 17), market.index)
    assert day1["status"] == "COMPLETED"
    data = _top(client)
    rel = _item(data, "RELIANCE.NS")
    assert data["run"]["run_id"] == day1["run_id"] and data["ranking_date"] == DAY1.isoformat()
    assert rel["rank"] == 3 and rel["reference_price"] == 1200.0 and rel["current_price"] == 1200.0
    with Session(db) as s:
        new = s.exec(select(Notification).where(Notification.user_id == uid,
                                                Notification.type == "NEW_TOP_CANDIDATE")).all()
        assert any(n.symbol == "RELIANCE.NS" for n in new)

    # DAY 2, before the ranking run: the market moved to 1,400
    market.set(DAY2, {sym: (1400.0 if sym == "RELIANCE.NS" else 510.0) for sym in SYMBOLS})
    jobs.prices_job(db, at(DAY2, 16, 5))                       # the scheduled prices job
    data = _top(client)
    rel = _item(data, "RELIANCE.NS")
    assert data["run"]["run_id"] == day1["run_id"] and data["ranking_date"] == DAY1.isoformat()
    assert rel["rank"] == 3                                    # ranking unchanged
    assert rel["reference_price"] == 1200.0 and rel["reference_price_as_of"] == DAY1.isoformat()
    assert rel["current_price"] == 1400.0 and rel["current_price_status"] == "LAST_CLOSE"
    assert rel["current_price_date"] == DAY2.isoformat()
    assert rel["current_price_as_of"] == at(DAY2, 15, 30).astimezone(UTC).isoformat()

    # DAY 2 ranking run: Top Picks moves to the Day-2 ranking and reference prices
    day2 = jobs.ranking_job(db, at(DAY2, 17), market.index)
    assert day2["status"] == "COMPLETED" and day2["run_id"] != day1["run_id"]
    data = _top(client)
    rel = _item(data, "RELIANCE.NS")
    assert data["run"]["run_id"] == day2["run_id"] and data["ranking_date"] == DAY2.isoformat()
    assert rel["rank"] == 1 and rel["reference_price"] == 1400.0 and rel["reference_price_as_of"] == DAY2.isoformat()
    assert rel["current_price"] == 1400.0
    with Session(db) as s:
        # Day-1 history is untouched
        snap = s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == day1["run_id"],
                                                    RankingSnapshot.symbol == "RELIANCE.NS")).one()
        assert snap.reference_price == 1200.0 and snap.rank == 3
        nrun = s.exec(select(NotificationRun).where(NotificationRun.source_key == day2["run_id"])).one()
        assert nrun.previous_run_id == day1["run_id"]
        slots = s.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == "ranking")).all()
        assert sorted((r.slot, r.status) for r in slots) == [(DAY1.isoformat(), "COMPLETED"),
                                                              (DAY2.isoformat(), "COMPLETED")]
