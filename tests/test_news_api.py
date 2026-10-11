"""
tests/test_news_api.py — news API: access control (shadow), facts vs
inference separation, market summary, hypotheses, stock-linked events, and
the audited admin ingestion trigger.
"""
import datetime as dt

from sqlmodel import Session, select

from catalysts import pipeline, providers as prov
from db.models.admin import AdminAuditLog
from db.models.stock import Company
from prediction_v2 import universe
from tests.test_prediction_api import ADMIN, H, client, db  # noqa: F401  (fixtures)

NOW = dt.datetime.now(dt.timezone.utc)


class _Wire:
    name = "testwire"

    def fetch(self, since, until):
        return [prov.RawArticle("n1", "https://www.reuters.com/brent", "Brent crude surges 6% after attack on tankers",
                                NOW - dt.timedelta(hours=2), excerpt="Oil prices jumped in Asian trade."),
                prov.RawArticle("n2", "https://www.rbi.org.in/p/2", "RBI unexpectedly cuts repo rate by 25 basis points",
                                NOW - dt.timedelta(hours=1))]


def _seed(db):
    with Session(db) as s:
        s.add(Company(symbol="IOC.NS", name="Indian Oil Corporation Limited", sector="Energy",
                      industry="Oil & Gas Refining & Marketing"))
        s.add(Company(symbol="DLF.NS", name="DLF Limited", sector="Real Estate", industry="Real Estate - Development"))
        s.commit()
        universe.add_symbols(s, ["IOC.NS", "DLF.NS"])
        pipeline.ingest(s, _Wire(), NOW - dt.timedelta(days=1), NOW, now=NOW)


def test_news_routes_are_admin_only_in_shadow_mode(client, db):
    for path in ("/api/v1/news/events", "/api/v1/news/market-summary", "/api/v1/news/hypotheses"):
        assert client.get(path, headers=H()).status_code == 403
        assert client.get(path, headers=ADMIN()).status_code == 200
        assert client.get(path).status_code == 401
    assert client.get("/api/v1/admin/news/ingestion-runs", headers=H()).status_code == 403


def test_events_separate_facts_from_inference(client, db):
    _seed(db)
    data = client.get("/api/v1/news/events", headers=ADMIN()).json()["data"]
    assert data["shadow"] is True and "not predictions" in data["notice"]
    brent = next(e for e in data["events"] if "Brent" in e["title"])
    assert brent["sources"][0]["tier"] == "TIER1" and brent["sources"][0]["ingested_at"]
    assert brent["classification"]["category"] == "COMMODITY"
    assert {"type": "COMMODITY", "key": "CRUDE_OIL", "symbol": None, "relation": "MENTIONED"} in brent["facts"]["entities"]
    ioc = next(x for x in brent["inference"]["exposures"] if x["symbol"] == "IOC.NS")
    assert ioc["sign"] == -1 and ioc["hypothesis_id"] == "H-CRUDE-OMC" and "marketing margins" in ioc["mechanism"]
    detail = client.get(f"/api/v1/news/events/{brent['event_id']}", headers=ADMIN()).json()["data"]
    assert detail["articles"][0]["excerpt"] == "Oil prices jumped in Asian trade." and detail["classification_history"]
    assert client.get("/api/v1/news/events/99999", headers=ADMIN()).status_code == 404
    only = client.get("/api/v1/news/events?symbol=DLF.NS", headers=ADMIN()).json()["data"]["events"]
    assert [e["title"] for e in only] == ["RBI unexpectedly cuts repo rate by 25 basis points"]


def test_market_summary_groups_exposed_stocks_by_sign(client, db):
    _seed(db)
    data = client.get("/api/v1/news/market-summary", headers=ADMIN()).json()["data"]
    cats = {c["category"]: c for c in data["catalysts"]}
    assert "IOC.NS" in cats["COMMODITY"]["exposed"]["negative"]
    assert "DLF.NS" in cats["MONETARY_POLICY"]["exposed"]["positive"]
    assert data["coverage"]["latest_ingestion"]["status"] == "COMPLETED"
    assert "not connected" in data["coverage"]["available_sources"]


def test_hypotheses_and_stock_linked_events(client, db):
    _seed(db)
    hyps = client.get("/api/v1/news/hypotheses", headers=ADMIN()).json()["data"]["hypotheses"]
    omc = next(h for h in hyps if h["id"] == "H-CRUDE-OMC")
    assert omc["status"] in ("SUPPORTED", "INCONCLUSIVE", "UNVALIDATED", "CONTRARY") and omc["factor"] == "BZ=F"
    ev = client.get("/api/v1/stocks/IOC.NS/events", headers=ADMIN()).json()["data"]
    assert ev["provider_configured"] is True and ev["linked_news_events"][0]["inferred"] is True
    assert "events" in ev                                                  # existing field kept


def test_admin_ingest_is_audited(client, db, monkeypatch):
    from scheduling import jobs
    started = []
    monkeypatch.setattr(jobs, "news_ingestion_job", lambda engine, *a, **k: started.append(engine))
    r = client.post("/api/v1/admin/news/ingest", headers=ADMIN())
    assert r.status_code == 202
    with Session(db) as s:
        assert s.exec(select(AdminAuditLog).where(AdminAuditLog.action == "NEWS_INGESTION_TRIGGERED")).first()


def test_prediction_detail_keeps_current_price_separate_from_the_reference(client, db):
    from tests.test_prediction_api import _run
    from db.models.market import PriceQuote
    with Session(db) as s:
        _run(s, "R-T", "TOMORROW_EOD", dt.date(2026, 10, 12))
        s.add(PriceQuote(symbol="AAA.NS", price=104.5, bar_date=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                         as_of=dt.datetime.now(dt.timezone.utc), status="DELAYED_INTRADAY"))
        s.commit()
    d = client.get("/api/v1/predictions/R-T-AAA.NS", headers=ADMIN()).json()["data"]
    assert d["prediction"]["reference_price"] == 100.0 and d["current_price"]["current_price"] == 104.5
    assert d["current_price"]["current_price_as_of"] and "not recalculated" in d["current_price"]["note"]
