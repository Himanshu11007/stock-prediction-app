"""
tests/test_news_e2e.py — the complete research flow, end to end, with fake
HTTP only:

  providers (official RBI RSS with zone-less IST times, Marketaux with entity
  tags and a syndicated copy, a rate-limited GDELT, an unconfigured Alpha
  Vantage) -> NEWS_INGESTION job -> pipeline (timestamps, dedup / grouping,
  stock mapping) -> v2 TOMORROW_EOD snapshot (news-first) -> admin API
  (/admin/news/candidates, /admin/news/status, /predictions)

and the API payloads are checked against the contract fixtures in
tests/contracts/, which the web tests (StockLens.Web) render. Set
UPDATE_CONTRACTS=1 to regenerate the fixtures after a deliberate contract
change.
"""
import datetime as dt
import json
import os
from pathlib import Path

from sqlmodel import Session, select

from catalysts import adapters as ad, providers as prov
from db.models.news import EventEntity, NewsArticle
from db.models.prediction import Prediction
from db.models.stock import Company
from prediction_v2 import service, universe
from scheduling import jobs
from tests.test_prediction_api import ADMIN, client, db  # noqa: F401  (fixtures)
from tests.test_prediction_v2 import FRI, SYMS, eod, market

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UTC = dt.timezone.utc
NOW = dt.datetime.combine(FRI, dt.time(14, 0), tzinfo=IST).astimezone(UTC)
CONTRACTS = Path(__file__).parent / "contracts"

RBI_FEED = (b'<?xml version="1.0"?><rss><channel><item><title>RBI announces special window for public sector oil '
            b'marketing companies</title><link>https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1</link>'
            b'<pubDate>Fri, 09 Oct 2026 13:05:00</pubDate></item></channel></rss>')


def _mx(i, url):
    return {"uuid": f"mx-{i}", "title": "Sone Steelworks quarterly results: net profit jumps to a record, beats estimates",
            "url": url, "published_at": "2026-10-09T08:00:00.000000Z", "source": "reuters.com",
            "description": "Sone Steelworks reported record quarterly profit.",
            "entities": [{"symbol": "S1.NS", "name": "Sone Steelworks", "exchange": "NSE", "country": "in",
                          "type": "equity", "match_score": 55.0}]}


MX_BODY = json.dumps({"meta": {"found": 2, "returned": 2, "limit": 3, "page": 1},
                      "data": [_mx(1, "https://www.reuters.com/sone-results"),
                               _mx(2, "https://www.livemint.com/sone-results-syndicated")]}).encode()


def _providers():
    return [prov.RssProvider({"rbi_press": ("https://www.rbi.org.in/pressreleases_rss.xml", "Reserve Bank of India")},
                             get=lambda url, t=20.0, h=None: (200, RBI_FEED), sleep=lambda s: None, name="official_rss"),
            ad.MarketauxProvider(api_token="MX-FAKE", get=lambda url, t=20.0, h=None: (200, MX_BODY), sleep=lambda s: None),
            prov.GdeltProvider(get=lambda url, t=20.0, h=None: (429, b""), sleep=lambda s: None),
            ad.AlphaVantageNewsProvider(api_key="", sleep=lambda s: None)]


NAMES = {"S1.NS": "Sone Steelworks Limited", "S2.NS": "Stwo Pharma Limited"}


def _universe(db):
    with Session(db) as s:
        for i, sym in enumerate(SYMS):
            s.add(Company(symbol=sym, name=NAMES.get(sym, sym), sector="Energy" if i < 5 else "IT"))
        s.commit()
        universe.add_symbols(s, SYMS)


def _shape(x):
    """Key structure of a JSON value (lists: the union over their items)."""
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, list):
        merged: dict = {}
        for item in x:
            s = _shape(item)
            if isinstance(s, dict):
                merged.update(s)
        return [merged] if merged else []
    return None


def _missing(contract, actual, path=""):
    out = []
    if isinstance(contract, dict):
        for k, v in contract.items():
            if not isinstance(actual, dict) or k not in actual:
                out.append(f"{path}.{k}")
            else:
                out += _missing(v, actual[k], f"{path}.{k}")
    elif isinstance(contract, list) and contract and isinstance(actual, list) and actual:
        out += _missing(contract[0], actual[0], f"{path}[]")
    return out


def _contract(name: str, payload: dict):
    f = CONTRACTS / name
    if os.environ.get("UPDATE_CONTRACTS") == "1" or not f.exists():
        CONTRACTS.mkdir(exist_ok=True)
        f.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    contract = _shape(json.loads(f.read_text(encoding="utf-8")))
    missing = _missing(contract, _shape(payload))
    assert not missing, f"API no longer provides fields the web relies on ({name}): {missing}"


def test_ingestion_to_screen_snapshot_api_and_contract(client, db):
    _universe(db)

    # 1. ingestion: every provider handled; nothing fabricated for the failing / unconfigured ones
    out = jobs.news_ingestion_job(db, now=NOW, providers=_providers())
    st = {r["provider"]: r["status"] for r in out["providers"]}
    assert st == {"official_rss": "COMPLETED", "marketaux": "COMPLETED", "gdelt": "RATE_LIMITED",
                  "alphavantage": "NOT_CONFIGURED"} and out["degraded"] is True

    with Session(db) as s:
        arts = {a.provider_article_id: a for a in s.exec(select(NewsArticle)).all()}
        rbi = next(a for a in arts.values() if a.provider == "official_rss")
        # 2. timestamps normalised: RBI's zone-less 13:05 is IST (07:35 UTC); original kept
        assert rbi.published_at.replace(tzinfo=UTC) == dt.datetime(2026, 10, 9, 7, 35, tzinfo=UTC)
        assert (rbi.published_raw, rbi.published_tz) == ("Fri, 09 Oct 2026 13:05:00", "NONE->IST")
        mx = [a for a in arts.values() if a.provider == "marketaux"]
        assert len(mx) == 2 and len({a.event_id for a in mx}) == 1              # syndicated copy: one event
        assert all(a.published_tz == "Z" and a.ingested_at for a in mx)
        link = s.exec(select(EventEntity).where(EventEntity.symbol == "S1.NS", EventEntity.relation == "SUBJECT")).first()
        assert link.evidence["source"] in ("provider_tag", "title")              # mapping evidence stored

    # 3. the v2 snapshot uses the combined news
    snap = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: market(FRI, SYMS))
    assert snap["status"] == "COMPLETED"
    with Session(db) as s:
        p1 = s.exec(select(Prediction).where(Prediction.symbol == "S1.NS")).one()
    assert p1.direction == "UP" and p1.setup_type == "NEWS_CATALYST"

    # 4. API: the screen for tomorrow, next to the v2 call; status of the inputs
    r = client.get(f"/api/v1/admin/news/candidates?session=tomorrow&as_of={eod(FRI).isoformat().replace('+', '%2B')}",
                   headers=ADMIN())
    assert r.status_code == 200
    body = r.json()
    data = body["data"]
    rows = {x["symbol"]: x for x in data["candidates"]}
    s1 = rows["S1.NS"]
    assert s1["call"] == "UP" and s1["v2"]["prediction_id"] == p1.prediction_id and s1["v2"]["direction"] == "UP"
    a = s1["evidence"][0]["articles"][0]
    assert a["url"] == "https://www.reuters.com/sone-results" and a["provider"] == "marketaux"
    assert a["published_at"].startswith("2026-10-09T08:00") and a["ingested_at"].startswith("2026-10-09T08:30")
    assert s1["confidence"] is None and "not calibrated" in data["confidence_note"]
    for sym, row in rows.items():                              # no direction without evidence
        if row["call"] in ("UP", "DOWN"):
            assert row["evidence"], sym
        else:
            assert row["exclusion_reasons"], sym
    assert data["snapshot"]["run_type"] == "TOMORROW_EOD" and data["session"] == "tomorrow"
    health = {p["provider"]: p["health"] for p in data["news_status"]["providers"]}
    assert health["gdelt"] == "FAILING" and health["alphavantage"] == "NOT_CONFIGURED"
    assert data["news_status"]["data_basis"] in ("REAL", "STALE")                # real ingestion, possibly old
    st = client.get("/api/v1/admin/news/status", headers=ADMIN()).json()
    assert st["data"]["coverage"]["universe"] == len(SYMS)          # the endpoint counts against the real clock
    # coverage at the test's own clock (the endpoint uses "now", so its window moves with real time)
    from catalysts import status as news_status_mod
    with Session(db) as s:
        cov = news_status_mod.news_status(s, NOW + dt.timedelta(hours=2), {"enabled": [], "refused": {}},
                                          list(SYMS))["coverage"]
    assert cov["universe"] == len(SYMS) and "S1.NS" not in cov["without_news"] and cov["with_news"] >= 1
    _contract("news_candidates_tomorrow.json", body)
    _contract("news_status.json", st)
    # the v2 snapshot and its explanation, as the web prediction pages receive them
    snap_api = client.get("/api/v1/predictions?session=tomorrow&page_size=500", headers=ADMIN()).json()
    assert {p["prediction_id"] for p in snap_api["data"]["predictions"]} >= {p1.prediction_id}
    detail = client.get(f"/api/v1/predictions/{p1.prediction_id}", headers=ADMIN()).json()
    news = detail["data"]["prediction"]["features"]["news"]
    assert news["status"] == "DIRECTIONAL" and news["evidence"][0]["articles"][0]["url"] == a["url"]
    assert news["confidence"] is None
    _contract("predictions_tomorrow.json", snap_api)
    _contract("prediction_detail.json", detail)


def test_without_any_ingestion_the_screen_says_unavailable(client, db):
    _universe(db)
    data = client.get("/api/v1/admin/news/candidates?session=today", headers=ADMIN()).json()["data"]
    assert data["news_status"]["data_basis"] == "UNAVAILABLE" and data["snapshot"] is None
    assert data["counts"]["UP"] == data["counts"]["DOWN"] == 0
    assert all(r["call"] == "NO_CALL" and r["inputs"]["prices"] == "UNAVAILABLE" for r in data["candidates"])
