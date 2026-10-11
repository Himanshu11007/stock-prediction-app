"""
tests/test_news_candidates.py — the explainable news screen: UP / DOWN only
with decisive, fresh, credible evidence; NEUTRAL with reasons; NO_CALL
without news or prices; evidence carries links, all three timestamps and
the mapping evidence; sentiment is kept apart from the call. No network.
"""
import datetime as dt

from sqlmodel import Session, select

from catalysts import candidates, impact
from db.models.news import EventEntity, NewsArticle
from tests.test_catalysts import T0, _ingest, art, news_db  # noqa: F401  (fixture)
from tests.test_prediction_v2 import db  # noqa: F401  (fixture)

COMPANIES = [("IOC.NS", "Energy"), ("ONGC.NS", "Energy"), ("CHENNPETRO.NS", "Energy")]


def test_screen_calls_reasons_evidence_and_provenance(news_db):
    _ingest(news_db, [art(1, "Indian Oil quarterly results: net profit jumps to a record, beats estimates")],
            T0 + dt.timedelta(minutes=30))
    with Session(news_db) as s:
        out = candidates.screen(s, T0 + dt.timedelta(hours=2), COMPANIES)
        link = s.exec(select(EventEntity).where(EventEntity.symbol == "IOC.NS", EventEntity.relation == "SUBJECT")).first()
        stored = s.exec(select(NewsArticle)).one()
    by = {r["symbol"]: r for r in out["candidates"]}
    ioc = by["IOC.NS"]
    assert ioc["call"] == "UP" and ioc["expected_return_direction"] == "UP" and ioc["exclusion_reasons"] == []
    assert ioc["scores"]["credibility"] == 0.9 and ioc["scores"]["novelty"] == 1.0 and 0 < ioc["scores"]["freshness"] <= 1
    assert ioc["horizon_sessions"] == 3 and "Energy" in ioc["affected_sectors"]
    assert "priced-in check not performed (no price data supplied)" in ioc["risks"]       # stated, not assumed
    a = ioc["evidence"][0]["articles"][0]
    assert a["url"].startswith("https://www.reuters.com/") and a["published_at"] and a["ingested_at"] and a["available_at"]
    assert a["available_at"] == stored.effective_available_at.replace(tzinfo=dt.timezone.utc).isoformat()
    assert ioc["evidence"][0]["mapping"]["evidence"]["source"] == "title"
    assert link.evidence["matched"] == "indian oil"                                        # why IOC is linked
    assert by["CHENNPETRO.NS"]["call"] in ("NEUTRAL", "NO_CALL") and by["CHENNPETRO.NS"]["exclusion_reasons"]
    assert out["counts"]["UP"] == 1 and out["mode"] == "SHADOW" and ioc["confidence"] is None


def test_no_news_no_prices_stale_and_priced_in_never_get_a_direction(news_db):
    _ingest(news_db, [art(1, "Indian Oil quarterly results: net profit jumps to a record, beats estimates")],
            T0 + dt.timedelta(minutes=30))
    with Session(news_db) as s:
        early = candidates.screen(s, T0 + dt.timedelta(minutes=10), COMPANIES)
        noprice = candidates.screen(s, T0 + dt.timedelta(hours=2), COMPANIES, has_prices=lambda sym: sym != "IOC.NS")
        stale = candidates.screen(s, T0 + dt.timedelta(days=3, hours=12), COMPANIES)
        moved = candidates.screen(s, T0 + dt.timedelta(hours=2), COMPANIES,
                                  price_move=lambda sym: (lambda since: (0.06, 0.02)))
    r = {x["symbol"]: x for x in early["candidates"]}["IOC.NS"]
    assert r["call"] == "NO_CALL" and "no material news" in r["exclusion_reasons"][0]   # not yet available
    r = {x["symbol"]: x for x in noprice["candidates"]}["IOC.NS"]
    assert r["call"] == "NO_CALL" and r["exclusion_reasons"] == ["no usable price data for this stock"]
    r = {x["symbol"]: x for x in stale["candidates"]}["IOC.NS"]
    assert r["call"] in ("NEUTRAL", "NO_CALL") and r["expected_return_direction"] is None
    r = {x["symbol"]: x for x in moved["candidates"]}["IOC.NS"]
    assert r["call"] == "NEUTRAL" and any("priced in" in x for x in r["risks"])
    assert r["news_sentiment"] > 0 and r["expected_return_direction"] is None    # positive wording != positive return


def test_freshness_horizon_comes_from_the_category():
    assert impact.IMPACT_SESSIONS["EARNINGS"] == 3 and impact.IMPACT_SESSIONS["ECONOMIC_DATA"] == 1
