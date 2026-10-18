"""
tests/test_drishti_announcements.py — Drishti announcements (NSE/BSE filings)
as a news-pipeline provider: asks only about our stocks, important filings by
default, a per-run item cap (1 credit per item), paced requests, the key in a
header only, a unique non-web identifier per filing, exchange credibility,
off by default behind its own switch, and end to end through the pipeline.
"""
import datetime as dt
import json
import urllib.parse

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from catalysts import adapters as ad
from catalysts import pipeline
from catalysts import providers as prov
from catalysts import registry
from db.models.news import EventEntity, NewsArticle
from db.models.prediction import V2UniverseMember
from db.models.stock import Company

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
KEY = "DRISHTI-ANN-SECRET"


def _row(i, sym="TRENT", date="2026-10-18T10:41:00+00:00", **kw):
    return {"id": f"a{i}", "symbol": sym, "company_name": "Trent Limited", "image": None, "date": date,
            "summary": f"Trent reports Q2FY27 revenue of Rs. 5,788 Crs (23% YoY growth) #{i}",
            "category": "Media Release of Financial Results", "related_categories": ["Financial Results"], **kw}


class Http:
    def __init__(self, pages):
        self.pages, self.seen = pages, []

    def __call__(self, url, timeout=20.0, headers=None):
        self.seen.append((url, headers))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return 200, json.dumps(self.pages(q)).encode()


def test_asks_only_about_our_stocks_with_a_cap_and_the_key_in_a_header():
    http = Http(lambda q: {"data": [_row(f"{q['symbols'][0][:3]}{q['page'][0]}{n}") for n in range(int(q["limit"][0]))],
                           "has_next": True})
    sleeps = []
    p = ad.DrishtiAnnouncementsProvider(api_key=KEY, max_items=60, get=http, sleep=sleeps.append)
    p.bind_universe([f"S{i:03d}.NS" for i in range(150)] + ["TRENT.NS", "OLD.BO"])
    items = list(p.fetch(NOW - dt.timedelta(hours=1), NOW))
    assert len(items) == 60                                                    # capped: 1 credit per item
    q1 = urllib.parse.parse_qs(urllib.parse.urlparse(http.seen[0][0]).query)
    assert len(q1["symbols"][0].split(",")) == 100 and "OLD" not in http.seen[-1][0]   # chunks of 100; no BSE-only
    assert q1["important"] == ["true"] and q1["limit"] == ["50"] and q1["from"] == ["2026-10-18T11:00:00Z"]
    assert urllib.parse.parse_qs(urllib.parse.urlparse(http.seen[1][0]).query)["limit"] == ["10"]   # what is left
    assert sleeps == [6.5]                                                     # paced for 10 requests a minute
    assert all(KEY not in u and h["X-API-Key"] == KEY for u, h in http.seen)
    a = items[0]
    assert a.url.startswith("urn:drishti:announcement:") and a.provider_article_id.startswith("ann:")
    assert a.symbols == ("TRENT.NS",) and a.source_domain == "nseindia.com" and a.title.startswith("Trent Limited: Trent")
    assert a.raw["category"] == "Media Release of Financial Results" and a.time_precision == "exact"


def test_refuses_without_a_key_or_a_stock_list_and_maps_errors():
    with pytest.raises(prov.NotConfigured):
        list(ad.DrishtiAnnouncementsProvider(api_key="", symbols=["TRENT"]).fetch(NOW, NOW))
    with pytest.raises(prov.NotConfigured) as e:
        list(ad.DrishtiAnnouncementsProvider(api_key=KEY).fetch(NOW, NOW))       # never the whole market
    assert "stock list" in str(e.value)
    for status, exc in ((401, prov.NotConfigured), (403, prov.NotPermitted), (402, prov.RateLimited),
                        (429, prov.RateLimited), (500, prov.ProviderError)):
        with pytest.raises(exc) as e:
            list(ad.DrishtiAnnouncementsProvider(api_key=KEY, symbols=["TRENT"], sleep=lambda s: None,
                                                 get=lambda u, t=20.0, h=None: (status, b"{}")).fetch(NOW, NOW))
        assert KEY not in str(e.value)
    imp = Http(lambda q: {"data": [], "has_next": False})
    list(ad.DrishtiAnnouncementsProvider(api_key=KEY, symbols=["TRENT"], important=False, get=imp).fetch(NOW, NOW))
    assert "important=false" in imp.seen[0][0]


def test_off_by_default_behind_its_own_switch():
    base = {"NEWS_PROVIDERS": "official_rss,drishti_announcements", "DRISHTI_API_KEY": "k",
            "NEWS_ALLOW_RESEARCH_PROVIDERS": "true"}
    assert "DRISHTI_ANNOUNCEMENTS_ENABLED" in registry.select(base).refused["drishti_announcements"]
    assert registry.select({**base, "DRISHTI_ENABLED": "true"}).enabled == ["official_rss"]   # news switch is separate
    on = registry.select({**base, "DRISHTI_ANNOUNCEMENTS_ENABLED": "true"})
    assert on.enabled == ["official_rss", "drishti_announcements"]


def test_pipeline_binds_our_stocks_and_stores_filings_as_primary_company_events():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Company(symbol="TRENT.NS", name="Trent Limited"))
        s.add(Company(symbol="TCS.NS", name="Tata Consultancy Services"))
        s.flush()
        s.add(V2UniverseMember(symbol="TRENT.NS", source="SEED"))
        s.add(V2UniverseMember(symbol="TCS.NS", source="SEED"))
        s.commit()
        http = Http(lambda q: {"data": [_row(1), _row(2, date="2026-10-18T10:45:00+00:00")], "has_next": False})
        p = ad.DrishtiAnnouncementsProvider(api_key=KEY, get=http, sleep=lambda x: None)
        run = pipeline.ingest(s, p, NOW - dt.timedelta(hours=2), NOW, now=NOW)
        assert run.status == "COMPLETED" and run.inserted == 2                 # distinct ids are not duplicates
        assert set(urllib.parse.parse_qs(urllib.parse.urlparse(http.seen[0][0]).query)["symbols"][0].split(",")) == \
            {"TRENT", "TCS"}
        arts = s.exec(select(NewsArticle).where(NewsArticle.provider == "drishti_announcements")).all()
        assert {a.source_tier for a in arts} == {"PRIMARY"}
        links = s.exec(select(EventEntity).where(EventEntity.symbol == "TRENT.NS", EventEntity.relation == "SUBJECT")).all()
        assert links                                                            # the company is the subject
        again = pipeline.ingest(s, p, NOW - dt.timedelta(hours=2), NOW, now=NOW)
        assert again.inserted == 0 and again.duplicates == 2                   # overlap windows never double-count
