"""
tests/test_drishti_provider.py — Drishti news connector (research only):
parsing, the key in a header (never in the URL or an error), cautious time
zones, pagination, error mapping, off-by-default selection, and benchmark
testability (symbol-only: events without a company are never "missed").
"""
import datetime as dt
import json
import urllib.parse

import pytest

from catalysts import adapters as ad
from catalysts import benchmark as bm
from catalysts import providers as prov
from catalysts import registry

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime(2026, 10, 18, 6, 0, tzinfo=UTC)
KEY = "DRISHTI-SECRET-KEY"


def _item(i, date, symbol="HDFCAMC", **kw):
    return {"id": f"n{i}", "title": f"HDFC AMC Q2 profit rises {i}%", "specific_title": None, "summary": "AUM up",
            "long_summary": None, "company": "HDFC Asset Management", "symbol": symbol, "sentiment": "positive",
            "source": "economictimes.indiatimes.com", "article_type": "results", "link": f"https://et.example/{i}",
            "scrip_code": "541729", "date": date, **kw}


def _getter(pages, seen):
    def get(url, timeout=20.0, headers=None):
        seen.append((url, headers))
        page = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["page"][0])
        return 200, json.dumps(pages[page - 1]).encode()
    return get


def test_parses_items_with_the_key_in_a_header_and_cautious_times():
    seen = []
    pages = [{"data": [_item(1, "2026-10-15T10:31:00Z"), _item(2, "2026-10-15T16:05:00"),     # no zone -> UTC
                       _item(3, "2026-10-15"), _item(4, "2026-10-15T11:00:00+05:30", symbol="532540"),
                       _item(5, None), _item(6, "2026-10-15T10:00:00Z", title=None, specific_title=None)],
              "has_next": True},
             {"data": [_item(7, "2026-10-16T09:00:00Z")], "has_next": False}]
    items = list(ad.DrishtiNewsProvider(api_key=KEY, symbols=["HDFCAMC.NS", " tcs "], get=_getter(pages, seen),
                                        sleep=lambda s: None).fetch(NOW - dt.timedelta(days=7), NOW))
    assert [a.provider_article_id for a in items] == ["n1", "n2", "n3", "n4", "n7"]       # no date / no title skipped
    a1, a2, a3, a4 = items[:4]
    assert a1.published_at == dt.datetime(2026, 10, 15, 10, 31, tzinfo=UTC) and a1.published_tz == "Z"
    assert a2.published_at == dt.datetime(2026, 10, 15, 16, 5, tzinfo=UTC) and a2.published_tz == "NONE->UTC"
    assert a3.time_precision == "date" and a3.published_at == dt.datetime(2026, 10, 15, 23, 59, 59, tzinfo=IST)
    assert a4.published_at == dt.datetime(2026, 10, 15, 5, 30, tzinfo=UTC) and a4.symbols == ()   # numeric code: no NSE symbol
    assert a1.symbols == ("HDFCAMC.NS",) and a1.raw == {"sentiment": "positive", "article_type": "results"}
    assert a1.provider_entities[0]["scrip_code"] == "541729" and a1.source_domain == "economictimes.indiatimes.com"
    assert len(seen) == 2                                                                   # followed has_next
    q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0][0]).query)
    assert q["symbols"] == ["HDFCAMC,TCS"] and q["limit"] == ["50"] and q["from"] == ["2026-10-11T06:00:00Z"]
    assert all(KEY not in url and h["X-API-Key"] == KEY for url, h in seen)                # header only


def test_page_cap_and_errors_never_leak_the_key():
    seen = []
    pages = [{"data": [_item(1, "2026-10-15T10:31:00Z")], "has_next": True}] * 5
    list(ad.DrishtiNewsProvider(api_key=KEY, get=_getter(pages, seen), sleep=lambda s: None, max_pages=2)
         .fetch(NOW - dt.timedelta(days=1), NOW))
    assert len(seen) == 2 and "symbols" not in seen[0][0]                                 # cap; no symbol filter
    with pytest.raises(prov.NotConfigured):
        list(ad.DrishtiNewsProvider(api_key="").fetch(NOW, NOW))
    for status, exc in ((401, prov.NotConfigured), (403, prov.NotPermitted), (402, prov.RateLimited),
                        (429, prov.RateLimited), (500, prov.ProviderError)):
        with pytest.raises(exc) as e:
            list(ad.DrishtiNewsProvider(api_key=KEY, get=lambda u, t=20.0, h=None: (status, b"{}"),
                                        sleep=lambda s: None).fetch(NOW, NOW))
        assert KEY not in str(e.value)

    def boom(url, timeout=20.0, headers=None):
        raise OSError(f"failed {url} {headers}")
    with pytest.raises(prov.ProviderError) as e:
        list(ad.DrishtiNewsProvider(api_key=KEY, get=boom, sleep=lambda s: None).fetch(NOW, NOW))
    assert KEY not in str(e.value)


def test_off_by_default_and_needs_its_own_switch_the_key_and_the_research_flag():
    base = {"NEWS_PROVIDERS": "official_rss,drishti", "DRISHTI_API_KEY": "k"}
    assert registry.select(base).enabled == ["official_rss"]
    assert "DRISHTI_ENABLED" in registry.select(base).refused["drishti"]
    assert "research-only" in registry.select({**base, "DRISHTI_ENABLED": "true"}).refused["drishti"]
    no_key = {**base, "DRISHTI_ENABLED": "true", "NEWS_ALLOW_RESEARCH_PROVIDERS": "true", "DRISHTI_API_KEY": ""}
    assert "DRISHTI_API_KEY is not set" in registry.select(no_key).refused["drishti"]
    ok = {**base, "DRISHTI_ENABLED": "true", "NEWS_ALLOW_RESEARCH_PROVIDERS": "true"}
    assert registry.select(ok).enabled == ["official_rss", "drishti"]
    assert registry.health("drishti", env={})["status"] == "DISABLED"                       # never spends a credit


def _event(symbols=()):
    t = dt.datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    return bm.Event(id="e", kind="nse_filing" if symbols else "rbi", title="x", official_time=t, precision="exact",
                    source="s", evidence_url=None, query="q", candidate="x", must=[], expected_symbols=symbols)


def test_symbol_only_providers_are_not_testable_without_a_company():
    cap = bm.Capability("drishti", True, symbol_only=True)
    assert bm.testability(cap, _event(), NOW) == bm.NOT_SUPPORTED
    assert bm.testability(cap, _event(("HDFCAMC.NS",)), NOW) is None
    assert bm.testability(bm.Capability("drishti", False, symbol_only=True), _event(), NOW) == bm.NOT_CONFIGURED
    assert bm.testability(bm.Capability("google_news", True), _event(), NOW) is None        # others unchanged
    out = [bm.Outcome("drishti", "e", "rbi", bm.NOT_SUPPORTED)]
    sm = bm.summarize(out)["drishti"]
    assert (sm["tested"], sm["missed"], sm["recall"]) == (0, 0, None)                      # never counted as a miss
