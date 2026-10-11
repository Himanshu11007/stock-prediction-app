"""
tests/test_news_providers.py — provider registry and adapters: configuration
gating (research-only, credentials), request budgets, rate limiting,
parsing and failure modes of each adapter, health checks, and the pipeline's
use of provider symbols, publisher domains and timestamp precision.
No network: every adapter gets a fake HTTP function.
"""
import datetime as dt
import json

import pytest
from sqlmodel import Session, select

from catalysts import adapters as ad, pipeline, providers as prov, registry
from db.models.news import EventEntity, NewsArticle
from db.models.stock import Company
from prediction_v2 import universe
from tests.test_prediction_v2 import db  # noqa: F401  (fixture)

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def fake(body, status=200, calls=None):
    def get(url, timeout=20, headers=None):
        if calls is not None:
            calls.append((url, headers))
        return status, body if isinstance(body, bytes) else json.dumps(body).encode()
    return get


# ── registry ─────────────────────────────────────────────────────────────────

def test_default_selection_is_the_official_feeds_and_research_providers_are_refused():
    assert registry.select({}).enabled == ["official_rss"]
    assert registry.select({"NEWS_PROVIDERS": "rss"}).enabled == ["official_rss"]          # alias
    sel = registry.select({"NEWS_PROVIDERS": "official_rss,google_news,nse_announcements,newsapi,bogus"})
    assert sel.enabled == ["official_rss"]
    assert "research-only" in sel.refused["google_news"] and "research-only" in sel.refused["nse_announcements"]
    assert sel.refused["bogus"] == "unknown provider"
    research = registry.select({"NEWS_PROVIDERS": "google_news,newsapi,federal_register",
                                "NEWS_ALLOW_RESEARCH_PROVIDERS": "true"})
    assert research.enabled == ["google_news", "federal_register"]
    assert research.refused["newsapi"] == "NEWSAPI_KEY is not set"


def test_request_budgets_fit_provider_quotas():
    assert registry.budget_problems(list(registry.SPECS)) == []                           # 17 hourly runs/day
    assert registry.budget_problems(["alphavantage"], runs_per_day=48)                     # 48 > 25/day quota
    for s in registry.SPECS.values():
        assert s.archive in ("NONE", "LIMITED", "FULL") and s.time_precision in ("exact", "date", "mixed")


def test_rate_limited_get_spaces_requests_and_caps_them():
    t, slept = [100.0], []
    g = registry.RateLimitedGet(2.0, 3, fake(b"ok"), clock=lambda: t[0], sleep=lambda s: slept.append(s))
    g("https://a"); g("https://a"); t[0] += 5; g("https://a")
    assert slept == [2.0]                                         # second call waited; third did not need to
    with pytest.raises(prov.RateLimited):
        g("https://a")


# ── adapters ─────────────────────────────────────────────────────────────────

FR = {"total_pages": 1, "results": [
    {"document_number": "2026-19999", "title": "Adjusting Imports of Steel into the United States",
     "abstract": "Proclamation increasing tariffs under section 232", "html_url": "https://www.federalregister.gov/d/2026-19999",
     "publication_date": "2026-10-08", "type": "Presidential Document", "agencies": [{"name": "Executive Office of the President"}]}]}


def test_federal_register_is_date_only_and_never_early():
    p = ad.FederalRegisterProvider(terms=["tariff", "duty"], get=fake(FR), sleep=lambda s: None)
    items = list(p.fetch(NOW - dt.timedelta(days=5), NOW))
    assert len(items) == 1                                         # deduplicated across search terms
    a = items[0]
    assert a.time_precision == "date" and a.source_domain == "federalregister.gov"
    assert a.published_at == dt.datetime(2026, 10, 9, 4, 59, tzinfo=UTC)          # 23:59 on 8 Oct, UTC-5


NSE = [{"seq_id": "106813731", "symbol": "TRENT", "sm_name": "Trent Limited", "desc": "Press Release",
        "an_dt": "05-Oct-2026 18:42:52", "exchdisstime": "05-Oct-2026 18:42:54",
        "attchmntText": "Trent Limited has informed the Exchange regarding Update for Q2FY27 and H1FY27.",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/TRENT_05102026.pdf"},
       {"seq_id": "1", "symbol": "OLD", "sm_name": "Old", "desc": "x", "exchdisstime": "01-Jan-2020 10:00:00"},
       {"seq_id": "2", "symbol": "BAD", "sm_name": "Bad", "desc": "x", "exchdisstime": "not a time"}]


def test_nse_filings_use_the_exchange_dissemination_time(monkeypatch):
    monkeypatch.setenv(ad.NSE_PERMISSION_ENV, "TEST-PERMISSION-REF")      # parsing only; the gate is tested below
    calls = []
    p = ad.NseAnnouncementsProvider(get=fake(NSE, calls=calls), sleep=lambda s: None)
    items = list(p.fetch(dt.datetime(2026, 10, 1, tzinfo=UTC), NOW))
    assert len(items) == 1
    a = items[0]
    assert a.published_at == dt.datetime(2026, 10, 5, 18, 42, 54, tzinfo=IST).astimezone(UTC)
    assert a.symbols == ("TRENT.NS",) and a.source_domain == "nseindia.com" and "Q2FY27" in a.excerpt
    assert "from_date=01-10-2026" in calls[0][0] and calls[0][1]["Accept"] == "application/json"
    with pytest.raises(prov.ProviderError):
        list(ad.NseAnnouncementsProvider(get=fake({"error": "x"}), sleep=lambda s: None).fetch(NOW, NOW))
    with pytest.raises(prov.ProviderError):
        list(ad.NseAnnouncementsProvider(get=fake(b"", status=403), sleep=lambda s: None).fetch(NOW, NOW))


GN = b"""<?xml version="1.0"?><rss><channel>
<item><title>Trent shares jump on Q2 update - The Economic Times</title><link>https://news.google.com/rss/articles/abc</link>
<guid>abc</guid><pubDate>Mon, 05 Oct 2026 13:34:32 GMT</pubDate><source url="https://economictimes.indiatimes.com">The Economic Times</source></item>
<item><title>RBI cuts repo rate - Reuters</title><link>https://news.google.com/rss/articles/def</link><guid>def</guid>
<pubDate>Sat, 10 Oct 2026 08:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
</channel></rss>"""


def test_google_news_attributes_publishers_and_marks_placeholder_times():
    p = ad.GoogleNewsProvider(queries=["Trent"], get=fake(GN), sleep=lambda s: None)
    a, b = list(p.fetch(dt.datetime(2026, 10, 1, tzinfo=UTC), NOW))
    assert a.title == "Trent shares jump on Q2 update" and a.source_domain == "economictimes.indiatimes.com"
    assert a.time_precision == "exact" and a.published_at == dt.datetime(2026, 10, 5, 13, 34, 32, tzinfo=UTC)
    assert b.time_precision == "date" and b.published_at == dt.datetime(2026, 10, 10, 23, 59, tzinfo=IST).astimezone(UTC)
    with pytest.raises(prov.ProviderError):
        list(ad.GoogleNewsProvider(queries=["x"], get=fake(b"<rss><unclosed"), sleep=lambda s: None).fetch(NOW, NOW))


def test_alphavantage_needs_a_real_key_and_reports_quota_messages():
    with pytest.raises(prov.NotConfigured):
        list(ad.AlphaVantageNewsProvider(api_key="demo").fetch(NOW, NOW))
    with pytest.raises(prov.RateLimited):
        list(ad.AlphaVantageNewsProvider(api_key="k", get=fake({"Information": "rate limit is 25 requests per day"}),
                                         sleep=lambda s: None).fetch(NOW, NOW))
    feed = {"feed": [{"title": "Oil jumps", "url": "https://x.com/a", "time_published": "20261009T120000",
                      "summary": "s", "source": "X", "source_domain": "x.com", "ticker_sentiment": [{"ticker": "RELIANCE.BSE"}]}]}
    (a,) = list(ad.AlphaVantageNewsProvider(api_key="k", get=fake(feed), sleep=lambda s: None).fetch(NOW, NOW))
    assert a.symbols == ("RELIANCE.NS",) and a.provider_entities[0]["symbol"] == "RELIANCE.BSE"   # mapped + raw tag kept
    assert a.published_at == dt.datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def test_health_states():
    rss = b"""<?xml version="1.0"?><rss><channel><item><title>RBI press release</title><link>https://www.rbi.org.in/a</link>
    <pubDate>Sat, 10 Oct 2026 10:00:00 +0530</pubDate></item></channel></rss>"""
    ok = registry.health("official_rss", get=fake(rss), now=NOW)
    assert ok["status"] == "OK" and ok["items"] == 7 and ok["newest_item_age_hours"] == 7.5   # 7 feeds, 04:30 UTC item
    down = registry.health("federal_register", get=fake(b"", status=500), now=NOW)
    assert down["status"] == "DOWN" and "HTTP 500" in down["detail"]
    assert registry.health("newsapi", env={}, now=NOW)["status"] == "NOT_CONFIGURED"
    secret = "AV-SECRET-123456"
    res = registry.health("alphavantage", env={"ALPHAVANTAGE_API_KEY": secret}, now=NOW,
                          get=fake({"Information": "Thank you for using Alpha Vantage! rate limit"}))
    assert res["status"] in ("RATE_LIMITED", "NOT_CONFIGURED") and secret not in json.dumps(res)


# ── pipeline uses provider metadata ─────────────────────────────────────────

def test_filing_symbol_is_the_subject_and_publisher_domain_sets_the_tier(db, monkeypatch):
    monkeypatch.setenv(ad.NSE_PERMISSION_ENV, "TEST-PERMISSION-REF")
    with Session(db) as s:
        s.add(Company(symbol="TRENT.NS", name="Trent Limited", sector="Consumer Cyclical", industry="Apparel Retail"))
        s.commit()
        universe.add_symbols(s, ["TRENT.NS"])
    filing = list(ad.NseAnnouncementsProvider(get=fake(NSE), sleep=lambda s: None).fetch(dt.datetime(2026, 10, 1, tzinfo=UTC), NOW))[0]
    gnews = list(ad.GoogleNewsProvider(queries=["Trent"], get=fake(GN), sleep=lambda s: None).fetch(dt.datetime(2026, 10, 1, tzinfo=UTC), NOW))[0]

    class L:
        name = "mixed"

        def fetch(self, since, until):
            return [filing, gnews]
    with Session(db) as s:
        pipeline.ingest(s, L(), dt.datetime(2026, 10, 1, tzinfo=UTC), NOW, now=NOW)
        arts = {a.provider_article_id: a for a in s.exec(select(NewsArticle)).all()}
        f = arts["106813731"]
        assert f.source_tier == "PRIMARY" and f.raw["provider_symbols"] == ["TRENT.NS"] and f.raw["time_precision"] == "exact"
        assert arts["abc"].source_domain == "economictimes.indiatimes.com" and arts["abc"].source_tier == "TIER2"
        subj = s.exec(select(EventEntity).where(EventEntity.event_id == f.event_id, EventEntity.relation == "SUBJECT")).all()
        assert [(e.symbol, e.method) for e in subj] == [("TRENT.NS", "PROVIDER")]


def test_news_job_uses_the_registry_and_reports_refusals(monkeypatch):
    from scheduling import jobs
    monkeypatch.setenv("NEWS_PROVIDERS", "official_rss,google_news")
    monkeypatch.delenv("NEWS_ALLOW_RESEARCH_PROVIDERS", raising=False)
    provs = jobs.news_providers()
    assert [p.name for p in provs] == ["official_rss"]
    assert "google_news" in jobs.news_selection()["refused"]


def test_official_feed_quirks_sebi_date_only_pib_undated_and_plain_user_agent():
    assert "http" not in prov.USER_AGENT                       # pib.gov.in refuses user agents containing a URL
    t, precision = prov.parse_pub_date("09 Oct, 2026 +0530")   # SEBI's format
    assert precision == "date" and t == dt.datetime(2026, 10, 9, 23, 59, tzinfo=IST).astimezone(UTC)
    assert prov.parse_pub_date("Fri, 09 Oct 2026 12:30:00 +0530")[1] == "exact"
    pib = (b'<?xml version="1.0"?><rss><channel><item><title>Cabinet approves new export scheme</title>'
           b'<link>https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1</link></item></channel></rss>')
    (a,) = list(prov.RssProvider({"pib": ("https://pib.gov.in/x", "PIB")}, get=fake(pib), sleep=lambda s: None)
                .fetch(NOW - dt.timedelta(hours=1), NOW))
    assert a.published_at == NOW and a.first_seen_at == NOW and a.raw["published_is_first_seen"] is True
    undated_elsewhere = list(prov.RssProvider({"rbi_press": ("https://www.rbi.org.in/x", "RBI")}, get=fake(pib),
                                              sleep=lambda s: None).fetch(NOW - dt.timedelta(hours=1), NOW))
    assert undated_elsewhere == []                             # only declared feeds may be first-seen dated


def test_rbi_zone_less_pub_dates_are_ist_not_utc():
    # RBI writes 'Sat, 10 Oct 2026 01:25:00' (IST, no zone); read as UTC it would be 5.5 hours late.
    assert prov.naive_tz_for("https://www.rbi.org.in/pressreleases_rss.xml") is not None
    assert prov.naive_tz_for("https://www.federalreserve.gov/feeds/press_all.xml") is None
    t, precision = prov.parse_pub_date("Sat, 10 Oct 2026 01:25:00", prov.naive_tz_for("https://www.rbi.org.in/x"))
    assert precision == "exact" and t == dt.datetime(2026, 10, 9, 19, 55, tzinfo=UTC)
    assert prov.parse_pub_date("Fri, 9 Oct 2026 14:00:00 GMT", prov.IST_TZ)[0] == dt.datetime(2026, 10, 9, 14, 0, tzinfo=UTC)
    feed = (b'<?xml version="1.0"?><rss><channel><item><title>RBI announces special window</title>'
            b'<link>https://www.rbi.org.in/p?id=1</link><pubDate>Sat, 10 Oct 2026 01:25:00</pubDate></item></channel></rss>')
    (a,) = list(prov.RssProvider({"rbi_press": ("https://www.rbi.org.in/x", "RBI")}, get=fake(feed), sleep=lambda s: None)
                .fetch(dt.datetime(2026, 10, 9, tzinfo=UTC), dt.datetime(2026, 10, 11, tzinfo=UTC)))
    assert a.published_at == dt.datetime(2026, 10, 9, 19, 55, tzinfo=UTC)


def test_original_timestamp_and_zone_are_kept_without_inventing_one():
    assert prov.zone_of("Fri, 09 Oct 2026 12:30:00 +0530") == "+0530"
    assert prov.zone_of("Fri, 9 Oct 2026 14:00:00 GMT") == "GMT"
    assert prov.zone_of("2026-10-09T12:30:00Z") == "Z"
    assert prov.zone_of("Sat, 10 Oct 2026 01:25:00", "IST") == "NONE->IST"
    assert prov.zone_of(None) is None
    feed = (b'<?xml version="1.0"?><rss><channel><item><title>RBI announces special window</title>'
            b'<link>https://www.rbi.org.in/p?id=1</link><pubDate>Sat, 10 Oct 2026 01:25:00</pubDate></item>'
            b'<item><title>Fed statement</title><link>https://www.federalreserve.gov/x</link></item></channel></rss>')
    (a,) = list(prov.RssProvider({"rbi_press": ("https://www.rbi.org.in/x", "RBI")}, get=fake(feed), sleep=lambda s: None)
                .fetch(dt.datetime(2026, 10, 9, tzinfo=UTC), dt.datetime(2026, 10, 11, tzinfo=UTC)))
    assert (a.published_raw, a.published_tz) == ("Sat, 10 Oct 2026 01:25:00", "NONE->IST")


def test_nse_sources_refuse_without_recorded_permission_and_classify_filings(monkeypatch):
    from catalysts import filings
    monkeypatch.delenv(ad.NSE_PERMISSION_ENV, raising=False)
    calls = []
    for p in (ad.NseAnnouncementsProvider(get=fake(NSE, calls=calls), sleep=lambda s: None),
              ad.NseAnnouncementsRssProvider(get=fake(b"<rss/>", calls=calls), sleep=lambda s: None)):
        with pytest.raises(prov.NotPermitted):
            list(p.fetch(dt.datetime(2026, 10, 1, tzinfo=UTC), NOW))
    assert calls == []                                       # nothing was requested from NSE
    monkeypatch.setenv(ad.NSE_PERMISSION_ENV, "TEST-PERMISSION-REF")
    feed = ('<?xml version="1.0"?><rss><channel>'
            '<item><title>Trent Limited</title><link>https://nsearchives.nseindia.com/a.xml</link><pubDate>11-Oct-2026 13:37:50</pubDate>'
            '<description>Trent Limited has informed the Exchange regarding Acquisition |SUBJECT: Acquisition</description></item>'
            '<item><title>Fedbank Limited</title><link>https://nsearchives.nseindia.com/b.xml</link><pubDate>11-Oct-2026 13:37:41</pubDate>'
            '<description>Fedbank has informed the Exchange about Schedule of Analysts or Institutional Investors Meet '
            '|SUBJECT: Schedule of Analysts or Institutional Investors Meet</description></item></channel></rss>').encode()
    p = ad.NseAnnouncementsRssProvider(get=fake(feed), sleep=lambda s: None)
    (a,) = list(p.fetch(dt.datetime(2026, 10, 11, tzinfo=UTC), dt.datetime(2026, 10, 12, tzinfo=UTC)))
    assert a.published_at == dt.datetime(2026, 10, 11, 8, 7, 50, tzinfo=UTC) and a.published_tz == "NONE->IST"
    assert a.raw["materiality"] == "HIGH" and p.excluded == ["Fedbank Limited: ANALYST_MEET"]
    assert filings.classify("Financial Results for the quarter ended 30-Sep-2026").materiality == "HIGH"
    assert filings.classify("Intimation of board meeting to consider financial results").category == "PRIOR_INTIMATION"
    assert filings.classify("Copy of newspaper publication").keep is False
    assert filings.classify("Creation of pledge by promoter").category == "PLEDGE"
    assert filings.classify("Something else entirely").materiality == "UNKNOWN"


def test_network_errors_are_retried_then_reported_and_http_errors_keep_their_status(monkeypatch):
    import urllib.error
    import urllib.request
    calls, slept = [], []

    def down(url, timeout=20.0):
        calls.append(url)
        raise urllib.error.URLError("connection refused")
    with pytest.raises(prov.ProviderError, match="URLError"):
        prov.fetch("https://feed.example.org/rss", get=down, attempts=3, sleep=slept.append)
    assert len(calls) == 3 and slept == [5.0, 10.0]            # bounded retries with backoff

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"ok"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Resp())
    assert prov.http_get("https://x.org") == (200, b"ok")

    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert prov.http_get("https://x.org") == (503, b"")         # an HTTP error is a status, not an exception


def test_gdelt_parses_first_seen_times_and_rejects_non_json():
    body = json.dumps({"articles": [
        {"url": "https://www.reuters.com/a", "title": "RBI keeps repo rate unchanged", "seendate": "20261009T081500Z",
         "domain": "reuters.com", "language": "English", "sourcecountry": "India"},
        {"url": "https://x.com/b", "title": "bad time", "seendate": "yesterday"}]}).encode()
    (a,) = list(prov.GdeltProvider(get=fake(body), sleep=lambda s: None).fetch(NOW - dt.timedelta(days=3), NOW))
    assert a.published_at == dt.datetime(2026, 10, 9, 8, 15, tzinfo=UTC) == a.first_seen_at
    assert (a.published_raw, a.published_tz, a.raw["published_is_first_seen"]) == ("20261009T081500Z", "Z", True)
    with pytest.raises(prov.ProviderError, match="non-JSON"):
        list(prov.GdeltProvider(get=fake(b"<html>error</html>"), sleep=lambda s: None).fetch(NOW, NOW))
    with pytest.raises(prov.RateLimited):
        list(prov.GdeltProvider(get=fake(b"Please limit requests to one every 5 seconds"), sleep=lambda s: None)
             .fetch(NOW, NOW))
