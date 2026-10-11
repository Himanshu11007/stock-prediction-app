"""
tests/test_news_benchmark.py — provider benchmark logic (missed vs
untestable, false matches, latency and market-reaction timing, mapping,
summary, recommendation criteria), Marketaux / Alpha Vantage adapters,
symbol mapping, and the new article provenance fields. No network.
"""
import datetime as dt
import json

import pytest
from sqlmodel import Session, select

from catalysts import adapters as ad, benchmark as bm, entities as ent, pipeline, providers as prov
from db.models.news import NewsArticle
from db.models.stock import Company
from prediction_v2 import universe
from tests.test_prediction_v2 import db  # noqa: F401  (fixture)

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
FILED = dt.datetime(2026, 10, 5, 18, 42, 54, tzinfo=IST).astimezone(UTC)        # Monday, after the close
NOW = dt.datetime(2026, 10, 11, 6, 0, tzinfo=UTC)


def trent_event():
    return bm.Event(id="nse:TRENT:1", kind="nse_filing", title="Trent: Update for Q2FY27", official_time=FILED,
                    precision="exact", source="NSE filing", evidence_url="https://nsearchives.nseindia.com/x",
                    query='"Trent"', candidate=r"\bTrent\b", must=[r"q2|quarter|update|revenue|result"],
                    expected_symbols=("TRENT.NS",))


def art(title, minutes, precision="exact", symbols=(), tags=(), url=None):
    return prov.RawArticle(f"id-{title[:8]}-{minutes}", url or f"https://x.com/{abs(hash(title))}-{minutes}", title,
                           FILED + dt.timedelta(minutes=minutes), time_precision=precision, symbols=tuple(symbols),
                           provider_entities=tuple(tags))


def test_market_reaction_time_rules():
    assert bm.market_reaction_time(FILED) == dt.datetime(2026, 10, 6, 9, 15, tzinfo=IST).astimezone(UTC)      # next open
    during = dt.datetime(2026, 10, 6, 11, 0, tzinfo=IST)
    assert bm.market_reaction_time(during) == during                                                        # in session
    fri = dt.datetime(2026, 10, 9, 20, 0, tzinfo=IST)
    assert bm.market_reaction_time(fri, ["2026-10-12"]) == dt.datetime(2026, 10, 13, 9, 15, tzinfo=IST).astimezone(UTC)


def test_untestable_is_never_a_miss():
    ev = trent_event()
    assert bm.testability(bm.Capability("x", configured=False), ev, NOW) == bm.NOT_CONFIGURED
    assert bm.testability(bm.Capability("x", True, max_age=dt.timedelta(days=2)), ev, NOW) == bm.NO_HISTORY
    assert bm.testability(bm.Capability("x", True, live_since=NOW - dt.timedelta(days=1)), ev, NOW) == bm.NO_HISTORY
    assert bm.testability(bm.Capability("x", True, of_record_kinds=("nse_filing",)), ev, NOW) == bm.OF_RECORD
    assert bm.testability(bm.Capability("x", True, max_age=dt.timedelta(days=31)), ev, NOW) is None
    s = bm.summarize([bm.Outcome("x", ev.id, ev.kind, bm.NO_HISTORY), bm.Outcome("x", "e2", "rbi", bm.MISSED),
                      bm.Outcome("x", "e3", "rbi", bm.RATE_LIMITED)])["x"]
    assert s["tested"] == 1 and s["recall"] == 0.0 and s["rate_limited"] == 1    # untestable and outages excluded


def test_detection_delay_actionability_false_matches_and_mapping():
    ev = trent_event()
    index = ent.CompanyIndex([("TRENT.NS", "Trent Limited"), ("TITAN.NS", "Titan Company Limited")])
    items = [art("Trent Q2 results preview: what to expect", -300),                    # preview: not verified
             art("Trent shares: brokerages see upside", 10),                           # candidate, no event words
             art("Trent Q2 update: revenue grows 23%", 22, symbols=("TRENT.NS",), tags=({"symbol": "TRENT.NS"},)),
             art("Trent Q2 update: revenue grows 23% (syndicated)", 40),
             art("Trent, Titan Q2 revenue update", 60 * 24 * 5)]                      # outside the window
    o = bm.evaluate("p", ev, items, index=index, canonical=lambda arts: 1)
    assert o.status == bm.DETECTED and o.verified == 2 and o.candidates == 4
    assert o.delay_minutes == 22.0 and o.actionable is True
    assert "brokerages see upside" in " ".join(o.false_matches)
    assert o.mapping_provider is True and o.mapping_stocklens is True and o.wrong_symbols == []
    late = bm.evaluate("p", ev, [art("Trent Q2 update: revenue grows", 60 * 16)], index=index)   # after the next open
    assert late.actionable is False
    date_only = bm.evaluate("p", ev, [art("Trent Q2 update: revenue grows", 30, precision="date")])
    assert date_only.delay_minutes is None and date_only.exact_share == 0.0               # no latency from dates
    wrong = bm.evaluate("p", ev, [art("Trent Q2 revenue update and Titan Company sales", 30,
                                      tags=({"symbol": "TITAN.NS"},), symbols=("TITAN.NS",))], index=index)
    assert wrong.mapping_provider is False and "TITAN.NS" in wrong.wrong_symbols


def test_summary_percentiles_and_recommendation_need_measured_timing_and_licence():
    rows = [bm.Outcome("good", f"e{i}", "nse_filing", bm.DETECTED, verified=1, exact_share=1.0, delay_minutes=float(10 + i),
                       actionable=True, mapping_stocklens=True, canonical_events=1) for i in range(20)]
    rows += [bm.Outcome("good", "m1", "rbi", bm.MISSED)]
    rows += [bm.Outcome("loud", f"e{i}", "nse_filing", bm.DETECTED, verified=50, exact_share=0.0) for i in range(30)]
    s = bm.summarize(rows)
    assert s["good"]["latency_minutes"]["p50"] == 19.5 and s["good"]["recall"] == round(20 / 21, 3)
    rec = bm.recommend(s, {"good": True, "loud": True})
    assert rec["recommended"] == ["good"]
    assert any("median delay" in r for r in rec["verdicts"]["loud"]["reasons"])         # volume alone is not enough
    assert bm.recommend(s, {"good": False})["recommended"] == []                          # licence required


def test_symbol_mapping():
    assert ad.to_nse_symbol("RELIANCE.BSE") == "RELIANCE.NS" and ad.to_nse_symbol("TCS.NS") == "TCS.NS"
    assert ad.to_nse_symbol("INFY", "NSE") == "INFY.NS" and ad.to_nse_symbol("AAPL", "NASDAQ") is None
    assert ad.to_nse_symbol("500325.BO") is None                                       # numeric BSE code: unmappable


MX = {"meta": {"found": 1, "returned": 1, "limit": 3, "page": 1}, "data": [
    {"uuid": "u-1", "title": "Trent Q2 update: revenue grows 23%", "description": "Zudio store count crosses 1,000",
     "url": "https://www.livemint.com/trent", "published_at": "2026-10-05T13:34:32.000000Z", "source": "livemint.com",
     "entities": [{"symbol": "TRENT.NS", "name": "Trent Limited", "exchange": "NSE", "country": "in", "type": "equity",
                   "match_score": 41.2}]}]}


def test_marketaux_parses_entities_and_never_leaks_the_token():
    seen = []

    def get(url, timeout=20, headers=None):
        seen.append(url)
        return 200, json.dumps(MX).encode()
    items = list(ad.MarketauxProvider(api_token="MX-SECRET", get=get, sleep=lambda s: None)
                 .fetch(FILED - dt.timedelta(days=1), FILED + dt.timedelta(days=1)))
    (a,) = items
    assert a.symbols == ("TRENT.NS",) and a.provider_entities[0]["match_score"] == 41.2
    assert a.published_at == dt.datetime(2026, 10, 5, 13, 34, 32, tzinfo=UTC) and a.source_domain == "livemint.com"
    assert "countries=in" in seen[0]
    with pytest.raises(prov.NotConfigured):
        list(ad.MarketauxProvider(api_token="").fetch(NOW, NOW))
    for status, exc in ((401, prov.NotConfigured), (402, prov.RateLimited), (429, prov.RateLimited), (500, prov.ProviderError)):
        with pytest.raises(exc) as e:
            list(ad.MarketauxProvider(api_token="MX-SECRET", get=lambda u, t=20, h=None: (status, b"{}"),
                                      sleep=lambda s: None).fetch(NOW, NOW))
        assert "MX-SECRET" not in str(e.value)

    def boom(url, timeout=20, headers=None):
        raise OSError(f"failed {url}")
    with pytest.raises(prov.ProviderError) as e:
        list(ad.MarketauxProvider(api_token="MX-SECRET", get=boom, sleep=lambda s: None).fetch(NOW, NOW))
    assert "MX-SECRET" not in str(e.value)


def test_alphavantage_rejected_key_is_not_configured_and_tags_are_kept():
    with pytest.raises(prov.NotConfigured):
        list(ad.AlphaVantageNewsProvider(api_key="k", sleep=lambda s: None,
                                         get=lambda u, t=20, h=None: (200, b'{"Information": "Invalid API key. Please retry"}'))
             .fetch(NOW, NOW))
    feed = {"feed": [{"title": "Reliance Q2", "url": "https://x.com/r", "time_published": "20261009T120000",
                      "ticker_sentiment": [{"ticker": "RELIANCE.BSE", "relevance_score": "0.9"}]}]}
    (a,) = list(ad.AlphaVantageNewsProvider(api_key="k", sleep=lambda s: None,
                                            get=lambda u, t=20, h=None: (200, json.dumps(feed).encode())).fetch(NOW, NOW))
    assert a.symbols == ("RELIANCE.NS",) and a.provider_entities[0]["symbol"] == "RELIANCE.BSE"


def test_articles_store_first_seen_category_flags_and_provider_entities(db):
    with Session(db) as s:
        s.add(Company(symbol="TRENT.NS", name="Trent Limited", sector="Consumer Cyclical"))
        s.commit()
        universe.add_symbols(s, ["TRENT.NS"])
    items = [prov.RawArticle("g1", "https://news.google.com/rss/articles/abc", "Trent Q2 update: revenue grows 23%",
                             FILED, time_precision="date", source_domain="livemint.com", source_name="Mint"),
             prov.RawArticle("d1", "https://example.org/a", "Unmapped tag story", FILED,
                             first_seen_at=FILED, raw={"published_is_first_seen": True},
                             provider_entities=({"symbol": "AAPL", "exchange": "NASDAQ"},))]

    class L:
        name = "bench"

        def fetch(self, since, until):
            return items
    with Session(db) as s:
        pipeline.ingest(s, L(), FILED - dt.timedelta(days=1), NOW, now=FILED + dt.timedelta(hours=8))
        rows = {a.provider_article_id: a for a in s.exec(select(NewsArticle)).all()}
    g, d = rows["g1"], rows["d1"]
    assert g.category == "BUSINESS_UPDATE" and g.first_seen_at is None
    assert {"DATE_ONLY_TIMESTAMP", "AGGREGATOR_LINK", "LATE_INGESTION", "NO_EXCERPT"} <= set(g.quality_flags)
    assert d.first_seen_at is not None and "PUBLISHED_TIME_IS_FIRST_SEEN" in d.quality_flags
    assert "PROVIDER_ENTITIES_UNMAPPED" in d.quality_flags and d.provider_entities[0]["symbol"] == "AAPL"


def test_pre_release_articles_and_excerpt_only_mentions_are_not_detections():
    ev = trent_event()
    items = [art("Trent Q2 revenue update from yesterday", -6 * 60),            # 6 h before the filing: not this event
             prov.RawArticle("r1", "https://x.com/roundup", "Stocks to watch today: Modern Engineering, Zudio",
                             FILED + dt.timedelta(minutes=5), excerpt="Trent Q2 update also in focus",
                             time_precision="exact"),                         # subject only in the excerpt
             art("Trent Q2 update: revenue up 23%", 20)]
    out = bm.evaluate("p", ev, items)
    assert out.status == bm.DETECTED and out.verified == 1 and out.delay_minutes == 20.0
    assert len(out.false_matches) == 2
    assert bm.evaluate("p", ev, items[:2]).status == bm.MISSED
    assert bm.evaluate("p", ev, [art("Trent Q2 update filed", -10)]).status == bm.DETECTED   # within the 30 min tolerance


def test_value_is_measured_on_events_stocklens_does_not_already_receive():
    def o(kind, delay, actionable):
        x = bm.Outcome("p", f"{kind}-{delay}", kind, bm.DETECTED)
        x.delay_minutes, x.actionable = delay, actionable
        return x
    rows = [o("rbi", 30.0, True), o("rbi", 50.0, True), o("nse_filing", 20.0, True), o("nse_filing", 900.0, False),
            bm.Outcome("p", "g", "gold", bm.DETECTED), bm.Outcome("p", "m", "nse_filing", bm.MISSED),
            bm.Outcome("p", "u", "nse_filing", bm.NO_HISTORY)]
    v = bm.value_over_sources(rows, ["rbi"])["p"]
    assert v["lag_behind_existing_sources_minutes"] == {"n": 2, "p50": 40.0}      # RBI is already read at the official time
    assert (v["other_events_tested"], v["other_events_detected"], v["other_events_timed"], v["other_events_actionable"]) == (4, 3, 2, 1)
    summary = bm.summarize(rows)
    reasons = bm.recommend(summary, {"p": True}, value={"p": v})["verdicts"]["p"]["reasons"]
    assert any("own sources do not deliver" in r for r in reasons)
    assert not any("own sources" in r for r in bm.recommend(summary, {"p": True})["verdicts"]["p"]["reasons"])


def test_roundups_and_day_late_reports_are_ambiguous_not_detections():
    ev = trent_event()
    index = ent.CompanyIndex([("TRENT.NS", "Trent Limited"), ("TITAN.NS", "Titan Company Limited"),
                              ("ITC.NS", "ITC Limited")])
    roundup = art("Stocks to watch: Trent Q2 update, Titan, ITC", 15)
    late = art("Trent Q2 update: what it means for investors", 26 * 60)
    out = bm.evaluate("p", ev, [roundup, late], index=index)
    assert out.status == bm.AMBIGUOUS and out.ambiguous_reason == "roundup / multi-company headline"
    assert out.delay_minutes is None and out.actionable is None              # no timing claimed from a weak match
    good = art("Trent Q2 update: revenue up 23%", 30)
    out2 = bm.evaluate("p", ev, [roundup, late, good], index=index)
    assert out2.status == bm.DETECTED and out2.delay_minutes == 30.0 and out2.verified == 1
    s = bm.summarize([out, out2, bm.Outcome("p", "m", "nse_filing", bm.MISSED)])["p"]
    assert (s["tested"], s["detected"], s["ambiguous"], s["recall"], s["recall_lenient"]) == (3, 1, 1, 0.333, 0.667)
    assert s["article_precision"] == round(1 / 5, 3)                          # 1 strong of 5 candidate articles
    seg = bm.by_segment([out, out2])
    assert seg["indian_company"]["p"]["tested"] == 2 and seg["commodity_currency"] == {}
    rows = bm.audit([out, out2], {ev.id: {"title": ev.title, "precision": "exact"}})
    assert "WEAK_MATCH" in rows[0]["flags"] and rows[1]["status"] == bm.DETECTED
