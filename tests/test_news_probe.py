"""
tests/test_news_probe.py — the live provider probe: unconfigured providers
are NOT_CONFIGURED (never failed), keys never reach the report, schema and
timestamp checks, rejected keys and rate limits. No network.
"""
import datetime as dt
import json

from catalysts import probe

NOW = dt.datetime(2026, 10, 11, 9, 0, tzinfo=dt.timezone.utc)
SECRET = "MX-SECRET-VALUE"


def mx_body(n=2, published="2026-10-11T06:30:15.000000Z"):
    return json.dumps({"meta": {"found": n, "returned": n, "limit": 3, "page": 1}, "data": [
        {"uuid": f"u{i}", "title": f"Reliance Industries Q2 results {i}", "url": f"https://x.com/{i}", "published_at": published,
         "source": "economictimes.indiatimes.com", "description": "d",
         "entities": [{"symbol": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE", "country": "in",
                       "type": "equity", "match_score": 40.1}]} for i in range(n)]}).encode()


def test_unconfigured_is_not_tested_never_failed():
    r = probe.probe("marketaux", NOW, get=lambda *a, **k: (500, b""), env={})
    assert r["status"] == "NOT_CONFIGURED" and r["configured"] is False
    assert {c["status"] for c in r["checks"].values()} == {"NOT_TESTED"}


def test_configured_probe_checks_schema_timestamps_history_and_india_without_leaking_the_key():
    calls = []

    def get(url, timeout=20.0, headers=None):
        calls.append(url)
        return 200, mx_body()
    r = probe.probe("marketaux", NOW, get=get, env={"MARKETAUX_API_TOKEN": SECRET})
    assert r["status"] == "OK"
    c = r["checks"]
    assert c["schema"]["status"] == "PASS" and c["timestamps"]["stated_zones"] == ["Z"]
    assert c["timestamps"]["future_dated"] == 0 and set(c["history"]) == {"30d_ago", "365d_ago", "730d_ago"}
    assert c["india"]["symbols_found"] == ["RELIANCE.NS"] and c["india"]["tagged_indian_share_recent"] == 1.0
    assert c["rate_limit"]["requests"] == len(calls) == 5
    assert SECRET not in json.dumps(r)                                     # the key never reaches the report


def test_schema_drift_rejected_key_and_rate_limit():
    bad = json.dumps({"meta": {}, "data": [{"uuid": "1", "title": "t", "url": "u", "published_at": "2026-10-11T06:30:00Z"}]}).encode()
    r = probe.probe("marketaux", NOW, get=lambda *a, **k: (200, bad), env={"MARKETAUX_API_TOKEN": SECRET})
    assert r["checks"]["schema"]["status"] == "FAIL" and "entities" in r["checks"]["schema"]["missing_item_fields"]
    assert r["status"] == "DEGRADED"
    assert probe.probe("marketaux", NOW, get=lambda *a, **k: (401, b"{}"),
                       env={"MARKETAUX_API_TOKEN": SECRET})["status"] == "REJECTED"
    assert probe.probe("marketaux", NOW, get=lambda *a, **k: (429, b"{}"),
                       env={"MARKETAUX_API_TOKEN": SECRET})["status"] == "RATE_LIMITED"
    av_quota = json.dumps({"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}).encode()
    r = probe.probe("alphavantage", NOW, get=lambda *a, **k: (200, av_quota), env={"ALPHAVANTAGE_API_KEY": SECRET})
    assert r["status"] == "RATE_LIMITED" and r["checks"]["authentication"]["status"] == "PASS"


def test_alpha_vantage_unstated_zone_is_flagged():
    body = json.dumps({"feed": [{"title": "Reliance Q2", "url": "https://x.com/r", "time_published": "20261011T063000",
                                 "source": "Reuters", "summary": "s",
                                 "ticker_sentiment": [{"ticker": "RELIANCE.BSE", "relevance_score": "0.9"}]}]}).encode()
    r = probe.probe("alphavantage", NOW, get=lambda *a, **k: (200, body), env={"ALPHAVANTAGE_API_KEY": SECRET})
    assert r["status"] == "OK" and r["checks"]["timestamps"]["stated_zones"] == ["NONE->UTC (unverified)"]
    assert "UTC is assumed" in r["checks"]["timestamps"]["note"]
    assert r["checks"]["india"]["symbols_found"] == ["RELIANCE.NS"]


def test_alpha_vantage_premium_endpoint_is_a_plan_restriction_not_a_rate_limit():
    premium = json.dumps({"Information": "Thank you for using Alpha Vantage! This is a premium endpoint. You may subscribe "
                                         "to any of the premium plans at https://www.alphavantage.co/premium/"}).encode()
    r = probe.probe("alphavantage", NOW, get=lambda *a, **k: (200, premium), env={"ALPHAVANTAGE_API_KEY": SECRET})
    assert r["status"] == "PLAN_RESTRICTED" and r["checks"]["authentication"]["status"] == "PASS"
    assert "premium endpoint" in r["checks"]["connectivity"]["detail"] and SECRET not in json.dumps(r)
