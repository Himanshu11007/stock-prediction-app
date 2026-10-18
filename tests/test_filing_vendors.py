"""
tests/test_filing_vendors.py — vendor filing / results-calendar clients and
the check that judges them: keys only in headers (never in URLs or errors),
time parsing, request shapes, error mapping, and the matched / late /
not_found rule with the before-market-reaction flag.
"""
import datetime as dt
import json
import urllib.parse

import pytest

from catalysts import filing_vendors as fv
from catalysts import providers as prov
from scripts.research import filings_vendor_check as chk

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OFFICIAL = dt.datetime(2026, 10, 5, 13, 34, 32, tzinfo=UTC)      # 19:04 IST, after the close


def _rec(payload, seen):
    def get(url, timeout=20.0, headers=None):
        seen.append((url, headers))
        return 200, json.dumps(payload).encode()
    return get


def test_drishti_filings_and_calendar_keep_the_key_in_the_header():
    seen = []
    c = fv.DrishtiFilings(api_key="D-KEY", get=_rec({"data": [
        {"id": "a1", "symbol": "TRENT", "company_name": "Trent", "date": "2026-10-05T13:40:00+00:00",
         "summary": "Q2 update: revenue +23%", "category": "Press Release"},
        {"id": "a2", "symbol": "TRENT", "date": None}]}, seen))
    (f,) = c.filings("TRENT.NS", OFFICIAL - dt.timedelta(hours=2), OFFICIAL + dt.timedelta(hours=8))
    assert (f.symbol, f.time, f.category, f.precision) == ("TRENT.NS", dt.datetime(2026, 10, 5, 13, 40, tzinfo=UTC),
                                                           "Press Release", "exact")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0][0]).query)
    assert q["symbols"] == ["TRENT"] and q["from"] == ["2026-10-05T11:34:32Z"] and "/announcements?" in seen[0][0]
    c.get = _rec({"data": [{"symbol": "CANBK", "date": "2026-10-29", "quarter": "Q2FY27", "purpose": "Financial Results"}]},
                 seen)
    (e,) = c.results_calendar(["CANBK.NS", "TCS"])
    assert (e.symbol, e.date, e.detail["quarter"]) == ("CANBK.NS", dt.date(2026, 10, 29), "Q2FY27")
    assert "symbols=CANBK%2CTCS" in seen[1][0]
    assert all("D-KEY" not in u and h["X-API-Key"] == "D-KEY" for u, h in seen) and c.calls == 2


def test_stockinsights_filings_and_calendar_use_a_bearer_header():
    seen = []
    c = fv.StockInsightsFilings(api_key="S-KEY", get=_rec({"status": "success", "data": [
        {"id": "s1", "ticker": "TRENT", "company_name": "Trent Ltd",
         "exchange_tickers": [{"exchange": "BSE", "ticker": "500251"}, {"exchange": "NSE", "ticker": "TRENT"}],
         "source_link": "https://nsearchives.example/x.pdf", "published_date": "2026-10-05T13:35:10Z",
         "ai_insights": {"announcement_type": "Business Update", "summary_header": "Revenue up 23%",
                         "sentiment": "positive"}}], "meta": {}}, seen))
    (f,) = c.filings("TRENT", OFFICIAL, OFFICIAL + dt.timedelta(days=1))
    assert (f.symbol, f.sentiment, f.category, f.link) == ("TRENT.NS", "positive", "Business Update",
                                                          "https://nsearchives.example/x.pdf")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0][0]).query)
    assert q["ticker"] == ["NSE:TRENT"] and q["from_date"] == ["2026-10-05"] and q["to_date"] == ["2026-10-06"]
    c.get = _rec({"data": [{"company_name": "Canara Bank", "ticker": "CANBK", "security_code": "532483",
                            "result_date": "2026-10-29T00:00:00.000Z"}]}, seen)
    (e,) = c.results_calendar(["CANBK"])
    assert (e.symbol, e.date) == ("CANBK.NS", dt.date(2026, 10, 29))                   # midnight UTC = that day
    assert all("S-KEY" not in u and h["Authorization"] == "Bearer S-KEY" for u, h in seen)


@pytest.mark.parametrize("cls", [fv.DrishtiFilings, fv.StockInsightsFilings])
def test_errors_map_cleanly_and_never_leak_the_key(cls):
    with pytest.raises(prov.NotConfigured):
        cls(api_key="").filings("TRENT", OFFICIAL, OFFICIAL)
    for status, exc in ((401, prov.NotConfigured), (403, prov.NotPermitted), (402, prov.RateLimited),
                        (429, prov.RateLimited), (500, prov.ProviderError)):
        with pytest.raises(exc) as e:
            cls(api_key="SECRET-K", get=lambda u, t=20.0, h=None: (status, b"{}")).filings("TRENT", OFFICIAL, OFFICIAL)
        assert "SECRET-K" not in str(e.value)

    def boom(url, timeout=20.0, headers=None):
        raise OSError(f"{url} {headers}")
    with pytest.raises(prov.ProviderError) as e:
        cls(api_key="SECRET-K", get=boom).filings("TRENT", OFFICIAL, OFFICIAL)
    assert "SECRET-K" not in str(e.value)


def _f(minutes, precision="exact"):
    return fv.Filing("v", "id", "TRENT.NS", OFFICIAL + dt.timedelta(minutes=minutes), precision, "t", "Press Release")


def test_judge_matched_late_not_found_and_before_reaction():
    assert chk.judge(OFFICIAL, []) == {"status": "not_found", "returned": 0}
    assert chk.judge(OFFICIAL, [_f(-120), _f(60 * 50)])["status"] == "not_found"          # outside -30 min .. +48 h
    m = chk.judge(OFFICIAL, [_f(400), _f(6)])
    assert (m["status"], m["delay_minutes"], m["before_reaction"]) == ("matched", 6.0, True)  # next open is hours later
    late = chk.judge(OFFICIAL, [_f(60 * 20)])
    assert late["status"] == "late" and late["before_reaction"] is False                    # after the 09:15 reaction
    assert chk.judge(OFFICIAL, [_f(5, "date")])["before_reaction"] is None                  # date-only: unknown


def test_check_refuses_to_run_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("STOCKINSIGHTS_API_KEY", raising=False)
    assert chk.main(["--vendor", "stockinsights"]) == 2
    assert "nothing was called" in capsys.readouterr().out
