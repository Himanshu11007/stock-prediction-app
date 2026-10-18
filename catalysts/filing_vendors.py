"""
catalysts/filing_vendors.py — exchange-filing (corporate announcement) and
results-calendar feeds from data vendors, for evaluation (RESEARCH ONLY).

Our missed gainers needed the filings themselves (results, orders, board
outcomes) and results dates known in advance, not general news. These small
clients answer exactly two questions per vendor, the same way for each:

  filings(symbol, since, until)  -> the vendor's filings for one NSE symbol
  results_calendar(symbols)      -> (symbol, result date) entries

Vendors (documentation read 18 Oct 2026):
  drishti        https://developers.manasija.in/v1  header X-API-Key
                 GET /announcements (1 credit per returned item), GET /earnings/upcoming
  stockinsights  https://stockinsights-ai-main-95a26a0.zuplo.app  header Authorization: Bearer
                 GET /api/in/v0/documents/announcement (AI summary, category, sentiment),
                 GET /api/in/v0/results-calendar; trial: 100 calls in 7 days

Keys come from DRISHTI_API_KEY / STOCKINSIGHTS_API_KEY and are only ever sent
in a header: never in a URL, a log line or an error message. A time without
a zone is read as UTC (never earlier than reality); a date-only value counts
as the end of that IST day.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from catalysts.providers import HttpGet, NotConfigured, NotPermitted, ProviderError, RateLimited, http_get

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


@dataclass
class Filing:
    vendor: str
    vendor_id: str
    symbol: str                        # NSE symbol with .NS, '' when the vendor gave none
    time: dt.datetime                  # UTC
    precision: str                     # exact | date
    title: str
    category: Optional[str] = None
    sentiment: Optional[str] = None
    link: Optional[str] = None
    raw_time: Optional[str] = None


@dataclass
class CalendarEntry:
    vendor: str
    symbol: str
    date: dt.date
    detail: dict[str, Any] = field(default_factory=dict)


def parse_time(raw: Any) -> tuple[Optional[dt.datetime], str]:
    s = str(raw or "").strip()
    if not s:
        return None, "exact"
    if len(s) == 10:
        try:
            return dt.datetime.combine(dt.date.fromisoformat(s), dt.time(23, 59, 59), tzinfo=IST).astimezone(UTC), "date"
        except ValueError:
            return None, "exact"
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None, "exact"
    return (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC), "exact"


def _bare(symbol: str) -> str:
    return symbol.strip().upper().removesuffix(".NS")


class _Client:
    vendor = ""
    env = ""

    def __init__(self, api_key: Optional[str] = None, get: HttpGet = http_get):
        self.key = api_key if api_key is not None else os.environ.get(self.env, "")
        self.get = get
        self.calls = 0

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _json(self, url: str) -> dict:
        if not self.key:
            raise NotConfigured(f"{self.env} is not set")
        self.calls += 1
        try:
            status, body = self.get(url, 20.0, self._headers())
        except OSError as e:                              # never surface request details (headers carry the key)
            raise ProviderError(f"{self.vendor} connection error: {type(e).__name__}") from None
        if status == 401:
            raise NotConfigured(f"{self.vendor} rejected the API key (401)")
        if status == 403:
            raise NotPermitted(f"{self.vendor}: this product is not enabled for the key (403)")
        if status in (402, 429):
            raise RateLimited(f"{self.vendor} HTTP {status} ({'insufficient credits' if status == 402 else 'rate limit'})")
        if status != 200:
            raise ProviderError(f"{self.vendor} HTTP {status}")
        try:
            return json.loads(body or b"{}")
        except ValueError as e:
            raise ProviderError(f"{self.vendor} returned non-JSON") from e


class DrishtiFilings(_Client):
    vendor, env = "drishti", "DRISHTI_API_KEY"
    BASE = "https://developers.manasija.in/v1"

    def _headers(self):
        return {"X-API-Key": self.key, "Accept": "application/json"}

    def filings(self, symbol: str, since: dt.datetime, until: dt.datetime, limit: int = 10) -> list[Filing]:
        """One page; 1 credit per returned item - keep windows narrow."""
        q = {"symbols": _bare(symbol), "from": f"{since.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}",
             "to": f"{until.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}", "page": 1, "limit": limit}
        out = []
        for a in self._json(f"{self.BASE}/announcements?{urllib.parse.urlencode(q)}").get("data") or []:
            t, prec = parse_time(a.get("date"))
            if t is None:
                continue
            sym = _bare(str(a.get("symbol") or ""))
            out.append(Filing("drishti", str(a.get("id") or ""), f"{sym}.NS" if sym and not sym.isdigit() else "", t, prec,
                              str(a.get("summary") or a.get("company_name") or "")[:300], a.get("category"),
                              None, None, str(a.get("date"))))
        return out

    def results_calendar(self, symbols: Iterable[str], limit: int = 50) -> list[CalendarEntry]:
        q = {"symbols": ",".join(_bare(s) for s in symbols), "page": 1, "limit": limit}
        out = []
        for a in self._json(f"{self.BASE}/earnings/upcoming?{urllib.parse.urlencode(q)}").get("data") or []:
            t, _ = parse_time(a.get("date"))
            if t is not None and a.get("symbol"):
                out.append(CalendarEntry("drishti", f"{_bare(a['symbol'])}.NS", t.astimezone(IST).date(),
                                         {"quarter": a.get("quarter"), "purpose": a.get("purpose")}))
        return out

    def usage(self) -> Optional[int]:
        """Credit balance (free route), or None."""
        try:
            d = self._json(f"{self.BASE}/account/usage")
        except ProviderError:
            return None
        self.calls -= 1                                     # account routes are free; not counted as data calls
        data = d.get("data", d)
        return data.get("balance") if isinstance(data, dict) else None


class StockInsightsFilings(_Client):
    vendor, env = "stockinsights", "STOCKINSIGHTS_API_KEY"
    BASE = "https://stockinsights-ai-main-95a26a0.zuplo.app/api/in/v0"

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}

    def filings(self, symbol: str, since: dt.datetime, until: dt.datetime, limit: int = 50) -> list[Filing]:
        """Date-granular filter (YYYY-MM-DD); one call per page (trial: 100 calls)."""
        q = {"ticker": f"NSE:{_bare(symbol)}", "from_date": f"{since.astimezone(IST):%Y-%m-%d}",
             "to_date": f"{until.astimezone(IST):%Y-%m-%d}", "limit": limit, "page": 1}
        out = []
        for a in self._json(f"{self.BASE}/documents/announcement?{urllib.parse.urlencode(q)}").get("data") or []:
            t, prec = parse_time(a.get("published_date"))
            if t is None:
                continue
            ai = a.get("ai_insights") or {}
            nse = next((x.get("ticker") for x in a.get("exchange_tickers") or [] if str(x.get("exchange")).upper() == "NSE"),
                       None)
            sym = _bare(str(nse or a.get("ticker") or ""))
            out.append(Filing("stockinsights", str(a.get("id") or ""), f"{sym}.NS" if sym else "", t, prec,
                              str(ai.get("summary_header") or a.get("company_name") or "")[:300],
                              ai.get("announcement_type"), ai.get("sentiment"), a.get("source_link"),
                              str(a.get("published_date"))))
        return out

    def results_calendar(self, symbols: Iterable[str]) -> list[CalendarEntry]:
        q = {"ticker": ",".join(f"NSE:{_bare(s)}" for s in symbols)}
        out = []
        for a in self._json(f"{self.BASE}/results-calendar?{urllib.parse.urlencode(q)}").get("data") or []:
            t, _ = parse_time(a.get("result_date"))
            if t is not None and a.get("ticker"):
                # a midnight-UTC date stands for that calendar day
                day = t.date() if t.time() == dt.time(0) else t.astimezone(IST).date()
                out.append(CalendarEntry("stockinsights", f"{_bare(a['ticker'])}.NS", day,
                                         {"security_code": a.get("security_code")}))
        return out

    def usage(self) -> Optional[int]:
        return None                                          # no documented usage route


VENDORS = {"drishti": DrishtiFilings, "stockinsights": StockInsightsFilings}
