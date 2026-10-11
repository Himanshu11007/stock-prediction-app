"""
catalysts/adapters.py — additional news / event providers (11 Oct 2026 audit).

All return catalysts.providers.RawArticle, so the pipeline and the
prediction logic are provider-independent. Each adapter is HTTP-injectable
(tests use fakes) and raises ProviderError / RateLimited / NotConfigured.

  FederalRegisterProvider   US executive actions, tariffs, sanctions, trade
                            rules (federalregister.gov API; keyless; history
                            since 1994). Date-only: published_at = end of the
                            publication date (US Eastern), conservative.
  NseAnnouncementsProvider  NSE corporate filings with the exchange's
                            dissemination time to the second (history at
                            least to 2024). RESEARCH-ONLY until NSE's terms
                            for automated use / redistribution are reviewed.
  GoogleNewsProvider        aggregated international and Indian news search
                            (keyless; historical windows with after:/before:).
                            RESEARCH-ONLY (Google News terms); older items carry
                            placeholder times -> treated as date-only.
  PublisherRssProvider      Economic Times, Mint, Business Standard market
                            feeds (live only, about 2 days). RESEARCH-ONLY
                            (publisher terms).
  AlphaVantageNewsProvider  NEWS_SENTIMENT (needs ALPHAVANTAGE_API_KEY; the demo
                            key only serves AAPL). Not configured.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Callable, Iterable, Optional

from catalysts import text as tx
from catalysts.providers import HttpGet, NotConfigured, ProviderError, RateLimited, RawArticle, _as_utf8, fetch, http_get

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockLens-research/0.1"


def _end_of_day_et(day: dt.date) -> dt.datetime:
    """23:59 on `day` in US Eastern, taking the LATER of EST/EDT in UTC
    (UTC-5), so a date-only item is never treated as available too early."""
    return dt.datetime.combine(day, dt.time(23, 59), tzinfo=dt.timezone(dt.timedelta(hours=-5))).astimezone(dt.timezone.utc)


def _end_of_day_ist(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(23, 59), tzinfo=IST).astimezone(dt.timezone.utc)


class FederalRegisterProvider:
    name = "federal_register"
    URL = "https://www.federalregister.gov/api/v1/documents.json"
    TERMS = ("tariff", "duty", "sanction", "export control", "trade agreement", "section 232", "section 301",
             "reciprocal", "india")

    def __init__(self, terms: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep, max_pages: int = 3):
        self.terms, self.get, self.sleep, self.max_pages = tuple(terms or self.TERMS), get, sleep, max_pages

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        seen: set[str] = set()
        for term in self.terms:
            for page in range(1, self.max_pages + 1):
                q = urllib.parse.urlencode({"conditions[term]": term, "per_page": 100, "page": page, "order": "newest",
                                            "conditions[publication_date][gte]": since.date().isoformat(),
                                            "conditions[publication_date][lte]": until.date().isoformat(),
                                            "fields[]": ["title", "abstract", "html_url", "publication_date",
                                                         "document_number", "type", "agencies"]}, doseq=True)
                data = json.loads(fetch(f"{self.URL}?{q}", self.get, sleep=self.sleep) or b"{}")
                for d in data.get("results", []):
                    num = d.get("document_number")
                    if not num or num in seen or not d.get("publication_date"):
                        continue
                    seen.add(num)
                    pub = _end_of_day_et(dt.date.fromisoformat(d["publication_date"]))
                    agencies = ", ".join(a.get("name", "") for a in d.get("agencies") or [] if a.get("name"))
                    yield RawArticle(provider_article_id=num, url=d.get("html_url") or "", title=tx.sanitize(d.get("title"), tx.MAX_TITLE),
                                     published_at=pub, excerpt=tx.sanitize(d.get("abstract")), source_name=agencies or "Federal Register",
                                     raw={"type": d.get("type"), "term": term}, time_precision="date",
                                     source_domain="federalregister.gov")
                if page >= int(data.get("total_pages") or 1):
                    break


class NseAnnouncementsProvider:
    name = "nse_announcements"
    URL = "https://www.nseindia.com/api/corporate-announcements"
    research_only = True

    def __init__(self, symbols: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep):
        self.symbols, self.get, self.sleep = tuple(s.split(".")[0].upper() for s in symbols or ()), get, sleep

    def _rows(self, since: dt.datetime, until: dt.datetime, symbol: Optional[str]) -> list[dict]:
        params = {"index": "equities", "from_date": f"{since.astimezone(IST):%d-%m-%Y}",
                  "to_date": f"{until.astimezone(IST):%d-%m-%Y}"}
        if symbol:
            params["symbol"] = symbol
        body = fetch(f"{self.URL}?{urllib.parse.urlencode(params)}", self.get, sleep=self.sleep,
                     headers={"User-Agent": BROWSER_UA, "Accept": "application/json"})
        data = json.loads(body or b"[]")
        if not isinstance(data, list):
            raise ProviderError("NSE returned an unexpected payload")
        return data

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        rows: list[dict] = []
        for sym in (self.symbols or (None,)):
            rows += self._rows(since, until, sym)
        for r in rows:
            stamp = r.get("exchdisstime") or r.get("an_dt")
            try:
                pub = dt.datetime.strptime(stamp, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).astimezone(dt.timezone.utc)
            except (TypeError, ValueError):
                continue
            if not (since <= pub <= until):
                continue
            sym = (r.get("symbol") or "").upper()
            title = tx.sanitize(f"{r.get('sm_name') or sym}: {r.get('desc') or 'Announcement'}", tx.MAX_TITLE)
            yield RawArticle(provider_article_id=str(r.get("seq_id") or f"{sym}:{stamp}"),
                             url=r.get("attchmntFile") or f"https://www.nseindia.com/companies-listing/corporate-filings-announcements#{sym}",
                             title=title, published_at=pub, excerpt=tx.sanitize(r.get("attchmntText")),
                             source_name="National Stock Exchange of India (filing)", symbols=(f"{sym}.NS",) if sym else (),
                             source_domain="nseindia.com", raw={"filed_at": r.get("an_dt"), "category": r.get("desc")})


PLACEHOLDER_TIMES = {"07:00:00", "08:00:00", "00:00:00"}


class GoogleNewsProvider:
    name = "google_news"
    research_only = True
    URL = "https://news.google.com/rss/search"
    QUERIES = ("India stock market", "Nifty Sensex", "tariff India", "US tariffs", "crude oil prices", "Brent crude",
               "US Treasury yields", "Federal Reserve", "RBI", "rupee dollar", "geopolitical tensions oil",
               "China economy", "Indian economy GDP inflation")

    def __init__(self, queries: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep, min_interval: float = 2.0):
        self.queries, self.get, self.sleep, self.min_interval = tuple(queries or self.QUERIES), get, sleep, min_interval

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        for k, query in enumerate(self.queries):
            if k:
                self.sleep(self.min_interval)
            q = f"{query} after:{since.date().isoformat()} before:{(until.date() + dt.timedelta(days=1)).isoformat()}"
            body = _as_utf8(fetch(f"{self.URL}?{urllib.parse.urlencode({'q': q, 'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'})}",
                                  self.get, sleep=self.sleep))
            try:
                root = ET.fromstring(body)
            except ET.ParseError as e:
                raise ProviderError("Google News returned malformed XML") from e
            for item in root.iter("item"):
                raw_title = tx.sanitize(item.findtext("title"), tx.MAX_TITLE)
                src = item.find("source")
                src_name = (src.text or "").strip() if src is not None else None
                src_url = src.get("url") if src is not None else None
                title = raw_title[: -(len(src_name) + 3)] if src_name and raw_title.endswith(f" - {src_name}") else raw_title
                try:
                    pub = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
                except (TypeError, ValueError):
                    continue
                precision = "date" if pub.strftime("%H:%M:%S") in PLACEHOLDER_TIMES else "exact"
                if precision == "date":
                    pub = _end_of_day_ist(pub.date())
                pub = pub.astimezone(dt.timezone.utc)
                if not (since <= pub <= until + dt.timedelta(days=1)):
                    continue
                link = (item.findtext("link") or "").strip()
                yield RawArticle(provider_article_id=(item.findtext("guid") or link)[:300], url=link, title=title,
                                 published_at=pub, source_name=src_name, time_precision=precision,
                                 source_domain=tx.domain(src_url) if src_url else None, raw={"query": query})


PUBLISHER_FEEDS = {
    "et_markets": ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "The Economic Times"),
    "livemint_markets": ("https://www.livemint.com/rss/markets", "Mint"),
    "bs_markets": ("https://www.business-standard.com/rss/markets-106.rss", "Business Standard"),
}


class PublisherRssProvider:
    """Indian financial press market feeds (Moneycontrol's feed was found stale
    since April 2024 and is excluded)."""
    name = "publisher_rss"
    research_only = True

    def __init__(self, feeds: Optional[dict] = None, get: HttpGet = http_get, sleep: Callable[[float], None] = time.sleep):
        from catalysts.providers import RssProvider
        self._rss = RssProvider(feeds or PUBLISHER_FEEDS, get=get, sleep=sleep, name=self.name)

    @property
    def errors(self) -> list[str]:
        return self._rss.errors

    def fetch(self, since, until):
        return self._rss.fetch(since, until)


class AlphaVantageNewsProvider:
    name = "alphavantage"
    research_only = True                 # terms for redistribution not reviewed
    URL = "https://www.alphavantage.co/query"
    TOPICS = ("economy_macro", "economy_monetary", "economy_fiscal", "energy_transportation", "financial_markets")

    def __init__(self, api_key: Optional[str] = None, tickers: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep):
        self.api_key = api_key if api_key is not None else os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.tickers, self.get, self.sleep = tuple(tickers or ()), get, sleep

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        if not self.api_key or self.api_key == "demo":
            raise NotConfigured("ALPHAVANTAGE_API_KEY is not set (the demo key only serves documented examples)")
        params = {"function": "NEWS_SENTIMENT", "time_from": f"{since:%Y%m%dT%H%M}", "time_to": f"{until:%Y%m%dT%H%M}",
                  "limit": 1000, "sort": "EARLIEST", "apikey": self.api_key}
        if self.tickers:
            params["tickers"] = ",".join(self.tickers)
        else:
            params["topics"] = ",".join(self.TOPICS)
        data = json.loads(fetch(f"{self.URL}?{urllib.parse.urlencode(params)}", self.get, sleep=self.sleep) or b"{}")
        for k in ("Information", "Note"):
            if k in data:                                    # quota or plan messages arrive as HTTP 200
                raise RateLimited(f"Alpha Vantage: {str(data[k])[:120]}")
        if "Error Message" in data:
            raise ProviderError(f"Alpha Vantage: {str(data['Error Message'])[:120]}")
        for a in data.get("feed", []):
            try:   # documented format YYYYMMDDTHHMMSS; time zone not stated in the response -> treated as UTC (unverified)
                pub = dt.datetime.strptime(a["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)
            except (KeyError, ValueError):
                continue
            yield RawArticle(provider_article_id=a.get("url", "")[:300], url=a.get("url", ""),
                             title=tx.sanitize(a.get("title"), tx.MAX_TITLE), published_at=pub,
                             excerpt=tx.sanitize(a.get("summary")), source_name=a.get("source"),
                             source_domain=a.get("source_domain"),
                             symbols=tuple(t.get("ticker") for t in a.get("ticker_sentiment") or [] if t.get("ticker")),
                             raw={"topics": [t.get("topic") for t in a.get("topics") or []]})
