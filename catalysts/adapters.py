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

from catalysts import filings
from catalysts import text as tx
from catalysts.providers import (HttpGet, NotConfigured, NotPermitted, ProviderError, RateLimited, RawArticle, _as_utf8,
                                 fetch, http_get, zone_of)

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
                                     published_raw=d["publication_date"], published_tz="NONE->ET (date only)",
                                     source_domain="federalregister.gov")
                if page >= int(data.get("total_pages") or 1):
                    break


# NSE Terms of Use (read 11 Oct 2026, https://www.nseindia.com/static/nse-terms-of-use):
#   "User is prohibited to conduct any systematic or automated data collection activities
#    (including scraping, data mining, data extraction and data harvesting) on or in relation
#    to our Website" and content may not be "stored ... without prior written permission of NSE".
# Commercial use of market data is licensed by NSE Data. Every automated NSE source therefore
# refuses to run unless the reference of NSE's written permission or licence is recorded here:
NSE_PERMISSION_ENV = "NSE_AUTOMATED_ACCESS_PERMISSION"
BSE_PERMISSION_ENV = "BSE_AUTOMATED_ACCESS_PERMISSION"


def require_permission(env_name: str, source: str) -> None:
    if not os.environ.get(env_name, "").strip():
        raise NotPermitted(f"{source}: automated collection needs the exchange's written permission or a licence; "
                           f"record its reference in {env_name} (not set)")


class NseAnnouncementsProvider:
    name = "nse_announcements"
    URL = "https://www.nseindia.com/api/corporate-announcements"
    research_only = True

    def __init__(self, symbols: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep):
        self.symbols, self.get, self.sleep = tuple(s.split(".")[0].upper() for s in symbols or ()), get, sleep

    def _rows(self, since: dt.datetime, until: dt.datetime, symbol: Optional[str]) -> list[dict]:
        require_permission(NSE_PERMISSION_ENV, "NSE corporate announcements")
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
                             source_domain="nseindia.com", raw={"filed_at": r.get("an_dt"), "category": r.get("desc")},
                             published_raw=stamp, published_tz="NONE->IST")


class NseAnnouncementsRssProvider:
    """NSE's official 'Online announcements' RSS feed (live: the latest ~35
    filings; no history). pubDate is IST without a zone ('11-Oct-2026
    13:37:50'). Routine notices are dropped by catalysts.filings. Gated by
    the same written-permission requirement as every automated NSE source."""
    name = "nse_rss"
    research_only = True
    URL = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"

    def __init__(self, get: HttpGet = http_get, sleep: Callable[[float], None] = time.sleep, keep_routine: bool = False):
        self.get, self.sleep, self.keep_routine = get, sleep, keep_routine
        self.excluded: list[str] = []

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        require_permission(NSE_PERMISSION_ENV, "NSE announcements RSS")
        root = ET.fromstring(_as_utf8(fetch(self.URL, self.get, sleep=self.sleep)))
        for item in root.iter("item"):
            stamp = (item.findtext("pubDate") or "").strip()
            try:
                pub = dt.datetime.strptime(stamp, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).astimezone(dt.timezone.utc)
            except ValueError:
                continue
            if not (since <= pub <= until):
                continue
            company = tx.sanitize(item.findtext("title"), 200)
            desc = tx.sanitize(item.findtext("description"))
            subject = desc.split("|SUBJECT:", 1)[1].strip() if "|SUBJECT:" in desc else desc
            fc = filings.classify(subject, desc)
            if not fc.keep and not self.keep_routine:
                self.excluded.append(f"{company}: {fc.category}")
                continue
            link = (item.findtext("link") or "").strip()
            yield RawArticle(provider_article_id=f"nse_rss:{link or company + stamp}"[:300], url=link or self.URL,
                             title=tx.sanitize(f"{company}: {subject}", tx.MAX_TITLE), published_at=pub,
                             excerpt=desc, source_name="National Stock Exchange of India (filing)",
                             source_domain="nseindia.com", published_raw=stamp, published_tz="NONE->IST",
                             raw={"filing_category": fc.category, "materiality": fc.materiality,
                                  "filings_version": filings.FILINGS_VERSION})


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
                    pub_raw = item.findtext("pubDate") or ""
                    pub = email.utils.parsedate_to_datetime(pub_raw)
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
                                 source_domain=tx.domain(src_url) if src_url else None, raw={"query": query},
                                 published_raw=pub_raw, published_tz=zone_of(pub_raw))


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
            if k in data:                                    # quota, plan and key messages arrive as HTTP 200
                msg = str(data[k])
                if re.search(r"premium endpoint|premium plan", msg, re.I):
                    raise NotPermitted("Alpha Vantage: NEWS_SENTIMENT is a premium endpoint for this key "
                                       "(plan restriction; a paid plan is needed)")
                if re.search(r"api ?key|apikey", msg, re.I) and not re.search(r"rate limit|per day|per minute", msg, re.I):
                    raise NotConfigured("Alpha Vantage rejected the API key")
                raise RateLimited(f"Alpha Vantage: {msg[:120]}")
        if "Error Message" in data:
            raise ProviderError(f"Alpha Vantage: {str(data['Error Message'])[:120]}")
        for a in data.get("feed", []):
            try:   # documented format YYYYMMDDTHHMMSS; time zone not stated in the response -> treated as UTC (unverified)
                pub = dt.datetime.strptime(a["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)
            except (KeyError, ValueError):
                continue
            tags = tuple({"symbol": t.get("ticker"), "relevance": t.get("relevance_score")}
                         for t in a.get("ticker_sentiment") or [] if t.get("ticker"))
            yield RawArticle(provider_article_id=a.get("url", "")[:300], url=a.get("url", ""),
                             title=tx.sanitize(a.get("title"), tx.MAX_TITLE), published_at=pub,
                             excerpt=tx.sanitize(a.get("summary")), source_name=a.get("source"),
                             source_domain=a.get("source_domain"),
                             symbols=tuple(s for s in (to_nse_symbol(t["symbol"]) for t in tags) if s),
                             provider_entities=tags,
                             raw={"topics": ",".join(t.get("topic") or "" for t in a.get("topics") or [])},
                             published_raw=a.get("time_published"), published_tz="NONE->UTC (unverified)")


def to_nse_symbol(symbol: Optional[str], exchange: Optional[str] = None) -> Optional[str]:
    """Provider ticker -> StockLens NSE symbol ('TCS.NS'), or None when the
    ticker is not an Indian listing. 'RELIANCE.BSE' / 'RELIANCE.BO' and
    exchange NSE/BSE map to 'RELIANCE.NS' (NSE and BSE symbols coincide for
    most large caps; mismatches are measured in the benchmark, not assumed)."""
    if not symbol:
        return None
    s = symbol.strip().upper()
    for suffix in (".NS", ".NSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)] + ".NS"
    for suffix in (".BSE", ".BO"):
        if s.endswith(suffix):
            base = s[: -len(suffix)]
            return None if base.isdigit() else base + ".NS"     # numeric BSE codes cannot be mapped
    ex = (exchange or "").upper()
    if "." not in s and ex in ("NSE", "BSE", "NSEI", "BOM", "NATIONAL STOCK EXCHANGE OF INDIA", "BOMBAY STOCK EXCHANGE"):
        return None if s.isdigit() else s + ".NS"
    return None


class MarketauxProvider:
    """Marketaux /v1/news/all (https://www.marketaux.com/documentation).
    Needs MARKETAUX_API_TOKEN; the token is a query parameter by the API's
    design, so URLs are never logged (errors carry status and domain only).
    Research-only: plan terms not reviewed. Free-plan limits are reported by
    Marketaux as about 100 requests a day and a few articles per request
    (UNVERIFIED here - no token)."""
    name = "marketaux"
    research_only = True
    URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_token: Optional[str] = None, countries: str = "in", search: Optional[str] = None,
                 symbols: Optional[Iterable[str]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep, max_pages: int = 3):
        self.token = api_token if api_token is not None else os.environ.get("MARKETAUX_API_TOKEN", "")
        self.countries, self.search, self.max_pages = countries, search, max_pages
        self.symbols, self.get, self.sleep = tuple(symbols or ()), get, sleep

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        if not self.token:
            raise NotConfigured("MARKETAUX_API_TOKEN is not set")
        for page in range(1, self.max_pages + 1):
            params = {"api_token": self.token, "language": "en", "page": page,
                      "published_after": f"{since.astimezone(dt.timezone.utc):%Y-%m-%dT%H:%M:%S}",
                      "published_before": f"{until.astimezone(dt.timezone.utc):%Y-%m-%dT%H:%M:%S}"}
            if self.countries:
                params["countries"] = self.countries
            if self.search:
                params["search"] = self.search
            if self.symbols:
                params["symbols"] = ",".join(self.symbols)
            try:
                status, body = self.get(f"{self.URL}?{urllib.parse.urlencode(params)}", 20.0)
            except OSError as e:                        # never surface the URL (it carries the token)
                raise ProviderError(f"Marketaux connection error: {type(e).__name__}") from None
            if status == 401:
                raise NotConfigured("Marketaux rejected the API token (invalid_api_token)")
            if status in (402, 429):
                raise RateLimited(f"Marketaux HTTP {status} (usage limit)")
            if status != 200:
                raise ProviderError(f"Marketaux HTTP {status}")
            try:
                data = json.loads(body or b"{}")
            except ValueError as e:
                raise ProviderError("Marketaux returned non-JSON") from e
            if "error" in data:
                raise ProviderError(f"Marketaux error {data['error'].get('code')}")
            for a in data.get("data", []):
                try:
                    pub = dt.datetime.fromisoformat(str(a["published_at"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                pub = pub if pub.tzinfo else pub.replace(tzinfo=dt.timezone.utc)
                tags = tuple({"symbol": e.get("symbol"), "name": e.get("name"), "exchange": e.get("exchange"),
                              "country": e.get("country"), "type": e.get("type"), "match_score": e.get("match_score")}
                             for e in a.get("entities") or [])
                yield RawArticle(provider_article_id=str(a.get("uuid") or a.get("url", ""))[:300], url=a.get("url", ""),
                                 title=tx.sanitize(a.get("title"), tx.MAX_TITLE), published_at=pub.astimezone(dt.timezone.utc),
                                 excerpt=tx.sanitize(a.get("description") or a.get("snippet")),
                                 source_name=a.get("source"), source_domain=a.get("source") if "." in str(a.get("source") or "") else None,
                                 symbols=tuple(s for s in (to_nse_symbol(t["symbol"], t.get("exchange")) for t in tags) if s),
                                 provider_entities=tags, published_raw=str(a.get("published_at")),
                                 published_tz=zone_of(str(a.get("published_at"))))
            meta = data.get("meta") or {}
            returned, found = int(meta.get("returned") or 0), int(meta.get("found") or 0)
            if not returned or page * max(returned, 1) >= found:
                break
            self.sleep(1.0)


class DrishtiNewsProvider:
    """Drishti (Manasija Technologies) GET /v1/news - Indian company news
    filterable by NSE symbol / BSE scrip code and date, with the provider's
    own sentiment (https://drishti.manasija.in/docs; spec
    https://developers.manasija.in/openapi.json, read 18 Oct 2026).

    Needs DRISHTI_API_KEY, sent in the X-API-Key header (never in the URL, so
    URLs carry no secret). Billing: 1 credit per page that returns items
    (sandbox: 1,000 credits once). There is no keyword search, so it can only
    answer "news about these symbols in this window". Research-only: terms
    and data source not reviewed; time zone of `date` not documented - a
    timestamp without a zone is read as UTC (never earlier than reality)."""
    name = "drishti"
    research_only = True
    URL = "https://developers.manasija.in/v1/news"
    PAGE_LIMIT = 50                                     # documented maximum

    def __init__(self, api_key: Optional[str] = None, symbols: Optional[Iterable[str]] = None,
                 sentiment: Optional[str] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep, max_pages: int = 2):
        self.key = api_key if api_key is not None else os.environ.get("DRISHTI_API_KEY", "")
        self.symbols = tuple(s.strip().upper().removesuffix(".NS") for s in (symbols or ()) if s and s.strip())
        self.sentiment, self.get, self.sleep, self.max_pages = sentiment, get, sleep, max_pages

    @staticmethod
    def _when(raw: Optional[str]) -> tuple[Optional[dt.datetime], str]:
        """(UTC time, precision). Date-only values count as available at the
        END of that IST day; values without a zone are read as UTC."""
        s = str(raw or "").strip()
        if not s:
            return None, "exact"
        if len(s) == 10:
            try:
                d = dt.date.fromisoformat(s)
            except ValueError:
                return None, "exact"
            return dt.datetime.combine(d, dt.time(23, 59, 59), tzinfo=IST).astimezone(dt.timezone.utc), "date"
        try:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None, "exact"
        return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(dt.timezone.utc), "exact"

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        if not self.key:
            raise NotConfigured("DRISHTI_API_KEY is not set")
        headers = {"X-API-Key": self.key, "Accept": "application/json"}
        for page in range(1, self.max_pages + 1):
            params = {"from": f"{since.astimezone(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
                      "to": f"{until.astimezone(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
                      "page": page, "limit": self.PAGE_LIMIT}
            if self.symbols:
                params["symbols"] = ",".join(self.symbols)
            if self.sentiment:
                params["sentiment"] = self.sentiment
            try:
                status, body = self.get(f"{self.URL}?{urllib.parse.urlencode(params)}", 20.0, headers)
            except OSError as e:                        # never surface request details (the header carries the key)
                raise ProviderError(f"Drishti connection error: {type(e).__name__}") from None
            if status == 401:
                raise NotConfigured("Drishti rejected the API key (401)")
            if status == 403:
                raise NotPermitted("Drishti: the news product is not enabled for this key (403)")
            if status in (402, 429):
                raise RateLimited(f"Drishti HTTP {status} ({'insufficient credits' if status == 402 else 'rate limit'})")
            if status != 200:
                raise ProviderError(f"Drishti HTTP {status}")
            try:
                data = json.loads(body or b"{}")
            except ValueError as e:
                raise ProviderError("Drishti returned non-JSON") from e
            for a in data.get("data") or []:
                pub, precision = self._when(a.get("date"))
                title = a.get("title") or a.get("specific_title")
                if pub is None or not title:
                    continue
                sym = str(a.get("symbol") or "").strip().upper()
                tag = {"symbol": sym or None, "scrip_code": a.get("scrip_code"), "name": a.get("company"),
                       "sentiment": a.get("sentiment"), "article_type": a.get("article_type")}
                src = a.get("source")
                yield RawArticle(provider_article_id=str(a.get("id") or a.get("link") or "")[:300],
                                 url=a.get("link") or "", title=tx.sanitize(title, tx.MAX_TITLE),
                                 published_at=pub, excerpt=tx.sanitize(a.get("summary")), source_name=src,
                                 source_domain=src if "." in str(src or "") else None, time_precision=precision,
                                 symbols=(sym + ".NS",) if sym and not sym.isdigit() else (),
                                 provider_entities=(tag,), published_raw=str(a.get("date")),
                                 published_tz=zone_of(str(a.get("date")), "UTC"),
                                 raw={"sentiment": a.get("sentiment"), "article_type": a.get("article_type")})
            if not data.get("has_next"):
                break
            self.sleep(1.0)
