"""
catalysts/providers.py — news providers (no paid subscription is used).

  RssProvider       official primary-source feeds: RBI press releases and
                    notifications, SEBI, US Federal Reserve. Public, free,
                    with publication times. Titles, links and a short excerpt
                    only (the feeds' copyright notices apply).
  NewsApiProvider   newsapi.org "everything" search. Needs NEWSAPI_KEY (a NEW
                    key: the old one is in public Git history and must be
                    revoked). The free plan is delayed and non-commercial, so it
                    is research-only; disabled unless the key is set.
  GdeltProvider     GDELT DOC 2.0 (free, global, keyless; "seendate" = when
                    GDELT first saw the article). Rate-limited to one request
                    per 5 s; from this network it frequently refuses, so it is
                    research-only and treated as unreliable.
  FixtureProvider   deterministic JSON for tests and manual curation.

Corporate filings (NSE/BSE announcements), broker ratings and licensed wires
are NOT available: the exchanges' terms restrict automated access and the
rest need a contract (BLOCKED; docs/NEWS_ENGINE.md).

Every provider returns RawArticle objects; failures raise ProviderError or
RateLimited after bounded retries so the ingestion run records them.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from catalysts import text as tx

# Plain product token: pib.gov.in answers 403 to user agents that contain a
# URL (measured 11 Oct 2026), so none is included.
USER_AGENT = "StockLens-research/0.1"
MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class RawArticle:
    provider_article_id: str
    url: str
    title: str
    published_at: dt.datetime
    excerpt: str = ""
    source_name: Optional[str] = None
    corrected_at: Optional[dt.datetime] = None
    raw: dict[str, Any] = field(default_factory=dict)
    # "exact" (time of day reliable) or "date" (only the day is reliable; the
    # provider then sets published_at to the END of that day, so a date-only
    # item can never look available earlier than it was).
    time_precision: str = "exact"
    symbols: tuple[str, ...] = ()               # provider-supplied tickers (e.g. the NSE symbol of a filing)
    source_domain: Optional[str] = None         # the original publisher when the URL is an aggregator link
    first_seen_at: Optional[dt.datetime] = None  # when the PROVIDER first saw it (e.g. GDELT seendate), if known
    provider_entities: tuple[dict, ...] = ()     # the provider's own entity tags, verbatim (symbol, name, exchange, score)


class ProviderError(RuntimeError):
    pass


class RateLimited(ProviderError):
    pass


class NotConfigured(ProviderError):
    pass


HttpGet = Callable[..., tuple[int, bytes]]


def http_get(url: str, timeout: float = 20.0, headers: Optional[dict[str, str]] = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(MAX_BYTES)
    except urllib.error.HTTPError as e:
        return e.code, e.read(10_000) if e.fp else b""


def fetch(url: str, get: HttpGet = http_get, attempts: int = 3, backoff: float = 5.0,
          sleep: Callable[[float], None] = time.sleep, headers: Optional[dict[str, str]] = None) -> bytes:
    """GET with bounded retries; 429 / 503 -> RateLimited after the last try."""
    last = None
    for i in range(attempts):
        try:
            status, body = get(url, 20.0, headers) if headers else get(url, 20.0)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = ProviderError(f"{type(e).__name__}: {e}"[:200])
        else:
            if status == 200:
                return body
            if status in (429, 503):
                last = RateLimited(f"HTTP {status} from {tx.domain(url)}")
            elif status == 403:
                raise ProviderError(f"HTTP 403 from {tx.domain(url)} (access refused / anti-bot)")
            else:
                raise ProviderError(f"HTTP {status} from {tx.domain(url)}")
        if i + 1 < attempts:
            sleep(backoff * (i + 1))
    raise last or ProviderError("no response")


def _utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


DATE_ONLY_FORMATS = ("%d %b, %Y %z", "%d %b %Y %z", "%d %b, %Y", "%d %b %Y", "%Y-%m-%d")


IST_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))
# Feeds whose pubDate carries no zone although it is local time. RBI writes
# 'Sat, 10 Oct 2026 01:25:00' meaning IST; read as UTC it is 5.5 hours late.
NAIVE_TZ_DOMAINS = {"rbi.org.in": IST_TZ}


def naive_tz_for(url: str) -> Optional[dt.tzinfo]:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return next((tz for d, tz in NAIVE_TZ_DOMAINS.items() if host == d or host.endswith("." + d)), None)


def parse_pub_date(s: Optional[str], naive_tz: Optional[dt.tzinfo] = None) -> tuple[Optional[dt.datetime], str]:
    """(UTC time, precision). Date-only values (SEBI writes '09 Oct, 2026
    +0530') become the END of that day in their own zone (IST if none), so
    they never look available early. Zone-less exact times are read in
    `naive_tz` when given (the feed's local zone), else as UTC."""
    if not s:
        return None, "exact"
    s = s.strip()
    for fmt in DATE_ONLY_FORMATS:
        try:
            d = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        tz = d.tzinfo or naive_tz or IST_TZ
        return dt.datetime.combine(d.date(), dt.time(23, 59), tzinfo=tz).astimezone(dt.timezone.utc), "date"
    if naive_tz is not None:
        try:
            d = email.utils.parsedate_to_datetime(s)
        except (TypeError, ValueError):
            d = None
        if d is not None and d.tzinfo is None:
            return d.replace(tzinfo=naive_tz).astimezone(dt.timezone.utc), "exact"
    return _rfc822(s), "exact"


def _rfc822(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return _utc(email.utils.parsedate_to_datetime(s.strip()))
    except (TypeError, ValueError):
        pass
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return _utc(dt.datetime.strptime(s.strip(), fmt))
        except ValueError:
            continue
    return None


def _as_utf8(body: bytes) -> bytes:
    """Feeds that declare UTF-8 but contain Windows-1252 bytes (seen on RBI)
    are re-decoded instead of producing replacement characters."""
    body = body.lstrip(b"\xef\xbb\xbf")
    try:
        body.decode("utf-8")
        return body
    except UnicodeDecodeError:
        text = body.decode("cp1252", errors="replace")
        text = re.sub(r'encoding=["\'][^"\']+["\']', 'encoding="utf-8"', text, count=1)
        return text.encode("utf-8")


UNDATED_FEEDS = {"pib"}          # feeds whose items have no publication time: availability = first sight


FEEDS: dict[str, tuple[str, str]] = {
    # key: (url, source name)
    "rbi_press": ("https://www.rbi.org.in/pressreleases_rss.xml", "Reserve Bank of India"),
    "rbi_notifications": ("https://www.rbi.org.in/notifications_rss.xml", "Reserve Bank of India"),
    "sebi": ("https://www.sebi.gov.in/sebirss.xml", "Securities and Exchange Board of India"),
    "fed_press": ("https://www.federalreserve.gov/feeds/press_all.xml", "US Federal Reserve"),
    # added after the 11 Oct 2026 provider audit (all verified reachable, exact times):
    # English feed needs reg=3 (without it PIB redirects to the Hindi feed). Its items carry NO pubDate:
    # they are recorded at first sight (see UNDATED_FEEDS), never backdated.
    "pib": ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3&reg=3", "Press Information Bureau (Government of India)"),
    "whitehouse_news": ("https://www.whitehouse.gov/news/feed/", "The White House"),
    "whitehouse_actions": ("https://www.whitehouse.gov/presidential-actions/feed/", "The White House (presidential actions)"),
}


class RssProvider:
    def __init__(self, feeds: Optional[dict[str, tuple[str, str]]] = None, get: HttpGet = http_get,
                 sleep: Callable[[float], None] = time.sleep, name: str = "rss"):
        self.feeds, self.get, self.sleep, self.name = feeds or FEEDS, get, sleep, name
        self.errors: list[str] = []

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        """Items published in [since, until]. A failing feed is recorded in
        `errors` and skipped; if every feed fails the run fails."""
        ok = 0
        for key, (url, source) in self.feeds.items():
            try:
                body = _as_utf8(fetch(url, self.get, sleep=self.sleep))
                root = ET.fromstring(body)
            except (ProviderError, ET.ParseError) as e:
                self.errors.append(f"{key}: {e}")
                continue
            ok += 1
            for item in root.iter("item"):
                title = tx.sanitize(item.findtext("title"), tx.MAX_TITLE)
                link = (item.findtext("link") or "").strip()
                published, precision = parse_pub_date(item.findtext("pubDate"), naive_tz_for(url))
                first_seen = False
                if published is None and key in UNDATED_FEEDS:
                    published, first_seen = min(until, dt.datetime.now(dt.timezone.utc)), True   # first sight
                if not title or not link or published is None or not (since <= published <= until):
                    continue
                guid = (item.findtext("guid") or link).strip()
                yield RawArticle(provider_article_id=f"{key}:{guid}"[:300], url=link, title=title,
                                 published_at=published, excerpt=tx.sanitize(item.findtext("description")),
                                 source_name=source, raw={"feed": key, **({"published_is_first_seen": True} if first_seen else {})},
                                 time_precision=precision, first_seen_at=published if first_seen else None)
        if ok == 0 and self.feeds:
            raise ProviderError("every feed failed: " + "; ".join(self.errors)[:400])


class NewsApiProvider:
    name = "newsapi"
    URL = "https://newsapi.org/v2/everything"

    def __init__(self, query: str = '(India OR Indian OR Nifty OR Sensex OR RBI OR SEBI) AND (stocks OR market OR '
                                    'economy OR shares OR tariff OR crude OR rupee)',
                 api_key: Optional[str] = None, get: HttpGet = http_get, sleep: Callable[[float], None] = time.sleep):
        self.query, self.get, self.sleep = query, get, sleep
        self.api_key = api_key if api_key is not None else os.environ.get("NEWSAPI_KEY", "")

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        if not self.api_key:
            raise NotConfigured("NEWSAPI_KEY is not set (and the previously committed key must not be reused)")
        q = urllib.parse.urlencode({"q": self.query, "from": since.isoformat(), "to": until.isoformat(),
                                    "language": "en", "sortBy": "publishedAt", "pageSize": 100})
        # The key is sent as a header, never in the URL, and never logged.
        body = fetch(f"{self.URL}?{q}", self.get, sleep=self.sleep, headers={"X-Api-Key": self.api_key})
        data = json.loads(body)
        if data.get("status") != "ok":
            raise ProviderError(f"newsapi status {data.get('code') or data.get('status')}")
        for a in data.get("articles", []):
            published = _rfc822(a.get("publishedAt")) or (
                _utc(dt.datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))) if a.get("publishedAt") else None)
            if not a.get("url") or not a.get("title") or published is None:
                continue
            yield RawArticle(provider_article_id=a["url"][:300], url=a["url"], title=tx.sanitize(a["title"], tx.MAX_TITLE),
                             published_at=published, excerpt=tx.sanitize(a.get("description")),
                             source_name=(a.get("source") or {}).get("name"))


class GdeltProvider:
    name = "gdelt"
    URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, query: str = '(India OR Nifty OR Sensex OR RBI) sourcelang:english',
                 get: HttpGet = http_get, sleep: Callable[[float], None] = time.sleep, max_records: int = 250):
        self.query, self.get, self.sleep, self.max_records = query, get, sleep, max_records

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        q = urllib.parse.urlencode({"query": self.query, "mode": "artlist", "format": "json",
                                    "maxrecords": self.max_records, "sort": "datedesc",
                                    "startdatetime": f"{since:%Y%m%d%H%M%S}", "enddatetime": f"{until:%Y%m%d%H%M%S}"})
        body = fetch(f"{self.URL}?{q}", self.get, sleep=self.sleep, backoff=6.0)
        if body.lstrip().startswith(b"Please limit requests"):
            raise RateLimited("GDELT asked to limit requests (one per 5 seconds)")
        try:
            data = json.loads(body or b"{}")
        except ValueError as e:
            raise ProviderError(f"GDELT returned non-JSON: {body[:80]!r}") from e
        for a in data.get("articles", []):
            seen = a.get("seendate")
            try:
                published = dt.datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            except (TypeError, ValueError):
                continue
            # GDELT gives no publication time: seendate is when GDELT first saw the article.
            yield RawArticle(provider_article_id=a.get("url", "")[:300], url=a.get("url", ""),
                             title=tx.sanitize(a.get("title"), tx.MAX_TITLE), published_at=published,
                             source_name=a.get("domain"), first_seen_at=published,
                             source_domain=a.get("domain") or None,
                             raw={"language": a.get("language"), "sourcecountry": a.get("sourcecountry"),
                                  "published_is_first_seen": True})


class FixtureProvider:
    def __init__(self, path: Path, name: str = "fixture"):
        self.path, self.name = Path(path), name

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawArticle]:
        for a in json.loads(self.path.read_text(encoding="utf-8")):
            published = _utc(dt.datetime.fromisoformat(a["published_at"]))
            if since <= published <= until:
                yield RawArticle(provider_article_id=str(a["id"]), url=a["url"], title=a["title"],
                                 published_at=published, excerpt=a.get("excerpt", ""), source_name=a.get("source"),
                                 corrected_at=_utc(dt.datetime.fromisoformat(a["corrected_at"]))
                                 if a.get("corrected_at") else None, raw={"fixture": True})
