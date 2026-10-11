"""
catalysts/registry.py — provider registry: specification, configuration,
rate limiting, request budgets and health checks.

Providers are enabled only through configuration; prediction logic never
knows which provider an article came from (everything is normalised to
RawArticle before the pipeline).

  NEWS_PROVIDERS                  comma-separated names (default "official_rss")
                                  ("rss" is accepted as an alias of official_rss)
  NEWS_ALLOW_RESEARCH_PROVIDERS   "true" to allow providers marked research-only
                                  (licence not reviewed for production use);
                                  default false, so they are refused and reported
  provider keys                   NEWSAPI_KEY, ALPHAVANTAGE_API_KEY (secret store
                                  only; never logged or returned)

Each spec states, from the 11 Oct 2026 audit (docs/NEWS_PROVIDERS.md), what the
provider covers, whether it has a genuine point-in-time archive, its timestamp
precision, its rate limits and its licence status.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from catalysts import providers as p
from catalysts import adapters as a

RUNS_PER_DAY = 17                       # hourly NEWS_INGESTION 06:10-22:10 IST (docs/SCHEDULER.md)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    factory: Callable[..., Any]
    kind: str                           # OFFICIAL | EXCHANGE | AGGREGATOR | PUBLISHER | API
    coverage: str
    research_only: bool                 # licence not reviewed for production use
    needs_env: Optional[str]            # env var with the credential, if any
    archive: str                        # NONE | LIMITED | FULL (point-in-time suitability below)
    archive_note: str
    time_precision: str                 # exact | date | mixed
    min_interval_s: float               # minimum seconds between HTTP requests to the provider
    daily_limit: Optional[int]          # provider's request quota per day (None = no stated quota)
    requests_per_run: int               # upper bound of requests one ingestion run makes
    licence: str
    cost: str


SPECS: dict[str, ProviderSpec] = {s.name: s for s in (
    ProviderSpec("official_rss", lambda **k: p.RssProvider(**k, name="official_rss"), "OFFICIAL",
                 "RBI press releases and notifications, SEBI, US Federal Reserve, PIB (Government of India), "
                 "White House news and presidential actions", False, None, "NONE",
                 "feeds hold only the latest 10-30 items; no history", "exact", 1.0, None, 7,
                 "official public releases; titles, links and short excerpts only", "free"),
    ProviderSpec("federal_register", a.FederalRegisterProvider, "OFFICIAL",
                 "US executive orders, proclamations, tariff and sanctions rules", False, None, "FULL",
                 "API back to 1994; availability = end of publication date (date-only, conservative)", "date",
                 1.0, None, 27, "US government works (public domain)", "free"),
    ProviderSpec("nse_announcements", a.NseAnnouncementsProvider, "EXCHANGE",
                 "NSE corporate filings (results, business updates, orders, board outcomes, ...)", True, None, "FULL",
                 "API returns filings by date range and symbol with the exchange dissemination time to the second "
                 "(verified back to 2024)", "exact", 2.0, None, 1,
                 "exchange website terms for automated use / redistribution NOT reviewed; licensed data via NSE "
                 "Data & Analytics", "free website API; licensed feed: paid (quote needed)"),
    ProviderSpec("google_news", a.GoogleNewsProvider, "AGGREGATOR",
                 "international and Indian news across publishers (search queries)", True, None, "LIMITED",
                 "historical search windows work, but results are capped (about 100 per query), ranked by Google, "
                 "and older items carry placeholder times (date-only); no first-seen time -> NOT leakage-free",
                 "mixed", 2.0, None, 13, "Google News terms: not licensed for commercial redistribution", "free"),
    ProviderSpec("publisher_rss", a.PublisherRssProvider, "PUBLISHER",
                 "Economic Times, Mint, Business Standard market news", True, None, "NONE",
                 "about 2 days of items", "exact", 1.0, None, 3, "publisher RSS terms (generally personal use)", "free"),
    ProviderSpec("newsapi", p.NewsApiProvider, "API", "aggregated news search", True, "NEWSAPI_KEY", "LIMITED",
                 "free plan: last 31 days only (verified 11 Oct 2026), publishedAt only, no first-seen time", "exact",
                 1.0, 100, 1, "developer plan is non-commercial and delayed; business plan paid",
                 "free (developer) / paid (business)"),
    ProviderSpec("alphavantage", a.AlphaVantageNewsProvider, "API", "NEWS_SENTIMENT with tickers and topics",
                 True, "ALPHAVANTAGE_API_KEY", "LIMITED",
                 "time_from/time_to supported; depth and Indian coverage NOT verified (demo key serves AAPL only)",
                 "exact", 12.0, 25, 1, "terms not reviewed", "free key (signup) / paid tiers"),
    ProviderSpec("marketaux", a.MarketauxProvider, "API",
                 "financial news with entity tags (symbols, exchanges, countries); country filter 'in'", True,
                 "MARKETAUX_API_TOKEN", "LIMITED",
                 "published_after / published_before supported per documentation; depth, Indian coverage and "
                 "timestamp quality NOT verified (no token)", "exact", 1.0, 100, 3,
                 "plan terms not reviewed", "free plan (signup) / paid tiers"),
    ProviderSpec("gdelt", p.GdeltProvider, "AGGREGATOR", "global news, first-seen times", True, None, "LIMITED",
                 "DOC API: about 3 months, first-seen 'seendate' would be point-in-time, but refused from this "
                 "network (shared carrier IP); raw event files lack headlines", "exact", 6.0, None, 1,
                 "GDELT terms (open, attribution)", "free"),
)}
ALIASES = {"rss": "official_rss"}


@dataclass
class Selection:
    enabled: list[str]
    refused: dict[str, str]             # name -> reason (unknown / research-only / missing credential)


def select(env: Optional[dict[str, str]] = None) -> Selection:
    env = os.environ if env is None else env
    names = [ALIASES.get(n.strip().lower(), n.strip().lower())
             for n in (env.get("NEWS_PROVIDERS") or "official_rss").split(",") if n.strip()]
    allow_research = (env.get("NEWS_ALLOW_RESEARCH_PROVIDERS") or "false").strip().lower() == "true"
    out = Selection([], {})
    for n in dict.fromkeys(names):
        spec = SPECS.get(n)
        if spec is None:
            out.refused[n] = "unknown provider"
        elif spec.research_only and not allow_research:
            out.refused[n] = "research-only (licence not reviewed); set NEWS_ALLOW_RESEARCH_PROVIDERS=true only for research"
        elif spec.needs_env and not env.get(spec.needs_env):
            out.refused[n] = f"{spec.needs_env} is not set"
        else:
            out.enabled.append(n)
    return out


def budget_problems(names: list[str], runs_per_day: int = RUNS_PER_DAY) -> list[str]:
    """Providers whose scheduled request volume would exceed their daily quota."""
    probs = []
    for n in names:
        s = SPECS[n]
        if s.daily_limit is not None and s.requests_per_run * runs_per_day > s.daily_limit:
            probs.append(f"{n}: {s.requests_per_run} x {runs_per_day} runs = {s.requests_per_run * runs_per_day} "
                         f"requests/day > quota {s.daily_limit}")
    return probs


class RateLimitedGet:
    """Wraps an HTTP get: at least `min_interval` seconds between requests to
    one provider (shared across threads), and at most `max_requests` per run."""

    def __init__(self, min_interval: float, max_requests: int, get: p.HttpGet = p.http_get,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.min_interval, self.max_requests, self.get = min_interval, max_requests, get
        self.clock, self.sleep, self.count, self._last, self._lock = clock, sleep, 0, None, threading.Lock()

    def __call__(self, url, timeout=20.0, headers=None):
        with self._lock:
            if self.count >= self.max_requests:
                raise p.RateLimited(f"request budget for this run exhausted ({self.max_requests})")
            if self._last is not None:
                wait = self.min_interval - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            self._last = self.clock()
            self.count += 1
        return self.get(url, timeout, headers) if headers else self.get(url, timeout)


def build(name: str, get: p.HttpGet = p.http_get, **kwargs):
    spec = SPECS[name]
    limited = RateLimitedGet(spec.min_interval_s, max(spec.requests_per_run * 3, 3), get)   # x3: retries
    return spec.factory(get=limited, **kwargs)


def configured_providers(env: Optional[dict[str, str]] = None) -> tuple[list, Selection]:
    sel = select(env)
    return [build(n) for n in sel.enabled], sel


def health(name: str, get: p.HttpGet = p.http_get, now: Optional[dt.datetime] = None,
           window: dt.timedelta = dt.timedelta(days=2), env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Live check of one provider: fetch a recent window and report status,
    latency, item count and the newest item's age. Never returns secrets."""
    env = os.environ if env is None else env
    spec = SPECS[name]
    now = now or dt.datetime.now(dt.timezone.utc)
    out: dict[str, Any] = {"provider": name, "kind": spec.kind, "research_only": spec.research_only,
                           "archive": spec.archive, "time_precision": spec.time_precision}
    if spec.needs_env and not env.get(spec.needs_env):
        return {**out, "status": "NOT_CONFIGURED", "detail": f"{spec.needs_env} is not set"}
    t0 = time.monotonic()
    try:
        prov = build(name, get=get)
        items = list(prov.fetch(now - window, now))
        errors = list(getattr(prov, "errors", []))
    except p.NotConfigured as e:
        return {**out, "status": "NOT_CONFIGURED", "detail": str(e)[:200]}
    except p.RateLimited as e:
        return {**out, "status": "RATE_LIMITED", "detail": str(e)[:200], "latency_s": round(time.monotonic() - t0, 2)}
    except p.ProviderError as e:
        return {**out, "status": "DOWN", "detail": str(e)[:200], "latency_s": round(time.monotonic() - t0, 2)}
    newest = max((i.published_at for i in items), default=None)
    status = "DEGRADED" if errors else ("OK" if items else "EMPTY")
    return {**out, "status": status, "latency_s": round(time.monotonic() - t0, 2), "items": len(items),
            "newest_item_age_hours": round((now - newest).total_seconds() / 3600, 1) if newest else None,
            "date_only_items": sum(1 for i in items if i.time_precision == "date"), "errors": errors[:5]}
