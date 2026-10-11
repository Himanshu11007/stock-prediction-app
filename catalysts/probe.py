"""
catalysts/probe.py — live validation of keyed news providers (Marketaux,
Alpha Vantage) before they are benchmarked. RESEARCH ONLY.

For each provider, with a small fixed request budget (Marketaux 6, Alpha
Vantage 5, inside both free daily quotas):

  configured     the credential variable is set (its value is never read
                 into the report, printed or logged)
  connectivity   a recent-window request returns HTTP 200
  authentication the provider accepts the key (401 / key messages -> REJECTED)
  rate_limit     402 / 429 / quota messages observed (-> RATE_LIMITED)
  schema         required fields present in the raw JSON
  timestamps     parse rate, stated zone, future-dated items, placeholder
                 times (00:00:00), share with an exact time of day
  history        articles returned for windows 30, 365 and 730 days back
  india          articles with an Indian listing among the provider's own
                 entity tags, and per-symbol results for large caps

Every check is PASS, FAIL, NOT_TESTED (with the reason) or an observed
value. An unconfigured provider is NOT_CONFIGURED: never FAIL.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Callable, Optional

from catalysts import adapters as ad
from catalysts.providers import NotConfigured, ProviderError, RateLimited, RawArticle, http_get

UTC = dt.timezone.utc
REQUIRED = {
    "marketaux": {"top": ("data", "meta"), "item": ("uuid", "title", "url", "published_at", "source", "entities"),
                  "entity": ("symbol", "exchange", "country", "match_score")},
    "alphavantage": {"top": ("feed",), "item": ("title", "url", "time_published", "source", "ticker_sentiment"),
                     "entity": ("ticker", "relevance_score")},
}
LARGE_CAPS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
ENV = {"marketaux": "MARKETAUX_API_TOKEN", "alphavantage": "ALPHAVANTAGE_API_KEY"}


class Recorder:
    """Wraps an HTTP get: keeps status codes and the last body for the schema
    check. URLs are never stored (they can carry the key)."""

    def __init__(self, get=http_get):
        self.get, self.statuses, self.last_body = get, [], b""

    def __call__(self, url, timeout=20.0, headers=None):
        status, body = self.get(url, timeout, headers) if headers else self.get(url, timeout)
        self.statuses.append(status)
        self.last_body = body
        return status, body


def schema_check(provider: str, body: bytes) -> dict[str, Any]:
    req = REQUIRED[provider]
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        return {"status": "FAIL", "detail": "response is not JSON"}
    if not isinstance(data, dict):
        return {"status": "FAIL", "detail": "response is not a JSON object"}
    missing_top = [k for k in req["top"] if k not in data]
    items = data.get("data") if provider == "marketaux" else data.get("feed")
    items = items if isinstance(items, list) else []
    if missing_top:
        return {"status": "FAIL", "missing": missing_top, "keys": sorted(data)[:10]}
    if not items:
        return {"status": "NOT_TESTED", "detail": "no items returned to check"}
    miss_item = sorted({k for it in items for k in req["item"] if k not in it})
    ents = [e for it in items for e in (it.get("entities") if provider == "marketaux" else it.get("ticker_sentiment")) or []]
    miss_ent = sorted({k for e in ents for k in req["entity"] if k not in e})
    return {"status": "PASS" if not miss_item and not miss_ent else "FAIL", "items_checked": len(items),
            "missing_item_fields": miss_item, "missing_entity_fields": miss_ent}


def timestamp_check(arts: list[RawArticle], now: dt.datetime) -> dict[str, Any]:
    if not arts:
        return {"status": "NOT_TESTED", "detail": "no articles"}
    zones = sorted({a.published_tz or "NONE" for a in arts})
    future = sum(1 for a in arts if a.published_at > now + dt.timedelta(minutes=10))
    midnight = sum(1 for a in arts if a.published_at.astimezone(UTC).strftime("%H:%M:%S") == "00:00:00")
    return {"status": "PASS" if not future else "FAIL", "articles": len(arts), "stated_zones": zones,
            "future_dated": future, "midnight_placeholder_share": round(midnight / len(arts), 3),
            "seconds_resolution_share": round(sum(1 for a in arts if a.published_at.second) / len(arts), 3),
            "note": "Alpha Vantage states no zone; UTC is assumed until verified against a known release time"
            if any((a.published_tz or "").startswith("NONE") for a in arts) else None}


def _call(fn: Callable[[], list[RawArticle]]) -> tuple[str, list[RawArticle], Optional[str]]:
    try:
        return "OK", fn(), None
    except NotConfigured as e:
        return "REJECTED", [], str(e)[:160]
    except RateLimited as e:
        return "RATE_LIMITED", [], str(e)[:160]
    except ProviderError as e:
        return "ERROR", [], str(e)[:160]


def probe(provider: str, now: Optional[dt.datetime] = None, get=http_get, env: Optional[dict] = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    now = now or dt.datetime.now(UTC)
    out: dict[str, Any] = {"provider": provider, "probed_at": now.isoformat(), "credential_variable": ENV[provider],
                           "configured": bool(env.get(ENV[provider]))}
    if not out["configured"]:
        out["status"] = "NOT_CONFIGURED"
        out["checks"] = {k: {"status": "NOT_TESTED", "detail": f"{ENV[provider]} is not set"}
                         for k in ("connectivity", "authentication", "rate_limit", "schema", "timestamps", "history", "india")}
        return out
    key = env[ENV[provider]]
    rec = Recorder(get)

    def make(**kw):
        if provider == "marketaux":
            return ad.MarketauxProvider(api_token=key, get=rec, sleep=lambda s: None, max_pages=1, **kw)
        return ad.AlphaVantageNewsProvider(api_key=key, get=rec, sleep=lambda s: None, **kw)

    recent = (now - dt.timedelta(days=1), now)
    kw = {"countries": "in"} if provider == "marketaux" else {}
    status, arts, err = _call(lambda: list(make(**kw).fetch(*recent)))
    checks: dict[str, Any] = {
        "connectivity": {"status": "PASS" if rec.statuses and rec.statuses[-1] == 200 else "FAIL",
                         "http_statuses": rec.statuses[:], "detail": err},
        "authentication": {"status": "FAIL" if status == "REJECTED" else "PASS" if status in ("OK", "RATE_LIMITED")
                           else "NOT_TESTED", "detail": err if status == "REJECTED" else None},
    }
    checks["schema"] = schema_check(provider, rec.last_body) if status == "OK" else {"status": "NOT_TESTED", "detail": status}
    checks["timestamps"] = timestamp_check(arts, now)
    limited = status == "RATE_LIMITED"
    hist: dict[str, Any] = {}
    if status == "OK":
        for days in (30, 365, 730):
            lo = now - dt.timedelta(days=days)
            st, a, e = _call(lambda lo=lo: list(make(**kw).fetch(lo, lo + dt.timedelta(days=1))))
            hist[f"{days}d_ago"] = {"status": st, "articles": len(a), "detail": e}
            limited |= st == "RATE_LIMITED"
    checks["history"] = hist or {"status": "NOT_TESTED", "detail": status}
    india: dict[str, Any] = {"tagged_indian_share_recent": round(sum(1 for a in arts if a.symbols) / len(arts), 3) if arts else None}
    if status == "OK":
        syms = LARGE_CAPS if provider == "marketaux" else LARGE_CAPS[:1]          # Alpha Vantage: 25 requests a day
        if provider == "marketaux":
            st, a, e = _call(lambda: list(make(symbols=[f"{s}.NS" for s in syms], countries="").fetch(now - dt.timedelta(days=7), now)))
        else:
            st, a, e = _call(lambda: list(make(tickers=[f"{s}.BSE" for s in syms]).fetch(now - dt.timedelta(days=7), now)))
        found = sorted({s for x in a for s in x.symbols})
        india.update(status=st, symbols_requested=[f"{s}.NS" for s in syms], articles=len(a), symbols_found=found,
                     detail=e)
        limited |= st == "RATE_LIMITED"
    checks["india"] = india
    checks["rate_limit"] = {"status": "OBSERVED" if limited else "NOT_REACHED", "requests": len(rec.statuses),
                            "http_statuses": rec.statuses}
    out["checks"] = checks
    out["status"] = "REJECTED" if status == "REJECTED" else "RATE_LIMITED" if status == "RATE_LIMITED" else \
        "OK" if all(checks[k].get("status") in ("PASS", None) for k in ("connectivity", "authentication", "schema")) else "DEGRADED"
    return out
