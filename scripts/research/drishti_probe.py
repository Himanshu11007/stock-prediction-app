"""
scripts/research/drishti_probe.py — first check of a Drishti API key
(research only; network; writes nothing).

    python scripts/research/drishti_probe.py [--symbol RELIANCE] [--days 7]

Reads DRISHTI_API_KEY from the environment and never prints it. Calls the
free account routes (plan, credits, enabled products) and ONE news page for
one symbol (1 credit if it returns items), then shows what the items look
like: fields present, the raw `date` format and whether it states a time
zone, sources, sentiment values and links - the facts the connector
(catalysts.adapters.DrishtiNewsProvider) assumes but the documentation does
not state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from catalysts import providers as prov  # noqa: E402
from catalysts.adapters import DrishtiNewsProvider  # noqa: E402

BASE = "https://developers.manasija.in/v1"


def _get(path: str, key: str) -> tuple[int, dict]:
    status, body = prov.http_get(BASE + path, 20.0, {"X-API-Key": key, "Accept": "application/json"})
    try:
        return status, json.loads(body or b"{}")
    except ValueError:
        return status, {"_non_json": True}


def _safe(d, keep=("status", "plan", "tier", "credits", "credit_balance", "balance", "products", "apis", "enabled",
                   "rest_products", "credits_used_today", "requests_today", "limits", "rpm", "daily")):
    """Only plan / credit / product fields; anything else (ids, e-mail) is dropped."""
    if isinstance(d, dict):
        inner = d.get("data", d)
        if isinstance(inner, dict):
            return {k: v for k, v in inner.items() if any(w in k.lower() for w in keep)}
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol", default="RELIANCE")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args(argv)
    key = os.environ.get("DRISHTI_API_KEY", "")
    if not key:
        print("DRISHTI_API_KEY is not set in this environment (nothing was called).")
        return 2
    st, acct = _get("/account", key)
    print(f"account: HTTP {st}")
    if st != 200:
        print("  the key was not accepted; check it in the Drishti developer portal")
        return 1
    print("  " + json.dumps(_safe(acct), default=str)[:800])
    st, usage = _get("/account/usage", key)
    print(f"usage: HTTP {st}\n  " + json.dumps(_safe(usage), default=str)[:800])

    now = dt.datetime.now(dt.timezone.utc)
    p = DrishtiNewsProvider(api_key=key, symbols=[args.symbol], max_pages=1)
    try:
        items = list(p.fetch(now - dt.timedelta(days=args.days), now))
    except prov.ProviderError as e:
        print(f"news: {type(e).__name__}: {e}")
        return 1
    print(f"\nnews for {args.symbol}, last {args.days} days: {len(items)} item(s) (1 credit if any)")
    for a in items[:5]:
        print(f"  - raw date {a.published_raw!r} (zone {a.published_tz}, {a.time_precision}) -> {a.published_at:%Y-%m-%d %H:%M} UTC")
        print(f"    {a.title[:100]}")
        print(f"    source {a.source_name!r} | sentiment {a.raw.get('sentiment')!r} | type {a.raw.get('article_type')!r} "
              f"| link {'yes' if a.url else 'no'} | symbols {a.symbols}")
    if items:
        zones = Counter(a.published_tz for a in items)
        print(f"\ntime zones stated: {dict(zones)}"
              + ("  <- no zone: the connector reads these as UTC (cautious); confirm with Drishti" if
                 any(str(z).startswith("NONE") for z in zones) else ""))
        print(f"sources: {dict(Counter(a.source_name for a in items).most_common(5))}")
    st, usage = _get("/account/usage", key)
    print(f"\nusage after: HTTP {st}\n  " + json.dumps(_safe(usage), default=str)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
