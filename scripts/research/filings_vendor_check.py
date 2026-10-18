"""
scripts/research/filings_vendor_check.py — does a vendor deliver the exchange
filings and results dates we need? (RESEARCH ONLY; network; writes one JSON.)

    python scripts/research/filings_vendor_check.py --vendor drishti
    python scripts/research/filings_vendor_check.py --vendor stockinsights [--since 2026-09-01] [--max-events 10]

Ground truth: the verified NSE filings (official dissemination time to the
second) stored in the news-provider benchmark report. For each recent filing
the vendor is asked for that symbol's filings around the official time, and
the closest vendor filing from 30 min before to 48 h after it is taken:

  matched     within 6 h of the official time  (delay = vendor time - official)
  late        6-48 h after
  not_found   nothing in the window

`before_reaction` = the vendor had it by the market reaction time (the official
time if NSE was open, else the next 09:15 IST) + 15 min. Results calendar:
the vendor's next result dates for a few symbols next to yfinance's.

Cost: drishti 1 credit per returned filing (narrow windows; balance shown
before / after); stockinsights 1 call per filing checked + 1 calendar call
(trial: 100 calls). Keys: DRISHTI_API_KEY / STOCKINSIGHTS_API_KEY, never printed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from catalysts import providers as prov  # noqa: E402
from catalysts.benchmark import market_reaction_time  # noqa: E402
from catalysts.filing_vendors import VENDORS, Filing  # noqa: E402

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OUT_DIR = Path(__file__).resolve().parent / "output" / "prediction_v2"
REPORT = OUT_DIR / "news_provider_benchmark.json"
CALENDAR_SYMBOLS = ("HDFCAMC", "CANBK", "RELIANCE", "AARTIIND", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
MATCH_BEFORE, MATCH_NEAR, MATCH_LATE = dt.timedelta(minutes=30), dt.timedelta(hours=6), dt.timedelta(hours=48)


def recent_filings(report: Path, since: str, max_events: int) -> list[dict[str, Any]]:
    d = json.loads(report.read_text(encoding="utf-8"))
    evs = [e for e in d.get("events") or [] if e.get("kind") == "nse_filing" and e.get("expected_symbols")
           and e.get("official_time", "") >= since]
    return sorted(evs, key=lambda e: e["official_time"], reverse=True)[:max_events]


def judge(official: dt.datetime, items: list[Filing], holidays: tuple[str, ...] = ()) -> dict[str, Any]:
    near = [f for f in items if official - MATCH_BEFORE <= f.time <= official + MATCH_LATE]
    if not near:
        return {"status": "not_found", "returned": len(items)}
    best = min(near, key=lambda f: abs((f.time - official).total_seconds()))
    delay = (best.time - official).total_seconds() / 60
    status = "matched" if best.time <= official + MATCH_NEAR else "late"
    reaction = market_reaction_time(official, holidays)
    return {"status": status, "returned": len(items), "delay_minutes": round(delay, 1),
            "before_reaction": best.time <= reaction + dt.timedelta(minutes=15) if best.precision == "exact" else None,
            "precision": best.precision, "title": best.title[:160], "category": best.category,
            "sentiment": best.sentiment, "link": bool(best.link)}


def yfinance_dates(symbols) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    try:
        import yfinance as yf
    except ImportError:
        return {s: None for s in symbols}
    for s in symbols:
        try:
            cal = yf.Ticker(f"{s}.NS").calendar
            ed = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
            out[s] = ed[0].isoformat() if ed else None
        except Exception:  # noqa: BLE001 - research comparison only
            out[s] = None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vendor", required=True, choices=sorted(VENDORS))
    ap.add_argument("--since", default="2026-09-01", help="only filings on or after this date (vendor history)")
    ap.add_argument("--max-events", type=int, default=10)
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--no-calendar", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    client = VENDORS[args.vendor]()
    if not client.key:
        print(f"{client.env} is not set in this environment (nothing was called).")
        return 2
    before = client.usage()
    rows = []
    for e in recent_filings(Path(args.report), args.since, args.max_events):
        official = dt.datetime.fromisoformat(e["official_time"]).astimezone(UTC)
        sym = e["expected_symbols"][0]
        if args.vendor == "drishti":                         # 1 credit per returned item: keep it narrow
            lo, hi = official - dt.timedelta(hours=2), official + dt.timedelta(hours=8)
        else:                                                # date filter: the filing day and the next
            lo, hi = official, official + dt.timedelta(days=1)
        try:
            res = judge(official, client.filings(sym, lo, hi))
        except prov.RateLimited as ex:
            res = {"status": "rate_limited", "detail": str(ex)[:120]}
        except prov.ProviderError as ex:
            res = {"status": "error", "detail": f"{type(ex).__name__}: {str(ex)[:120]}"}
        rows.append({"symbol": sym, "official_time": e["official_time"], "event": e.get("title"), **res})
        print(f"  {e['official_time'][:16]}  {sym:14} {res['status']:12} "
              + (f"delay {res['delay_minutes']:>7} min  before reaction: {res.get('before_reaction')}  | {res.get('category')}"
                 if "delay_minutes" in res else f"returned {res.get('returned', '-')}  {res.get('detail', '')}"))
        if res["status"] in ("rate_limited",) or str(res.get("detail", "")).startswith(("NotConfigured", "NotPermitted")):
            break
    calendar = []
    if not args.no_calendar:
        try:
            got = {c.symbol.removesuffix(".NS"): c for c in client.results_calendar(CALENDAR_SYMBOLS)}
            yfd = yfinance_dates(CALENDAR_SYMBOLS)
            for s in CALENDAR_SYMBOLS:
                c = got.get(s)
                calendar.append({"symbol": s, "vendor_date": c.date.isoformat() if c else None,
                                 "yfinance_date": yfd.get(s), **({"detail": c.detail} if c else {})})
        except prov.ProviderError as ex:
            calendar = [{"error": f"{type(ex).__name__}: {str(ex)[:120]}"}]
        print("\nresults calendar (vendor vs yfinance):")
        for c in calendar:
            print(f"  {c}")
    after = client.usage()
    n = len(rows)
    summary = {
        "filings_checked": n, "matched": sum(r["status"] == "matched" for r in rows),
        "late": sum(r["status"] == "late" for r in rows), "not_found": sum(r["status"] == "not_found" for r in rows),
        "errors": sum(r["status"] in ("error", "rate_limited") for r in rows),
        "median_delay_minutes": median(delays) if (delays := [r["delay_minutes"] for r in rows if "delay_minutes" in r])
        else None,
        "before_reaction": sum(1 for r in rows if r.get("before_reaction")),
        "calls": client.calls, "credits_before": before, "credits_after": after,
    }
    print(f"\nsummary: {json.dumps(summary)}")
    out = Path(args.out) if args.out else OUT_DIR / f"filings_vendor_{args.vendor}.json"
    out.write_text(json.dumps({"vendor": args.vendor, "generated_at": dt.datetime.now(UTC).isoformat(),
                               "research_only": True, "since": args.since, "summary": summary,
                               "filings": rows, "results_calendar": calendar}, indent=1), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
