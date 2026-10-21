"""
scripts/research/results_surprise_study.py — do quarterly results surprises
predict Indian stock returns once the surprise is known? (RESEARCH ONLY;
network for --fetch; writes one JSON report and a cache outside the repo.)

    python scripts/research/results_surprise_study.py --fetch      # download (cached), then analyse
    python scripts/research/results_surprise_study.py              # analyse the cache only

Why: the price/technical models of Phases 12-15 showed no skill (accuracy
~50%, AUC ~0.50), and point-in-time news could not be replayed. Results
events can: yfinance's earnings history gives, per quarter, the announcement
time, the analysts' EPS estimate and the reported EPS (about 6 years).

Event timing (announcement time converted to IST; NSE session 09:15-15:30):
  PRE_OPEN     announced before 09:15 on a trading day -> known before that open
  AFTER_CLOSE  announced at/after 15:30 -> known before the next session's open
  INTRADAY     announced 09:15-15:30 -> the market reacts the same day
  UNTIMED      midnight timestamp (time unknown) -> date only, reported apart
T0 = the first session whose open comes after the announcement.

Surprise: Yahoo's Surprise(%) and an ADJUSTED surprise = surprise minus the
median surprise of the same stock's previous 8 quarters (point in time, at
least 4 needed). This removes systematic offsets (e.g. estimates on another
basis: HDFC AMC shows about -30% every quarter). Rows with |estimate| < 0.1
or |surprise| > 300% are dropped.

Returns (adjusted prices), each minus NIFTY 50 over the same window:
  gap          close(T0-1) -> open(T0)      not tradable; shows the reaction
  day0         open(T0)    -> close(T0)     buy at the open after the news
  d5 / d20     open(T0)    -> close(T0+4) / close(T0+19)
  drift5/20    close(T0)   -> close(T0+5) / close(T0+20)   after the first day
Net = excess minus a 0.20% round-trip cost (one-day and multi-day alike).

Periods: DEV 2021-2023, TEST 2024-2026 (an effect counts only if it holds in
both). Universe: today's list (survivorship bias: delisted stocks missing).
Yahoo's estimates and timestamps are not exchange data; results are a
research lead, never a guarantee.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "research" / "output" / "results_surprise_study.json"
CACHE = Path(os.environ.get("STOCKLENS_RESULTS_CACHE") or Path(tempfile.gettempdir()) / "stocklens_results_cache")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OPEN, CLOSE = dt.time(9, 15), dt.time(15, 30)
COST = 0.0020
START = "2020-06-01"
BUCKETS = [(-1e9, -20, "big miss (< -20%)"), (-20, -5, "miss (-20% to -5%)"), (-5, 5, "in line (-5% to +5%)"),
           (5, 20, "beat (+5% to +20%)"), (20, 1e9, "big beat (> +20%)")]
WINDOWS = ("gap", "day0", "d5", "d20", "drift5", "drift20")


def universe() -> list[str]:
    syms = set()
    for f in ("largecap", "midcap", "smallcap", "nifty500"):
        p = ROOT / "data" / f"{f}.csv"
        if p.exists():
            syms |= {r["Symbol"].strip() for r in csv.DictReader(p.open(encoding="utf-8")) if r.get("Symbol")}
    return sorted(syms)


# ── fetch (cached) ───────────────────────────────────────────────────────────

def fetch(symbols: list[str]) -> None:
    import yfinance as yf
    (CACHE / "earnings").mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not (CACHE / "earnings" / f"{s}.json").exists()]
    print(f"earnings: {len(symbols) - len(todo)} cached, {len(todo)} to fetch")
    for i, s in enumerate(todo, 1):
        rows: list[dict] = []
        for attempt in range(3):
            try:
                df = yf.Ticker(s).get_earnings_dates(limit=28)
                if df is not None and not df.empty:
                    for ts, r in df.iterrows():
                        rows.append({"ts": ts.isoformat(), "estimate": _num(r.get("EPS Estimate")),
                                     "reported": _num(r.get("Reported EPS")), "surprise": _num(r.get("Surprise(%)"))})
                break
            except Exception as e:  # noqa: BLE001 - research fetch: retry, then record as empty
                print(f"  {s}: {type(e).__name__} {str(e)[:80]} (try {attempt + 1})")
                time.sleep(5 * (attempt + 1))
        (CACHE / "earnings" / f"{s}.json").write_text(json.dumps(rows), encoding="utf-8")
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(1.0)
    prices = CACHE / "prices.pkl"
    if not prices.exists():
        print("prices: downloading")
        frames = []
        allsyms = symbols + ["^NSEI"]
        for i in range(0, len(allsyms), 80):
            chunk = allsyms[i:i + 80]
            frames.append(yf.download(chunk, start=START, auto_adjust=True, progress=False, group_by="ticker",
                                      threads=True))
            time.sleep(2)
        pd.concat(frames, axis=1).to_pickle(prices)


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ── analysis ─────────────────────────────────────────────────────────────────

def classify(ts: dt.datetime) -> str:
    t = ts.astimezone(IST)
    if t.time() == dt.time(0, 0) or ts.time() == dt.time(0, 0):
        return "UNTIMED"
    if t.time() < OPEN:
        return "PRE_OPEN"
    if t.time() >= CLOSE:
        return "AFTER_CLOSE"
    return "INTRADAY"


def first_session(sessions: pd.DatetimeIndex, ts: dt.datetime, kind: str) -> Optional[int]:
    """Index of T0: the first session whose open is after the announcement."""
    d = ts.astimezone(IST).date()
    if kind == "AFTER_CLOSE":
        d = d + dt.timedelta(days=1)
    i = sessions.searchsorted(pd.Timestamp(d))
    return int(i) if i < len(sessions) else None


def adjusted_surprises(rows: list[dict]) -> list[dict]:
    rows = sorted((r for r in rows if r["reported"] is not None and r["estimate"] is not None
                   and r["surprise"] is not None and abs(r["estimate"]) >= 0.1 and abs(r["surprise"]) <= 300),
                  key=lambda r: r["ts"])
    for i, r in enumerate(rows):
        prior = [p["surprise"] for p in rows[max(0, i - 8):i]]
        r["adj"] = r["surprise"] - statistics.median(prior) if len(prior) >= 4 else None
    return rows


def events(symbols: list[str]) -> list[dict]:
    px = pd.read_pickle(CACHE / "prices.pkl")
    nifty = px["^NSEI"].dropna(subset=["Close"])
    sessions = nifty.index
    out = []
    for s in symbols:
        f = CACHE / "earnings" / f"{s}.json"
        if not f.exists() or s not in px.columns.get_level_values(0):
            continue
        p = px[s].reindex(sessions)
        for r in adjusted_surprises(json.loads(f.read_text(encoding="utf-8"))):
            ts = dt.datetime.fromisoformat(r["ts"])
            if ts > dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=35):
                continue                                       # d20 window not complete yet
            kind = classify(ts)
            i = first_session(sessions, ts, kind if kind != "UNTIMED" else "PRE_OPEN")
            if i is None or i < 1 or i + 20 >= len(sessions):
                continue
            o, c = p["Open"], p["Close"]
            no, nc = nifty["Open"], nifty["Close"]

            def ret(a, b, sa, sb):
                x0, x1 = sa.iloc[a], sb.iloc[b]
                return None if pd.isna(x0) or pd.isna(x1) or x0 <= 0 else float(x1 / x0 - 1)
            row = {"symbol": s, "date": sessions[i].date().isoformat(), "kind": kind, "surprise": r["surprise"],
                   "adj": r["adj"], "year": sessions[i].year}
            spans = {"gap": (i - 1, i, c, o, nc, no), "day0": (i, i, o, c, no, nc), "d5": (i, i + 4, o, c, no, nc),
                     "d20": (i, i + 19, o, c, no, nc), "drift5": (i, i + 5, c, c, nc, nc),
                     "drift20": (i, i + 20, c, c, nc, nc)}
            for w, (a, b, sa, sb, na, nb) in spans.items():
                st, ix = ret(a, b, sa, sb), ret(a, b, na, nb)
                row[w] = None if st is None or ix is None or abs(st) > 0.6 else st - ix
            out.append(row)
    return out


def summarize(vals: list[float], cost: float) -> Optional[dict]:
    v = [x for x in vals if x is not None]
    if len(v) < 20:
        return {"n": len(v)} if v else None
    m = statistics.fmean(v)
    sd = statistics.pstdev(v)
    se = sd / math.sqrt(len(v))
    return {"n": len(v), "mean_excess_pct": round(100 * m, 2), "net_after_cost_pct": round(100 * (m - cost), 2),
            "median_pct": round(100 * statistics.median(v), 2), "hit_rate_pct": round(100 * sum(x > 0 for x in v) / len(v), 1),
            "ci95_pct": [round(100 * (m - 1.96 * se), 2), round(100 * (m + 1.96 * se), 2)]}


def report(evts: list[dict]) -> dict:
    def bucket(x):
        return next((name for lo, hi, name in BUCKETS if x is not None and lo <= x < hi), None)
    out: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "research_only": True,
                           "events": len(evts), "by_kind": {}, "method": __doc__.split("\n\n")[2:5]}
    for k in ("PRE_OPEN", "AFTER_CLOSE", "INTRADAY", "UNTIMED"):
        out["by_kind"][k] = sum(e["kind"] == k for e in evts)
    known = [e for e in evts if e["kind"] in ("PRE_OPEN", "AFTER_CLOSE")]       # known before an open
    tables: dict[str, Any] = {}
    for label, pool, key in (("known_before_open_adjusted", known, "adj"), ("known_before_open_raw", known, "surprise"),
                             ("intraday_adjusted", [e for e in evts if e["kind"] == "INTRADAY"], "adj")):
        t: dict[str, Any] = {}
        for period, yrs in (("DEV 2021-2023", range(2021, 2024)), ("TEST 2024-2026", range(2024, 2027))):
            rows = [e for e in pool if e["year"] in yrs]
            t[period] = {name: {w: summarize([e[w] for e in rows if bucket(e[key]) == name],
                                             0.0 if w == "gap" else COST) for w in WINDOWS}
                         for _, _, name in BUCKETS}
        tables[label] = t
    out["tables"] = tables
    # the simplest tradable rule: buy at the open after a big adjusted beat, hold 5 / 20 sessions
    rule = {}
    for period, yrs in (("DEV 2021-2023", range(2021, 2024)), ("TEST 2024-2026", range(2024, 2027))):
        big = [e for e in known if e["year"] in yrs and e["adj"] is not None and e["adj"] >= 20]
        miss = [e for e in known if e["year"] in yrs and e["adj"] is not None and e["adj"] <= -20]
        rule[period] = {"buy_big_beat_hold5": summarize([e["d5"] for e in big], COST),
                        "buy_big_beat_hold20": summarize([e["d20"] for e in big], COST),
                        "big_miss_hold5_for_contrast": summarize([e["d5"] for e in miss], COST)}
    out["simple_rule"] = rule
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download missing earnings and prices first (cached)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols (quick trial)")
    args = ap.parse_args(argv)
    syms = universe()[: args.limit] if args.limit else universe()
    print(f"universe {len(syms)} symbols; cache {CACHE}")
    if args.fetch:
        fetch(syms)
    evts = events(syms)
    rep = report(evts)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"events": rep["events"], "by_kind": rep["by_kind"], "simple_rule": rep["simple_rule"]}, indent=1))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
