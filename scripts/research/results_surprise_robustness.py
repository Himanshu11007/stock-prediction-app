"""
scripts/research/results_surprise_robustness.py — robustness of the results
surprise study, plus the pre-results test (RESEARCH ONLY; reads the cache
written by results_surprise_study.py --fetch; no network).

    python scripts/research/results_surprise_robustness.py [--db storage/app.db]

A. Is "moderate beat -> outperform NIFTY over 20 sessions" (and "miss ->
   underperform") stable? Same events and definitions as the study (known
   before an open, adjusted surprise, buy at the open of T0, 0.20% cost):
     * by year (2021 ... 2026)
     * by liquidity: median traded value of the 60 sessions before T0
       (top half vs bottom half)
     * by sector (where the local stock master knows it)
     * clustering: results arrive in seasons, so one CI per EVENT is too
       optimistic; here each calendar week's events are averaged first and
       the CI is taken across weeks
B. Pre-results: do companies that beat in at least 3 of their previous 4
   quarters rise in the 5 sessions before the next announcement (close of
   T0-6 to close of T0-1, all before the news) more than companies that beat
   in at most 1 of 4? The announcement date itself is what a results
   calendar shows in advance (Yahoo: expected dates; exchange: confirmed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import results_surprise_study as st  # noqa: E402

OUT = st.ROOT / "scripts" / "research" / "output" / "results_surprise_robustness.json"
BEAT, MISS = (5, 20), (-20, -5)


def ci(vals, cost=st.COST):
    v = [x for x in vals if x is not None]
    if len(v) < 15:
        return {"n": len(v)}
    m, se = statistics.fmean(v), statistics.pstdev(v) / math.sqrt(len(v))
    return {"n": len(v), "net_pct": round(100 * (m - cost), 2),
            "ci95_net_pct": [round(100 * (m - cost - 1.96 * se), 2), round(100 * (m - cost + 1.96 * se), 2)],
            "hit_pct": round(100 * sum(x > 0 for x in v) / len(v), 1)}


def weekly_ci(rows, key="d20", cost=st.COST):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            y, w, _ = dt.date.fromisoformat(r["date"]).isocalendar()
            by[(y, w)].append(r[key])
    wk = [statistics.fmean(v) for v in by.values()]
    out = ci(wk, cost)
    out["weeks"] = out.pop("n")
    return out


def enrich(symbols):
    """Events plus T0 index facts: liquidity before T0 and the pre-results move."""
    px = pd.read_pickle(st.CACHE / "prices.pkl")
    nifty = px["^NSEI"].dropna(subset=["Close"])
    sessions = nifty.index
    out = []
    for s in symbols:
        f = st.CACHE / "earnings" / f"{s}.json"
        if not f.exists() or s not in px.columns.get_level_values(0):
            continue
        p = px[s].reindex(sessions)
        tv = (p["Close"] * p["Volume"])
        rows = st.adjusted_surprises(json.loads(f.read_text(encoding="utf-8")))
        for k, r in enumerate(rows):
            ts = dt.datetime.fromisoformat(r["ts"])
            if ts > dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=35):
                continue
            kind = st.classify(ts)
            i = st.first_session(sessions, ts, kind if kind != "UNTIMED" else "PRE_OPEN")
            if i is None or i < 61 or i + 20 >= len(sessions):
                continue
            o, c, nc, no = p["Open"], p["Close"], nifty["Close"], nifty["Open"]

            def ex(a, b, sa, sb, na, nb):
                x0, x1, y0, y1 = sa.iloc[a], sb.iloc[b], na.iloc[a], nb.iloc[b]
                if any(pd.isna(v) for v in (x0, x1, y0, y1)) or x0 <= 0 or y0 <= 0:
                    return None
                d = float(x1 / x0 - 1)
                return None if abs(d) > 0.6 else d - float(y1 / y0 - 1)
            prior = rows[max(0, k - 4):k]
            beats = sum(1 for q in prior if q["surprise"] > 0) if len(prior) == 4 else None
            out.append({"symbol": s, "date": sessions[i].date().isoformat(), "year": sessions[i].year, "kind": kind,
                        "adj": r["adj"], "d5": ex(i, i + 4, o, c, no, nc), "d20": ex(i, i + 19, o, c, no, nc),
                        "pre5": ex(i - 6, i - 1, c, c, nc, nc), "beats_prev4": beats,
                        "liq": float(tv.iloc[i - 60:i].median()) if tv.iloc[i - 60:i].notna().sum() > 30 else None})
    return out


def in_bucket(x, b):
    return x is not None and b[0] <= x < b[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(Path("C:/Python/Git/stock-prediction-app/storage/app.db")))
    args = ap.parse_args(argv)
    ev = enrich(st.universe())
    known = [e for e in ev if e["kind"] in ("PRE_OPEN", "AFTER_CLOSE")]
    sectors = {}
    try:
        con = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
        sectors = dict(con.execute("select symbol, sector from companies where sector is not null"))
    except sqlite3.Error:
        pass
    rep: dict = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "research_only": True,
                 "events": len(ev), "known_before_open": len(known)}
    beat = [e for e in known if in_bucket(e["adj"], BEAT)]
    miss = [e for e in known if in_bucket(e["adj"], MISS)]
    rep["by_year"] = {str(y): {"beat_d20": ci([e["d20"] for e in beat if e["year"] == y]),
                               "miss_d20": ci([e["d20"] for e in miss if e["year"] == y])}
                      for y in range(2021, 2027)}
    liq = sorted(e["liq"] for e in known if e["liq"] is not None)
    mid = liq[len(liq) // 2] if liq else 0
    rep["by_liquidity"] = {name: {"beat_d20": ci([e["d20"] for e in beat if e["liq"] is not None and f(e["liq"])]),
                                  "miss_d20": ci([e["d20"] for e in miss if e["liq"] is not None and f(e["liq"])])}
                           for name, f in (("more liquid half", lambda x: x >= mid), ("less liquid half", lambda x: x < mid))}
    sec: dict = defaultdict(lambda: {"beat": [], "miss": []})
    for e in beat:
        if e["symbol"] in sectors:
            sec[sectors[e["symbol"]]]["beat"].append(e["d20"])
    for e in miss:
        if e["symbol"] in sectors:
            sec[sectors[e["symbol"]]]["miss"].append(e["d20"])
    rep["by_sector"] = {k: {"beat_d20": ci(v["beat"]), "miss_d20": ci(v["miss"])} for k, v in sorted(sec.items())}
    rep["clustered_by_week"] = {
        p: {"beat_d20": weekly_ci([e for e in beat if e["year"] in yrs]),
            "miss_d20": weekly_ci([e for e in miss if e["year"] in yrs]),
            "beat_d5": weekly_ci([e for e in beat if e["year"] in yrs], "d5")}
        for p, yrs in (("2021-2023", range(2021, 2024)), ("2024-2026", range(2024, 2027)))}
    rep["pre_results"] = {
        p: {"beat_3_or_4_of_last_4": ci([e["pre5"] for e in ev if e["year"] in yrs and (e["beats_prev4"] or 0) >= 3]),
            "beat_0_or_1_of_last_4": ci([e["pre5"] for e in ev if e["year"] in yrs and e["beats_prev4"] is not None
                                         and e["beats_prev4"] <= 1])}
        for p, yrs in (("2021-2023", range(2021, 2024)), ("2024-2026", range(2024, 2027)))}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
