"""
scripts/research/factor_shock_replay.py — point-in-time replay of the factor
transmission hypotheses on price-defined shocks (read-only research).

  python scripts/research/factor_shock_replay.py [--period 3y]

Why price-defined shocks: no historical news archive with original
availability times is accessible (docs/NEWS_ENGINE.md). Large overnight
moves in Brent, gold, USD/INR and US yields are the market's own record of
macro / geopolitical / commodity news, with exact timestamps, so they test
the transmission half of the news chain without look-ahead.

Rules fixed before running (no tuning; thresholds are a priori):
  shock      the factor's last COMPLETED daily bar dated before Indian session
             t has |return| >= 2 x the stdev of the previous 60 bars (past only)
  call       for every target of every registered hypothesis on that factor:
             direction = target sign x sign(factor move) x factor_sign;
             sign-0 (ambiguous) targets make no call
  trade      TODAY_PREOPEN semantics: enter at the open of t, exit at the close
             of t (the overnight gap is known at entry and is NOT counted);
             intraday shorts are allowed in Indian cash equities
  priced     gap = open(t) / close(t-1) - 1, reported per call: how much of the
             move was in the price before anyone could act
  costs      15 and 25 bps round trip (incl. 5 bps slippage)
  splits     chronological by shock date: 60% development, 20% validation,
             20% holdout (the holdout is read once)
  baselines  random direction on the same stock-days (seeded); always neutral;
             gap-following (trade in the gap's direction) on the same stock-days
  stats      Wilson interval for hit rate; date-clustered bootstrap for mean
             net return; excess vs NIFTY open-to-close the same day
Limitations: Yahoo daily bars (adjusted, delayed); today's classifications;
survivorship; the factor bar boundary is approximate (part of a factor's
daily move can occur during the previous Indian session).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from catalysts.transmission import HYPOTHESES  # noqa: E402
from fundamentals.provider import fetch_price_history  # noqa: E402
from prediction_v2.performance import mean_ci, wilson  # noqa: E402

DB = Path(r"C:\Python\Git\stock-prediction-app\storage\app.db")
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "factor_shock_replay.json"
Z = 2.0
LOOKBACK = 60
COSTS = (15, 25)


def cluster_ci(rows: list[dict], cost: float, n: int = 1000, seed: int = 3):
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["date"], []).append(r["ret"] - cost / 1e4)
    days = [(sum(v), len(v)) for v in by.values()]
    if len(days) < 8:
        return None
    rng, means = random.Random(seed), []
    for _ in range(n):
        tot = cnt = 0
        for _ in days:
            a, b = days[rng.randrange(len(days))]
            tot, cnt = tot + a, cnt + b
        means.append(tot / cnt)
    means.sort()
    return (round(means[int(0.025 * n)], 5), round(means[int(0.975 * n) - 1], 5))


def summarize(rows: list[dict], key: str = "ret") -> dict:
    out = {"calls": len(rows), "shock_days": len({r["date"] for r in rows})}
    if not rows:
        return out
    hits = sum(1 for r in rows if r[key] > 0)
    out.update(hit_rate=round(hits / len(rows), 4), hit_ci95=wilson(hits, len(rows)),
               mean_gross=round(sum(r[key] for r in rows) / len(rows), 5),
               median_gross=round(sorted(r[key] for r in rows)[len(rows) // 2], 5),
               mean_excess_vs_nifty=round(sum(r["excess"] for r in rows) / len(rows), 5),
               mean_gap_in_call_direction=round(sum(r["gap_dir"] for r in rows) / len(rows), 5))
    for c in COSTS:
        net = [r[key] - c / 1e4 for r in rows]
        ci = mean_ci(net)
        out[f"net_{c}bps"] = {"mean": round(ci[0], 5), "ci95_naive": [round(x, 5) for x in ci[1:]] if ci else None,
                              "ci95_clustered": cluster_ci(rows, c) if key == "ret" else None,
                              "worst": round(min(net), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    comps = list(con.execute("SELECT symbol, sector, industry FROM companies"))
    hyps = [h for h in HYPOTHESES if h.factor]
    targets: dict[str, list[tuple]] = {}
    syms: set[str] = set()
    for h in hyps:
        for tg in h.targets:
            if tg.sign == 0:
                continue
            mem = sorted(set(tg.symbols) | {s for s, sec, ind in comps
                                            if (ind and ind in tg.industries) or (sec and sec in tg.sectors)})
            targets.setdefault(h.id, []).append((tg, mem))
            syms |= set(mem)
    data = {}
    sl = sorted(syms)
    for k in range(0, len(sl), 50):
        data.update(fetch_price_history(sl[k:k + 50], period=args.period))
    fx = fetch_price_history(sorted({h.factor for h in hyps}) + ["^NSEI"], period=args.period)
    nifty = fx["^NSEI"]
    nifty_oc = (nifty["Close"] / nifty["Open"] - 1)
    nifty_oc.index = nifty_oc.index.normalize()
    sessions = [d.normalize() for d in nifty.index]
    rows_by_h: dict[str, list[dict]] = {}
    shock_log: dict[str, list[dict]] = {}
    rng = random.Random(7)
    for h in hyps:
        f = fx.get(h.factor)
        if f is None or f.empty:
            continue
        r = f["Close"].pct_change()
        sd = r.rolling(LOOKBACK).std().shift(1)               # past-only volatility
        fdates = [d.normalize() for d in f.index]
        for t in sessions[1:]:
            prior = [i for i, d in enumerate(fdates) if d < t]
            if not prior:
                continue
            i = prior[-1]
            if pd.isna(r.iloc[i]) or pd.isna(sd.iloc[i]) or sd.iloc[i] == 0 or abs(r.iloc[i]) < Z * sd.iloc[i]:
                continue
            move = 1 if r.iloc[i] > 0 else -1
            shock_log.setdefault(h.id, []).append({"session": t.date().isoformat(), "factor_bar": fdates[i].date().isoformat(),
                                                   "factor_return": round(float(r.iloc[i]), 4),
                                                   "z": round(float(r.iloc[i] / sd.iloc[i]), 2)})
            for tg, mem in targets.get(h.id, []):
                d = tg.sign * move * h.factor_sign
                for s in mem:
                    df = data.get(s)
                    if df is None:
                        continue
                    idx = df.index.normalize()
                    if t not in idx:
                        continue
                    k = list(idx).index(t)
                    if k == 0:
                        continue
                    o, c, pc = float(df["Open"].iloc[k]), float(df["Close"].iloc[k]), float(df["Close"].iloc[k - 1])
                    if not (o > 0 and pc > 0):
                        continue
                    oc, gap = c / o - 1, o / pc - 1
                    n_oc = float(nifty_oc.get(t, float("nan")))
                    rows_by_h.setdefault(h.id, []).append({
                        "date": t.date().isoformat(), "symbol": s, "direction": "UP" if d > 0 else "DOWN",
                        "ret": d * oc, "excess": d * (oc - n_oc) if n_oc == n_oc else 0.0, "gap_dir": d * gap,
                        "random": rng.choice((1, -1)) * oc,
                        "gap_follow": (1 if gap > 0 else -1) * oc if gap != 0 else 0.0})
    report: dict = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "period": args.period,
                    "rules": __doc__.split("Rules fixed before running")[1].split("Limitations")[0].strip(),
                    "first_session": sessions[0].date().isoformat(), "last_session": sessions[-1].date().isoformat(),
                    "hypotheses": {}}
    allrows = []
    for hid, rows in rows_by_h.items():
        dates = sorted({r["date"] for r in rows})
        a, b = dates[int(len(dates) * 0.6)] if dates else None, dates[int(len(dates) * 0.8)] if dates else None
        split = lambda r: "development" if r["date"] < a else "validation" if r["date"] < b else "holdout"  # noqa: E731
        res = {"shocks": len(shock_log.get(hid, [])), "split_dates": {"validation_from": a, "holdout_from": b}}
        for sp in ("development", "validation", "holdout"):
            sub = [r for r in rows if split(r) == sp]
            res[sp] = {"hypothesis": summarize(sub), "random_same_stock_days": summarize(sub, "random"),
                       "gap_follow_same_stock_days": summarize(sub, "gap_follow")}
        res["all"] = summarize(rows)
        res["shock_sample"] = shock_log.get(hid, [])[-10:]
        report["hypotheses"][hid] = res
        allrows += [{**r, "split": split(r)} for r in rows]
    report["pooled"] = {sp: summarize([r for r in allrows if r["split"] == sp]) for sp in ("development", "validation", "holdout")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for hid, res in report["hypotheses"].items():
        hd = res["holdout"]["hypothesis"]
        print(f"{hid:18s} shocks={res['shocks']:3d} all_calls={res['all']['calls']:4d} hit={res['all'].get('hit_rate')} "
              f"net25={res['all'].get('net_25bps', {}).get('mean')} | holdout calls={hd['calls']} hit={hd.get('hit_rate')} "
              f"net25={hd.get('net_25bps', {}).get('mean')} gapdir={res['all'].get('mean_gap_in_call_direction')}")
    print("pooled:", {k: (v["calls"], v.get("hit_rate"), v.get("net_25bps", {}).get("mean")) for k, v in report["pooled"].items()})


if __name__ == "__main__":
    main()
