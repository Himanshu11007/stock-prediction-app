"""
scripts/research/transmission_validation.py — price-based tests of the
factor transmission hypotheses (catalysts/transmission.py). Read-only;
writes catalysts/transmission_validation.json (consumed at runtime as each
hypothesis' status).

  python scripts/research/transmission_validation.py [--period 3y]

For every hypothesis with a factor series and every target with a non-zero
sign: the equal-weight basket's daily return in excess of NIFTY 50 is
regressed on the factor's daily return,
  lag 0  same calendar date (does the channel exist at all?)
  lag 1  factor return of the previous factor session (usable for a
         next-session forecast made after the Indian close)
Rules fixed before running (no tuning):
  development = first 2/3 of dates, confirmation = last 1/3
  SUPPORTED     lag-0 beta has the hypothesised sign with |t| >= 2 in
                development AND the same sign in confirmation
  CONTRARY      lag-0 beta has the opposite sign with |t| >= 2 in development
                AND the opposite sign in confirmation
  INCONCLUSIVE  otherwise
  predictive    the same test passes for lag 1 (reported separately)
Hypotheses without a factor (RBI easing, tariffs, capex, peer read-through)
need dated event history and stay UNVALIDATED.
Limitations: OLS t-statistics with daily data (no autocorrelation
correction); today's company classifications; Yahoo data; survivorship.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from catalysts.transmission import HYPOTHESES, TRANSMISSION_VERSION, VALIDATION_FILE  # noqa: E402
from fundamentals.provider import fetch_price_history  # noqa: E402

DB = Path(r"C:\Python\Git\stock-prediction-app\storage\app.db")


def ols(y: pd.Series, x: pd.Series) -> dict:
    d = pd.concat([y, x], axis=1).dropna()
    n = len(d)
    if n < 60:
        return {"n": n, "beta": None, "t": None}
    yy, xx = d.iloc[:, 0], d.iloc[:, 1]
    xm, ym = xx.mean(), yy.mean()
    sxx = ((xx - xm) ** 2).sum()
    beta = ((xx - xm) * (yy - ym)).sum() / sxx
    resid = yy - ym - beta * (xx - xm)
    se = math.sqrt((resid ** 2).sum() / (n - 2) / sxx) if sxx > 0 else float("nan")
    return {"n": n, "beta": round(float(beta), 5), "t": round(float(beta / se), 2) if se else None}


def classify(sign: int, dev: dict, conf: dict) -> str:
    if dev["t"] is None or conf["beta"] is None:
        return "INCONCLUSIVE"
    if sign * dev["t"] >= 2 and sign * conf["beta"] > 0:
        return "SUPPORTED"
    if sign * dev["t"] <= -2 and sign * conf["beta"] < 0:
        return "CONTRARY"
    return "INCONCLUSIVE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    comps = list(con.execute("SELECT symbol, sector, industry FROM companies"))
    plan, symbols, factors = [], set(), set()
    for h in HYPOTHESES:
        if not h.factor:
            continue
        factors.add(h.factor)
        for i, tg in enumerate(h.targets):
            members = sorted(set(tg.symbols) | {s for s, sec, ind in comps
                                                if (ind and ind in tg.industries) or (sec and sec in tg.sectors)})
            plan.append((h, i, tg, members))
            symbols |= set(members)
    data = {}
    syms = sorted(symbols)
    for k in range(0, len(syms), 50):
        data.update(fetch_price_history(syms[k:k + 50], period=args.period))
    fx = fetch_price_history(sorted(factors) + ["^NSEI"], period=args.period)
    nifty = fx["^NSEI"]["Close"].pct_change()
    out: dict = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "transmission_version": TRANSMISSION_VERSION,
                 "period": args.period, "method": __doc__.split("Rules fixed")[0].strip().splitlines()[-1],
                 "rules": "SUPPORTED: lag-0 sign matches, |t|>=2 in development and same sign in confirmation",
                 "hypotheses": {}}
    for h in HYPOTHESES:
        if not h.factor:
            out["hypotheses"][h.id] = {"status": "UNVALIDATED", "reason": "needs dated event history (no factor series)"}
    for h, i, tg, members in plan:
        rets = [data[s]["Close"].pct_change().rename(s) for s in members if data.get(s) is not None]
        if not rets:
            res = {"members": members, "with_data": 0, "status": "INCONCLUSIVE", "reason": "no price data"}
        else:
            basket = pd.concat(rets, axis=1).mean(axis=1) - nifty
            f = fx[h.factor]["Close"].pct_change() * h.factor_sign
            f.index = f.index.normalize()
            basket.index = basket.index.normalize()
            f1 = f.shift(1)
            dates = basket.dropna().index
            cut = dates[int(len(dates) * 2 / 3)]
            res = {"members": members, "with_data": len(rets), "sign": tg.sign, "split": cut.date().isoformat()}
            for lag, series in (("lag0", f), ("lag1", f1)):
                dev = ols(basket[basket.index < cut], series)
                conf = ols(basket[basket.index >= cut], series)
                res[lag] = {"development": dev, "confirmation": conf,
                            "status": classify(tg.sign, dev, conf) if tg.sign else "AMBIGUOUS_BY_DESIGN"}
            res["status"] = res["lag0"]["status"]
            res["predictive"] = res["lag1"]["status"] == "SUPPORTED"
        hyp = out["hypotheses"].setdefault(h.id, {"factor": h.factor, "targets": []})
        hyp["targets"].append({"index": i, **res})
    for hid, hyp in out["hypotheses"].items():
        if "targets" not in hyp:
            continue
        sts = [t["status"] for t in hyp["targets"] if t.get("sign")]
        hyp["status"] = ("CONTRARY" if "CONTRARY" in sts else "SUPPORTED" if sts and all(s == "SUPPORTED" for s in sts)
                         else "INCONCLUSIVE" if sts else "AMBIGUOUS_BY_DESIGN")
        hyp["predictive"] = any(t.get("predictive") for t in hyp["targets"] if t.get("sign"))
    VALIDATION_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for hid, hyp in out["hypotheses"].items():
        tl = [(t.get("lag0", {}).get("development", {}).get("t"), t.get("lag0", {}).get("confirmation", {}).get("beta"),
               t.get("lag1", {}).get("development", {}).get("t")) for t in hyp.get("targets", [])]
        print(f"{hid:22s} {hyp['status']:20s} predictive={hyp.get('predictive')} lag0 dev t / conf beta / lag1 dev t: {tl}")


if __name__ == "__main__":
    main()
