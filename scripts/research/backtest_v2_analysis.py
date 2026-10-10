"""
scripts/research/backtest_v2_analysis.py — investigation of the negative
Prediction v2 baseline backtest (read-only research; writes a JSON report).

  python scripts/research/backtest_v2_analysis.py [--period 3y] [--cache PATH]

What it adds to backtest_v2.py:
  * horizons 1, 3 and 5 market sessions, with the corrected calendar
    alignment (prediction_v2.backtest.run, fwd());
  * a count of the stock-sessions the earlier alignment got wrong;
  * an empirical look-ahead check on real data: decisions recomputed from
    bars truncated at the cutoff must equal the decisions made by the run;
  * breakdowns by setup, direction, sector, market regime (trend and
    volatility) and liquidity bucket, for v2 and the baselines;
  * session-clustered bootstrap intervals, a per-session equal-weight
    portfolio (drawdown, concentration of gains) and the smallest edge the
    holdout could have detected.

Nothing is tuned: rules.THRESHOLDS are used as written, and no individual
stock is singled out. The holdout of baseline-v0.1 has already been seen
once (10 Oct report); this run re-reads it after a verified defect fix only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fundamentals.provider import fetch_price_history  # noqa: E402
from prediction_v2 import backtest, features as feat, rules  # noqa: E402
from prediction_v2.features import FEATURE_SET_VERSION  # noqa: E402

VALIDATION_CSV = ROOT / "scripts" / "research" / "output" / "ranking_validation" / "ranking_snapshots_production.csv"
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "backtest_analysis.json"
BASELINES = ("v2", "previous_day", "sector", "random_matched", "v1_rank", "always_up")
COSTS = (10, 15, 20)
SLIPPAGE_BPS = 5          # added on top of each cost level (round trip)


def load(period: str, cache: Path | None):
    con = sqlite3.connect(f"file:{ROOT / 'storage' / 'app.db'}?mode=ro", uri=True)
    symbols = sorted({r[0] for r in con.execute("SELECT symbol FROM stock_universe")})
    sectors = dict(con.execute("SELECT symbol, sector FROM companies"))
    if cache and cache.exists():
        data, nifty = pickle.loads(cache.read_bytes())
    else:
        data = {}
        for i in range(0, len(symbols), 50):
            data.update(fetch_price_history(symbols[i:i + 50], period=period))
        nifty = fetch_price_history(["^NSEI"], period=period)["^NSEI"]
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((data, nifty)))
    hist = {s: df for s, df in data.items() if df is not None and len(df) > 80}
    return symbols, sectors, hist, nifty


def alignment_audit(hist: dict, nifty: pd.DataFrame, horizon: int) -> dict:
    """How many (stock, session) windows would the earlier own-bar stepping
    have measured over a different end session than the market calendar?"""
    cal = [d.date() for d in nifty.index]
    cal_pos = {d: i for i, d in enumerate(cal)}
    total = wrong = 0
    for df in hist.values():
        own = [d.date() for d in df.index]
        for i, d in enumerate(own[:-horizon]):
            k = cal_pos.get(d)
            if k is None or k + horizon >= len(cal):
                continue
            total += 1
            if own[i + horizon] != cal[k + horizon]:
                wrong += 1
    return {"windows": total, "misaligned_under_old_method": wrong, "share": wrong / total if total else None}


def lookahead_check(hist: dict, nifty: pd.DataFrame, res: backtest.BacktestResult, n: int = 300) -> dict:
    """Re-decide sampled v2 trades from data truncated at the cutoff."""
    rng = random.Random(5)
    v2 = [t for t in res.trades if t.strategy == "v2"]
    sample = rng.sample(v2, min(n, len(v2)))
    mismatches = []
    for t in sample:
        cut = pd.Timestamp(t.cutoff)
        df = hist[t.symbol][hist[t.symbol].index <= cut]
        nf = nifty[nifty.index <= cut]
        f, flags = feat.compute(df, t.cutoff, nf, expected_session=t.cutoff)
        d = rules.decide(f, flags)
        # sector-relative features need peers; the rules do not use them, so the call must match exactly
        if d["direction"] != t.direction or d["setup_type"] != t.setup:
            mismatches.append({"symbol": t.symbol, "cutoff": t.cutoff.isoformat(), "run": t.direction,
                               "truncated": d["direction"]})
    return {"checked": len(sample), "mismatches": mismatches}


def portfolio(res: backtest.BacktestResult, strategy: str, split: str, cost_bps: float) -> dict:
    """Equal weight across each session's calls; one value per session."""
    by_day: dict[dt.date, list[float]] = {}
    for t in res.trades:
        if t.strategy == strategy and t.split == split:
            by_day.setdefault(t.cutoff, []).append(t.ret - cost_bps / 1e4)
    if not by_day:
        return {"sessions": 0}
    days = sorted(by_day)
    daily = [sum(by_day[d]) / len(by_day[d]) for d in days]
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in daily:
        equity *= 1 + r
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    gross = sorted((t.ret for t in res.trades if t.strategy == strategy and t.split == split), reverse=True)
    top = gross[:max(1, len(gross) // 10)]
    pos_total = sum(x for x in gross if x > 0)
    return {"sessions": len(days), "mean_daily": sum(daily) / len(daily),
            "positive_sessions": sum(1 for r in daily if r > 0) / len(daily),
            "compounded_return_non_overlapping_note": "overlapping windows when horizon > 1; indicative only",
            "compounded_return": equity - 1, "max_drawdown": mdd,
            "top_decile_share_of_positive_gross": sum(top) / pos_total if pos_total else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--cache", default=str(ROOT / "storage" / "price_cache" / "backtest_v2_3y.pkl"))
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    symbols, sectors, hist, nifty = load(args.period, Path(args.cache) if args.cache else None)
    v1_top = None
    if VALIDATION_CSV.exists():
        v = pd.read_csv(VALIDATION_CSV)
        v = v[(v["eligible"] == True) & (v["rank"] <= 20)]  # noqa: E712
        v1_top = {dt.date.fromisoformat(d): set(g["symbol"]) for d, g in v.groupby("date")}
    report: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "period": args.period,
        "rule_version": rules.RULE_VERSION, "thresholds": rules.THRESHOLDS, "feature_set_version": FEATURE_SET_VERSION,
        "universe_requested": len(symbols), "universe_with_data": len(hist),
        "symbols_without_data": sorted(set(symbols) - set(hist)),
        "first_session": nifty.index[0].date().isoformat(), "last_session": nifty.index[-1].date().isoformat(),
        "nifty_sessions": len(nifty),
        "costs_bps": {"commission_and_taxes": list(COSTS), "slippage_added": SLIPPAGE_BPS,
                      "applied_round_trip": [c + SLIPPAGE_BPS for c in COSTS]},
        "entry_exit": "close of the cutoff session -> close of the h-th market session after it (NIFTY calendar)",
        "embargo_sessions": 5, "splits": "chronological 60/20/20 with embargo + horizon dropped at each boundary",
        "horizons": {},
    }
    for h in [int(x) for x in args.horizons.split(",")]:
        res = backtest.run(hist, nifty, sectors, horizon=h, embargo=5, seed=7, v1_top=v1_top)
        hr: dict = {"alignment_audit": alignment_audit(hist, nifty, h),
                    "costs": {f"{c + SLIPPAGE_BPS}bps": res.report(cost_bps=c + SLIPPAGE_BPS) for c in COSTS},
                    "breakdowns_holdout_20bps": {}, "breakdowns_all_splits_v2_20bps": {},
                    "portfolio_holdout_20bps": {s: portfolio(res, s, "holdout", 20 + SLIPPAGE_BPS) for s in BASELINES},
                    "v2_setup_counts": {}}
        for by in ("setup", "direction", "sector", "regime_trend", "regime_vol", "liquidity"):
            hr["breakdowns_holdout_20bps"][by] = res.breakdown(by, "holdout", 20 + SLIPPAGE_BPS, BASELINES)
            hr["breakdowns_all_splits_v2_20bps"][by] = {
                sp: res.breakdown(by, sp, 20 + SLIPPAGE_BPS, ("v2",))["v2"] for sp in res.sessions}
        for t in res.trades:
            if t.strategy == "v2":
                key = f"{t.split}:{t.setup}"
                hr["v2_setup_counts"][key] = hr["v2_setup_counts"].get(key, 0) + 1
        if h == 1:
            report["lookahead_check"] = lookahead_check(hist, nifty, res)
        hold = hr["costs"][f"{20 + SLIPPAGE_BPS}bps"]["splits"]["holdout"]["strategies"]["v2"]
        ci = hold.get("mean_net_return_ci95_clustered")
        hr["minimum_detectable_edge"] = {
            "note": "half-width of the session-clustered 95% interval of v2's mean net return in the holdout; "
                    "a true edge smaller than this could not be distinguished from zero",
            "half_width": (ci[1] - ci[0]) / 2 if ci else None}
        report["horizons"][str(h)] = hr
        print(f"h={h}: v2 holdout calls={hold['calls']} hit={hold['hit_rate']} net@25bps={hold['mean_net_return']} "
              f"clustered={ci} ({time.time() - t0:.0f}s)")
    report["runtime_seconds"] = round(time.time() - t0, 1)
    report["limitations"] = [
        "Survivorship: today's universe list is applied to the whole period; delisted or renamed symbols are absent.",
        "Daily Yahoo Finance bars (delayed, split/dividend adjusted after the fact); not exchange data.",
        "Sector labels are today's companies.sector values; peers are not point-in-time.",
        "No corporate-event, consensus or intraday inputs; catalyst setups are not tested.",
        "v1 baseline uses month-end research reproductions (scripts/research/output/ranking_validation), "
        "not live production runs.",
        "Costs are fixed per trade (no market impact); short (DOWN) trades assume borrow is available, "
        "which is not generally true for Indian cash equities overnight.",
        "Windows overlap for horizons above 1, so portfolio figures are indicative only.",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out} in {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
