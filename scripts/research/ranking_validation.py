"""
scripts/research/ranking_validation.py — validate the production StockLens
ranking out of sample (docs/RANKING_VALIDATION_V1.md).

PRE-REGISTERED PROTOCOL (fixed before results were computed)
  Ranking dates  last NIFTY 50 session of each month, 2023-07 .. 2026-09
  DEV            ranking dates 2023-07-31 .. 2024-12-31; an observation is used
                 only if its forward window ends before 2025-01-01
  FINAL          ranking dates 2025-01-01 .. 2026-09-30; used once
  Population     production-eligible stocks, ranked by the production engine
  Primary metric mean cross-sectional Spearman rank-IC between StockLens Score
                 and 3-month forward return (Newey-West t, lags = 2)
  Selection      production weights (A) are kept unless an alternative beats
                 A on DEV by >= 0.02 mean 3M IC with a Newey-West t >= 2 on
                 the per-date IC difference; at most one change.
  Alternatives   A production 25/20/15/10/10/10/5/5; B equal weights over
                 the 8 non-ML components; C quality-heavy 35/15/20/5/5/10/5/5;
                 D quality-valuation balanced 25/25/15/10/5/10/5/5
                 (order: quality/valuation/financial_health/trend/momentum/
                 risk/sector_outlook/market_regime; ml_signal 0 throughout)

Usage:  python scripts/research/ranking_validation.py [--refresh-prices]
Output: scripts/research/output/ranking_validation/*.json|csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import common
from evaluation import ranking_validation as rv
from ranking.service import DEFAULT_RULES, DEFAULT_WEIGHTS
from ranking.technical import technical_snapshot

OUT = common.OUTPUT / "ranking_validation"
PRICE_CACHE = common.CACHE / "ranking_validation_prices.pkl"
BENCH = "^NSEI"
SPLIT = rv.Split(pd.Timestamp("2023-07-01"), pd.Timestamp("2024-12-31"),
                 pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30"))
NW_LAGS = {"1M": 0, "3M": 2, "6M": 5, "12M": 11}
COMPONENTS = ["quality", "valuation", "financial_health", "technical_trend", "momentum", "risk",
              "sector_outlook", "market_regime"]
VARIANTS = {
    "A_production": dict(DEFAULT_WEIGHTS),
    "B_equal": {**{k: 12.5 for k in COMPONENTS}, "ml_signal": 0.0},
    "C_quality_heavy": {"quality": 35, "valuation": 15, "financial_health": 20, "technical_trend": 5,
                        "momentum": 5, "risk": 10, "sector_outlook": 5, "market_regime": 5, "ml_signal": 0},
    "D_quality_value": {"quality": 25, "valuation": 25, "financial_health": 15, "technical_trend": 10,
                        "momentum": 5, "risk": 10, "sector_outlook": 5, "market_regime": 5, "ml_signal": 0},
}


# ── data ─────────────────────────────────────────────────────────────────────

def load_universe() -> dict:
    con = sqlite3.connect(f"file:{common.ROOT / 'storage' / 'app.db'}?mode=ro", uri=True)
    comps = {s: (n, sec, ind) for s, n, sec, ind in con.execute(
        "SELECT symbol, name, sector, industry FROM companies WHERE active = 1 AND symbol IN "
        "(SELECT symbol FROM stock_universe)")}
    funds = {}
    for sym, data in con.execute("SELECT symbol, data FROM fundamental_snapshots WHERE status IN ('OK','PARTIAL') "
                                 "ORDER BY fetched_at"):
        funds[sym] = json.loads(data)
    con.close()
    return {s: {"name": n, "sector": (funds.get(s) or {}).get("sector") or sec,
                "industry": (funds.get(s) or {}).get("industry") or ind,
                "annual": (funds.get(s) or {}).get("annual") or []} for s, (n, sec, ind) in comps.items()}


def load_prices(symbols: list[str], refresh: bool) -> dict:
    if PRICE_CACHE.exists() and not refresh:
        return pickle.loads(PRICE_CACHE.read_bytes())
    import yfinance as yf
    raw = yf.download(symbols + [BENCH], start="2021-06-01", end="2026-10-03", auto_adjust=False,
                      actions=True, group_by="ticker", threads=True, progress=False)
    out = {}
    for s in symbols + [BENCH]:
        try:
            d = raw[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if d.empty:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None)
        out[s] = d
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PRICE_CACHE.write_bytes(pickle.dumps(out))
    return out


def production_frame(d: pd.DataFrame) -> pd.DataFrame:
    """OHLCV as production downloads it (yfinance auto_adjust=True)."""
    f = (d["Adj Close"] / d["Close"]).where(d["Close"] > 0)
    out = pd.DataFrame({c: d[c] * f for c in ("Open", "High", "Low", "Close")})
    out["Volume"] = d["Volume"]
    return out.dropna(subset=["Close"])


# ── analysis helpers ─────────────────────────────────────────────────────────

def summarise_returns(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None,
            "median": float(x.median()) if len(x) else None,
            "hit_rate": float((x > 0).mean()) if len(x) else None,
            "std": float(x.std()) if len(x) > 1 else None}


def segment_frame(df: pd.DataFrame, seg: str, h: str) -> pd.DataFrame:
    d = df[(df["segment"] == seg) & df[f"ret_{h}"].notna()]
    if seg == "DEV":
        d = d[d[f"exit_{h}"] < SPLIT.final_start]
    return d


def ic_series(d: pd.DataFrame, h: str, score_col: str = "stockai_score") -> pd.Series:
    return d.groupby("date").apply(lambda g: rv.spearman(g[score_col], g[f"ret_{h}"]),
                                   include_groups=False).dropna()


def ic_stats(ics: pd.Series, h: str) -> dict:
    m, se, t = rv.newey_west_mean(ics.to_numpy(), NW_LAGS[h])
    return {"dates": int(len(ics)), "mean_ic": round(m, 4) if not np.isnan(m) else None,
            "nw_se": round(se, 4) if not np.isnan(se) else None, "nw_t": round(t, 2) if not np.isnan(t) else None,
            "share_positive": round(float((ics > 0).mean()), 3) if len(ics) else None}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-prices", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    prices = load_prices(sorted(universe), args.refresh_prices)
    bench = prices[BENCH]
    calendar = bench.index
    stocks = {}
    for sym, meta in universe.items():
        if sym not in prices:
            continue
        p = prices[sym]
        # Valuation uses the quoted (split-adjusted, not dividend-adjusted)
        # close: a dividend-adjusted close at D would embed dividends paid
        # after D. Technicals use the production-style adjusted frame; a
        # later dividend rescales the whole history <= D uniformly, so
        # returns/volatility/trend at D are unaffected.
        stocks[sym] = {**meta, "prices": p[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]),
                       "tech_prices": production_frame(p), "close": p["Close"],
                       "dividends": p["Dividends"][p["Dividends"] > 0]}
    missing_prices = sorted(set(universe) - set(stocks))
    manifest = hashlib.sha256(b"".join(pd.util.hash_pandas_object(prices[s]["Close"]).values.tobytes()
                                       for s in sorted(prices))).hexdigest()

    dates = [g.index[-1] for _, g in bench.loc["2023-07-01":"2026-09-30"].groupby(
        bench.loc["2023-07-01":"2026-09-30"].index.to_period("M"))]
    print(f"{len(stocks)} stocks with prices, {len(dates)} ranking dates", flush=True)

    # Production technicals at each date (expensive; computed once).
    from joblib import Parallel, delayed

    def tech_for(sym):
        p = stocks[sym]["tech_prices"]
        out = {}
        for d in dates:
            hist = p.loc[p.index <= d]
            out[d] = technical_snapshot(hist) if len(hist) >= 60 else {}
        return sym, out
    tech = dict(Parallel(n_jobs=16)(delayed(tech_for)(s) for s in sorted(stocks)))

    def run_variant(weights, rules=DEFAULT_RULES):
        rows = []
        for d in dates:
            ranked = rv.rank_universe(stocks, d, weights, rules, {s: tech[s][d] for s in stocks})
            for r in ranked:
                rows.append({"date": d, "symbol": r["symbol"], "stockai_score": r["stockai_score"],
                             "score_coverage": r["score_coverage"], "eligible": r["eligible"], "rank": r["rank"],
                             "fqvf_coverage": r["fqvf_coverage"],
                             **{f"c_{k}": v["score"] for k, v in r["components"].items()}})
        return pd.DataFrame(rows)

    base = run_variant(DEFAULT_WEIGHTS)
    # Forward returns + NIFTY + regime (independent of weights)
    fr = []
    for d in dates:
        nifty = {h: rv.forward_return(bench["Close"], calendar, d, n) for h, n in rv.HORIZONS.items()}
        pos = calendar.searchsorted(d, side="right") - 1
        exits = {h: (calendar[pos + n] if pos + n < len(calendar) else pd.NaT) for h, n in rv.HORIZONS.items()}
        for sym, s in stocks.items():
            row = {"date": d, "symbol": sym}
            for h, n in rv.HORIZONS.items():
                row[f"ret_{h}"] = rv.forward_return(s["close"], calendar, d, n)
                row[f"bench_{h}"] = nifty[h]
                row[f"exit_{h}"] = exits[h]
            fr.append(row)
    fwd = pd.DataFrame(fr)
    regime = {d: technical_snapshot(bench.loc[bench.index <= d].assign(
        Open=lambda x: x["Open"], Volume=lambda x: x["Volume"].fillna(0))[["Open", "High", "Low", "Close", "Volume"]]
        ).get("regime") for d in dates}

    def enrich(scores):
        df = scores.merge(fwd, on=["date", "symbol"], how="left")
        df["segment"] = df["date"].map(SPLIT.segment)
        df["regime"] = df["date"].map(regime)
        for h in rv.HORIZONS:
            df[f"excess_{h}"] = df[f"ret_{h}"] - df[f"bench_{h}"]
        return df

    allrows = enrich(base)
    elig = allrows[allrows["eligible"]].copy()
    elig["bucket"] = None
    for d, g in elig.groupby("date"):
        elig.loc[g.index, "bucket"] = rv.assign_buckets(g)
    allrows.to_csv(OUT / "ranking_snapshots_production.csv", index=False)

    results = {"protocol": __doc__, "price_manifest_sha256": manifest, "ranking_dates": [str(d.date()) for d in dates],
               "stocks_with_prices": len(stocks), "universe_symbols": len(universe),
               "universe_without_price_history": missing_prices, "benchmark": "NIFTY 50 price index (^NSEI)"}

    # Coverage
    cov = allrows.groupby("date").agg(stocks=("symbol", "size"), scored=("stockai_score", lambda x: x.notna().sum()),
                                      eligible=("eligible", "sum"), mean_fqvf_coverage=("fqvf_coverage", "mean"),
                                      mean_score_coverage=("score_coverage", "mean")).reset_index()
    cov["date"] = cov["date"].dt.date.astype(str)
    cov.to_csv(OUT / "coverage_by_date.csv", index=False)
    results["coverage"] = cov.to_dict(orient="records")

    # Buckets
    bucket_rows = []
    for seg in ("DEV", "FINAL"):
        for h in rv.HORIZONS:
            d = segment_frame(elig, seg, h)
            if d.empty:
                continue
            per_date = d.groupby(["date", "bucket"])["excess_" + h].mean().unstack()
            for name, _, _ in rv.BUCKETS:
                b = d[d["bucket"] == name]
                m, se, t = rv.newey_west_mean(per_date[name].dropna().to_numpy(), NW_LAGS[h]) if name in per_date else (np.nan,) * 3
                bucket_rows.append({"segment": seg, "horizon": h, "bucket": name, "dates": int(b["date"].nunique()),
                                    **{f"ret_{k}": v for k, v in summarise_returns(b[f"ret_{h}"]).items()},
                                    "mean_excess_vs_nifty": float(b[f"excess_{h}"].mean()) if len(b) else None,
                                    "per_date_excess_mean": round(m, 4) if not np.isnan(m) else None,
                                    "per_date_excess_ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
                                    if not np.isnan(se) else None})
            if {"Top 10%", "Bottom 25%"} <= set(per_date.columns):
                spread = (per_date["Top 10%"] - per_date["Bottom 25%"]).dropna()
                m, se, t = rv.newey_west_mean(spread.to_numpy(), NW_LAGS[h])
                bucket_rows.append({"segment": seg, "horizon": h, "bucket": "Top10% minus Bottom25%",
                                    "dates": int(len(spread)), "per_date_excess_mean": round(m, 4),
                                    "per_date_excess_ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
                                    if not np.isnan(se) else None, "nw_t": round(t, 2) if not np.isnan(t) else None})
    pd.DataFrame(bucket_rows).to_csv(OUT / "bucket_results.csv", index=False)
    results["buckets"] = bucket_rows

    # Rank IC
    results["rank_ic"] = {seg: {h: ic_stats(ic_series(segment_frame(elig, seg, h), h), h) for h in rv.HORIZONS}
                          for seg in ("DEV", "FINAL")}

    # Top-N portfolios
    topn_rows, risk = [], {}
    for n_top in (5, 10, 20):
        for seg in ("DEV", "FINAL"):
            for h in rv.HORIZONS:
                d = segment_frame(elig, seg, h)
                top = d[d["rank"] <= n_top]
                if top.empty:
                    continue
                per_date = top.groupby("date").agg(ret=(f"ret_{h}", "mean"), bench=(f"bench_{h}", "first"))
                universe_avg = d.groupby("date")[f"ret_{h}"].mean()
                ex = per_date["ret"] - per_date["bench"]
                m, se, t = rv.newey_west_mean(ex.to_numpy(), NW_LAGS[h])
                exu = (per_date["ret"] - universe_avg.reindex(per_date.index)).dropna()
                mu, seu, tu = rv.newey_west_mean(exu.to_numpy(), NW_LAGS[h])
                topn_rows.append({
                    "portfolio": f"Top {n_top}", "segment": seg, "horizon": h, "dates": int(len(per_date)),
                    "mean_return": float(per_date["ret"].mean()), "median_return": float(per_date["ret"].median()),
                    "hit_rate_positive": float((per_date["ret"] > 0).mean()),
                    "mean_excess_vs_nifty": round(m, 4), "excess_vs_nifty_ci95":
                        [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)] if not np.isnan(se) else None,
                    "nw_t_vs_nifty": round(t, 2) if not np.isnan(t) else None,
                    "mean_excess_vs_eligible_universe": round(mu, 4),
                    "nw_t_vs_universe": round(tu, 2) if not np.isnan(tu) else None,
                    "hit_rate_vs_nifty": float((ex > 0).mean()),
                    "stock_level_median": float(top[f"ret_{h}"].median())})
        # risk metrics from non-overlapping 1M rebalanced portfolio over each segment
        for seg in ("DEV", "FINAL"):
            d = segment_frame(elig, seg, "1M")
            top = d[d["rank"] <= n_top]
            if top.empty:
                continue
            pr = top.groupby("date")["ret_1M"].mean()
            br = d.groupby("date")["bench_1M"].first().reindex(pr.index)
            stats = rv.portfolio_stats(pr, br)
            members = top.groupby("date")["symbol"].apply(set)
            turns = [1 - len(a & b) / n_top for a, b in zip(members.iloc[:-1], members.iloc[1:])]
            stats["avg_monthly_turnover"] = float(np.mean(turns)) if turns else None
            risk[f"Top {n_top} {seg}"] = stats
    for seg in ("DEV", "FINAL"):
        d = segment_frame(elig, seg, "1M")
        ew = d.groupby("date")["ret_1M"].mean()
        br = d.groupby("date")["bench_1M"].first()
        risk[f"Eligible universe equal-weight {seg}"] = rv.portfolio_stats(ew, br)
        risk[f"NIFTY 50 {seg}"] = rv.portfolio_stats(br, br)
    pd.DataFrame(topn_rows).to_csv(OUT / "top_n_results.csv", index=False)
    results["top_n"] = topn_rows
    results["risk_1m_rebalanced"] = risk

    # Regime analysis (NIFTY regime at the ranking date, production detect_regime)
    reg_rows = []
    for seg in ("DEV", "FINAL", "ALL"):
        for h in ("1M", "3M"):
            d = elig[elig[f"ret_{h}"].notna()] if seg == "ALL" else segment_frame(elig, seg, h)
            for reg, g in d.groupby("regime"):
                ics = ic_series(g, h)
                top = g[g["rank"] <= 10].groupby("date").agg(r=(f"ret_{h}", "mean"), b=(f"bench_{h}", "first"))
                reg_rows.append({"segment": seg, "horizon": h, "regime": reg, "dates": int(g["date"].nunique()),
                                 "mean_ic": float(ics.mean()) if len(ics) else None,
                                 "top10_mean_excess_vs_nifty": float((top["r"] - top["b"]).mean()) if len(top) else None})
    pd.DataFrame(reg_rows).to_csv(OUT / "regime_results.csv", index=False)
    results["regime"] = reg_rows
    results["regime_by_date"] = {str(d.date()): r for d, r in regime.items()}

    # Component analysis: each component as a ranking signal, and leave-one-out
    comp_rows = []
    for comp in COMPONENTS:
        col = f"c_{comp}"
        for seg in ("DEV", "FINAL"):
            for h in ("1M", "3M", "6M"):
                d = segment_frame(elig, seg, h)
                ics = ic_series(d[d[col].notna()], h, col) if col in d else pd.Series(dtype=float)
                comp_rows.append({"component": comp, "test": "component_alone", "segment": seg, "horizon": h,
                                  "coverage": float(d[col].notna().mean()) if len(d) else None,
                                  **ic_stats(ics, h)})
    loo = {}
    for comp in COMPONENTS:
        w = dict(DEFAULT_WEIGHTS)
        w[comp] = 0.0
        loo[comp] = enrich(run_variant(w))
    for comp, df in loo.items():
        e = df[df["eligible"]]
        for seg in ("DEV", "FINAL"):
            for h in ("1M", "3M", "6M"):
                a = ic_series(segment_frame(elig, seg, h), h)
                b = ic_series(segment_frame(e, seg, h), h)
                diff = (b - a).dropna()
                m, se, t = rv.newey_west_mean(diff.to_numpy(), NW_LAGS[h])
                comp_rows.append({"component": comp, "test": "leave_one_out", "segment": seg, "horizon": h,
                                  "mean_ic_without": round(float(b.mean()), 4) if len(b) else None,
                                  "mean_ic_production": round(float(a.mean()), 4) if len(a) else None,
                                  "ic_change": round(m, 4) if not np.isnan(m) else None,
                                  "nw_t": round(t, 2) if not np.isnan(t) else None})
    pd.DataFrame(comp_rows).to_csv(OUT / "component_results.csv", index=False)
    results["components"] = comp_rows

    # Weight variants: select on DEV only (pre-registered rule), then FINAL once
    var_rows, var_ic = [], {}
    for name, w in VARIANTS.items():
        df = allrows if name == "A_production" else enrich(run_variant(w))
        e = df[df["eligible"]]
        var_ic[name] = {seg: ic_series(segment_frame(e, seg, "3M"), "3M") for seg in ("DEV", "FINAL")}
        for seg in ("DEV", "FINAL"):
            for h in rv.HORIZONS:
                st = ic_stats(ic_series(segment_frame(e, seg, h), h), h)
                d = segment_frame(e, seg, h)
                top = d[d["rank"] <= 10].groupby("date").agg(r=(f"ret_{h}", "mean"), b=(f"bench_{h}", "first"))
                var_rows.append({"variant": name, "segment": seg, "horizon": h, **st,
                                 "top10_mean_excess_vs_nifty": float((top["r"] - top["b"]).mean()) if len(top) else None})
    pd.DataFrame(var_rows).to_csv(OUT / "weight_variants.csv", index=False)
    selection = {"rule": "keep A unless an alternative beats A on DEV 3M mean IC by >= 0.02 with NW t >= 2",
                 "comparisons": {}}
    chosen = "A_production"
    best_margin = 0.0
    for name in VARIANTS:
        if name == "A_production":
            continue
        diff = (var_ic[name]["DEV"] - var_ic["A_production"]["DEV"]).dropna()
        m, se, t = rv.newey_west_mean(diff.to_numpy(), NW_LAGS["3M"])
        qualifies = bool(m >= 0.02 and t >= 2)
        selection["comparisons"][name] = {"dev_ic_diff_vs_A": round(m, 4), "nw_t": round(t, 2) if not np.isnan(t) else None,
                                          "qualifies": qualifies}
        if qualifies and m > best_margin:
            chosen, best_margin = name, m
    selection["selected"] = chosen
    results["weight_variants"] = var_rows
    results["selection"] = selection
    results["final_out_of_sample_selected"] = [r for r in var_rows if r["variant"] == chosen and r["segment"] == "FINAL"]

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"selection": selection, "rank_ic": results["rank_ic"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
