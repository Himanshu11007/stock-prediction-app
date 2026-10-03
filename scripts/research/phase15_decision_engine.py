"""
scripts/research/phase15_decision_engine.py — Phase 15b: decision-engine research.

Counterfactuals over the production decision engine, using full production
replays (evaluation/walk_forward.py, news neutral):
  DEV    scripts/audit/output/walk_forward_benchmark/research_dev/  (2025-01 .. 2025-09)
  FINAL  scripts/audit/output/walk_forward_benchmark/full/          (2025-10 .. 2026-09)
Thresholds are chosen on DEV only and then evaluated once on FINAL.
No production threshold is changed.

What is (and is not) exactly counterfactual:
  * Signal thresholds act on the stored `confluence` (score/100), so any
    alternative BUY/SELL threshold is exact.
  * Filters are ANDed gates; the replay stores `passes_quality_filters`
    with the CURRENT thresholds. A STRICTER threshold is exact
    (passes AND new gate); a LOOSER one would need the other gates' inputs
    (RSI, volume, volatility), which are not stored, so it is not evaluated.
  * scanner/filters.py has no risk/reward gate; calculate_risk() fixes the
    target at 2x the stop distance, so R/R is constant (2.0, HOLD 1.0).

Pre-registered selection (docs/DECISION_ENGINE_RESEARCH.md): for bullish
calls, the BUY threshold maximising DEV mean 5D excess return (bullish calls
minus all rows on the same dates) with n >= 50; mirror image for bearish.

Usage:
    python scripts/research/phase15_decision_engine.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from config import BUY_MIN, HOLD_MIN, MIN_ACCURACY, MIN_CONFIDENCE, MIN_CONFLUENCE_SCORE, SELL_MIN, STRONG_BUY_MIN
from evaluation.walk_forward import HORIZONS, wilson_interval

BENCH = common.ROOT / "scripts" / "audit" / "output" / "walk_forward_benchmark"
BULL, BEAR = ("BUY", "STRONG BUY"), ("SELL", "STRONG SELL")


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(BENCH / name / "raw_model_predictions.csv", parse_dates=["prediction_timestamp"])
    df["score100"] = df["confluence"] * 100
    df["date"] = df["prediction_timestamp"]
    # Excess return vs the equal-weight mean of all replayed rows on that date.
    for h in HORIZONS:
        df[f"excess_{h}d"] = df[f"return_{h}d"] - df.groupby("date")[f"return_{h}d"].transform("mean")
    return df


def rate(series) -> dict:
    s = series.dropna().astype(int)
    lo, hi = wilson_interval(int(s.sum()), len(s))
    return {"n": len(s), "rate": round(s.mean(), 4) if len(s) else None, "ci95": [lo, hi]}


def cluster_mean_ci(df, col, n_boot=1000, seed=0):
    d = df[["date", col]].dropna()
    if d.empty:
        return [None, None]
    codes, uniq = pd.factorize(d["date"])
    sums = np.zeros((len(uniq), 2))
    np.add.at(sums, codes, np.c_[d[col].to_numpy(), np.ones(len(d))])
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    tot = sums[draws].sum(axis=1)
    m = tot[:, 0] / tot[:, 1]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def group_stats(rows: pd.DataFrame, h: int = 5, bullish: bool = True) -> dict:
    d = rows[rows[f"return_{h}d"].notna()]
    succ = (d[f"return_{h}d"] > 0) if bullish else (d[f"return_{h}d"] < 0)
    return {
        "n": len(d),
        "success": rate(succ.astype(int)),
        "mean_return_pct": round(float(d[f"return_{h}d"].mean()), 4) if len(d) else None,
        "mean_excess_pct": round(float(d[f"excess_{h}d"].mean()), 4) if len(d) else None,
        "mean_excess_ci95": cluster_mean_ci(d, f"excess_{h}d"),
    }


def alignment(df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["signal"], df["predicted_direction"]).rename(columns={0: "model_down", 1: "model_up"})
    by_signal = df.groupby("signal").agg(
        n=("signal", "size"), mean_p_up=("ensemble_probability", "mean"),
        mean_confidence=("confidence", "mean"), mean_confluence=("confluence", "mean"))
    return {
        "signal_x_model_direction": ct.to_dict(orient="index"),
        "by_signal": by_signal.round(4).to_dict(orient="index"),
        "spearman_confluence_vs_p_up": round(float(df["confluence"].corr(df["ensemble_probability"], method="spearman")), 4),
        "bullish_signal_while_model_down_pct": round(float(
            ((df["signal"].isin(BULL)) & (df["predicted_direction"] == 0)).sum()
            / max(1, df["signal"].isin(BULL).sum()) * 100), 1),
    }


def signal_table(df: pd.DataFrame) -> list[dict]:
    out = []
    for h in HORIZONS:
        for sig, g in df.groupby("signal"):
            r = rate(g[f"signal_success_{h}d"])
            out.append({"horizon_days": h, "signal": sig, **r,
                        "mean_return_pct": round(float(g[f"return_{h}d"].mean()), 4),
                        "mean_excess_pct": round(float(g[f"excess_{h}d"].mean()), 4)})
    return out


def threshold_sweep(df: pd.DataFrame, bullish: bool) -> list[dict]:
    grid = range(50, 76, 2) if bullish else range(26, 52, 2)
    out = []
    for t in grid:
        rows = df[df["score100"] >= t] if bullish else df[df["score100"] < t]
        out.append({"threshold": t, **{f"h{h}": group_stats(rows, h, bullish) for h in (1, 5, 10)}})
    return out


def pick(sweep: list[dict], bullish: bool, min_n: int = 50) -> int | None:
    ok = [r for r in sweep if r["h5"]["n"] >= min_n and r["h5"]["mean_excess_pct"] is not None]
    if not ok:
        return None
    key = (lambda r: r["h5"]["mean_excess_pct"]) if bullish else (lambda r: -r["h5"]["mean_excess_pct"])
    return max(ok, key=key)["threshold"]


def filter_counterfactuals(df: pd.DataFrame) -> dict:
    base = df["passes_quality_filters"]
    bull = df["signal"].isin(BULL)
    out = {"MIN_CONFIDENCE": [], "MIN_CONFLUENCE_SCORE": [], "MIN_ACCURACY": []}
    for t in (MIN_CONFIDENCE, 60, 65, 70, 75):
        keep = base & (df["confidence"] >= t)
        out["MIN_CONFIDENCE"].append({"threshold": t, "n_kept": int(keep.sum()),
                                      "bullish": group_stats(df[keep & bull]),
                                      "bearish": group_stats(df[keep & df["signal"].isin(BEAR)], bullish=False)})
    for t in (MIN_CONFLUENCE_SCORE, 0.60, 0.65, 0.70):
        keep = base & (~bull | (df["confluence"] >= t))
        out["MIN_CONFLUENCE_SCORE"].append({"threshold": t, "n_kept": int(keep.sum()),
                                            "bullish": group_stats(df[keep & bull])})
    for t in (MIN_ACCURACY, 0.48, 0.52, 0.56):
        keep = base & (~bull | (df["production_accuracy_fast"] >= t))
        out["MIN_ACCURACY"].append({"threshold": t, "n_kept": int(keep.sum()),
                                    "bullish": group_stats(df[keep & bull])})
    return out


def main() -> None:
    dev, final = load("research_dev"), load("full")
    res = {"current_thresholds": {"STRONG_BUY_MIN": STRONG_BUY_MIN, "BUY_MIN": BUY_MIN,
                                  "HOLD_MIN": HOLD_MIN, "SELL_MIN": SELL_MIN,
                                  "MIN_CONFIDENCE": MIN_CONFIDENCE,
                                  "MIN_CONFLUENCE_SCORE": MIN_CONFLUENCE_SCORE,
                                  "MIN_ACCURACY": MIN_ACCURACY}}
    for name, df in (("DEV", dev), ("FINAL", final)):
        res[name] = {
            "n_rows": len(df), "n_symbols": int(df["symbol"].nunique()),
            "first": str(df["date"].min().date()), "last": str(df["date"].max().date()),
            "alignment": alignment(df),
            "signals_all": signal_table(df),
            "signals_filtered": signal_table(df[df["passes_quality_filters"]]),
            "bullish_sweep": threshold_sweep(df, True),
            "bearish_sweep": threshold_sweep(df, False),
            "filters": filter_counterfactuals(df),
            "hold_band_unconditional": {f"{h}d": rate((df[f"return_{h}d"].abs() <= 3.0)
                                                     .where(df[f"return_{h}d"].notna()))
                                        for h in HORIZONS},
        }
    bt = pick(res["DEV"]["bullish_sweep"], True)
    st = pick(res["DEV"]["bearish_sweep"], False)
    res["selection_on_dev"] = {"bullish_threshold": bt, "bearish_threshold": st}

    def at(sweep, t):
        return next(r for r in sweep if r["threshold"] == t)

    res["final_check"] = {
        "bullish_selected": at(res["FINAL"]["bullish_sweep"], bt) if bt else None,
        "bullish_current_58": at(res["FINAL"]["bullish_sweep"], BUY_MIN),
        "bearish_selected": at(res["FINAL"]["bearish_sweep"], st) if st else None,
        "bearish_current_42": at(res["FINAL"]["bearish_sweep"], HOLD_MIN),
    }
    common.write_json("phase15_decision_engine.json", res)
    print("selected on DEV:", res["selection_on_dev"])
    for k, v in res["final_check"].items():
        if v:
            print(k, v["h5"])


if __name__ == "__main__":
    main()
