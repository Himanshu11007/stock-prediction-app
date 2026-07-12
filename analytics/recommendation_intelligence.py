"""
analytics/recommendation_intelligence.py — Recommendation Intelligence Engine

READ-ONLY analytics engine. This module NEVER:
  - Changes model weights
  - Changes thresholds
  - Retrains models
  - Modifies the database
  - Deletes recommendations

Its sole purpose is to read historical validated recommendations and
generate structured statistics + deterministic developer recommendations.

Main entry point:
    generate_engine_report() -> dict

All functions return safe empty/fallback values when data is missing —
the engine never crashes the application.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Optional

import numpy as np
import pandas as pd

from config import TRACKER_DB, BUY_MIN, STRONG_BUY_MIN, HOLD_MIN, SELL_MIN
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)

ENGINE_VERSION = "v1.0"

# ── Thresholds to analyse (read-only — never modified) ────────────────────────
_ANALYSE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

# ── Pillar column map: DB column → human label ────────────────────────────────
_PILLAR_COLS = {
    "pillar_ml_dir":    "ML Direction",
    "pillar_ml_conf":   "ML Confidence",
    "pillar_tech":      "Technical Analysis",
    "pillar_news":      "News Sentiment",
    "pillar_volume":    "Volume",
    "pillar_regime":    "Market Regime",
    "pillar_timeframe": "Multi-Timeframe",
    "pillar_momentum":  "Momentum",
}


# ══════════════════════════════════════════════════════════════════════════════
# Database reader — strict read-only
# ══════════════════════════════════════════════════════════════════════════════

def _load_validated(extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Load all validated recommendations from the database into a DataFrame.
    Returns an empty DataFrame if the table does not exist or has no rows.
    This function never writes to the database.
    """
    base_cols = [
        "id", "saved_date", "symbol", "stock", "signal",
        "cmp", "confluence_score", "ml_confidence", "news_score",
        "accuracy", "target", "stop_loss",
        "validation_date", "validation_price", "return_pct", "success",
        "scan_id", "sector", "market_regime", "engine_version",
        "weighted_score",
    ] + list(_PILLAR_COLS.keys())

    if extra_cols:
        base_cols += [c for c in extra_cols if c not in base_cols]

    try:
        TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(TRACKER_DB))
        con.row_factory = sqlite3.Row

        # Confirm table exists
        exists = con.execute("""
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='recommendation_validation'
        """).fetchone()
        if not exists:
            con.close()
            return pd.DataFrame()

        # Only select columns that actually exist (migration-safe)
        real_cols = {
            row[1] for row in
            con.execute("PRAGMA table_info(recommendation_validation)")
        }
        select_cols = [c for c in base_cols if c in real_cols]
        if not select_cols:
            con.close()
            return pd.DataFrame()

        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM recommendation_validation "
            f"WHERE is_validated = 1",
            con,
        )
        con.close()

        # Coerce numeric columns
        for col in ["confluence_score", "ml_confidence", "news_score", "accuracy",
                    "return_pct", "success", "weighted_score"] + list(_PILLAR_COLS.keys()):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        log_exception(logger, "Failed to load validated recommendations", e)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation with safe fallback for empty/constant series."""
    try:
        valid = pd.concat([a, b], axis=1).dropna()
        if len(valid) < 3:
            return 0.0
        r = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        return round(float(r) if not np.isnan(r) else 0.0, 4)
    except Exception:
        return 0.0


def _stats(series: pd.Series, success: pd.Series) -> dict:
    """
    Compute standard stats for a group: trades, success_count, success_rate,
    avg_return, median_return, win_loss_ratio.
    """
    total   = len(series)
    s_count = int(success.sum()) if not success.empty else 0
    avg     = round(float(series.mean()), 4) if total > 0 else 0.0
    med     = round(float(series.median()), 4) if total > 0 else 0.0
    rate    = round(s_count / total * 100, 1) if total > 0 else 0.0
    fails   = total - s_count
    wl      = round(s_count / fails, 2) if fails > 0 else float("inf")
    return {
        "trades":       total,
        "successful":   s_count,
        "failed":       fails,
        "success_rate": rate,
        "avg_return":   avg,
        "median_return": med,
        "win_loss_ratio": wl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Summary
# ══════════════════════════════════════════════════════════════════════════════

def _summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total": 0, "successful": 0, "failed": 0,
            "success_rate": 0.0, "avg_return": 0.0,
            "best_return": 0.0, "worst_return": 0.0,
            "median_return": 0.0,
            "data_quality": "No validated recommendations available.",
        }
    ret = df["return_pct"].dropna()
    suc = df["success"].dropna()
    total = len(df)
    s_count = int(suc.sum())

    pillar_coverage = sum(
        1 for col in _PILLAR_COLS
        if col in df.columns and df[col].notna().any()
    )
    data_quality = (
        f"{total} validated records. "
        f"Pillar data available for {pillar_coverage}/8 pillars. "
        f"Sector data: {'yes' if df.get('sector', pd.Series()).notna().any() else 'no'}."
    )
    return {
        "total":        total,
        "successful":   s_count,
        "failed":       total - s_count,
        "success_rate": round(s_count / total * 100, 1) if total else 0.0,
        "avg_return":   round(float(ret.mean()), 2) if not ret.empty else 0.0,
        "best_return":  round(float(ret.max()), 2) if not ret.empty else 0.0,
        "worst_return": round(float(ret.min()), 2) if not ret.empty else 0.0,
        "median_return": round(float(ret.median()), 2) if not ret.empty else 0.0,
        "data_quality": data_quality,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Threshold analysis
# ══════════════════════════════════════════════════════════════════════════════

def _threshold_analysis(df: pd.DataFrame) -> list[dict]:
    """
    For each candidate threshold, calculate performance of all
    recommendations whose confluence_score >= threshold.
    READ-ONLY — thresholds are never changed.
    """
    if df.empty or "confluence_score" not in df.columns:
        return []

    results = []
    for t in _ANALYSE_THRESHOLDS:
        subset = df[df["confluence_score"] >= t].copy()
        if subset.empty:
            results.append({
                "threshold": t, "trades": 0, "successful": 0,
                "failed": 0, "success_rate": 0.0,
                "avg_return": 0.0, "median_return": 0.0,
            })
            continue
        st = _stats(subset["return_pct"].dropna(), subset["success"].dropna())
        results.append({"threshold": t, **st})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Confidence analysis
# ══════════════════════════════════════════════════════════════════════════════

_CONF_BINS   = [50, 60, 70, 80, 90, 101]
_CONF_LABELS = ["50-60", "60-70", "70-80", "80-90", "90-100"]


def _confidence_analysis(df: pd.DataFrame) -> list[dict]:
    if df.empty or "ml_confidence" not in df.columns:
        return []

    df = df.copy()
    df["conf_band"] = pd.cut(
        df["ml_confidence"],
        bins=_CONF_BINS, labels=_CONF_LABELS, right=False,
    )
    results = []
    for band in _CONF_LABELS:
        subset = df[df["conf_band"] == band]
        if subset.empty:
            results.append({"confidence_band": band, "trades": 0,
                            "success_rate": 0.0, "avg_return": 0.0,
                            "median_return": 0.0, "win_loss_ratio": 0.0})
            continue
        st = _stats(subset["return_pct"].dropna(), subset["success"].dropna())
        results.append({"confidence_band": band, **st})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Pillar analysis
# ══════════════════════════════════════════════════════════════════════════════

def _pillar_analysis(df: pd.DataFrame) -> list[dict]:
    """
    For each pillar with stored scores, compute avg score,
    correlation with success, and correlation with return_pct.
    Pillars with no stored data are flagged clearly.
    """
    results = []
    for col, label in _PILLAR_COLS.items():
        if col not in df.columns or df[col].isna().all():
            results.append({
                "pillar": label,
                "avg_score": None,
                "corr_with_success": None,
                "corr_with_return": None,
                "interpretation": (
                    f"No stored data for {label} yet. "
                    "Pillar scores are persisted from new recommendations going forward."
                ),
            })
            continue

        series = df[col].dropna()
        avg    = round(float(series.mean()), 4)
        c_suc  = _safe_corr(df[col], df["success"])
        c_ret  = _safe_corr(df[col], df["return_pct"])

        if abs(c_suc) >= 0.5:
            strength = "strong"
        elif abs(c_suc) >= 0.25:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if c_suc >= 0 else "negative"

        if c_suc >= 0.5:
            interp = (
                f"{label} has the strongest positive relationship with "
                "successful recommendations."
            )
        elif c_suc >= 0.25:
            interp = f"{label} has a moderate positive influence on success."
        elif c_suc <= -0.25:
            interp = (
                f"{label} shows a {strength} negative correlation with success — "
                "review how this pillar is weighted."
            )
        elif abs(c_suc) < 0.10:
            interp = (
                f"{label} currently contributes very little to predicting success "
                f"(correlation: {c_suc:.2f})."
            )
        else:
            interp = (
                f"{label} has a {strength} {direction} relationship with "
                f"success (correlation: {c_suc:.2f})."
            )

        results.append({
            "pillar":            label,
            "avg_score":         avg,
            "corr_with_success": c_suc,
            "corr_with_return":  c_ret,
            "interpretation":    interp,
        })

    # Sort by absolute correlation descending
    results.sort(
        key=lambda r: abs(r["corr_with_success"] or 0.0),
        reverse=True,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 6 — Sector analysis
# ══════════════════════════════════════════════════════════════════════════════

def _sector_analysis(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    df = df.copy()
    if "sector" not in df.columns or df["sector"].isna().all():
        return [{
            "sector": "Unknown",
            "trades": len(df),
            "success_rate": 0.0,
            "avg_return": 0.0,
            "best_stock": None,
            "worst_stock": None,
            "note": (
                "No sector data available yet. Add a 'Sector' column to "
                "data/nse_stocks.csv and sector analytics will populate "
                "automatically going forward."
            ),
        }]

    df["sector"] = df["sector"].fillna("Unknown")
    results = []
    for sector, grp in df.groupby("sector", sort=False):
        st = _stats(grp["return_pct"].dropna(), grp["success"].dropna())

        # Best/worst stock by avg return within sector
        stock_perf = grp.groupby("symbol")["return_pct"].mean()
        best  = stock_perf.idxmax() if not stock_perf.empty else None
        worst = stock_perf.idxmin() if not stock_perf.empty else None

        results.append({"sector": sector, **st, "best_stock": best, "worst_stock": worst})

    results.sort(key=lambda r: r["success_rate"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 7 — Regime analysis
# ══════════════════════════════════════════════════════════════════════════════

def _regime_analysis(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    df = df.copy()
    if "market_regime" not in df.columns or df["market_regime"].isna().all():
        return [{"regime": "Unknown", "trades": len(df), "success_rate": 0.0,
                 "avg_return": 0.0,
                 "note": "Market regime not yet stored. New recommendations will include it."}]

    df["market_regime"] = df["market_regime"].fillna("Unknown")
    results = []
    for regime, grp in df.groupby("market_regime", sort=False):
        st = _stats(grp["return_pct"].dropna(), grp["success"].dropna())
        results.append({"regime": regime, **st})

    results.sort(key=lambda r: r["success_rate"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 8 — Signal analysis
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_ORDER = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]


def _signal_analysis(df: pd.DataFrame) -> list[dict]:
    if df.empty or "signal" not in df.columns:
        return []

    results = []
    for sig in _SIGNAL_ORDER:
        grp = df[df["signal"] == sig]
        if grp.empty:
            results.append({"signal": sig, "trades": 0, "success_rate": 0.0,
                            "avg_return": 0.0, "median_return": 0.0})
            continue
        st = _stats(grp["return_pct"].dropna(), grp["success"].dropna())
        results.append({"signal": sig, **st})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 9 — Deterministic recommendations
# ══════════════════════════════════════════════════════════════════════════════

def _generate_recommendations(
    df:            pd.DataFrame,
    threshold_data: list[dict],
    conf_data:      list[dict],
    pillar_data:    list[dict],
    signal_data:    list[dict],
) -> list[dict]:
    """
    Generate deterministic, evidence-based developer recommendations.
    Never suggests changing weights/thresholds — only observes and highlights.
    """
    recs: list[dict] = []

    if df.empty:
        return [{
            "title":          "Insufficient Data",
            "priority":       "Low",
            "confidence":     0,
            "recommendation": "No validated recommendations available to analyze yet.",
            "evidence":       "Run 'Validate Old Recommendations' after 5 trading days.",
        }]

    total = len(df)

    # ── Threshold insight ─────────────────────────────────────────────────────
    if len(threshold_data) >= 2:
        t_sorted = sorted(threshold_data, key=lambda r: r["success_rate"], reverse=True)
        best_t = t_sorted[0]
        lowest_t = threshold_data[0]   # 0.50 — the most permissive
        improvement = best_t["success_rate"] - lowest_t["success_rate"]
        if improvement >= 3.0 and best_t["trades"] >= 5:
            recs.append({
                "title":          "BUY Threshold Observation",
                "priority":       "High" if improvement >= 10 else "Medium",
                "confidence":     min(95, int(60 + improvement * 2)),
                "recommendation": (
                    f"Recommendations with confluence ≥ {best_t['threshold']} "
                    f"achieve {best_t['success_rate']}% success rate across "
                    f"{best_t['trades']} trades."
                ),
                "evidence": (
                    f"Success rate improves from {lowest_t['success_rate']}% "
                    f"(threshold 0.50) to {best_t['success_rate']}% "
                    f"(threshold {best_t['threshold']})."
                ),
            })

    # ── Pillar insights ───────────────────────────────────────────────────────
    pillars_with_data = [p for p in pillar_data if p["corr_with_success"] is not None]

    if pillars_with_data:
        strongest = max(pillars_with_data, key=lambda p: p["corr_with_success"] or 0.0)
        weakest   = min(pillars_with_data, key=lambda p: p["corr_with_success"] or 0.0)

        if (strongest["corr_with_success"] or 0.0) >= 0.50:
            recs.append({
                "title":          f"Strong Pillar: {strongest['pillar']}",
                "priority":       "High",
                "confidence":     85,
                "recommendation": (
                    f"{strongest['pillar']} appears to be one of the strongest "
                    "contributors to successful recommendations."
                ),
                "evidence": (
                    f"Correlation with success: {strongest['corr_with_success']:.2f}. "
                    f"Avg pillar score: {strongest['avg_score']:.3f}."
                ),
            })

        news_pillar = next((p for p in pillars_with_data if p["pillar"] == "News Sentiment"), None)
        if news_pillar and abs(news_pillar["corr_with_success"] or 0.0) < 0.10:
            recs.append({
                "title":          "Weak Pillar: News Sentiment",
                "priority":       "Medium",
                "confidence":     70,
                "recommendation": (
                    "News sentiment currently contributes very little to "
                    "recommendation success."
                ),
                "evidence": (
                    f"Correlation with success: {news_pillar['corr_with_success']:.2f} "
                    f"(below 0.10 threshold). Consider reviewing the news source "
                    "or sentiment weighting."
                ),
            })

        if (weakest["corr_with_success"] or 0.0) <= -0.25:
            recs.append({
                "title":          f"Negative Pillar: {weakest['pillar']}",
                "priority":       "Medium",
                "confidence":     75,
                "recommendation": (
                    f"{weakest['pillar']} shows a negative relationship with "
                    "successful recommendations — it may be hurting overall performance."
                ),
                "evidence": (
                    f"Correlation with success: {weakest['corr_with_success']:.2f}."
                ),
            })

    # ── Signal performance insights ───────────────────────────────────────────
    sig_map = {s["signal"]: s for s in signal_data}
    buy_perf  = sig_map.get("BUY", {})
    hold_perf = sig_map.get("HOLD", {})
    sell_perf = sig_map.get("SELL", {})

    if (hold_perf.get("trades", 0) >= 5 and buy_perf.get("trades", 0) >= 5
            and hold_perf.get("success_rate", 0) < buy_perf.get("success_rate", 0) - 5):
        recs.append({
            "title":          "HOLD vs BUY Performance",
            "priority":       "Medium",
            "confidence":     75,
            "recommendation": "HOLD recommendations underperform BUY recommendations.",
            "evidence": (
                f"BUY success rate: {buy_perf['success_rate']}% "
                f"({buy_perf['trades']} trades). "
                f"HOLD success rate: {hold_perf['success_rate']}% "
                f"({hold_perf['trades']} trades)."
            ),
        })

    if (sell_perf.get("trades", 0) >= 5
            and sell_perf.get("success_rate", 0) < 40):
        recs.append({
            "title":          "SELL Signal Accuracy",
            "priority":       "Medium",
            "confidence":     70,
            "recommendation": (
                f"SELL signals have a low success rate "
                f"({sell_perf.get('success_rate', 0)}%). "
                "The downside-prediction logic may need review."
            ),
            "evidence": (
                f"{sell_perf.get('trades', 0)} SELL recommendations validated. "
                f"Avg return: {sell_perf.get('avg_return', 0):.2f}%."
            ),
        })

    # ── Confidence band insight ───────────────────────────────────────────────
    conf_with_data = [c for c in conf_data if c.get("trades", 0) >= 3]
    if len(conf_with_data) >= 2:
        best_conf = max(conf_with_data, key=lambda c: c.get("success_rate", 0))
        if best_conf.get("success_rate", 0) >= 70:
            recs.append({
                "title":          "High-Confidence Zone",
                "priority":       "Medium",
                "confidence":     80,
                "recommendation": (
                    f"Stocks with ML confidence in the "
                    f"{best_conf['confidence_band']}% band achieve "
                    f"{best_conf['success_rate']}% success rate — "
                    "the strongest confidence zone."
                ),
                "evidence": (
                    f"{best_conf['trades']} trades in this band. "
                    f"Avg return: {best_conf.get('avg_return', 0):.2f}%."
                ),
            })

    # ── General health insight ────────────────────────────────────────────────
    if total >= 10:
        overall_rate = df["success"].mean() * 100 if "success" in df else 0
        avg_ret = df["return_pct"].mean() if "return_pct" in df else 0
        recs.append({
            "title":          "Overall Engine Health",
            "priority":       "Low",
            "confidence":     90,
            "recommendation": (
                f"Engine has generated {total} validated recommendations with "
                f"{overall_rate:.1f}% success rate and {avg_ret:.2f}% avg return."
            ),
            "evidence": (
                "Based on all validated records in the recommendation_validation table."
            ),
        })

    return recs if recs else [{
        "title":          "Insufficient Data",
        "priority":       "Low",
        "confidence":     50,
        "recommendation": "More validated recommendations needed for meaningful insights.",
        "evidence":       f"Only {total} validated records available. Need at least 10.",
    }]


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_engine_report() -> dict:
    """
    Generate the complete Recommendation Intelligence Report.

    READ-ONLY: queries the database, computes statistics, returns a
    structured dict. Never writes to the database or modifies any
    configuration.

    Returns:
        dict with keys:
            summary, threshold_analysis, confidence_analysis,
            pillar_analysis, sector_analysis, regime_analysis,
            signal_analysis, recommendations,
            meta (execution_time_ms, records_analyzed, engine_version)
    """
    t0 = time.time()
    logger.info("INTELLIGENCE_STARTED")

    _FALLBACK = {
        "summary":             {},
        "threshold_analysis":  [],
        "confidence_analysis": [],
        "pillar_analysis":     [],
        "sector_analysis":     [],
        "regime_analysis":     [],
        "signal_analysis":     [],
        "recommendations":     [],
        "meta": {
            "execution_time_ms": 0,
            "records_analyzed":  0,
            "engine_version":    ENGINE_VERSION,
            "error": "Intelligence engine failed — see logs.",
        },
    }

    try:
        df = _load_validated()
        n  = len(df)

        summary   = _summary(df)
        threshold = _threshold_analysis(df)
        conf      = _confidence_analysis(df)
        pillar    = _pillar_analysis(df)
        sector    = _sector_analysis(df)
        regime    = _regime_analysis(df)
        signal    = _signal_analysis(df)
        recs      = _generate_recommendations(df, threshold, conf, pillar, signal)

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            "INTELLIGENCE_COMPLETED | records=%d | time=%.1fms",
            n, elapsed_ms,
        )

        return {
            "summary":             summary,
            "threshold_analysis":  threshold,
            "confidence_analysis": conf,
            "pillar_analysis":     pillar,
            "sector_analysis":     sector,
            "regime_analysis":     regime,
            "signal_analysis":     signal,
            "recommendations":     recs,
            "meta": {
                "execution_time_ms": elapsed_ms,
                "records_analyzed":  n,
                "engine_version":    ENGINE_VERSION,
            },
        }

    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        log_exception(logger, "INTELLIGENCE_FAILED", e)
        logger.error("INTELLIGENCE_FAILED | time=%.1fms | error=%s", elapsed_ms, e)
        _FALLBACK["meta"]["execution_time_ms"] = elapsed_ms
        return _FALLBACK