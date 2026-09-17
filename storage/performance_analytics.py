"""
storage/performance_analytics.py — Read-only analytics over validated recommendations.

All functions are pure reads — they never modify the database.
Every function returns an empty/safe default when the table is missing or empty
so the dashboard never crashes on a fresh install.

Public API
──────────
load_validated_df()             → pd.DataFrame   (all validated rows)
summary_metrics(df)             → dict
signal_performance(df)          → pd.DataFrame
confidence_performance(df)      → pd.DataFrame
confluence_performance(df)      → pd.DataFrame
sentiment_performance(df)       → pd.DataFrame
monthly_trend(df)               → pd.DataFrame
top_winners(df, n)              → pd.DataFrame
top_losers(df, n)               → pd.DataFrame
generate_insights(df, sig, con) → list[str]
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from config import TRACKER_DB


# ══════════════════════════════════════════════════════════════════════════════
# Database read
# ══════════════════════════════════════════════════════════════════════════════

def load_validated_df() -> pd.DataFrame:
    """
    Load all validated recommendations into a DataFrame.

    Returns an empty DataFrame with the correct columns when the table
    does not exist or contains no validated rows.
    """
    _EMPTY = pd.DataFrame(columns=[
        "Date", "Symbol", "Stock", "Signal",
        "CMP", "Confluence Score", "ML Confidence", "News Score",
        "Target", "Stop Loss",
        "Validation Date", "Validation Price", "Return %", "Success",
    ])

    try:
        TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(TRACKER_DB))
        con.row_factory = sqlite3.Row

        # Check table exists before querying
        exists = con.execute("""
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='recommendation_validation'
        """).fetchone()

        if not exists:
            con.close()
            return _EMPTY

        rows = con.execute("""
            SELECT
                saved_date       AS "Date",
                symbol           AS "Symbol",
                stock            AS "Stock",
                signal           AS "Signal",
                cmp              AS "CMP",
                confluence_score AS "Confluence Score",
                ml_confidence    AS "ML Confidence",
                news_score       AS "News Score",
                target           AS "Target",
                stop_loss        AS "Stop Loss",
                validation_date  AS "Validation Date",
                validation_price AS "Validation Price",
                return_pct       AS "Return %",
                success          AS "Success"
            FROM  recommendation_validation
            WHERE is_validated = 1
            ORDER BY validation_date DESC, saved_date DESC
        """).fetchall()

        con.close()

        if not rows:
            return _EMPTY

        df = pd.DataFrame([dict(r) for r in rows])
        # Ensure numeric columns are typed correctly
        for col in ["CMP", "Confluence Score", "ML Confidence",
                    "News Score", "Return %", "Validation Price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Success"] = pd.to_numeric(df["Success"], errors="coerce")
        return df

    except Exception:
        return _EMPTY


# ══════════════════════════════════════════════════════════════════════════════
# Summary metrics
# ══════════════════════════════════════════════════════════════════════════════

def summary_metrics(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPIs from validated recommendations.

    Returns:
        dict with keys:
            total, successful, failed, success_rate,
            avg_return, best_return, worst_return
    """
    if df.empty:
        return {
            "total": 0, "successful": 0, "failed": 0,
            "success_rate": 0.0, "avg_return": 0.0,
            "best_return": 0.0, "worst_return": 0.0,
        }

    total      = len(df)
    successful = int(df["Success"].sum())
    failed     = total - successful
    ret        = df["Return %"].dropna()

    return {
        "total":        total,
        "successful":   successful,
        "failed":       failed,
        "success_rate": round(successful / total * 100, 1) if total else 0.0,
        "avg_return":   round(ret.mean(), 2) if not ret.empty else 0.0,
        "best_return":  round(ret.max(),  2) if not ret.empty else 0.0,
        "worst_return": round(ret.min(),  2) if not ret.empty else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Signal performance
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_ORDER = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]


def signal_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by signal type.

    Returns DataFrame with columns:
        Signal | Count | Success Rate % | Avg Return %
    """
    if df.empty:
        return pd.DataFrame(columns=["Signal", "Count", "Success Rate %", "Avg Return %"])

    grp = df.groupby("Signal").agg(
        Count          = ("Success",  "count"),
        Successful     = ("Success",  "sum"),
        avg_return     = ("Return %", "mean"),
    ).reset_index()

    grp["Success Rate %"] = (grp["Successful"] / grp["Count"] * 100).round(1)
    grp["Avg Return %"]   = grp["avg_return"].round(2)
    grp = grp[["Signal", "Count", "Success Rate %", "Avg Return %"]]

    # Order by predefined signal hierarchy
    grp["_order"] = grp["Signal"].map(
        {s: i for i, s in enumerate(_SIGNAL_ORDER)}
    ).fillna(99)
    return grp.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Confidence band analysis
# ══════════════════════════════════════════════════════════════════════════════

_CONF_BINS   = [0, 60, 70, 80, 90, 101]
_CONF_LABELS = ["50–60%", "60–70%", "70–80%", "80–90%", "90%+"]


def confidence_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by ML Confidence band and compute success rate + avg return.

    Returns DataFrame with columns:
        Confidence Band | Count | Success Rate % | Avg Return %
    """
    if df.empty or "ML Confidence" not in df.columns:
        return pd.DataFrame(
            columns=["Confidence Band", "Count", "Success Rate %", "Avg Return %"]
        )

    tmp = df.copy()
    tmp["Confidence Band"] = pd.cut(
        tmp["ML Confidence"], bins=_CONF_BINS, labels=_CONF_LABELS, right=False
    )
    grp = tmp.groupby("Confidence Band", observed=True).agg(
        Count      = ("Success",  "count"),
        Successful = ("Success",  "sum"),
        avg_return = ("Return %", "mean"),
    ).reset_index()

    grp["Success Rate %"] = (grp["Successful"] / grp["Count"] * 100).round(1)
    grp["Avg Return %"]   = grp["avg_return"].round(2)
    return grp[["Confidence Band", "Count", "Success Rate %", "Avg Return %"]]


# ══════════════════════════════════════════════════════════════════════════════
# Confluence score band analysis
# ══════════════════════════════════════════════════════════════════════════════

_CONF_SCORE_BINS   = [0.0, 0.50, 0.60, 0.70, 0.80, 1.01]
_CONF_SCORE_LABELS = ["<0.50", "0.50–0.60", "0.60–0.70", "0.70–0.80", "0.80+"]


def confluence_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Confluence Score band and compute success rate + avg return.

    Returns DataFrame with columns:
        Confluence Band | Count | Success Rate % | Avg Return %
    """
    if df.empty or "Confluence Score" not in df.columns:
        return pd.DataFrame(
            columns=["Confluence Band", "Count", "Success Rate %", "Avg Return %"]
        )

    tmp = df.copy()
    tmp["Confluence Band"] = pd.cut(
        tmp["Confluence Score"],
        bins=_CONF_SCORE_BINS,
        labels=_CONF_SCORE_LABELS,
        right=False,
    )
    grp = tmp.groupby("Confluence Band", observed=True).agg(
        Count      = ("Success",  "count"),
        Successful = ("Success",  "sum"),
        avg_return = ("Return %", "mean"),
    ).reset_index()

    grp["Success Rate %"] = (grp["Successful"] / grp["Count"] * 100).round(1)
    grp["Avg Return %"]   = grp["avg_return"].round(2)
    return grp[["Confluence Band", "Count", "Success Rate %", "Avg Return %"]]


# ══════════════════════════════════════════════════════════════════════════════
# News sentiment bucket analysis
# ══════════════════════════════════════════════════════════════════════════════

def _news_bucket(score: float) -> str:
    if score > 0.25:
        return "🟢 Positive"
    if score < -0.25:
        return "🔴 Negative"
    return "🟡 Neutral"


def sentiment_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket by News Score (Positive / Neutral / Negative) and compute metrics.

    Returns DataFrame with columns:
        Sentiment | Count | Success Rate % | Avg Return %
    """
    if df.empty or "News Score" not in df.columns:
        return pd.DataFrame(
            columns=["Sentiment", "Count", "Success Rate %", "Avg Return %"]
        )

    tmp = df.copy()
    tmp["Sentiment"] = tmp["News Score"].apply(_news_bucket)
    grp = tmp.groupby("Sentiment").agg(
        Count      = ("Success",  "count"),
        Successful = ("Success",  "sum"),
        avg_return = ("Return %", "mean"),
    ).reset_index()

    grp["Success Rate %"] = (grp["Successful"] / grp["Count"] * 100).round(1)
    grp["Avg Return %"]   = grp["avg_return"].round(2)
    _order = {"🟢 Positive": 0, "🟡 Neutral": 1, "🔴 Negative": 2}
    grp["_o"] = grp["Sentiment"].map(_order).fillna(9)
    return (
        grp.sort_values("_o")
        .drop(columns="_o")
        [["Sentiment", "Count", "Success Rate %", "Avg Return %"]]
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Monthly trend
# ══════════════════════════════════════════════════════════════════════════════

def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly success rate (validated date bucketed to YYYY-MM).

    Returns DataFrame with columns:
        Month | Count | Success Rate %
    Sorted chronologically.
    """
    if df.empty or "Validation Date" not in df.columns:
        return pd.DataFrame(columns=["Month", "Count", "Success Rate %"])

    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(
        tmp["Validation Date"], errors="coerce"
    ).dt.to_period("M").astype(str)

    grp = tmp.groupby("Month").agg(
        Count      = ("Success", "count"),
        Successful = ("Success", "sum"),
    ).reset_index()

    grp["Success Rate %"] = (grp["Successful"] / grp["Count"] * 100).round(1)
    return grp[["Month", "Count", "Success Rate %"]].sort_values("Month").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Top winners / losers
# ══════════════════════════════════════════════════════════════════════════════

def top_winners(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-n recommendations by Return %."""
    if df.empty:
        return pd.DataFrame(columns=["Date", "Stock", "Signal", "Return %"])
    return (
        df[["Date", "Stock", "Signal", "Return %"]]
        .dropna(subset=["Return %"])
        .sort_values("Return %", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_losers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the bottom-n recommendations by Return %."""
    if df.empty:
        return pd.DataFrame(columns=["Date", "Stock", "Signal", "Return %"])
    return (
        df[["Date", "Stock", "Signal", "Return %"]]
        .dropna(subset=["Return %"])
        .sort_values("Return %", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Automatic insights
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    df:       pd.DataFrame,
    sig_df:   pd.DataFrame,
    conf_df:  pd.DataFrame,
    confl_df: pd.DataFrame,
    sent_df:  pd.DataFrame,
) -> list[str]:
    """
    Auto-generate 3–5 plain-English observations from the analytics DataFrames.
    Returns an empty list when there is insufficient data.
    """
    insights: list[str] = []
    if df.empty:
        return insights

    # ── 1. Best-performing signal type ───────────────────────────────────────
    if not sig_df.empty and len(sig_df) > 1:
        best_sig = sig_df.loc[sig_df["Success Rate %"].idxmax()]
        worst_sig = sig_df.loc[sig_df["Success Rate %"].idxmin()]
        insights.append(
            f"**{best_sig['Signal']}** recommendations have the highest success rate "
            f"({best_sig['Success Rate %']}%) — outperforming "
            f"**{worst_sig['Signal']}** ({worst_sig['Success Rate %']}%)."
        )

    # ── 2. Confidence band insight ────────────────────────────────────────────
    if not conf_df.empty and len(conf_df) > 1:
        conf_valid = conf_df[conf_df["Count"] >= 3]   # ignore thin bands
        if len(conf_valid) > 1:
            best_band = conf_valid.loc[conf_valid["Success Rate %"].idxmax()]
            insights.append(
               f"The **{best_band ['Confidence Band']}** ML Confidence band has the "
               f"highest observed success rate at **{best_band ['Success Rate %']}%** "
               f"across **{best_band['Count']} ** recommendations."
            )
            # Check if high confidence truly beats low confidence
            first_rate = conf_valid["Success Rate %"].iloc[0]
            last_rate  = conf_valid["Success Rate %"].iloc[-1]
            if last_rate > first_rate + 5:
                insights.append(
                    "Higher ML confidence is correlated with better outcomes — "
                    "the model's certainty is a useful signal."
                )
            elif last_rate < first_rate - 5:
                insights.append(
                    "Surprisingly, lower ML confidence bands are outperforming higher ones — "
                    "consider reviewing the confidence calibration."
                )

    # ── 3. Confluence score insight ───────────────────────────────────────────
    if not confl_df.empty and len(confl_df) > 1:
        confl_valid = confl_df[confl_df["Count"] >= 3]
        if not confl_valid.empty:
            best_confl = confl_valid.loc[confl_valid["Success Rate %"].idxmax()]
            insights.append(
                f"Confluence band **{best_confl['Confluence Band']} ** has the highest"
                f"observed success rate at ** {best_confl['Success Rate %']}%** "
                f"across **{best_confl['Count']}** recommendations "
                f"(avg return {best_confl['Avg Return %']:+.2f}%)."
            )

    # ── 4. News sentiment insight ─────────────────────────────────────────────
    if not sent_df.empty and len(sent_df) > 1:
        best_sent = sent_df.loc[sent_df["Success Rate %"].idxmax()]
        insights.append(
            f"{best_sent['Sentiment']} news sentiment has the highest observed success rate"
            f"among the sentiment buckets({best_sent['Success Rate %']}%)"
        )

    # ── 5. Overall return insight ─────────────────────────────────────────────
    avg = df["Return %"].mean()

    if "Signal" in df.columns:
        direction_correct = (df["Signal"].isin(["BUY","STRONG BUY"]) & (df["Return %"] > 0)) |(df["Signal"].isin(["SELL","STRONG SELL"]) & (df["Return %"] < 0))|((df["Signal"] == "HOLD") & (df["Return %"].abs() <= 3.0))
        predicted_direction_pct = direction_correct.mean() * 100
    else:
        predicted_direction_pct = None
    if predicted_direction_pct is not None:
        insights.append(
            f"Across all {len(df)} validated recommendation, the average return is"
            f"**{avg:+.2f}%** with **{predicted_direction_pct:.0f}%** of picks"
            f"matching the validation outcome for their signal"
        )
    else:
        insights.append(
            f"Across all{len(df)} validated recommendation, the average return is"
            f"**{avg:+2.f}%**."
        )

    return insights