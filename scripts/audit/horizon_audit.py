"""
scripts/audit/horizon_audit.py — Phase 9 recommendation-quality audit.

READ-ONLY diagnostic. Does not modify storage/tracker.db, does not change
the production validation engine, does not touch is_validated/
validation_date/return_pct/success on any row.

Why this script exists
───────────────────────
storage/recommendation_validation.py validates a recommendation whenever
an admin happens to call POST /tracker/validate-old - there is no
scheduled job. The actual number of trading days elapsed at validation
time in the current database ranges from 5 to 39 (median 13, mean ~17.7),
so the stored `return_pct` does NOT represent a clean, comparable "N-day
forward return" across rows - see docs/RECOMMENDATION_QUALITY_AUDIT.md
"Train/test & validation methodology" for the full measurement.

This script reconstructs TRUE, consistent 1/3/5/10-trading-day forward
returns for a sample of past recommendations, straight from each symbol's
own historical daily price series (fetched fresh via yfinance - never
touches the app's 1-hour rolling price cache, which only holds ~1 year of
"current" data, not point-in-time history). It reuses the EXISTING,
UNMODIFIED calculate_return()/calculate_success() functions from
storage.recommendation_validation, so the success definition is identical
to production - only the horizon measurement itself is new.

Usage:
    python scripts/audit/horizon_audit.py

Output:
    scripts/audit/output/horizon_audit.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import TRACKER_DB  # noqa: E402
from storage.recommendation_validation import calculate_return, calculate_success  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
HORIZONS = [1, 3, 5, 10]

# Deterministic sample: the 25 symbols with the most recommendation rows
# in the current database, so this script is reproducible against a given
# snapshot of storage/tracker.db and keeps the number of yfinance calls
# bounded.
_MAX_SYMBOLS = 25


def load_sample_recommendations() -> pd.DataFrame:
    con = sqlite3.connect(str(TRACKER_DB))
    df = pd.read_sql_query(
        "SELECT symbol, saved_date, signal, cmp FROM recommendation_validation "
        "WHERE is_validated = 1",
        con,
    )
    con.close()

    top_symbols = df["symbol"].value_counts().head(_MAX_SYMBOLS).index.tolist()
    return df[df["symbol"].isin(top_symbols)].copy()


def fetch_close_series(symbol: str) -> pd.Series | None:
    df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def forward_return_rows(symbol: str, close: pd.Series, recs: pd.DataFrame) -> list[dict]:
    rows = []
    dates = close.index
    for _, rec in recs.iterrows():
        saved = pd.Timestamp(rec["saved_date"])
        # First trading day at or after saved_date in the fetched series.
        pos_candidates = np.searchsorted(dates.values, saved.to_datetime64(), side="left")
        if pos_candidates >= len(dates):
            continue
        entry_pos = pos_candidates
        cmp_ = float(rec["cmp"])
        signal = rec["signal"]

        for h in HORIZONS:
            target_pos = entry_pos + h
            if target_pos >= len(dates):
                continue  # not enough future history fetched for this horizon
            future_price = float(close.iloc[target_pos])
            ret = calculate_return(cmp_, future_price)
            success = calculate_success(signal, ret)
            rows.append({
                "symbol": symbol,
                "saved_date": rec["saved_date"],
                "signal": signal,
                "horizon_days": h,
                "return_pct": ret,
                "success": success,
            })
    return rows


def main() -> None:
    sample = load_sample_recommendations()
    symbols = sorted(sample["symbol"].unique())

    all_rows: list[dict] = []
    fetch_failures: list[str] = []
    for symbol in symbols:
        close = fetch_close_series(symbol)
        if close is None or len(close) < 30:
            fetch_failures.append(symbol)
            continue
        recs = sample[sample["symbol"] == symbol]
        all_rows.extend(forward_return_rows(symbol, close, recs))

    df = pd.DataFrame(all_rows)

    by_horizon = []
    for h in HORIZONS:
        subset = df[df["horizon_days"] == h] if not df.empty else df
        n = len(subset)
        if n == 0:
            by_horizon.append({"horizon_days": h, "n": 0})
            continue
        by_horizon.append({
            "horizon_days": h,
            "n": n,
            "success_rate_pct": round(float(subset["success"].mean()) * 100, 1),
            "avg_return_pct": round(float(subset["return_pct"].mean()), 3),
            "median_return_pct": round(float(subset["return_pct"].median()), 3),
        })

    by_horizon_and_signal = []
    if not df.empty:
        for (h, sig), grp in df.groupby(["horizon_days", "signal"]):
            by_horizon_and_signal.append({
                "horizon_days": int(h),
                "signal": sig,
                "n": len(grp),
                "success_rate_pct": round(float(grp["success"].mean()) * 100, 1),
                "avg_return_pct": round(float(grp["return_pct"].mean()), 3),
            })

    output = {
        "method": (
            "Reconstructs true 1/3/5/10-trading-day forward returns from each "
            "symbol's own historical Close series (fresh yfinance fetch), using "
            "the stored recommendation's cmp as entry price and the UNMODIFIED "
            "production calculate_return()/calculate_success() functions. "
            "Independent of whenever the admin-triggered validator actually ran."
        ),
        "n_symbols_sampled": len(symbols),
        "n_symbols_fetch_failed": len(fetch_failures),
        "fetch_failures": fetch_failures,
        "n_recommendation_rows_in_sample": len(sample),
        "n_horizon_observations_total": len(df),
        "by_horizon": by_horizon,
        "by_horizon_and_signal": by_horizon_and_signal,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "horizon_audit.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(json.dumps(by_horizon, indent=2))


if __name__ == "__main__":
    main()
