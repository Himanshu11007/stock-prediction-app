"""
evaluation/walk_forward.py — core of the Phase 10 clean walk-forward benchmark.

Evaluation contract (full text: docs/WALK_FORWARD_BENCHMARK.md)
────────────────────────────────────────────────────────────────
  T (prediction_timestamp)  End of trading day D: the last daily bar the
                            production scanner would have seen had it run
                            after D's close.
  Information set at T      Daily bars dated <= D only. replay_production_
                            prediction() slices the history to <= D before
                            any production code sees it.
  Production prediction     The unmodified scanner path (scanner/engine.py
                            _scan_one): prepare_data -> train_model(fast=True)
                            -> ensemble_predict(X.iloc[-1:]) -> detect_regime
                            -> get_trend_signal (daily + weekly) ->
                            generate_signal -> passes_quality_filters.
                            News is the documented exception: no point-in-time
                            news source exists, so the production no-headlines
                            path (score 0.0) is used and flagged.
  Outcomes                  Adjusted Close exactly h trading days after D
                            (h in 1/3/5/10) in the symbol's own trading
                            calendar. Entry price = Close[D]. NULL when bar
                            D+h does not exist — never substituted or shortened.

Nothing in this module changes production behaviour; it only calls the
production functions on point-in-time slices of history.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.engineer import create_features, get_trend_signal
from models.trainer import (
    ENSEMBLE_WEIGHTS,
    component_probabilities,
    ensemble_predict,
    ensemble_proba,
    train_model,
)
from news.sentiment import analyze_overall_sentiment
from scanner.filters import passes_quality_filters
from storage.recommendation_validation import calculate_return, calculate_success
from utils.decision_engine import generate_signal
from utils.explainability import compute_pillar_scores, compute_weighted_score
from utils.helpers import prepare_data
from utils.regime import detect_regime
from utils.risk import calculate_risk

HORIZONS = (1, 3, 5, 10)

# Production look-back windows: data.loader._fetch downloads period="1y"
# daily bars; load_multi_timeframe_data downloads period="2y" weekly bars.
DAILY_LOOKBACK  = pd.DateOffset(years=1)
WEEKLY_LOOKBACK = pd.DateOffset(years=2)

NEWS_STATUS = "unavailable_no_point_in_time_source"

_SHORT = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb"}
MODEL_VERSION = "ensemble_" + "_".join(
    f"{_SHORT[k]}{v:.2f}" for k, v in ENSEMBLE_WEIGHTS.items()
)

CONFIDENCE_BUCKETS = [(lo, lo + 5) for lo in range(50, 100, 5)]  # [50,55) … [95,100]


class TemporalIntegrityError(AssertionError):
    """A benchmark row used information from after its prediction timestamp."""


# ══════════════════════════════════════════════════════════════════════════════
# Point-in-time inputs
# ══════════════════════════════════════════════════════════════════════════════

def weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monday-start weekly OHLCV bars built from daily bars, mirroring yfinance
    interval="1wk". The final week is partial (ends at the last daily bar),
    which is exactly what yfinance returns when called mid-week — so no
    weekly bar ever contains a day after the last daily bar.
    """
    g = daily.groupby(daily.index.to_period("W-SUN"))
    weekly = pd.DataFrame({
        "Open":   g["Open"].first(),
        "High":   g["High"].max(),
        "Low":    g["Low"].min(),
        "Close":  g["Close"].last(),
        "Volume": g["Volume"].sum(),
    })
    weekly.index = weekly.index.start_time
    return weekly


def production_inputs(full: pd.DataFrame, issue_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The (daily, weekly) frames production would have downloaded at T.
    Only bars dated <= issue_date are ever read.
    """
    issue_date = pd.Timestamp(issue_date)
    history = full.loc[full.index <= issue_date]
    daily = history.loc[history.index > issue_date - DAILY_LOOKBACK].copy()
    weekly = weekly_bars(history.loc[history.index > issue_date - WEEKLY_LOOKBACK])
    return daily, weekly


def issue_row_features(daily: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Feature row for the LAST bar of `daily` (bar D).

    create_features() drops bar D because its next-day label is unknown.
    Every feature is backward-looking, so appending a placeholder bar after
    D lets the bar-D row survive dropna() without any of its feature values
    depending on the placeholder; the placeholder-derived label is discarded.
    tests/test_walk_forward_benchmark.py verifies the result equals the
    features computed with the real next bar present.
    """
    issue_date = daily.index[-1]
    placeholder = daily.iloc[[-1]].copy()
    placeholder.index = pd.DatetimeIndex([issue_date + pd.Timedelta(days=1)])
    feats = create_features(pd.concat([daily, placeholder]))
    if issue_date not in feats.index:
        return None
    return feats.loc[[issue_date], feature_cols]


# ══════════════════════════════════════════════════════════════════════════════
# Production replay
# ══════════════════════════════════════════════════════════════════════════════

def replay_production_prediction(full: pd.DataFrame, issue_date) -> dict:
    """
    Re-run the unmodified scanner pipeline as it would have run after the
    close of `issue_date`. Returns a dict; on skip, {"skipped": reason}.
    """
    issue_date = pd.Timestamp(issue_date)
    daily, weekly = production_inputs(full, issue_date)
    if daily.empty or daily.index[-1] != issue_date:
        return {"skipped": "issue_date_not_a_trading_day"}

    data, X, y, _, _, y_train, _ = prepare_data(daily.copy())
    if len(X) < 2 or len(set(y_train)) < 2:
        return {"skipped": "single_class_or_insufficient_rows"}

    # ── ML: exactly scanner/engine.py:_scan_one ──────────────────────────────
    models, acc = train_model(X, y, fast=True)
    latest = X.iloc[-1:]
    pred, confidence, prob = ensemble_predict(models, latest)
    comps = component_probabilities(models, latest)

    # News: production's no-headlines path (analyze_overall_sentiment([]))
    _, news_score, _, _ = analyze_overall_sentiment([])

    regime_info     = detect_regime(data)
    weekly_trend    = get_trend_signal(weekly)
    daily_trend     = get_trend_signal(daily)
    timeframe_score = (weekly_trend["score"] + daily_trend["score"]) / 2

    signal, score, _reason, _factors = generate_signal(
        prediction=int(pred),
        confidence=confidence,
        news_score=news_score,
        timeframe_score=timeframe_score,
        data=data,
        regime_info=regime_info,
    )
    pillar_scores = compute_pillar_scores(
        prediction=int(pred), confidence=confidence,
        news_score=news_score, timeframe_score=timeframe_score,
        data=data, regime_info=regime_info,
    )
    passes = passes_quality_filters(data, signal, confidence, acc, score)
    risk = calculate_risk(data, signal)

    # ── Diagnostic only: same fitted models applied to the unseen bar-D row ──
    fwd_prob = fwd_pred = fwd_conf = None
    fwd_row = issue_row_features(daily, list(X.columns))
    if fwd_row is not None:
        fwd_prob = float(ensemble_proba(models, fwd_row)[0])
        fwd_pred = 1 if fwd_prob > 0.5 else 0
        fwd_conf = round(max(fwd_prob, 1 - fwd_prob) * 100, 2)

    # The label of the last training row is Close[next bar] > Close[row];
    # that next bar is the label's realisation time.
    last_train_pos = daily.index.get_loc(X.index[-1])
    label_end = daily.index[last_train_pos + 1] if last_train_pos + 1 < len(daily) else None

    return {
        "prediction_timestamp":         issue_date,
        "model_version":                MODEL_VERSION,
        "daily_window_start":           daily.index[0],
        "daily_window_end":             daily.index[-1],
        "weekly_window_end":            weekly.index[-1],
        "training_start_timestamp":     X.index[0],
        "training_end_timestamp":       X.index[-1],
        "training_label_end_timestamp": label_end,
        "n_training_rows":              int(len(X)),
        "feature_row_timestamp":        latest.index[-1],
        "prediction_row_in_training_set": bool(latest.index[-1] in X.index),
        "majority_class_training":      int(y.mean() > 0.5),
        "production_accuracy_fast":     float(acc),
        "logistic_probability":         float(comps["Logistic Regression"][0]),
        "rf_probability":               float(comps["Random Forest"][0]),
        "xgb_probability":              float(comps["XGBoost"][0]),
        "ensemble_probability":         float(prob),
        "predicted_direction":          int(pred),
        "confidence":                   float(confidence),
        "forward_row_ensemble_probability": fwd_prob,
        "forward_row_predicted_direction":  fwd_pred,
        "forward_row_confidence":           fwd_conf,
        "signal":                       signal,
        "confluence":                   float(score),
        "weighted_score":               float(compute_weighted_score(pillar_scores)),
        "market_regime":                regime_info.get("regime", "Unknown"),
        "timeframe_score":              float(timeframe_score),
        "news_score":                   float(news_score),
        "news_status":                  NEWS_STATUS,
        "passes_quality_filters":       bool(passes),
        "production_cmp":               float(risk["close"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Outcomes
# ══════════════════════════════════════════════════════════════════════════════

def session_closes(full: pd.DataFrame) -> pd.Series:
    """
    Close prices of real trading sessions only. Yahoo emits flat
    zero-volume placeholder bars on exchange holidays (Open=High=Low=Close,
    Volume=0); those are not trading days and must not count towards a
    trading-day horizon. The production replay still sees them, because
    production does.
    """
    placeholder = (
        (full["Volume"] == 0)
        & (full["Open"] == full["High"])
        & (full["High"] == full["Low"])
        & (full["Low"] == full["Close"])
    )
    return full.loc[~placeholder, "Close"]


def forward_outcomes(close: pd.Series, issue_date, horizons=HORIZONS) -> dict:
    """
    Price exactly h trading days after issue_date, in `close`'s own calendar.
    Missing future bars give None — never a nearer or later substitute.
    """
    issue_date = pd.Timestamp(issue_date)
    pos = close.index.get_loc(issue_date)
    entry = float(close.iloc[pos])
    out = {"prediction_price": entry}
    for h in horizons:
        tgt = pos + h
        if tgt < len(close):
            price = float(close.iloc[tgt])
            out[f"outcome_date_{h}d"]  = close.index[tgt]
            out[f"actual_price_{h}d"]  = price
            out[f"return_{h}d"]        = calculate_return(entry, price)
            out[f"outcome_{h}d"]       = int(price > entry)
        else:
            out[f"outcome_date_{h}d"]  = None
            out[f"actual_price_{h}d"]  = None
            out[f"return_{h}d"]        = None
            out[f"outcome_{h}d"]       = None
    return out


def build_prediction_record(symbol: str, full: pd.DataFrame, issue_date,
                            horizons=HORIZONS) -> dict:
    """
    One benchmark row: point-in-time production replay + forward outcomes
    + baselines + per-horizon correctness / signal success. Raises
    TemporalIntegrityError if any timestamp rule is violated.
    """
    issue_date = pd.Timestamp(issue_date)
    close = session_closes(full)
    if issue_date not in close.index:
        return {"symbol": symbol, "prediction_timestamp": issue_date,
                "skipped": "issue_date_not_a_trading_session"}
    rec = replay_production_prediction(full, issue_date)
    if "skipped" in rec:
        return {"symbol": symbol, "prediction_timestamp": issue_date, **rec}

    pos = close.index.get_loc(issue_date)
    prev_close = float(close.iloc[pos - 1]) if pos > 0 else None

    rec = {
        "prediction_id": f"{symbol}|{issue_date:%Y-%m-%d}|{MODEL_VERSION}",
        "symbol": symbol,
        **rec,
        **forward_outcomes(close, issue_date, horizons),
    }
    # Already realised at T: did bar D close above bar D-1?
    rec["realized_issue_move"] = (
        int(rec["prediction_price"] > prev_close) if prev_close is not None else None
    )

    for h in horizons:
        outcome = rec[f"outcome_{h}d"]
        ret = rec[f"return_{h}d"]
        if outcome is None:
            for k in ("correct", "forward_row_correct", "baseline_majority_correct",
                      "baseline_prev_direction_correct", "signal_success",
                      "legacy_cmp_return", "legacy_cmp_signal_success"):
                rec[f"{k}_{h}d"] = None
            continue
        rec[f"correct_{h}d"] = int(rec["predicted_direction"] == outcome)
        rec[f"forward_row_correct_{h}d"] = (
            int(rec["forward_row_predicted_direction"] == outcome)
            if rec["forward_row_predicted_direction"] is not None else None
        )
        rec[f"baseline_majority_correct_{h}d"] = int(rec["majority_class_training"] == outcome)
        rec[f"baseline_prev_direction_correct_{h}d"] = (
            int(rec["realized_issue_move"] == outcome)
            if rec["realized_issue_move"] is not None else None
        )
        rec[f"signal_success_{h}d"] = calculate_success(rec["signal"], ret)
        # Existing-system convention: entry = stored cmp (Close of bar D-1),
        # so this return includes bar D's move, already known at T.
        legacy_ret = calculate_return(rec["production_cmp"], rec[f"actual_price_{h}d"])
        rec[f"legacy_cmp_return_{h}d"] = legacy_ret
        rec[f"legacy_cmp_signal_success_{h}d"] = calculate_success(rec["signal"], legacy_ret)

    assert_temporal_integrity(rec, horizons)
    return rec


def assert_temporal_integrity(rec: dict, horizons=HORIZONS) -> None:
    """Fail loudly if a row used, or is scored with, the wrong side of T."""
    t = pd.Timestamp(rec["prediction_timestamp"])
    problems = []
    if not pd.Timestamp(rec["training_end_timestamp"]) < t:
        problems.append("training_end_timestamp >= prediction_timestamp")
    if rec["training_label_end_timestamp"] is None or not pd.Timestamp(rec["training_label_end_timestamp"]) <= t:
        problems.append("training label realised after prediction_timestamp")
    if not pd.Timestamp(rec["daily_window_end"]) <= t:
        problems.append("daily input window extends past prediction_timestamp")
    if not pd.Timestamp(rec["weekly_window_end"]) <= t:
        problems.append("weekly input window extends past prediction_timestamp")
    if not pd.Timestamp(rec["feature_row_timestamp"]) <= t:
        problems.append("feature row after prediction_timestamp")
    for h in horizons:
        d = rec.get(f"outcome_date_{h}d")
        if d is not None and not pd.Timestamp(d) > t:
            problems.append(f"outcome_date_{h}d not after prediction_timestamp")
    if problems:
        raise TemporalIntegrityError(f"{rec.get('prediction_id')}: " + "; ".join(problems))


# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics (descriptive only — no scores, no rankings)
# ══════════════════════════════════════════════════════════════════════════════

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1)


def rate(series: pd.Series) -> dict:
    """n, rate %, Wilson 95% CI for a 0/1 series (None values ignored)."""
    s = series.dropna().astype(int)
    n, k = int(len(s)), int(s.sum())
    lo, hi = wilson_interval(k, n)
    return {
        "n": n,
        "rate_pct": round(k / n * 100, 1) if n else None,
        "ci95_low_pct": lo,
        "ci95_high_pct": hi,
    }


def confidence_bucket(conf: float | None) -> str | None:
    if conf is None or (isinstance(conf, float) and math.isnan(conf)):
        return None
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= conf < hi or (hi == 100 and conf == 100):
            return f"{lo}-{hi}"
    return None


def direction_summary(df: pd.DataFrame, correct_prefix: str, pred_col: str,
                      horizons=HORIZONS) -> list[dict]:
    rows = []
    for h in horizons:
        sub = df[df[f"outcome_{h}d"].notna()]
        if pred_col in sub:
            sub = sub[sub[pred_col].notna()]
        r = rate(sub[f"{correct_prefix}_{h}d"])
        r["horizon_days"] = h
        if pred_col in sub and len(sub):
            r["predicted_up_pct"] = round(float(sub[pred_col].astype(int).mean()) * 100, 1)
            r["actual_up_pct"] = round(float(sub[f"outcome_{h}d"].astype(int).mean()) * 100, 1)
        rows.append(r)
    return rows


def bucket_summary(df: pd.DataFrame, bucket_col: str, value_col: str,
                   order: list[str] | None = None) -> list[dict]:
    rows = []
    keys = order if order is not None else sorted(df[bucket_col].dropna().unique())
    for key in keys:
        sub = df[(df[bucket_col] == key) & df[value_col].notna()]
        r = rate(sub[value_col])
        rows.append({"bucket": key, **r})
    return rows


def signal_summary(df: pd.DataFrame, success_prefix: str, return_prefix: str,
                   horizons=HORIZONS) -> list[dict]:
    rows = []
    for h in horizons:
        sub = df[df[f"{success_prefix}_{h}d"].notna()]
        for sig, grp in sorted(sub.groupby("signal"), key=lambda kv: kv[0]):
            r = rate(grp[f"{success_prefix}_{h}d"])
            rets = grp[f"{return_prefix}_{h}d"].astype(float)
            rows.append({
                "horizon_days": h,
                "signal": sig,
                **r,
                "mean_return_pct": round(float(rets.mean()), 3),
                "median_return_pct": round(float(rets.median()), 3),
            })
    return rows


def unconditional_success_rates(df: pd.DataFrame, horizons=HORIZONS) -> list[dict]:
    """
    How often each success rule is met by the whole replay population,
    regardless of the signal issued — a descriptive reference, computed
    on the same rows and with no information from after T.
    """
    rows = []
    for h in horizons:
        rets = df[f"return_{h}d"].dropna().astype(float)
        for label, sig in (("up_rule (BUY/STRONG BUY)", "BUY"),
                           ("down_rule (SELL/STRONG SELL)", "SELL"),
                           ("hold_band_rule (HOLD)", "HOLD")):
            r = rate(rets.map(lambda x, s=sig: calculate_success(s, x)))
            rows.append({"horizon_days": h, "rule": label, **r})
    return rows


def regime_summary(df: pd.DataFrame, value_prefix: str, horizons=HORIZONS) -> list[dict]:
    rows = []
    for h in horizons:
        col = f"{value_prefix}_{h}d"
        sub = df[df[col].notna()]
        for regime, grp in sorted(sub.groupby("market_regime"), key=lambda kv: kv[0]):
            rows.append({"horizon_days": h, "market_regime": regime, **rate(grp[col])})
    return rows


def confluence_buckets(df: pd.DataFrame, value_col: str, width: float = 0.05,
                       min_n: int = 30) -> list[dict]:
    """Observed-range confluence buckets; rates suppressed below min_n."""
    sub = df[df[value_col].notna()].copy()
    if sub.empty:
        return []
    sub["bucket_lo"] = (np.floor(sub["confluence"] / width) * width).round(2)
    rows = []
    for lo, grp in sub.groupby("bucket_lo"):
        r = rate(grp[value_col])
        if r["n"] < min_n:
            r = {"n": r["n"], "rate_pct": None, "ci95_low_pct": None, "ci95_high_pct": None,
                 "note": f"n < {min_n}: rate not reported"}
        rows.append({"confluence_bucket": f"{lo:.2f}-{lo + width:.2f}", **r})
    return rows
