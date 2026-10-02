"""
scripts/audit/model_metrics_audit.py — Phase 9 recommendation-quality audit.

READ-ONLY diagnostic. Does not modify the database, does not retrain or
save any production model artifact, does not change thresholds/weights.

Purpose
───────
The existing production code (models/trainer.py) only ever reports
accuracy (`model.score()`). It never computes precision/recall/F1/
confusion matrix/ROC-AUC, and it never compares against a naive baseline.
This script reuses the EXACT existing feature engineering
(features.engineer.create_features) and the EXACT existing model
candidates (models.trainer._make_candidates) — unmodified — against a
deterministic sample of real stocks, with a proper chronological
(non-shuffled) train/test split, to answer three audit questions:

  1. What do precision/recall/F1/confusion-matrix/ROC-AUC/PR-AUC actually
     look like, beyond the single accuracy number the app surfaces?
  2. How does out-of-sample (chronological holdout) accuracy compare to
     in-sample (fit-and-score-on-everything) accuracy? A large gap would
     indicate the walk-forward accuracy figure is meaningfully more
     honest than a naive in-sample number would be.
  3. How does model accuracy compare to two trivial baselines: always
     predict the majority class, and "tomorrow repeats today's direction"?

This script does NOT touch storage/tracker.db or any recommendation
validation data — it only downloads fresh OHLCV data via yfinance and
runs the existing, unmodified training code against it.

Usage:
    python scripts/audit/model_metrics_audit.py

Output:
    scripts/audit/output/model_metrics_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.engineer import create_features  # noqa: E402
from models.trainer import _make_candidates, ensemble_predict  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Deterministic, fixed sample — first 10 symbols (alphabetical, as they
# appear in the CSV) from each of the three existing universe files, so
# this script's results are reproducible on any run.
_SAMPLE_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",  # largecap.csv head
    "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AIAENG.NS",                  # midcap.csv head
    "AARTIIND.NS", "AAVAS.NS", "AEGISCHEM.NS", "AETHER.NS",             # smallcap.csv head
]

FEATURE_COLS = [
    "Close", "Volume", "Price_Change",
    "MA_5", "MA_10", "MA_Diff",
    "EMA_20", "EMA_50", "EMA_Cross", "Price_vs_EMA20",
    "RSI", "Momentum", "Volatility",
    "Volume_Change", "Volume_MA", "Volume_Ratio",
    "MACD", "MACD_Hist", "MACD_Cross",
    "BB_Width", "BB_Position",
    "ATR", "ATR_Pct",
    "ADX", "Plus_DI", "Minus_DI",
    "Vol_Breakout",
]  # identical list to utils/helpers.py:prepare_data — kept in sync deliberately


def fetch(symbol: str) -> pd.DataFrame | None:
    df = yf.download(symbol, period="2y", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def per_stock_metrics(symbol: str) -> dict | None:
    raw = fetch(symbol)
    if raw is None or len(raw) < 150:
        return {"symbol": symbol, "error": "insufficient raw data"}

    data = create_features(raw)
    cols = [c for c in FEATURE_COLS if c in data.columns]
    X = data[cols]
    y = data["Up"].astype(int)

    if len(X) < 120:
        return {"symbol": symbol, "error": "insufficient feature rows after dropna"}

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"symbol": symbol, "error": "single-class train or test split"}

    # ── Fit the exact production ensemble on the TRAIN split only ───────────
    models = {}
    for name, model in _make_candidates():
        model.fit(X_train, y_train)
        models[name] = model

    # Ensemble prediction for every row of the out-of-sample test split,
    # using the EXACT production blend weights (models.trainer.ensemble_predict).
    oos_preds, oos_probs = [], []
    for i in range(len(X_test)):
        row = X_test.iloc[i : i + 1]
        pred, _conf, prob = ensemble_predict(models, row)
        oos_preds.append(pred)
        oos_probs.append(prob)
    oos_preds = np.array(oos_preds)
    oos_probs = np.array(oos_probs)
    y_test_arr = y_test.to_numpy()

    # ── In-sample comparison: fit on ALL rows, score on the SAME rows ────────
    in_sample_models = {}
    for name, model in _make_candidates():
        model.fit(X, y)
        in_sample_models[name] = model
    in_preds = []
    for i in range(len(X)):
        row = X.iloc[i : i + 1]
        pred, _conf, _prob = ensemble_predict(in_sample_models, row)
        in_preds.append(pred)
    in_sample_acc = accuracy_score(y, in_preds)

    # ── Baselines (computed on the SAME out-of-sample test split) ────────────
    majority_class = int(y_train.mode().iloc[0])
    baseline_majority_acc = accuracy_score(y_test_arr, [majority_class] * len(y_test_arr))

    # "tomorrow repeats today's realized direction" — Up[t] predicted by Up[t-1]
    prev_direction = y.shift(1)
    prev_direction_test = prev_direction.iloc[split:].to_numpy()
    valid_mask = ~pd.isna(prev_direction_test)
    if valid_mask.sum() > 0:
        baseline_prev_dir_acc = accuracy_score(
            y_test_arr[valid_mask], prev_direction_test[valid_mask].astype(int)
        )
    else:
        baseline_prev_dir_acc = None

    cm = confusion_matrix(y_test_arr, oos_preds, labels=[0, 1]).tolist()

    try:
        roc_auc = roc_auc_score(y_test_arr, oos_probs) if len(set(y_test_arr)) == 2 else None
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_test_arr, oos_probs) if len(set(y_test_arr)) == 2 else None
    except ValueError:
        pr_auc = None

    return {
        "symbol": symbol,
        "_oos_probs_and_outcomes": list(zip(oos_probs.tolist(), y_test_arr.tolist())),
        "n_total_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_class_balance_up_pct": round(float(y_train.mean()) * 100, 1),
        "test_class_balance_up_pct": round(float(y_test.mean()) * 100, 1),
        "out_of_sample": {
            "accuracy":  round(accuracy_score(y_test_arr, oos_preds), 4),
            "precision": round(precision_score(y_test_arr, oos_preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test_arr, oos_preds, zero_division=0), 4),
            "f1":        round(f1_score(y_test_arr, oos_preds, zero_division=0), 4),
            "roc_auc":   round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc":    round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix_labels_0_1": cm,
        },
        "in_sample_accuracy_fit_on_all_score_on_all": round(in_sample_acc, 4),
        "baseline_majority_class_accuracy": round(baseline_majority_acc, 4),
        "baseline_previous_direction_accuracy": (
            round(baseline_prev_dir_acc, 4) if baseline_prev_dir_acc is not None else None
        ),
    }


_CALIBRATION_BINS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 101]
_CALIBRATION_LABELS = [
    "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90+",
]


def calibration_table(results: list[dict]) -> list[dict]:
    """
    Pools every out-of-sample (probability, actual-outcome) pair across all
    sampled stocks, converts each to the same confidence/prediction the
    production ensemble_predict() would report (confidence = max(p,1-p)*100,
    pred = 1 if p>0.5 else 0), and buckets by confidence band to check
    whether e.g. "70% confidence" predictions are actually correct ~70% of
    the time. This is the same question Task 10 of the audit asks, computed
    from fresh out-of-sample data rather than the (pre-bug-fix, contaminated)
    production validation table.
    """
    rows = []
    for r in results:
        pairs = r.get("_oos_probs_and_outcomes") or []
        for prob, actual in pairs:
            pred = 1 if prob > 0.5 else 0
            confidence = max(prob, 1 - prob) * 100
            rows.append({"confidence": confidence, "correct": int(pred == actual)})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df["confidence"], bins=_CALIBRATION_BINS, labels=_CALIBRATION_LABELS, right=False)

    table = []
    for band in _CALIBRATION_LABELS:
        subset = df[df["band"] == band]
        n = len(subset)
        if n == 0:
            table.append({"confidence_band": band, "n": 0, "observed_success_rate_pct": None})
            continue
        observed = subset["correct"].mean() * 100
        table.append({
            "confidence_band": band,
            "n": n,
            "observed_success_rate_pct": round(observed, 1),
        })
    return table


def main() -> None:
    results = [per_stock_metrics(sym) for sym in _SAMPLE_SYMBOLS]
    ok = [r for r in results if "error" not in r]
    calibration = calibration_table(ok)
    # Drop the bulky raw pair list from the persisted per-stock output now
    # that the pooled calibration table has been built from it.
    for r in ok:
        r.pop("_oos_probs_and_outcomes", None)

    agg = {}
    if ok:
        for key in ("accuracy", "precision", "recall", "f1"):
            vals = [r["out_of_sample"][key] for r in ok]
            agg[f"mean_oos_{key}"] = round(float(np.mean(vals)), 4)
        agg["mean_in_sample_accuracy"] = round(
            float(np.mean([r["in_sample_accuracy_fit_on_all_score_on_all"] for r in ok])), 4
        )
        agg["mean_baseline_majority_accuracy"] = round(
            float(np.mean([r["baseline_majority_class_accuracy"] for r in ok])), 4
        )
        prev_dir_vals = [r["baseline_previous_direction_accuracy"] for r in ok if r["baseline_previous_direction_accuracy"] is not None]
        agg["mean_baseline_previous_direction_accuracy"] = (
            round(float(np.mean(prev_dir_vals)), 4) if prev_dir_vals else None
        )
        agg["mean_in_sample_minus_out_of_sample_accuracy_gap"] = round(
            agg["mean_in_sample_accuracy"] - agg["mean_oos_accuracy"], 4
        )

    output = {
        "method": (
            "Single chronological 80/20 holdout per stock (no shuffling). "
            "Uses the UNMODIFIED features.engineer.create_features and "
            "models.trainer._make_candidates/ensemble_predict - identical "
            "code to production, different (fresh, out-of-DB) data and "
            "additional metrics computed on top."
        ),
        "sample_symbols": _SAMPLE_SYMBOLS,
        "n_symbols_requested": len(_SAMPLE_SYMBOLS),
        "n_symbols_with_valid_result": len(ok),
        "per_stock": results,
        "aggregate_mean_across_stocks": agg,
        "pooled_confidence_calibration": calibration,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "model_metrics_audit.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
