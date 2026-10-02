"""
scripts/audit/walk_forward_benchmark.py — Phase 10 clean walk-forward benchmark.

Single reproducible entry point. For every (symbol, issue date D) in the
configured evaluation period it replays the unmodified production scanner
pipeline using only daily bars dated <= D (evaluation/walk_forward.py),
measures 1/3/5/10-trading-day outcomes, and writes two SEPARATE datasets:

  raw_model_predictions.csv       every replayed prediction (raw model benchmark)
  production_recommendations.csv  only rows that pass the production quality
                                  filters (production recommendation benchmark)

plus summary.json (machine-readable), summary.md (human-readable) and
price_manifest.json (sha256 of every input price file).

READ-ONLY with respect to the application: never touches storage/tracker.db,
the price cache, or any production model artefact.

Reproducibility: prices are downloaded once into a snapshot directory
(gitignored) and ALWAYS re-read from that snapshot, so a re-run against the
same snapshot and code produces byte-identical outputs. --refresh
re-downloads (and will change results as Yahoo revises history).

Usage:
    python scripts/audit/walk_forward_benchmark.py --config dev
    python scripts/audit/walk_forward_benchmark.py --config full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from evaluation.walk_forward import (  # noqa: E402
    CONFIDENCE_BUCKETS,
    HORIZONS,
    MODEL_VERSION,
    NEWS_STATUS,
    TemporalIntegrityError,
    build_prediction_record,
    bucket_summary,
    confidence_bucket,
    confluence_buckets,
    direction_summary,
    production_inputs,
    rate,
    regime_summary,
    session_closes,
    signal_summary,
    unconditional_success_rates,
)
from models.trainer import ENSEMBLE_WEIGHTS, walk_forward_validate_ensemble  # noqa: E402
from utils.helpers import prepare_data  # noqa: E402

AUDIT_DIR    = Path(__file__).resolve().parent
SNAPSHOT_DIR = AUDIT_DIR / ".price_snapshot"
OUTPUT_ROOT  = AUDIT_DIR / "output" / "walk_forward_benchmark"

# Price history fetched once for every config: enough for a 2-year weekly
# look-back before the earliest issue date. End is exclusive.
DATA_START = "2023-06-01"
DATA_END   = "2026-10-02"

# Deterministic universe: head of each existing universe CSV (largecap /
# midcap / smallcap), hard-coded so later CSV edits cannot change results.
_LARGE = ["ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
          "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BPCL.NS"]
_MID   = ["ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AIAENG.NS", "AJANTPHARM.NS",
          "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUBANK.NS"]
_SMALL = ["AARTIIND.NS", "AAVAS.NS", "AEGISCHEM.NS", "AETHER.NS", "AKZOINDIA.NS",
          "ALEMBICLTD.NS", "ALEXOTYRES.NS", "AMBER.NS", "ANGELONE.NS", "APARINDS.NS"]

CONFIGS = {
    # Fast development check (~100 predictions, ~2 min).
    "dev": {
        "symbols": _LARGE[:2] + _MID[:2] + _SMALL[:2],
        "eval_start": "2026-06-01",
        "eval_end": "2026-09-30",
        "stride_trading_days": 5,
    },
    # Production evaluation (~1,400 predictions, ~25 min).
    "full": {
        "symbols": _LARGE + _MID + _SMALL,
        "eval_start": "2025-10-01",
        "eval_end": "2026-09-30",
        "stride_trading_days": 5,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Prices
# ══════════════════════════════════════════════════════════════════════════════

def _snapshot_path(symbol: str) -> Path:
    return SNAPSHOT_DIR / f"{symbol.replace('.', '_')}.csv"


def _download(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf
    # auto_adjust=True: the yfinance default used by data.loader._fetch.
    df = yf.download(symbol, start=DATA_START, end=DATA_END, interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[df["Close"].notna()]


def load_prices(symbol: str, refresh: bool) -> pd.DataFrame | None:
    path = _snapshot_path(symbol)
    if refresh or not path.exists():
        df = _download(symbol)
        if df is None:
            return None
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    # Always read back from the snapshot so first runs and re-runs see
    # byte-identical inputs.
    df = pd.read_csv(path, index_col="Date", parse_dates=True, float_precision="round_trip")
    return df if not df.empty else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def issue_dates(full: pd.DataFrame, start: str, end: str, stride: int) -> list[pd.Timestamp]:
    """Every `stride`-th real trading session in [start, end]."""
    sessions = session_closes(full).index
    days = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    return list(days[::stride])


def ensemble_cv_for_symbol(full: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    """Task 2 reporting: legacy vs actual-ensemble walk-forward accuracy on
    the production 1-year window ending at the last issue date."""
    daily, _ = production_inputs(full, as_of)
    _, X, y, *_ = prepare_data(daily.copy())
    if len(X) < 80 or len(set(y)) < 2:
        return None
    r = walk_forward_validate_ensemble(X, y)
    return {
        "as_of": as_of,
        "n_rows": int(len(X)),
        "n_folds": r["n_folds"],
        "n_test_rows": r["n_test_rows"],
        "ensemble_accuracy": r["ensemble_accuracy"],
        "legacy_mean_individual_model_accuracy": r["mean_individual_model_accuracy"],
        "per_model_accuracy": r["per_model_accuracy"],
    }


def run(config_name: str, refresh: bool) -> Path:
    cfg = CONFIGS[config_name]
    out_dir = OUTPUT_ROOT / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped, fetch_failures, manifest, cv_rows = [], [], [], {}, []
    for symbol in cfg["symbols"]:
        full = load_prices(symbol, refresh)
        if full is None or len(full) < 300:
            fetch_failures.append(symbol)
            continue
        path = _snapshot_path(symbol)
        manifest[symbol] = {
            "file": path.name, "sha256": _sha256(path), "rows": int(len(full)),
            "first_date": full.index[0], "last_date": full.index[-1],
        }
        dates = issue_dates(full, cfg["eval_start"], cfg["eval_end"], cfg["stride_trading_days"])
        print(f"{symbol}: {len(dates)} issue dates", flush=True)
        for d in dates:
            rec = build_prediction_record(symbol, full, d)  # raises on temporal violation
            (skipped if "skipped" in rec else records).append(rec)
        if dates:
            cv = ensemble_cv_for_symbol(full, dates[-1])
            if cv:
                cv_rows.append({"symbol": symbol, **cv})

    raw = pd.DataFrame(records).sort_values(["prediction_timestamp", "symbol"]).reset_index(drop=True)
    raw["confidence_bucket"] = raw["confidence"].map(confidence_bucket)
    raw["forward_row_confidence_bucket"] = raw["forward_row_confidence"].map(confidence_bucket)
    filtered = raw[raw["passes_quality_filters"]].reset_index(drop=True)

    raw.to_csv(out_dir / "raw_model_predictions.csv", index=False)
    filtered.to_csv(out_dir / "production_recommendations.csv", index=False)
    (out_dir / "price_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    summary = summarise(cfg, config_name, raw, filtered, skipped, fetch_failures, cv_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {out_dir}")
    return out_dir


def summarise(cfg, config_name, raw, filtered, skipped, fetch_failures, cv_rows) -> dict:
    bucket_order = [f"{lo}-{hi}" for lo, hi in CONFIDENCE_BUCKETS]
    agree = raw["predicted_direction"] == raw["realized_issue_move"]

    cv_ens = [r["ensemble_accuracy"] for r in cv_rows if r["ensemble_accuracy"] is not None]
    cv_leg = [r["legacy_mean_individual_model_accuracy"] for r in cv_rows
              if r["legacy_mean_individual_model_accuracy"] is not None]
    cv_n = sum(r["n_test_rows"] for r in cv_rows)

    return {
        "config": config_name,
        "config_detail": cfg,
        "data_window": {"start": DATA_START, "end_exclusive": DATA_END},
        "model_version": MODEL_VERSION,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "horizons_trading_days": list(HORIZONS),
        "news_status": NEWS_STATUS,
        "population": {
            "n_symbols_requested": len(cfg["symbols"]),
            "n_symbols_evaluated": int(raw["symbol"].nunique()),
            "fetch_failures": fetch_failures,
            "n_predictions": int(len(raw)),
            "n_skipped": len(skipped),
            "skip_reasons": pd.Series([s["skipped"] for s in skipped]).value_counts().to_dict()
                            if skipped else {},
            "first_prediction": raw["prediction_timestamp"].min(),
            "last_prediction": raw["prediction_timestamp"].max(),
            "n_production_recommendations": int(len(filtered)),
            "signal_counts_all": raw["signal"].value_counts().sort_index().to_dict(),
            "signal_counts_filtered": filtered["signal"].value_counts().sort_index().to_dict(),
            "n_with_outcome": {f"{h}d": int(raw[f"outcome_{h}d"].notna().sum()) for h in HORIZONS},
        },
        "temporal_integrity": {
            "all_rows_passed_assertions": True,  # any violation aborts the run
            "prediction_row_in_training_set_pct": round(
                float(raw["prediction_row_in_training_set"].mean()) * 100, 1),
            "as_deployed_prediction_equals_realized_issue_move": rate(agree.astype(int)),
        },
        "ensemble_walk_forward_cv": {
            "description": "Same folds per symbol; legacy = mean of individual model "
                           "accuracies (what train_model reports), ensemble = actual blend.",
            "n_symbols": len(cv_rows),
            "n_test_rows_total": cv_n,
            "mean_ensemble_accuracy": round(sum(cv_ens) / len(cv_ens), 4) if cv_ens else None,
            "mean_legacy_mean_individual_accuracy": round(sum(cv_leg) / len(cv_leg), 4) if cv_leg else None,
            "per_symbol": cv_rows,
        },
        "raw_model_benchmark": {
            "population": "every replayed prediction, before any production filter",
            "as_deployed_direction": direction_summary(raw, "correct", "predicted_direction"),
            "forward_row_direction_diagnostic": direction_summary(
                raw, "forward_row_correct", "forward_row_predicted_direction"),
            "baseline_majority_class": direction_summary(raw, "baseline_majority_correct", "majority_class_training"),
            "baseline_previous_direction": direction_summary(
                raw, "baseline_prev_direction_correct", "realized_issue_move"),
            "confidence_buckets_as_deployed_1d": bucket_summary(
                raw, "confidence_bucket", "correct_1d", bucket_order),
            "confidence_buckets_as_deployed_5d": bucket_summary(
                raw, "confidence_bucket", "correct_5d", bucket_order),
            "confidence_buckets_forward_row_1d": bucket_summary(
                raw, "forward_row_confidence_bucket", "forward_row_correct_1d", bucket_order),
            "by_regime_as_deployed": regime_summary(raw, "correct"),
            "signals_unfiltered": signal_summary(raw, "signal_success", "return"),
            "unconditional_success_rule_rates": unconditional_success_rates(raw),
        },
        "production_recommendation_benchmark": {
            "population": "replayed predictions that pass scanner.filters.passes_quality_filters "
                          "(the rows production would persist)",
            "signals": signal_summary(filtered, "signal_success", "return"),
            "signals_existing_cmp_convention": signal_summary(
                filtered, "legacy_cmp_signal_success", "legacy_cmp_return"),
            "confluence_buckets_5d": confluence_buckets(filtered, "signal_success_5d"),
            "confluence_buckets_unfiltered_5d": confluence_buckets(raw, "signal_success_5d"),
            "confluence_observed_range": {
                "min": round(float(raw["confluence"].min()), 4),
                "max": round(float(raw["confluence"].max()), 4),
            },
            "by_regime": regime_summary(filtered, "signal_success"),
        },
    }


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(s: dict) -> str:
    p, ti, cv = s["population"], s["temporal_integrity"], s["ensemble_walk_forward_cv"]
    rm, pr = s["raw_model_benchmark"], s["production_recommendation_benchmark"]
    dcols = ["horizon_days", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct", "predicted_up_pct", "actual_up_pct"]
    scols = ["horizon_days", "signal", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct",
             "mean_return_pct", "median_return_pct"]
    bcols = ["bucket", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct"]
    out = [
        f"# Walk-forward benchmark — `{s['config']}` config\n",
        "Generated by `scripts/audit/walk_forward_benchmark.py`. Descriptive only: "
        "no overall score, no verdict. See docs/WALK_FORWARD_BENCHMARK.md for the contract.\n",
        "## Population\n",
        f"- Symbols evaluated: {p['n_symbols_evaluated']} / {p['n_symbols_requested']} "
        f"(fetch failures: {', '.join(p['fetch_failures']) or 'none'})",
        f"- Predictions: {p['n_predictions']} (skipped: {p['n_skipped']})",
        f"- Issue dates: {p['first_prediction']} → {p['last_prediction']}",
        f"- Rows with outcome: {p['n_with_outcome']}",
        f"- Production recommendations (filter survivors): {p['n_production_recommendations']}",
        f"- Signals (all): {p['signal_counts_all']}",
        f"- Signals (filtered): {p['signal_counts_filtered']}",
        f"- News: `{s['news_status']}`\n",
        "## Temporal integrity\n",
        "- Every row passed the T-ordering assertions (a violation aborts the run).",
        f"- Prediction row is inside its own training set: {ti['prediction_row_in_training_set_pct']}% of rows",
        f"- As-deployed prediction equals the already-realised D-1→D move: "
        f"{ti['as_deployed_prediction_equals_realized_issue_move']}\n",
        "## Ensemble walk-forward CV (Task 2)\n",
        f"- Symbols: {cv['n_symbols']}, pooled test rows: {cv['n_test_rows_total']}",
        f"- Mean actual-ensemble accuracy: {cv['mean_ensemble_accuracy']}",
        f"- Mean legacy (mean of individual models) accuracy: {cv['mean_legacy_mean_individual_accuracy']}\n",
        "## A. Raw model benchmark (all replayed predictions)\n",
        "### As-deployed ensemble direction vs forward outcome\n", _table(rm["as_deployed_direction"], dcols),
        "### Diagnostic: same models, unseen bar-D feature row\n",
        _table(rm["forward_row_direction_diagnostic"], dcols),
        "### Baseline: majority class of training labels\n", _table(rm["baseline_majority_class"], dcols),
        "### Baseline: previous direction (D-1→D move)\n", _table(rm["baseline_previous_direction"], dcols),
        "### Confidence buckets (as deployed, 1D direction)\n",
        _table(rm["confidence_buckets_as_deployed_1d"], bcols),
        "### Confidence buckets (forward-row diagnostic, 1D direction)\n",
        _table(rm["confidence_buckets_forward_row_1d"], bcols),
        "### Signals before filtering\n", _table(rm["signals_unfiltered"], scols),
        "### Unconditional success-rule rates (same population)\n",
        _table(rm["unconditional_success_rule_rates"],
               ["horizon_days", "rule", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct"]),
        "## B. Production recommendation benchmark (filter survivors)\n",
        "### Signal success, entry = Close[D]\n", _table(pr["signals"], scols),
        "### Same rows, existing cmp convention (entry = Close[D-1]; includes bar D's known move)\n",
        _table(pr["signals_existing_cmp_convention"], scols),
        f"### Confluence (observed range {pr['confluence_observed_range']}), 5D success\n",
        _table(pr["confluence_buckets_5d"], ["confluence_bucket", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct"]),
        "### By market regime\n",
        _table(pr["by_regime"], ["horizon_days", "market_regime", "n", "rate_pct", "ci95_low_pct", "ci95_high_pct"]),
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", choices=sorted(CONFIGS), default="dev")
    ap.add_argument("--refresh", action="store_true", help="re-download the price snapshot")
    args = ap.parse_args()
    try:
        run(args.config, args.refresh)
    except TemporalIntegrityError as e:
        print(f"BENCHMARK FAILED — temporal integrity violation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
