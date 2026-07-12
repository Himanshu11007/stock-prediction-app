"""
api/services.py — Service layer wrapping the existing StockAI Pro engine.

This module contains ZERO new business logic. Every function here is a thin
pass-through to existing modules (scanner/, storage/, news/, utils/, data/,
models/, features/). Routes call these services; services call your engine.

No thresholds, weights, or scoring logic are modified anywhere in this file —
every numeric constant (MIN_ACCURACY, signal buckets, pillar weights, etc.)
continues to live exactly where it already does, in config.py and the
existing modules. This file only orchestrates calls in the same order
app.py already uses for its "Analyse Stock" flow.
"""
from __future__ import annotations

import time
import uuid
import threading
from typing import Optional

from data.loader import load_data, load_multi_timeframe_data
from utils.helpers import prepare_data
from models.trainer import train_model, ensemble_predict
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment
from utils.decision_engine import generate_signal
from features.engineer import get_trend_signal
from utils.regime import detect_regime
from utils.risk import calculate_risk
from utils.company_mapper import get_company_names

from scanner.background import start_background_scan, is_scan_running, scan_progress
from scanner.cache import load_category_cache, cache_age_minutes
from config import CATEGORIES

from storage.tracker import save_signal, save_recommendation, upsert_recommendation, get_recent_signals
from utils.explainability import build_recommendation_explanation, compute_pillar_scores, compute_weighted_score
from utils.company_mapper import get_sector
from storage.recommendation_validation import (
    validate_old_recommendations,
    migrate_schema,
)
from storage.performance_analytics import (
    load_validated_df,
    summary_metrics,
    signal_performance,
    confidence_performance,
    confluence_performance,
)

from utils.logger import get_logger, log_exception, read_last_log_lines, clear_log_file

logger = get_logger(__name__)

# Ensure the recommendation_validation table/columns exist before any
# service in this module touches it.
migrate_schema()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Stock analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock(symbol: str) -> dict:
    """
    Run the full single-stock analysis pipeline — identical step order to
    app.py's "Analyse Stock" tab: load → features → train → predict →
    news/sentiment → regime → multi-timeframe → confluence signal → risk.

    Raises:
        ValueError   — invalid symbol / no price data / bad target distribution
        RuntimeError — any downstream pipeline failure
    """
    symbol = symbol.strip().upper()

    data = load_data(symbol)
    if data is None or data.empty:
        raise ValueError(f"No price data found for symbol '{symbol}'")

    company_name = get_company_names(symbol)

    try:
        data, X, y, _, _, y_train, _ = prepare_data(data)
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}") from e

    if len(set(y_train)) < 2:
        raise ValueError(
            f"Insufficient class variety in target for '{symbol}' — cannot train model"
        )

    try:
        models, acc = train_model(X, y)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e

    try:
        pred, confidence, _ = ensemble_predict(models, X.tail(1))
    except Exception:
        pred, confidence = 0, 0.0

    try:
        headlines = fetch_news(symbol)
    except Exception:
        headlines = []

    try:
        _, overall_score, _, _ = analyze_overall_sentiment(headlines)
    except Exception:
        overall_score = 0.0

    try:
        regime_info = detect_regime(data)
    except Exception:
        regime_info = None

    try:
        multi_tf_data   = load_multi_timeframe_data(symbol)
        weekly_trend    = get_trend_signal(multi_tf_data["weekly"])
        daily_trend     = get_trend_signal(multi_tf_data["daily"])
        timeframe_score = (weekly_trend["score"] + daily_trend["score"]) / 2
    except Exception:
        weekly_trend    = {"trend": "UNKNOWN", "score": 0}
        daily_trend     = {"trend": "UNKNOWN", "score": 0}
        timeframe_score = 0

    try:
        final_signal, final_score, reason, factors = generate_signal(
            prediction=int(pred[0]) if hasattr(pred, "__len__") else int(pred),
            confidence=confidence,
            news_score=overall_score,
            timeframe_score=timeframe_score,
            data=data,
            regime_info=regime_info,
        )
    except Exception as e:
        raise RuntimeError(f"Signal generation failed: {e}") from e

    try:
        risk = calculate_risk(data, final_signal)
    except Exception:
        risk = {}

    close_price = float(data["Close"].iloc[-1])

    # Persist — same two calls app.py makes, same try/except-and-continue
    # behaviour (a save failure must never break the analysis response).
    try:
        save_signal(symbol, company_name, final_signal, final_score, confidence, acc, close_price)
    except Exception as e:
        log_exception(logger, f"save_signal failed for {symbol}", e)

    try:
        _pred_int = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
        _pillar_scores = compute_pillar_scores(
            prediction=_pred_int, confidence=confidence,
            news_score=overall_score, timeframe_score=timeframe_score,
            data=data, regime_info=regime_info,
        )
        _weighted_score = compute_weighted_score(_pillar_scores)
        _sector = None
        try:
            _sector = get_sector(symbol)
        except Exception:
            pass

        upsert_recommendation(
            symbol           = symbol,
            stock            = company_name,
            signal           = final_signal,
            cmp              = close_price,
            confluence_score = final_score,
            ml_confidence    = confidence,
            news_score       = overall_score,
            accuracy         = acc,
            target           = risk.get("target")    if risk else None,
            stop_loss        = risk.get("stop_loss") if risk else None,
            pillar_scores    = _pillar_scores,
            weighted_score   = _weighted_score,
            sector           = _sector,
            market_regime    = (regime_info or {}).get("regime"),
            engine_version   = "v1.0",
        )
    except Exception as e:
        log_exception(logger, f"save_recommendation failed for {symbol}", e)

    # ── Build the explanation panel ───────────────────────────────────────────
    # Purely additive: a failure here must NEVER break the analysis response.
    # build_recommendation_explanation() already has its own internal
    # try/except and returns a safe fallback dict on failure, but we wrap
    # the call itself defensively too, so a completely unexpected error
    # (e.g. a bad import) still degrades to explanation=None rather than
    # failing the whole /analyze-stock request.
    explanation = None
    try:
        explanation = build_recommendation_explanation(
            symbol=symbol,
            stock_name=company_name,
            signal=final_signal,
            score=final_score,
            confidence=confidence,
            accuracy=acc,
            prediction=int(pred[0]) if hasattr(pred, "__len__") else int(pred),
            news_score=overall_score,
            timeframe_score=timeframe_score,
            regime_info=regime_info,
            factors=factors,
            risk=risk,
            data=data,
        )
    except Exception as e:
        log_exception(logger, f"Explanation build failed for {symbol}", e)
        explanation = None

    return {
        "symbol":       symbol,
        "stock":        company_name,
        "signal":       final_signal,
        "score":        round(final_score, 4),
        "confidence":   round(confidence, 2),
        "accuracy":     round(acc, 4),
        "news_score":   round(overall_score, 4),
        "regime":       (regime_info or {}).get("regime", "Unknown"),
        "weekly_trend": weekly_trend["trend"],
        "daily_trend":  daily_trend["trend"],
        "target":       risk.get("target")    if risk else None,
        "stop_loss":    risk.get("stop_loss") if risk else None,
        "factors":      factors,
        "explanation":  explanation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Top Picks scan — scan_id tracking (API-layer only; scanner code untouched)
# ══════════════════════════════════════════════════════════════════════════════
#
# scanner.background.start_background_scan() always scans ALL categories
# (Large/Mid/Small Cap) together — it has no concept of "scan just one
# category" and no concept of a caller-facing scan_id. Per the task
# constraints we do not modify that module's logic. Instead, this in-memory
# registry maps an API-generated scan_id to the category the caller asked
# about, and reports status by reading the *existing* is_scan_running() /
# scan_progress() / load_category_cache() functions — no new scanning
# logic, just a translation layer for the REST contract.

_scan_registry_lock = threading.Lock()
_scan_registry: dict[str, dict] = {}   # scan_id -> {"category": str, "started_at": float}


def start_top_picks_scan(category: str) -> dict:
    """
    Trigger the existing background scan and register a scan_id bound to
    the requested category so the caller can poll status/result for it.

    Note: the underlying scanner always scans all 3 categories in one run
    (existing behaviour, unchanged). This call simply (a) kicks that off if
    nothing is running, and (b) hands back a scan_id the caller can use to
    track *this* category's portion of that run.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")

    scan_id = uuid.uuid4().hex

    # start_background_scan() is itself idempotent — returns False if a scan
    # is already running, which we surface as status="running" below rather
    # than treating as an error.
    already_running = is_scan_running()
    if not already_running:
        from utils.company_mapper import COMPANY_MAPPING
        start_background_scan(COMPANY_MAPPING)

    with _scan_registry_lock:
        _scan_registry[scan_id] = {"category": category, "started_at": time.time()}

    return {
        "scan_id":  scan_id,
        "status":   "started",
        "category": category,
    }


def get_scan_status(scan_id: str) -> dict:
    """
    Translate the scanner's global progress file into a per-scan_id view.

    Returns status: "running" | "completed" | "failed" | "not_found"
    """
    with _scan_registry_lock:
        entry = _scan_registry.get(scan_id)

    if entry is None:
        return {
            "scan_id": scan_id, "status": "not_found",
            "progress": 0, "total": 0,
            "message": "Unknown scan_id",
        }

    category = entry["category"]
    progress = scan_progress()   # {"category": ..., "done": int, "total": int}

    prog_category = progress.get("category", "")
    done          = int(progress.get("done", 0))
    total         = int(progress.get("total", 0))

    if isinstance(prog_category, str) and prog_category.startswith("error"):
        return {
            "scan_id": scan_id, "status": "failed",
            "progress": done, "total": total,
            "message": prog_category,
        }

    if not is_scan_running() and prog_category == "done":
        return {
            "scan_id": scan_id, "status": "completed",
            "progress": total, "total": total,
            "message": f"Scan completed for {category}",
        }

    if is_scan_running():
        return {
            "scan_id": scan_id, "status": "running",
            "progress": done, "total": total,
            "message": f"Scanning ({prog_category or category})",
        }

    # Not running, not explicitly "done" — treat as completed if a cache exists
    cached = load_category_cache(category)
    if cached is not None:
        return {
            "scan_id": scan_id, "status": "completed",
            "progress": total or len(cached), "total": total or len(cached),
            "message": f"Scan completed for {category}",
        }

    return {
        "scan_id": scan_id, "status": "running",
        "progress": done, "total": total,
        "message": f"Scanning {category} stocks",
    }


def get_scan_result(scan_id: str) -> dict:
    """
    Return the cached recommendations for the category bound to this scan_id.
    """
    with _scan_registry_lock:
        entry = _scan_registry.get(scan_id)

    if entry is None:
        raise ValueError(f"Unknown scan_id '{scan_id}'")

    category = entry["category"]
    status_info = get_scan_status(scan_id)

    results = load_category_cache(category) or []

    return {
        "scan_id": scan_id,
        "status":  status_info["status"],
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Tracker
# ══════════════════════════════════════════════════════════════════════════════

def get_saved_recommendations(limit: int = 50) -> list[dict]:
    """Return recently saved signals (storage.tracker.get_recent_signals)."""
    return get_recent_signals(limit=limit)


def save_manual_recommendation(
    symbol:     str,
    stock:      str,
    signal:     str,
    cmp:        float,
    score:      float,
    confidence: float,
    news_score: float,
    accuracy:   float = 0.0,
    target:     Optional[float] = None,
    stop_loss:  Optional[float] = None,
) -> int:
    """Manually save a recommendation row (storage.tracker.save_recommendation)."""
    return save_recommendation(
        symbol=symbol, stock=stock, signal=signal, cmp=cmp,
        confluence_score=score, ml_confidence=confidence,
        news_score=news_score, accuracy=accuracy,
        target=target, stop_loss=stop_loss,
    )


def run_validation() -> int:
    """Run the existing 5-trading-day validation engine, unchanged."""
    return validate_old_recommendations()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Performance
# ══════════════════════════════════════════════════════════════════════════════

def get_performance_summary() -> dict:
    df = load_validated_df()
    return summary_metrics(df)


def get_performance_by_signal() -> list[dict]:
    df = load_validated_df()
    return signal_performance(df).to_dict(orient="records")


def get_performance_by_confidence() -> list[dict]:
    df = load_validated_df()
    return confidence_performance(df).to_dict(orient="records")


def get_performance_by_confluence() -> list[dict]:
    df = load_validated_df()
    return confluence_performance(df).to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Logs
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_logs(lines: int = 100) -> list[str]:
    return read_last_log_lines(n=lines)


def clear_logs() -> bool:
    return clear_log_file()


# ══════════════════════════════════════════════════════════════════════════════
# Intelligence Engine
# ══════════════════════════════════════════════════════════════════════════════

def get_intelligence_report() -> dict:
    """
    Return the full Recommendation Intelligence Report.
    Read-only — delegates entirely to the analytics engine.
    """
    from analytics.recommendation_intelligence import generate_engine_report
    return generate_engine_report()