"""
scanner/engine.py — pure business logic, no Streamlit calls.
Safe to call from background threads.
"""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from utils.helpers import prepare_data
from models.trainer import train_model, ensemble_predict
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment
from utils.decision_engine import generate_signal
from utils.regime import detect_regime
from utils.risk import calculate_risk
from scanner.filters import passes_quality_filters
from config import SCAN_MAX_STOCKS, SCAN_MAX_WORKERS
from data.loader import load_multi_timeframe_data
from features.engineer import get_trend_signal
from utils.logger import get_logger, log_stock_diagnostics, log_exception
from storage.tracker import upsert_recommendation, generate_scan_id
from utils.explainability import compute_pillar_scores, compute_weighted_score
from utils.company_mapper import get_sector

logger = get_logger(__name__)

_FETCH_RETRIES = 2
_RETRY_DELAY   = 2  # seconds


def _load_with_retry(loader_fn, symbol: str):
    """Call loader_fn with retries on transient 401/crumb errors."""
    for attempt in range(_FETCH_RETRIES + 1):
        try:
            data = loader_fn(symbol)
            if data is not None and not data.empty:
                return data
            return None
        except Exception as e:
            err = str(e).lower()
            is_transient = "401" in err or "crumb" in err or "unauthorized" in err
            if is_transient and attempt < _FETCH_RETRIES:
                logger.warning(
                    "%s: transient error — %s (retry %d/%d in %ds)",
                    symbol, e, attempt + 1, _FETCH_RETRIES, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                continue
            raise
    return None


def _scan_one(symbol: str, company_map: dict, loader_fn) -> dict | None:
    """Scan a single symbol. Returns a result dict, or None on failure/filter."""
    try:
        data = _load_with_retry(loader_fn, symbol)
        if data is None or data.empty:
            logger.warning("%s: no data returned", symbol)
            return None

        data, X, y, _, _, y_train, _ = prepare_data(data)
        if len(set(y_train)) < 2:
            logger.info("%s: single-class target — skipping", symbol)
            return None

        models, acc = train_model(X, y, fast=True)

        latest = X.iloc[-1:]
        pred, confidence, _ = ensemble_predict(models, latest)

        headlines = fetch_news(symbol)
        _, overall_score, _, _ = analyze_overall_sentiment(headlines)

        regime_info = detect_regime(data)

        multi_tf_data   = load_multi_timeframe_data(symbol)
        weekly_trend    = get_trend_signal(multi_tf_data["weekly"])
        daily_trend     = get_trend_signal(multi_tf_data["daily"])
        timeframe_score = (weekly_trend["score"] + daily_trend["score"]) / 2

        signal, score, reason, factors = generate_signal(
            prediction=int(pred),
            confidence=confidence,
            news_score=overall_score,
            timeframe_score=timeframe_score,
            data=data,
            regime_info=regime_info,
        )

        # ── Recompute the 8 raw pillar scores for diagnostics + persistence ───
        # Reuses decision_engine's own pillar functions (see
        # utils/explainability.py module docstring) — does not duplicate or
        # alter the scoring logic, just recovers the intermediate values
        # generate_signal() already used internally.
        pillar_scores = compute_pillar_scores(
            prediction=int(pred), confidence=confidence,
            news_score=overall_score, timeframe_score=timeframe_score,
            data=data, regime_info=regime_info,
        )
        weighted_score = compute_weighted_score(pillar_scores)

        log_stock_diagnostics(
            symbol=symbol, prediction=pred, confidence=confidence,
            accuracy=acc, signal=signal, final_score=score,
            ml_dir=pillar_scores["ML Direction"],
            ml_conf=pillar_scores["ML Confidence"],
            tech_score=pillar_scores["Technical Analysis"],
            news_score=pillar_scores["News Sentiment"],
            volume_score=pillar_scores["Volume"],
            regime_score=pillar_scores["Market Regime"],
            timeframe_score=pillar_scores["Multi-Timeframe"],
            momentum_score=pillar_scores["Momentum"],
            weighted_score=weighted_score,
        )

        if not passes_quality_filters(data, signal, confidence, acc, score):
            logger.info(
                "FILTERED | %-20s | signal=%-11s | conf=%6.2f | acc=%.4f | score=%.4f",
                symbol, signal, confidence, acc, score,
            )
            return None

        risk = calculate_risk(data, signal)

        logger.info(
            "PASSED   | %-20s | signal=%-11s | score=%.4f | conf=%6.2f | acc=%.4f",
            symbol, signal, score, confidence, acc,
        )

        sector = None
        try:
            sector = get_sector(symbol)
        except Exception:
            sector = None

        return {
            "stock":           company_map.get(symbol, symbol.replace(".NS", "")),
            "symbol":          symbol,
            "signal":          signal,
            "score":           round(score, 4),
            "confidence":      round(confidence, 2),
            "accuracy":        round(acc * 100, 2),
            "reason":          reason,
            "factors":         factors,
            "close":           risk["close"],
            "stop_loss":       risk["stop_loss"],
            "target":          risk["target"],
            "rr_ratio":        risk["rr_ratio"],
            "regime":          regime_info.get("regime", "Unknown"),
            "weekly_trend":    weekly_trend["trend"],
            "daily_trend":     daily_trend["trend"],
            "timeframe_score": round(timeframe_score, 2),
            "model":           "Ensemble",
            "news_score":      round(overall_score, 2),
            # ── Recommendation Intelligence Engine fields ──────────────────────
            "pillar_scores":   pillar_scores,
            "weighted_score":  weighted_score,
            "sector":          sector,
            "engine_version":  "v1.0",
        }

    except Exception as e:
        log_exception(logger, f"{symbol}: scan failed", e)
        return None


def _rerank_top_with_news(results: list[dict], top_n: int = 20) -> list[dict]:
    """Re-score top-N using already-fetched news_score — no extra HTTP calls."""
    top      = sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]
    top_syms = {r["symbol"] for r in top}

    reranked = []
    for r in top:
        news_score = r.get("news_score", 0.0)
        r["score"] = round(min(1.0, max(0.0, r["score"] + news_score * 0.10)), 4)
        reranked.append(r)

    remaining = [r for r in results if r["symbol"] not in top_syms]
    return sorted(reranked + remaining, key=lambda r: r["score"], reverse=True)


def _persist_recommendation(result: dict, scan_id: str) -> None:
    """
    Upsert one scan result into recommendation_validation, tagged with the
    scan_id for this run. Uses (symbol, saved_date) as the dedup key — see
    storage.tracker.upsert_recommendation for full behaviour. A failure
    here is logged and swallowed; it must never abort the scan.
    """
    try:
        upsert_recommendation(
            symbol           = result["symbol"],
            stock            = result["stock"],
            signal           = result["signal"],
            cmp              = result["close"],
            confluence_score = result["score"],
            ml_confidence    = result["confidence"],
            news_score       = result.get("news_score", 0.0),
            accuracy         = result["accuracy"] / 100.0,  # stored as 0-1 in DB
            target           = result.get("target"),
            stop_loss        = result.get("stop_loss"),
            scan_id          = scan_id,
            pillar_scores    = result.get("pillar_scores"),
            weighted_score   = result.get("weighted_score"),
            sector           = result.get("sector"),
            market_regime    = result.get("regime"),
            engine_version   = result.get("engine_version"),
        )
    except Exception as e:
        log_exception(logger, f"Failed to persist recommendation for {result.get('symbol')}", e)


def get_recommendations(
    stock_list,
    company_map: dict,
    use_raw_loader: bool = False,
    save_callback: Callable[[list], None] | None = None,
    save_interval: int = 5,
    scan_id: str | None = None,
) -> list[dict]:
    """
    Args:
        scan_id: Identifier for this scan run, used to tag every persisted
                 recommendation for traceability. If not provided, one is
                 generated automatically (prefix "SCAN").
    """
    if use_raw_loader:
        from data.loader import load_data_raw as loader_fn
    else:
        from data.loader import load_data as loader_fn

    scan_id = scan_id or generate_scan_id("SCAN")

    stocks  = list(stock_list)[:SCAN_MAX_STOCKS]
    results: list[dict] = []
    done    = 0

    logger.info(
        "Parallel scan: %d stocks / %d workers | scan_id=%s",
        len(stocks), SCAN_MAX_WORKERS, scan_id,
    )

    with ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_one, s, company_map, loader_fn): s for s in stocks}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)
                _persist_recommendation(result, scan_id)
            if save_callback and done % save_interval == 0:
                save_callback(sorted(results, key=lambda r: r["score"], reverse=True))

    logger.info("Scan batch done: %d processed, %d passed filters", done, len(results))

    prelim = sorted(results, key=lambda r: r["score"], reverse=True)
    final  = _rerank_top_with_news(prelim, top_n=20)
    if save_callback:
        save_callback(final)
    return final