"""
scanner/engine.py — pure business logic, no Streamlit calls.
Safe to call from background threads.
"""
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


def _scan_one(symbol: str, company_map: dict, loader_fn) -> dict | None:
    """Scan a single symbol. Returns a result dict, or None on failure / filtered out."""
    try:
        data = loader_fn(symbol)
        if data is None or data.empty:
            return None

        data, X, y, _, _, y_train, _ = prepare_data(data)
        if len(set(y_train)) < 2:
            return None

        models, acc = train_model(X, y)
        model_name  = "Ensemble"

        latest = X.iloc[-1:]
        pred, confidence, _ = ensemble_predict(models, latest)

        headlines = fetch_news(symbol)
        _, overall_score, _ = analyze_overall_sentiment(headlines)

        regime_info = detect_regime(data)

        multi_tf_data = load_multi_timeframe_data(symbol)
        
        weekly_trend = get_trend_signal(multi_tf_data["weekly"])
        daily_trend = get_trend_signal(multi_tf_data["daily"])

        raw_tf_score = (
            weekly_trend["score"] + daily_trend["score"]
        )
        timeframe_score = raw_tf_score / 2


        signal, score, reason, factors = generate_signal(
           prediction=int(pred),
           confidence=confidence,
           news_score= overall_score,
           timeframe_score=timeframe_score,
           data=data,
           regime_info=regime_info
        )

        # print(
        #     f"{symbol} | {signal} | "
        #     f"Score = {score: .2f} | "
        #     f"Conf = {confidence: .2f} |"
        #     f"Acc = {acc * 100: .2f}"
        # )

        if not passes_quality_filters(data, signal, confidence, acc,score):
            return None

        risk = calculate_risk(data, signal)

        return {
            "stock":       company_map.get(symbol, symbol.replace(".NS", "")),
            "symbol":      symbol,
            "signal":      signal,
            "score":       round(score, 4),
            "confidence":  round(confidence, 2),
            "accuracy":    round(acc * 100, 2),
            "reason":      reason,
            "factors":     factors,
            "close":       risk["close"],
            "stop_loss":   risk["stop_loss"],
            "target":      risk["target"],
            "rr_ratio":    risk["rr_ratio"],
            "regime":      regime_info.get("regime", "Unknown"),
            "weekly_trend":weekly_trend["trend"],
            "daily_trend":daily_trend["trend"],
            "timeframe_score":round(timeframe_score,2),
            "model":       model_name,
        }
    except Exception:
        return None


def get_recommendations(
    stock_list,
    company_map: dict,
    use_raw_loader: bool = False,
    save_callback: Callable[[list], None] | None = None,
    save_interval: int = 5,
) -> list[dict]:
    """
    Scan stocks in parallel. Optionally calls save_callback every save_interval
    stocks so the UI can display partial results before the full scan ends.
    """
    if use_raw_loader:
        from data.loader import load_data_raw as loader_fn
    else:
        from data.loader import load_data as loader_fn

    stocks  = list(stock_list)[:SCAN_MAX_STOCKS]
    results: list[dict] = []
    done    = 0

    with ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_one, s, company_map, loader_fn): s for s in stocks}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)

            if save_callback and done % save_interval == 0:
                save_callback(sorted(results, key=lambda r: r["score"], reverse=True))

    final = sorted(results, key=lambda r: r["score"], reverse=True)
    if save_callback:
        save_callback(final)
    return final
