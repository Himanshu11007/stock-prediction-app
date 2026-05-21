"""
scanner/engine.py — pure business logic, no Streamlit calls.
Safe to call from background threads.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from utils.helpers import prepare_data
from models.trainer import train_model
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment
from utils.decision_engine import generate_signal
from scanner.filters import passes_quality_filters
from config import SCAN_MAX_STOCKS, SCAN_MAX_WORKERS


def _scan_one(symbol: str, company_map: dict, loader_fn) -> dict | None:
    """Scan a single symbol. Returns a result dict, or None on failure / filtered out."""
    try:
        data = loader_fn(symbol)
        if data is None or data.empty:
            return None

        data, X, _, X_train, X_test, y_train, y_test = prepare_data(data)
        if len(set(y_train)) < 2:
            return None

        model, acc, model_name = train_model(X_train, X_test, y_train, y_test)

        latest = X.iloc[-1:]
        pred   = model.predict(latest)

        try:
            proba      = model.predict_proba(latest)
            confidence = float(max(proba[0])) * 100
        except AttributeError:
            confidence = 50.0

        headlines = fetch_news(symbol)
        _, overall_score, _ = analyze_overall_sentiment(headlines)

        signal, score, reason = generate_signal(int(pred[0]), confidence, overall_score)

        if not passes_quality_filters(data, signal, confidence, acc):
            return None

        return {
            "stock":      company_map.get(symbol, symbol.replace(".NS", "")),
            "symbol":     symbol,
            "signal":     signal,
            "score":      round(score, 4),
            "confidence": round(confidence, 2),
            "accuracy":   round(acc * 100, 2),
            "reason":     reason,
            "close":      round(float(data["Close"].iloc[-1]), 2),
            "model":      model_name,
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

    Args:
        stock_list:    Iterable of Yahoo Finance symbols
        company_map:   Dict symbol → display name
        use_raw_loader: True when called from a background thread
        save_callback: Called with the current sorted results every save_interval stocks
        save_interval: How often (in completed stocks) to call save_callback
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

            # Progressive save — write partial results so UI doesn't wait
            if save_callback and done % save_interval == 0:
                save_callback(sorted(results, key=lambda r: r["score"], reverse=True))

    final = sorted(results, key=lambda r: r["score"], reverse=True)

    # Final save (captures any remaining results after last interval)
    if save_callback:
        save_callback(final)

    return final
