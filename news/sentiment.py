import threading
import streamlit as st

# ── FinBERT loaded lazily — not at import time — to avoid OOM on Streamlit Cloud
@st.cache_resource(show_spinner=False)
def _load_finbert():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None


# ── Thread-safe in-memory sentiment cache ─────────────────────────────────────
_sentiment_cache: dict[str, tuple[str, float]] = {}
_cache_lock = threading.Lock()


def _textblob_sentiment(text: str) -> tuple[str, float]:
    """Lightweight fallback when FinBERT is unavailable."""
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity  # -1 to +1
        if polarity > 0.1:
            return "Positive", round(polarity, 2)
        elif polarity < -0.1:
            return "Negative", round(polarity, 2)
    except Exception:
        pass
    return "Neutral", 0.0


def analyze_sentiment(headline: str) -> tuple[str, float]:
    with _cache_lock:
        if headline in _sentiment_cache:
            return _sentiment_cache[headline]

    finbert = _load_finbert()
    if finbert is None:
        out = _textblob_sentiment(headline)
    else:
        try:
            result = finbert(headline[:512])[0]
            label  = result["label"].lower()
            score  = result["score"]
            if label == "positive":
                out = ("Positive",  round(score, 2))
            elif label == "negative":
                out = ("Negative", round(-score, 2))
            else:
                out = ("Neutral", 0.0)
        except Exception:
            out = _textblob_sentiment(headline)

    with _cache_lock:
        _sentiment_cache[headline] = out
    return out


def analyze_overall_sentiment(headlines: list[str]) -> tuple[str, float, list[dict], dict]:
    """
    Analyze sentiment of a list of headlines.

    Returns:
        mood      (str)        — "Bullish" | "Neutral" | "Bearish"
        avg_score (float)      — mean sentiment score across all headlines
        details   (list[dict]) — per-headline breakdown
        counts    (dict)       — {"positive": int, "neutral": int, "negative": int}
    """
    if not headlines:
        return "Neutral", 0.0, [], {"positive": 0, "neutral": 0, "negative": 0}

    finbert = _load_finbert()

    # ── Batch mode: run all uncached headlines through FinBERT in one call ───
    # This is significantly faster than calling finbert() one headline at a time.
    with _cache_lock:
        uncached = [h for h in headlines if h not in _sentiment_cache]

    if finbert is not None and uncached:
        try:
            batch_results = finbert([h[:512] for h in uncached])
            with _cache_lock:
                for headline, res in zip(uncached, batch_results):
                    label = res["label"].lower()
                    score = res["score"]
                    if label == "positive":
                        out = ("Positive",  round(score, 2))
                    elif label == "negative":
                        out = ("Negative", round(-score, 2))
                    else:
                        out = ("Neutral", 0.0)
                    _sentiment_cache[headline] = out
        except Exception:
            # Fall back to per-headline TextBlob if batch fails
            for headline in uncached:
                out = _textblob_sentiment(headline)
                with _cache_lock:
                    _sentiment_cache[headline] = out

    # ── Read results from cache — no re-analysis ─────────────────────────────
    scores  = []
    details = []
    counts  = {"positive": 0, "neutral": 0, "negative": 0}

    for hl in headlines:
        sentiment, score = analyze_sentiment(hl)
        scores.append(score)
        details.append({"headline": hl, "sentiment": sentiment, "score": score})
        counts[sentiment.lower()] = counts.get(sentiment.lower(), 0) + 1

    avg = sum(scores) / len(scores)
    if avg > 0.25:
        mood = "Bullish"
    elif avg < -0.25:
        mood = "Bearish"
    else:
        mood = "Neutral"

    return mood, round(avg, 2), details, counts