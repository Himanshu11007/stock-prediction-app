import threading
import streamlit as st
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)

@st.cache_resource(show_spinner=False)
def _load_finbert():
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT sentiment model...")
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        logger.info("FinBERT loaded successfully")
        return model
    except Exception as e:
        log_exception(logger, "Failed to load FinBERT — falling back to TextBlob", e)
        return None

_sentiment_cache: dict[str, tuple[str, float]] = {}
_cache_lock = threading.Lock()


def _textblob_sentiment(text: str) -> tuple[str, float]:
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "Positive", round(polarity, 2)
        elif polarity < -0.1:
            return "Negative", round(polarity, 2)
    except Exception as e:
        logger.debug("TextBlob fallback failed: %s", e)
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
        except Exception as e:
            logger.debug("FinBERT inference failed for headline, using TextBlob: %s", e)
            out = _textblob_sentiment(headline)

    with _cache_lock:
        _sentiment_cache[headline] = out
    return out


def analyze_overall_sentiment(headlines: list[str]) -> tuple[str, float, list[dict], dict]:
    """
    Returns:
        mood, avg_score, details, counts
        counts = {"positive": int, "neutral": int, "negative": int}
    """
    if not headlines:
        return "Neutral", 0.0, [], {"positive": 0, "neutral": 0, "negative": 0}

    finbert = _load_finbert()

    # Batch uncached headlines through FinBERT in one call
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
            logger.debug("FinBERT batch: %d headlines analyzed", len(uncached))
        except Exception as e:
            logger.warning("FinBERT batch failed, falling back to TextBlob: %s", e)
            for headline in uncached:
                out = _textblob_sentiment(headline)
                with _cache_lock:
                    _sentiment_cache[headline] = out

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

    logger.debug(
        "Sentiment summary: mood=%s avg=%.3f pos=%d neu=%d neg=%d",
        mood, avg, counts["positive"], counts["neutral"], counts["negative"],
    )
    return mood, round(avg, 2), details, counts