import streamlit as st

# ── FinBERT loaded lazily — not at import time — to avoid OOM on Streamlit Cloud
@st.cache_resource(show_spinner=False)
def _load_finbert():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None

# ── In-memory sentiment cache — avoids re-running FinBERT on identical headlines
_sentiment_cache: dict[str, tuple[str, float]] = {}


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

    _sentiment_cache[headline] = out
    return out


def analyze_overall_sentiment(headlines: list[str]) -> tuple[str, float, list[dict]]:
    if not headlines:
        return "Neutral", 0.0, []

    scores  = []
    details = []
    for hl in headlines:
        sentiment, score = analyze_sentiment(hl)
        scores.append(score)
        details.append({"headline": hl, "sentiment": sentiment, "score": score})

    avg = sum(scores) / len(scores)
    if avg > 0.25:
        mood = "Bullish"
    elif avg < -0.25:
        mood = "Bearish"
    else:
        mood = "Neutral"

    return mood, round(avg, 2), details
