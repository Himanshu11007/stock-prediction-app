import streamlit as st
from transformers import pipeline

# ── FinBERT loaded once per process ───────────────────────────────────────────
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

finbert = load_finbert()

# ── In-memory sentiment cache — avoids re-running FinBERT on identical headlines
_sentiment_cache: dict[str, tuple[str, float]] = {}


def analyze_sentiment(headline: str) -> tuple[str, float]:
    if headline in _sentiment_cache:
        return _sentiment_cache[headline]

    try:
        result = finbert(headline[:512])[0]   # truncate to model limit
        label  = result["label"].lower()
        score  = result["score"]

        if label == "positive":
            out = ("Positive",  round(score, 2))
        elif label == "negative":
            out = ("Negative", round(-score, 2))
        else:
            out = ("Neutral", 0.0)
    except Exception:
        out = ("Neutral", 0.0)

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
