import streamlit as st
from transformers import pipeline

#Load FinBERT once
@st.cache_resource
def load_finbert():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )
finbert = load_finbert()


def analyze_sentiment(headline):
    try:
        result = finbert(headline)[0]
        label = result["label"].lower()
        score = result["score"]

        # Convert labels
        if label == "positive":
            sentiment = "Positive"
            final_score = score

        elif label == "negative":
            sentiment = "Negative"
            final_score = -score
        else:
            sentiment = "Neutral"
            final_score = 0
        return sentiment, round(final_score, 2)
    except Exception as e:
        return "Neutral", 0
    
def analyze_overall_sentiment(headlines):
    """Calculate overall sentiment from all headlines"""
    scores = []

    for headline in headlines:
        sentiment,score = analyze_sentiment(headline)
        scores.append(score)
    #No headline safety
    if len(scores) == 0:
        return "Neutral",0
    avg_score = sum(scores)/len(scores)

    #Final market mood
    if avg_score > 0.25:
        overrall ="Bullish"
    elif avg_score < -0.25:
        overrall ="Bearish"
    else:
        overrall ="Neutral"
    return overrall,round(avg_score,2)
