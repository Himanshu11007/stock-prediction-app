from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment of news headline"""
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive",polarity
    elif polarity < 0:
        return "Negative",polarity
    return "Neutral",polarity