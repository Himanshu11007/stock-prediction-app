from config import (
    MIN_ACCURACY,
    MIN_CONFIDENCE,
    MIN_AVG_VOLUME,
    RSI_BUY_MIN,
    RSI_BUY_MAX,
    VOLATILITY_SPIKE_MULTIPLIER,
    MIN_CONFLUENCE_SCORE,
)

# Signals considered bullish for RSI range check
_BULLISH = {"STRONG BUY", "BUY"}
_BEARISH = {"STRONG SELL", "SELL"}


def passes_quality_filters(data, signal: str, confidence: float, accuracy: float,score:float = 0.0) -> bool:
    """Return True only when this stock/signal meets all quality standards."""

    # Weak model
    if accuracy < MIN_ACCURACY:
        return False

    # Low confidence
    if confidence < MIN_CONFIDENCE:
        return False
    
    if signal in _BULLISH and score < MIN_CONFLUENCE_SCORE:
        return False

    # Illiquid stock
    if "Volume" in data.columns:
        avg_vol = float(data["Volume"].tail(20).mean())
        if avg_vol < MIN_AVG_VOLUME:
            return False

    # RSI check for bullish signals: avoid overbought / deeply oversold entries
    if signal in _BULLISH and "RSI" in data.columns:
        rsi = float(data["RSI"].iloc[-1])
        if not (RSI_BUY_MIN < rsi < RSI_BUY_MAX):
            return False

    # Volatility spike filter
    if "Volatility" in data.columns:
        recent_vol = float(data["Volatility"].iloc[-1])
        avg_vol_   = float(data["Volatility"].mean())
        if avg_vol_ > 0 and recent_vol > avg_vol_ * VOLATILITY_SPIKE_MULTIPLIER:
            return False

    return True