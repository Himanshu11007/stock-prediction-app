from config import (
    MIN_ACCURACY,
    MIN_CONFIDENCE,
    MIN_AVG_VOLUME,
    RSI_BUY_MIN,
    RSI_BUY_MAX,
    VOLATILITY_SPIKE_MULTIPLIER,
)


def passes_quality_filters(data, signal: str, confidence: float, accuracy: float) -> bool:
    """Return True only when this stock/signal meets all quality standards."""

    # Weak model — skip
    if accuracy < MIN_ACCURACY:
        return False

    # Low confidence — skip
    if confidence < MIN_CONFIDENCE:
        return False

    # Illiquid stock: 20-day avg volume must exceed threshold
    if "Volume" in data.columns:
        avg_vol = float(data["Volume"].tail(20).mean())
        if avg_vol < MIN_AVG_VOLUME:
            return False

    # RSI check for BUY: avoid overbought (>RSI_BUY_MAX) and deeply oversold (<RSI_BUY_MIN)
    if signal == "BUY" and "RSI" in data.columns:
        rsi = float(data["RSI"].iloc[-1])
        if not (RSI_BUY_MIN < rsi < RSI_BUY_MAX):
            return False

    # Volatility spike: recent volatility must not be an extreme outlier
    if "Volatility" in data.columns:
        recent_vol = float(data["Volatility"].iloc[-1])
        avg_vol_   = float(data["Volatility"].mean())
        if avg_vol_ > 0 and recent_vol > avg_vol_ * VOLATILITY_SPIKE_MULTIPLIER:
            return False

    return True
