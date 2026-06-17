from config import (
    MIN_ACCURACY, MIN_CONFIDENCE, MIN_AVG_VOLUME,
    RSI_BUY_MIN, RSI_BUY_MAX,
    VOLATILITY_SPIKE_MULTIPLIER, MIN_CONFLUENCE_SCORE,
)
from utils.logger import get_logger, log_filter_rejection

logger  = get_logger(__name__)
_BULLISH = {"STRONG BUY", "BUY"}
_BEARISH = {"STRONG SELL", "SELL"}


def passes_quality_filters(
    data, signal: str, confidence: float, accuracy: float, score: float = 0.0
) -> bool:
    """Return True only when this stock/signal meets all quality standards."""

    # ── Accuracy: only gate bullish predictions ───────────────────────────────
    if signal in _BULLISH and accuracy < MIN_ACCURACY:
        log_filter_rejection(
            symbol=str(data.index[-1]) if hasattr(data, "index") else "unknown",
            signal=signal, confidence=confidence, accuracy=accuracy, score=score,
            reason=f"accuracy {accuracy:.4f} below MIN_ACCURACY {MIN_ACCURACY}",
        )
        return False

    # ── Low confidence ────────────────────────────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        log_filter_rejection(
            symbol="?", signal=signal, confidence=confidence,
            accuracy=accuracy, score=score,
            reason=f"confidence {confidence:.2f} below MIN_CONFIDENCE {MIN_CONFIDENCE}",
        )
        return False

    # ── Min confluence score for BUY signals ──────────────────────────────────
    if signal in _BULLISH and score < MIN_CONFLUENCE_SCORE:
        log_filter_rejection(
            symbol="?", signal=signal, confidence=confidence,
            accuracy=accuracy, score=score,
            reason=f"score {score:.4f} below MIN_CONFLUENCE_SCORE {MIN_CONFLUENCE_SCORE}",
        )
        return False

    # ── Illiquid stock ────────────────────────────────────────────────────────
    if "Volume" in data.columns:
        avg_vol = float(data["Volume"].tail(20).mean())
        if avg_vol < MIN_AVG_VOLUME:
            log_filter_rejection(
                symbol="?", signal=signal, confidence=confidence,
                accuracy=accuracy, score=score,
                reason=f"avg_volume {avg_vol:.0f} below MIN_AVG_VOLUME {MIN_AVG_VOLUME}",
            )
            return False

    # ── RSI range for bullish signals ─────────────────────────────────────────
    if signal in _BULLISH and "RSI" in data.columns:
        rsi = float(data["RSI"].iloc[-1])
        if not (RSI_BUY_MIN < rsi < RSI_BUY_MAX):
            log_filter_rejection(
                symbol="?", signal=signal, confidence=confidence,
                accuracy=accuracy, score=score,
                reason=f"RSI {rsi:.1f} outside [{RSI_BUY_MIN}, {RSI_BUY_MAX}]",
            )
            return False

    # ── Volatility spike ──────────────────────────────────────────────────────
    if "Volatility" in data.columns:
        recent_vol = float(data["Volatility"].iloc[-1])
        avg_vol_   = float(data["Volatility"].mean())
        if avg_vol_ > 0 and recent_vol > avg_vol_ * VOLATILITY_SPIKE_MULTIPLIER:
            log_filter_rejection(
                symbol="?", signal=signal, confidence=confidence,
                accuracy=accuracy, score=score,
                reason=f"volatility spike {recent_vol:.4f} > {VOLATILITY_SPIKE_MULTIPLIER}× avg {avg_vol_:.4f}",
            )
            return False

    return True