from config import (
    MIN_ACCURACY,
    MIN_CONFIDENCE,
    MIN_AVG_VOLUME,
    RSI_BUY_MIN,
    RSI_BUY_MAX,
    VOLATILITY_SPIKE_MULTIPLIER,
    MIN_CONFLUENCE_SCORE,
)

_BULLISH = {"STRONG BUY", "BUY"}
_BEARISH = {"STRONG SELL", "SELL"}


def passes_quality_filters(
    data,
    signal: str,
    confidence: float,
    accuracy: float,
    score: float = 0.0,
) -> bool:
    """
    Return True only when this stock/signal meets all quality standards.

    Filter logic:
      - Accuracy threshold only applied to BUY/STRONG BUY signals — a weak
        model predicting SELL/HOLD is still useful information and shouldn't
        be silently dropped.
      - Confluence score (MIN_CONFLUENCE_SCORE) gating only for BUY signals.
      - RSI range check only for BUY signals (no point checking RSI on SELLs).
      - Volume and volatility checks apply to all signals.
    """

    # ── Accuracy: only gate bullish predictions ──────────────────────────────
    # Bearish/neutral signals from a weak model are still informative.
    if signal in _BULLISH and accuracy < MIN_ACCURACY:
        return False

    # ── Low model confidence (all signals) ──────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        return False

    # ── Minimum confluence score for BUY signals ─────────────────────────────
    if signal in _BULLISH and score < MIN_CONFLUENCE_SCORE:
        return False

    # ── Illiquid stock (all signals) ─────────────────────────────────────────
    if "Volume" in data.columns:
        avg_vol = float(data["Volume"].tail(20).mean())
        if avg_vol < MIN_AVG_VOLUME:
            return False

    # ── RSI range check: only for bullish signals ────────────────────────────
    if signal in _BULLISH and "RSI" in data.columns:
        rsi = float(data["RSI"].iloc[-1])
        if not (RSI_BUY_MIN < rsi < RSI_BUY_MAX):
            return False

    # ── Volatility spike (all signals) ──────────────────────────────────────
    if "Volatility" in data.columns:
        recent_vol = float(data["Volatility"].iloc[-1])
        avg_vol_   = float(data["Volatility"].mean())
        if avg_vol_ > 0 and recent_vol > avg_vol_ * VOLATILITY_SPIKE_MULTIPLIER:
            return False

    return True