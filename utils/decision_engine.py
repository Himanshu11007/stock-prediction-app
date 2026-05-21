"""
utils/decision_engine.py — Multi-factor confluence scoring engine.

Produces STRONG BUY / BUY / HOLD / SELL / STRONG SELL based on the
weighted alignment of six independent signal pillars.

Pillar weights (must sum to 1.0):
  ML Direction    35 %   — which way the model predicts
  ML Confidence   20 %   — how certain the model is
  Technical       25 %   — indicator composite (RSI, MACD, EMA, ADX, BB)
  News Sentiment  10 %   — FinBERT polarity
  Volume          5  %   — volume strength vs average
  Market Regime   5  %   — trend regime context

Each pillar returns a score in [-1.0, +1.0].
The weighted sum is mapped to [0, 100], then bucketed into a signal.
"""

from __future__ import annotations
import pandas as pd


# ── Score → Signal buckets ─────────────────────────────────────────────────
STRONG_BUY_MIN  = 72
BUY_MIN         = 58
HOLD_MIN        = 42
SELL_MIN        = 28
# < 28 → STRONG SELL

# ── Pillar weights ─────────────────────────────────────────────────────────
W_ML_DIR    = 0.35
W_ML_CONF   = 0.20
W_TECH      = 0.25
W_NEWS      = 0.10
W_VOLUME    = 0.05
W_REGIME    = 0.05


def _ml_direction_score(prediction: int) -> float:
    """+1 bullish, -1 bearish."""
    return 1.0 if prediction == 1 else -1.0


def _ml_confidence_score(confidence: float) -> float:
    """Map confidence 50–100% → [-1, +1]. Below 50% treated as 50%."""
    c = max(50.0, min(100.0, confidence))
    return (c - 50.0) / 50.0          # 50→0, 75→0.5, 100→1.0


def _technical_score(data: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Composite of RSI, MACD, EMA cross, ADX, Bollinger position.
    Returns (score in [-1,+1], list of human-readable factor strings).
    """
    row     = data.iloc[-1]
    factors = []
    votes   = []

    # RSI
    rsi = float(row.get("RSI", 50))
    if rsi < 35:
        votes.append(-0.8); factors.append(f"RSI {rsi:.0f} — oversold (bearish)")
    elif rsi < 45:
        votes.append(-0.4); factors.append(f"RSI {rsi:.0f} — below neutral")
    elif rsi < 60:
        votes.append( 0.3); factors.append(f"RSI {rsi:.0f} — neutral/bullish zone")
    elif rsi < 70:
        votes.append( 0.7); factors.append(f"RSI {rsi:.0f} — bullish momentum")
    else:
        votes.append(-0.5); factors.append(f"RSI {rsi:.0f} — overbought (caution)")

    # MACD histogram direction + recent crossover
    macd_hist  = float(row.get("MACD_Hist", 0))
    macd_cross = float(row.get("MACD_Cross", 0))
    if macd_cross == 1.0:
        votes.append(1.0); factors.append("MACD bullish crossover (strong signal)")
    elif macd_cross == -1.0:
        votes.append(-1.0); factors.append("MACD bearish crossover (strong signal)")
    elif macd_hist > 0:
        votes.append(0.5); factors.append("MACD histogram positive")
    else:
        votes.append(-0.5); factors.append("MACD histogram negative")

    # EMA stack (EMA20 vs EMA50)
    ema_cross = float(row.get("EMA_Cross", 0))
    price_vs_ema = float(row.get("Price_vs_EMA20", 0))
    if ema_cross == 1.0 and price_vs_ema > 0:
        votes.append(0.8); factors.append("Price above EMA20 & EMA20 > EMA50 (bullish stack)")
    elif ema_cross == -1.0 and price_vs_ema < 0:
        votes.append(-0.8); factors.append("Price below EMA20 & EMA20 < EMA50 (bearish stack)")
    elif ema_cross == 1.0:
        votes.append(0.4); factors.append("EMA20 > EMA50 (trend support)")
    else:
        votes.append(-0.4); factors.append("EMA20 < EMA50 (trend headwind)")

    # ADX trend strength
    adx = float(row.get("ADX", 0))
    if adx >= 30:
        factors.append(f"ADX {adx:.0f} — strong trend")
        # amplify existing direction, don't vote independently
    elif adx < 20:
        factors.append(f"ADX {adx:.0f} — weak trend (choppy)")

    # Bollinger position
    bb_pos = float(row.get("BB_Position", 0.5))
    if bb_pos < 0.2:
        votes.append(0.6); factors.append("Near lower Bollinger Band (mean-reversion buy zone)")
    elif bb_pos > 0.8:
        votes.append(-0.3); factors.append("Near upper Bollinger Band (extended)")
    else:
        votes.append(0.1); factors.append("Within Bollinger Bands (normal range)")

    raw = sum(votes) / len(votes) if votes else 0.0

    # Amplify if ADX confirms strong trend
    if adx >= 30:
        raw = raw * 1.2
    raw = max(-1.0, min(1.0, raw))

    return raw, factors


def _news_score(sentiment_score: float) -> tuple[float, str]:
    """Map FinBERT sentiment [-1,+1] to a labelled factor."""
    if sentiment_score > 0.3:
        return sentiment_score, f"News sentiment bullish ({sentiment_score:+.2f})"
    elif sentiment_score < -0.3:
        return sentiment_score, f"News sentiment bearish ({sentiment_score:+.2f})"
    return sentiment_score, f"News sentiment neutral ({sentiment_score:+.2f})"


def _volume_score(data: pd.DataFrame) -> tuple[float, str]:
    """Volume ratio vs 20-day average → [-1, +1]."""
    try:
        vol_ratio   = float(data.iloc[-1].get("Volume_Ratio", 1.0))
        price_up    = float(data.iloc[-1].get("Price_Change", 0)) > 0
        vol_breakout = float(data.iloc[-1].get("Vol_Breakout", 0)) == 1.0

        if vol_breakout:
            return 1.0, f"Volume breakout ({vol_ratio:.1f}× avg) with price surge"
        elif vol_ratio > 1.5 and price_up:
            return 0.7, f"High volume ({vol_ratio:.1f}× avg) on up-day"
        elif vol_ratio > 1.2:
            return 0.3, f"Above-average volume ({vol_ratio:.1f}× avg)"
        elif vol_ratio < 0.7:
            return -0.2, f"Low volume ({vol_ratio:.1f}× avg) — weak conviction"
        return 0.0, f"Normal volume ({vol_ratio:.1f}× avg)"
    except Exception:
        return 0.0, "Volume data unavailable"


def generate_signal(
    prediction:      int,
    confidence:      float,
    news_score:      float,
    data:            pd.DataFrame | None = None,
    regime_info:     dict | None         = None,
) -> tuple[str, float, str, list[str]]:
    """
    Confluence scoring engine.

    Args:
        prediction:  ML prediction (1 = bullish, 0 = bearish)
        confidence:  Model confidence 0–100
        news_score:  FinBERT sentiment score -1 to +1
        data:        Feature-engineered OHLCV DataFrame (optional but recommended)
        regime_info: Output of utils.regime.detect_regime (optional)

    Returns:
        (signal, score_0_to_1, summary_reason, factor_list)
        - signal: "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
        - score_0_to_1: 0.0 – 1.0 (normalised 100-point scale / 100)
        - summary_reason: one-sentence headline
        - factor_list: ordered list of contributing factor descriptions
    """
    factors: list[str] = []

    # ── Pillar 1: ML direction ────────────────────────────────────────────────
    ml_dir = _ml_direction_score(prediction)
    dir_label = "Bullish" if ml_dir > 0 else "Bearish"
    factors.append(f"ML prediction: {dir_label} ({confidence:.0f}% confidence)")

    # ── Pillar 2: ML confidence ────────────────────────────────────────────────
    ml_conf = _ml_confidence_score(confidence)
    # no separate factor line — already shown in pillar 1

    # ── Pillar 3: Technical indicators ────────────────────────────────────────
    tech_score = 0.0
    if data is not None and not data.empty:
        tech_score, tech_factors = _technical_score(data)
        factors.extend(tech_factors)
    else:
        factors.append("Technical indicators: insufficient data")

    # ── Pillar 4: News sentiment ──────────────────────────────────────────────
    news_s, news_label = _news_score(news_score)
    factors.append(news_label)

    # ── Pillar 5: Volume strength ─────────────────────────────────────────────
    vol_s, vol_label = 0.0, "Volume: no data"
    if data is not None and not data.empty:
        vol_s, vol_label = _volume_score(data)
    factors.append(vol_label)

    # ── Pillar 6: Market regime ───────────────────────────────────────────────
    regime_s = 0.0
    if regime_info:
        regime_s = float(regime_info.get("regime_score", 0.0))
        regime_label = regime_info.get("regime", "Unknown")
        factors.append(f"Market regime: {regime_label} — {regime_info.get('reason','')}")
    else:
        factors.append("Market regime: not analysed")

    # ── Weighted confluence score [-1, +1] ────────────────────────────────────
    weighted = (
        ml_dir   * W_ML_DIR  +
        ml_conf  * W_ML_CONF +
        tech_score * W_TECH  +
        news_s   * W_NEWS    +
        vol_s    * W_VOLUME  +
        regime_s * W_REGIME
    )

    # Map [-1, +1] → [0, 100]
    score_100 = (weighted + 1.0) * 50.0
    score_100 = max(0.0, min(100.0, score_100))

    # ── Signal bucket ──────────────────────────────────────────────────────────
    if score_100 >= STRONG_BUY_MIN:
        signal  = "STRONG BUY"
        summary = f"High-conviction bullish confluence (score {score_100:.0f}/100)"
    elif score_100 >= BUY_MIN:
        signal  = "BUY"
        summary = f"Bullish confluence across multiple factors (score {score_100:.0f}/100)"
    elif score_100 >= HOLD_MIN:
        signal  = "HOLD"
        summary = f"Mixed or insufficient confluence (score {score_100:.0f}/100)"
    elif score_100 >= SELL_MIN:
        signal  = "SELL"
        summary = f"Bearish confluence — caution advised (score {score_100:.0f}/100)"
    else:
        signal  = "STRONG SELL"
        summary = f"High-conviction bearish confluence (score {score_100:.0f}/100)"

    return signal, round(score_100 / 100, 4), summary, factors
