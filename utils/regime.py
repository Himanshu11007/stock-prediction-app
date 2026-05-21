"""
utils/regime.py — Rule-based market regime detection.

Returns one of: "Bullish", "Bearish", "Sideways", "High Volatility"
"""
import pandas as pd


def detect_regime(data: pd.DataFrame) -> dict:
    """
    Analyse the last row of a feature-engineered DataFrame and classify the
    current market regime.

    Returns:
        {
          "regime":      "Bullish" | "Bearish" | "Sideways" | "High Volatility",
          "regime_score": float   # +1.0 bullish .. -1.0 bearish  (used in scoring)
          "reason":      str
        }
    """
    try:
        row = data.iloc[-1]

        adx       = float(row.get("ADX", 0))
        close     = float(row.get("Close", 0))
        ema20     = float(row.get("EMA_20", close))
        ema50     = float(row.get("EMA_50", close))
        macd_hist = float(row.get("MACD_Hist", 0))
        atr_pct   = float(row.get("ATR_Pct", 0))
        rsi       = float(row.get("RSI", 50))

        # ── High Volatility override ──────────────────────────────────────────
        # ATR as % of price > 3% signals extreme choppiness
        if atr_pct > 0.03:
            return {
                "regime":       "High Volatility",
                "regime_score": 0.0,
                "reason":       f"ATR {atr_pct*100:.1f}% of price — elevated risk",
            }

        # ── Sideways: ADX < 20 means no real trend ────────────────────────────
        if adx < 20:
            return {
                "regime":       "Sideways",
                "regime_score": 0.0,
                "reason":       f"ADX {adx:.1f} — no clear directional trend",
            }

        # ── Trending: use EMA stack + MACD direction ──────────────────────────
        above_ema20 = close > ema20
        above_ema50 = close > ema50
        ema_aligned = ema20 > ema50   # golden stack

        bullish_count = sum([above_ema20, above_ema50, ema_aligned, macd_hist > 0, rsi > 50])
        bearish_count = sum([not above_ema20, not above_ema50, not ema_aligned, macd_hist < 0, rsi < 50])

        if bullish_count >= 4:
            return {
                "regime":       "Bullish",
                "regime_score":  1.0,
                "reason":       f"Price above EMA20/50, ADX {adx:.0f}, MACD positive",
            }

        if bearish_count >= 4:
            return {
                "regime":       "Bearish",
                "regime_score": -1.0,
                "reason":       f"Price below EMA20/50, ADX {adx:.0f}, MACD negative",
            }

        # Mixed — call it sideways with slight tilt
        score = round((bullish_count - bearish_count) / 5, 2)
        return {
            "regime":       "Sideways",
            "regime_score": score,
            "reason":       f"Mixed signals — ADX {adx:.0f}, {bullish_count} bullish vs {bearish_count} bearish factors",
        }

    except Exception:
        return {"regime": "Unknown", "regime_score": 0.0, "reason": "Regime calculation failed"}
