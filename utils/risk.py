"""
utils/risk.py — ATR-based risk management.

Provides stop-loss, target, and risk/reward calculations that are
attached to every recommendation so the UI can display a risk panel.
"""
from __future__ import annotations
import pandas as pd


def calculate_risk(data: pd.DataFrame, signal: str) -> dict:
    """
    Compute stop-loss, target, and risk/reward for the latest bar.

    Uses 1.5× ATR for stop distance and a 2:1 R/R minimum for target.

    Args:
        data:   Feature-engineered DataFrame (must contain ATR and Close).
        signal: "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"

    Returns dict with keys:
        close, stop_loss, target, risk_pct, reward_pct, rr_ratio, atr
    """
    try:
        close = float(data["Close"].iloc[-1])
        atr   = float(data["ATR"].iloc[-1])

        is_long  = signal in ("STRONG BUY", "BUY")
        is_short = signal in ("STRONG SELL", "SELL")

        # ATR multiplier: tighter for STRONG signals (more conviction)
        stop_mult   = 1.2 if "STRONG" in signal else 1.5
        target_mult = stop_mult * 2.0   # minimum 2:1 R/R

        if is_long:
            stop_loss = round(close - stop_mult  * atr, 2)
            target    = round(close + target_mult * atr, 2)
        elif is_short:
            stop_loss = round(close + stop_mult  * atr, 2)
            target    = round(close - target_mult * atr, 2)
        else:
            # HOLD — provide context but no directional levels
            stop_loss = round(close - 1.5 * atr, 2)
            target    = round(close + 1.5 * atr, 2)

        risk_pct   = round(abs(close - stop_loss) / close * 100, 2)
        reward_pct = round(abs(target - close)    / close * 100, 2)
        rr_ratio   = round(reward_pct / risk_pct, 2) if risk_pct > 0 else 0.0

        return {
            "close":      round(close, 2),
            "stop_loss":  stop_loss,
            "target":     target,
            "risk_pct":   risk_pct,
            "reward_pct": reward_pct,
            "rr_ratio":   rr_ratio,
            "atr":        round(atr, 2),
        }

    except Exception:
        close = float(data["Close"].iloc[-1]) if not data.empty else 0.0
        return {
            "close": round(close, 2),
            "stop_loss": None, "target": None,
            "risk_pct": None, "reward_pct": None,
            "rr_ratio": None, "atr": None,
        }
