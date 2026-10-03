"""
ranking/technical.py — technical, risk and data-quality metrics for one stock
from its daily price history (as of the last available bar).

Reuses the existing production feature code (features.engineer) and
regime/trend logic (utils.regime, get_trend_signal) rather than
re-implementing them. Every metric is None when it cannot be computed;
`issues` lists data problems found in the history.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from evaluation.walk_forward import DAILY_LOOKBACK, WEEKLY_LOOKBACK, session_closes, weekly_bars
from features.engineer import compute_features, get_trend_signal
from fundamentals.provider import clean_number
from utils.regime import detect_regime

TRADING_DAYS = 252


def _ret(closes: pd.Series, n: int) -> Optional[float]:
    if len(closes) <= n:
        return None
    return clean_number(closes.iloc[-1] / closes.iloc[-1 - n] - 1)


def price_issues(df: pd.DataFrame) -> list[str]:
    """Data-health findings in a daily OHLCV frame (not fatal on their own)."""
    issues: list[str] = []
    numeric = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        issues.append("infinite values in price history")
    nan_rows = int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    if nan_rows:
        issues.append(f"{nan_rows} bar(s) with missing OHLC values")
    if (df["Close"] <= 0).any():
        issues.append("non-positive close price(s)")
    sessions = session_closes(df)
    zero_vol = int((df.loc[sessions.index, "Volume"] <= 0).sum())
    if zero_vol:
        issues.append(f"{zero_vol} trading session(s) with zero volume")
    gaps = sessions.index.to_series().diff().dt.days
    long_gaps = int((gaps > 7).sum())
    if long_gaps:
        issues.append(f"{long_gaps} gap(s) of more than 7 days between sessions (missing history)")
    return issues


def technical_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """Metrics as of the last bar of `df` (daily OHLCV, >= 2 years ideal)."""
    as_of = df.index[-1]
    daily = df.loc[df.index > as_of - DAILY_LOOKBACK]
    weekly = weekly_bars(df.loc[df.index > as_of - WEEKLY_LOOKBACK])
    sessions = session_closes(daily)
    rets = sessions.pct_change().dropna()

    out: dict[str, Any] = {
        "as_of_date": as_of.date().isoformat(),
        "close": clean_number(df["Close"].iloc[-1]),
        "volume": clean_number(df["Volume"].iloc[-1]),
        "bars_1y": int(len(sessions)),
        "return_20d": _ret(sessions, 20),
        "return_60d": _ret(sessions, 60),
        "return_250d": _ret(sessions, 250),
        "volatility_annual": clean_number(rets.tail(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS)) if len(rets) > 20 else None,
        "avg_volume_20d": clean_number(daily["Volume"].tail(20).mean()),
        "max_drawdown_1y": None, "atr_pct": None, "rsi": None,
        "trend_daily": None, "trend_weekly": None, "trend_score": None,
        "regime": None, "regime_score": None, "regime_reason": None,
    }
    if len(sessions) > 20:
        peak = sessions.cummax()
        out["max_drawdown_1y"] = clean_number(((sessions - peak) / peak).min())

    feats = compute_features(daily).drop(columns="Up")
    feats = feats[feats.notna().all(axis=1)]
    if not feats.empty:
        out["atr_pct"] = clean_number(feats["ATR_Pct"].iloc[-1])
        out["rsi"] = clean_number(feats["RSI"].iloc[-1])
        regime = detect_regime(feats)
        out["regime"] = regime.get("regime")
        out["regime_score"] = clean_number(regime.get("regime_score"))
        out["regime_reason"] = regime.get("reason")

    if len(daily) >= 60:
        d = get_trend_signal(daily)
        out["trend_daily"] = {"trend": d["trend"], "score": d["score"]}
    if len(weekly) >= 60:
        w = get_trend_signal(weekly)
        out["trend_weekly"] = {"trend": w["trend"], "score": w["score"]}
    scores = [t["score"] for t in (out["trend_daily"], out["trend_weekly"]) if t]
    out["trend_score"] = round(sum(scores) / len(scores), 3) if scores else None
    return out


ML_DISCLAIMER = (
    "Informational only. The direction classifier showed no measurable "
    "out-of-sample discrimination in StockAI's walk-forward research "
    "(docs/ML_EXPERIMENT_REGISTRY.md); it carries no weight in the StockAI "
    "Score unless an administrator assigns one."
)


def ml_signal(df: pd.DataFrame) -> dict[str, Any]:
    """Production ensemble's next-day direction for the last bar (Phase 11A
    inference contract: trained on labelled rows only, predicts bar D)."""
    from models.trainer import ensemble_predict, train_model
    from utils.helpers import prepare_inference_data

    daily = df.loc[df.index > df.index[-1] - DAILY_LOOKBACK]
    inf = prepare_inference_data(daily.copy())
    if inf.X_pred is None:
        return {"available": False, "reason": f"incomplete features: {list(inf.X_pred_invalid)}"}
    if len(set(inf.y_train)) < 2:
        return {"available": False, "reason": "single-class training target"}
    models, _ = train_model(inf.X, inf.y, fast=True)
    pred, confidence, prob = ensemble_predict(models, inf.X_pred)
    return {
        "available": True,
        "direction": "UP" if pred == 1 else "DOWN",
        "probability_up": round(float(prob), 4),
        "confidence": float(confidence),
        "prediction_bar_date": inf.X_pred.index[0].date().isoformat(),
        "disclaimer": ML_DISCLAIMER,
    }
