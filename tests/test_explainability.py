"""
tests/test_explainability.py — Unit tests for utils/explainability.py

These tests exercise the real explainability engine against the real
config.py weights and decision_engine.py pillar functions — no mocking
of the logic under test.
"""
import json

import pandas as pd
import pytest

from config import (
    W_ML_DIR, W_ML_CONF, W_TECH, W_NEWS,
    W_VOLUME, W_REGIME, W_TIMEFRAME, W_MOMENTUM, BUY_MIN,
)
from utils.decision_engine import generate_signal
from utils.explainability import (
    build_recommendation_explanation,
    build_pillar_breakdown,
    classify_impact,
    explain_signal_type,
    explain_hold_reason,
    build_risk_summary,
)


def _make_data(rsi=55, macd_hist=0.0, macd_cross=0.0, ema_cross=0.0,
               price_vs_ema20=0.0, adx=20, bb_position=0.5) -> pd.DataFrame:
    return pd.DataFrame({
        "RSI": [rsi], "MACD_Hist": [macd_hist], "MACD_Cross": [macd_cross],
        "EMA_Cross": [ema_cross], "Price_vs_EMA20": [price_vs_ema20],
        "ADX": [adx], "BB_Position": [bb_position],
        "Close": [100.0], "Volume": [1_000_000],
        "Price_Change": [0.01], "Vol_Breakout": [0.0],
    })


# ── 1. Positive pillar creates strength ─────────────────────────────────────

def test_positive_pillar_creates_strength():
    pillar_scores = {
        "ML Direction": 0.0, "ML Confidence": 0.0, "Technical Analysis": 0.5,
        "News Sentiment": 0.0, "Volume": 0.0, "Market Regime": 0.0,
        "Multi-Timeframe": 0.0, "Momentum": 0.0,
    }
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.6,
        confidence=65.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores=pillar_scores, risk=None,
    )
    assert any("Technical setup is supportive" in s for s in exp["strengths"])


# ── 2. Negative pillar creates weakness ─────────────────────────────────────

def test_negative_pillar_creates_weakness():
    pillar_scores = {
        "ML Direction": -0.5, "ML Confidence": 0.0, "Technical Analysis": 0.0,
        "News Sentiment": 0.0, "Volume": 0.0, "Market Regime": 0.0,
        "Multi-Timeframe": 0.0, "Momentum": 0.0,
    }
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="SELL", score=0.3,
        confidence=65.0, accuracy=0.5, prediction=0, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores=pillar_scores, risk=None,
    )
    assert any("ML direction is bearish" in w for w in exp["weaknesses"])


# ── 3. HOLD explains why it is not BUY ──────────────────────────────────────

def test_hold_explains_why_not_buy():
    """Score below BUY_MIN must appear in the weaknesses/hold-reason list."""
    pillar_scores = {
        "ML Direction": 0.1, "ML Confidence": 0.0, "Technical Analysis": 0.1,
        "News Sentiment": 0.0, "Volume": 0.0, "Market Regime": 0.0,
        "Multi-Timeframe": 0.0, "Momentum": 0.0,
    }
    score_below_buy = (BUY_MIN - 5) / 100.0   # guaranteed below threshold
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="HOLD", score=score_below_buy,
        confidence=55.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores=pillar_scores, risk=None,
    )
    assert any("below the BUY" in w for w in exp["weaknesses"])
    assert exp["signal_explanation"].startswith("HOLD means")


def test_explain_hold_reason_helper_directly():
    pillar_scores = {
        "ML Direction": -0.1, "Volume": 0.0, "News Sentiment": 0.0,
        "Multi-Timeframe": 0.0, "Momentum": 0.0,
    }
    reasons = explain_hold_reason(score=0.40, pillar_scores=pillar_scores, confidence=55.0)
    assert len(reasons) > 0
    assert any("Confluence score" in r for r in reasons)


# ── 4. BUY includes target/stop-loss watch point ────────────────────────────

def test_buy_includes_target_stop_loss_watch_point():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.65,
        confidence=70.0, accuracy=0.55, prediction=1, news_score=0.2,
        timeframe_score=0.3, regime_info=None, factors=[],
        pillar_scores={"ML Direction": 0.5}, risk=None,
    )
    assert any("target and stop-loss" in wp.lower() for wp in exp["watch_points"])


def test_sell_includes_avoid_entry_watch_point():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="SELL", score=0.3,
        confidence=70.0, accuracy=0.55, prediction=0, news_score=-0.2,
        timeframe_score=-0.3, regime_info=None, factors=[],
        pillar_scores={"ML Direction": -0.5}, risk=None,
    )
    assert any("avoid fresh entry" in wp.lower() for wp in exp["watch_points"])


# ── 5. Missing risk returns safe risk message ───────────────────────────────

def test_missing_risk_returns_safe_message():
    assert build_risk_summary(None) == "Risk data is not available for this recommendation."
    assert build_risk_summary({}) == "Risk data is not available for this recommendation."

    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="HOLD", score=0.5,
        confidence=60.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert exp["risk_summary"] == "Risk data is not available for this recommendation."


def test_partial_risk_data_returns_safe_message():
    """Missing even one of close/stop_loss/target should fall back safely."""
    assert build_risk_summary({"close": 100.0, "stop_loss": None, "target": 110.0}) == (
        "Risk data is not available for this recommendation."
    )


# ── 6. Missing pillar scores does not crash ─────────────────────────────────

def test_missing_pillar_scores_does_not_crash():
    """pillar_scores=None triggers internal recomputation from data — must not crash."""
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.6,
        confidence=70.0, accuracy=0.55, prediction=1, news_score=0.1,
        timeframe_score=0.2, regime_info=None, factors=[],
        pillar_scores=None, risk=None, data=None,
    )
    assert len(exp["pillar_breakdown"]) == 8
    assert exp["summary"] != ""


def test_partial_pillar_scores_fills_missing_with_zero():
    """A pillar_scores dict missing some keys should not crash build_pillar_breakdown."""
    breakdown = build_pillar_breakdown({"ML Direction": 0.5})  # only one key provided
    assert len(breakdown) == 8
    names = {row["pillar"] for row in breakdown}
    assert "Technical Analysis" in names
    tech_row = next(r for r in breakdown if r["pillar"] == "Technical Analysis")
    assert tech_row["score"] == 0.0


# ── 7. Unknown signal does not crash ────────────────────────────────────────

def test_unknown_signal_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="MAYBE", score=0.5,
        confidence=60.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert "not a recognised signal" in exp["signal_explanation"] or exp["signal_explanation"]


def test_explain_signal_type_unknown():
    text = explain_signal_type("NOT_A_REAL_SIGNAL")
    assert "not a recognised signal type" in text


def test_explain_signal_type_empty_string():
    text = explain_signal_type("")
    assert text == "Signal type is unavailable for this recommendation."


# ── 8. Pillar contribution uses config weights ──────────────────────────────

def test_pillar_contribution_uses_config_weights():
    pillar_scores = {
        "ML Direction": 0.5, "ML Confidence": 0.3, "Technical Analysis": 0.4,
        "News Sentiment": 0.2, "Volume": 0.1, "Market Regime": -0.1,
        "Multi-Timeframe": 0.0, "Momentum": 0.0,
    }
    breakdown = build_pillar_breakdown(pillar_scores)

    weight_lookup = {
        "ML Direction": W_ML_DIR, "ML Confidence": W_ML_CONF,
        "Technical Analysis": W_TECH, "News Sentiment": W_NEWS,
        "Volume": W_VOLUME, "Market Regime": W_REGIME,
        "Multi-Timeframe": W_TIMEFRAME, "Momentum": W_MOMENTUM,
    }
    for row in breakdown:
        expected_weight = weight_lookup[row["pillar"]]
        assert row["weight"] == expected_weight, (
            f"{row['pillar']}: weight {row['weight']} != config weight {expected_weight}"
        )
        expected_contribution = round(pillar_scores[row["pillar"]] * expected_weight, 4)
        assert row["weighted_contribution"] == expected_contribution


def test_pillar_breakdown_sums_to_actual_confluence_score():
    """
    The most important correctness check: sum(weighted_contribution) across
    all 8 pillars must equal the actual `weighted` value generate_signal()
    used internally to produce the score — proving the recomputed pillar
    scores genuinely match what produced the real signal, not an
    approximation.
    """
    data = _make_data(rsi=65, macd_hist=0.5, macd_cross=1.0,
                       ema_cross=1.0, price_vs_ema20=0.02, adx=35, bb_position=0.3)
    regime_info = {"regime": "Bullish", "regime_score": 0.8, "reason": "uptrend"}

    signal, score, reason, factors = generate_signal(
        prediction=1, confidence=82.0, news_score=0.4,
        timeframe_score=0.6, data=data, regime_info=regime_info,
    )

    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal=signal, score=score,
        confidence=82.0, accuracy=0.55, prediction=1, news_score=0.4,
        timeframe_score=0.6, regime_info=regime_info, factors=factors,
        pillar_scores=None, risk=None, data=data,
    )

    total_contribution = sum(row["weighted_contribution"] for row in exp["pillar_breakdown"])
    implied_score_100 = (total_contribution + 1.0) * 50.0
    actual_score_100 = score * 100.0

    assert abs(implied_score_100 - actual_score_100) < 0.5, (
        f"Pillar contributions imply score {implied_score_100:.2f}, "
        f"but actual confluence score is {actual_score_100:.2f}"
    )


# ── 9. Explanation output contains all required keys ───────────────────────

def test_explanation_output_contains_all_required_keys():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.6,
        confidence=70.0, accuracy=0.55, prediction=1, news_score=0.1,
        timeframe_score=0.2, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    required_keys = {
        "summary", "signal_explanation", "strengths", "weaknesses",
        "watch_points", "pillar_breakdown", "risk_summary",
        "confidence_note", "final_interpretation",
    }
    assert required_keys.issubset(exp.keys())


def test_classify_impact_boundaries():
    assert classify_impact(0.25) == "positive"
    assert classify_impact(0.2499) == "neutral"
    assert classify_impact(-0.25) == "negative"
    assert classify_impact(-0.2499) == "neutral"
    assert classify_impact(0.0) == "neutral"
    assert classify_impact(None) == "neutral"   # must not raise


# ── 10. API response can serialize explanation ──────────────────────────────

def test_explanation_is_json_serializable():
    """The full explanation dict must round-trip cleanly through json.dumps,
    since it gets returned inside a FastAPI JSON response."""
    exp = build_recommendation_explanation(
        symbol="RELIANCE.NS", stock_name="Reliance Industries",
        signal="STRONG BUY", score=0.78, confidence=88.0, accuracy=0.6,
        prediction=1, news_score=0.5, timeframe_score=0.7,
        regime_info={"regime": "Bullish", "regime_score": 1.0, "reason": "strong trend"},
        factors=["factor 1", "factor 2"],
        pillar_scores={
            "ML Direction": 0.7, "ML Confidence": 0.5, "Technical Analysis": 0.6,
            "News Sentiment": 0.5, "Volume": 0.3, "Market Regime": 1.0,
            "Multi-Timeframe": 0.7, "Momentum": 0.4,
        },
        risk={"close": 2850.5, "stop_loss": 2710.0, "target": 3120.0, "rr_ratio": 1.9},
    )
    serialized = json.dumps(exp)
    deserialized = json.loads(serialized)
    assert deserialized["summary"] == exp["summary"]
    assert len(deserialized["pillar_breakdown"]) == 8


# ── Additional edge cases from Part 11 ──────────────────────────────────────

def test_empty_factors_list_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="HOLD", score=0.45,
        confidence=55.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert exp is not None


def test_none_score_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="HOLD", score=None,
        confidence=60.0, accuracy=0.5, prediction=1, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert exp["summary"] != ""


def test_negative_score_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="STRONG SELL", score=-0.1,
        confidence=80.0, accuracy=0.6, prediction=0, news_score=-0.5,
        timeframe_score=-0.6, regime_info=None, factors=[],
        pillar_scores={"ML Direction": -0.7}, risk=None,
    )
    assert exp is not None


def test_very_high_confidence_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="STRONG BUY", score=0.9,
        confidence=100.0, accuracy=0.7, prediction=1, news_score=0.8,
        timeframe_score=0.9, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert "high" in exp["confidence_note"].lower()


def test_very_low_confidence_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="HOLD", score=0.4,
        confidence=0.0, accuracy=0.3, prediction=0, news_score=0.0,
        timeframe_score=0.0, regime_info=None, factors=[],
        pillar_scores={}, risk=None,
    )
    assert "low" in exp["confidence_note"].lower()


def test_missing_regime_info_does_not_crash():
    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.6,
        confidence=70.0, accuracy=0.55, prediction=1, news_score=0.1,
        timeframe_score=0.2, regime_info=None, factors=[],
        pillar_scores=None, data=None, risk=None,
    )
    assert exp is not None


def test_genuine_internal_failure_returns_safe_fallback():
    """Force a real exception inside the function and confirm the fallback fires."""
    class Explodes:
        def get(self, *a, **k):
            raise RuntimeError("simulated failure")

    exp = build_recommendation_explanation(
        symbol="X.NS", stock_name="X Ltd", signal="BUY", score=0.6,
        confidence=70.0, accuracy=0.55, prediction=1, news_score=0.1,
        timeframe_score=0.2, regime_info=None, factors=[],
        pillar_scores=Explodes(), risk=None,
    )
    assert exp["summary"] == "Explanation is not available for this recommendation."
    assert exp["pillar_breakdown"] == []