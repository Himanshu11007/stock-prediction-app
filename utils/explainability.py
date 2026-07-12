"""
utils/explainability.py — Explainable AI engine for StockAI Pro.

Converts raw model/scoring values into structured, human-readable
explanations for every recommendation. Purely presentational — this
module performs no scoring, makes no predictions, and never influences
the signal/score that decision_engine.generate_signal() produces.

Design note — why pillar scores are recomputed here instead of being
returned by generate_signal():
    generate_signal() in utils/decision_engine.py only returns
    (signal, score, summary, factors) — it does not expose the eight
    individual pillar scores it computes internally. Per an explicit
    product decision, decision_engine.py is kept 100% untouched by this
    task (no signature changes, no risk to the four existing call sites
    in app.py, scanner/engine.py, recommendation_engine.py, and
    api/services.py).

    Instead, this module imports and reuses decision_engine's own private
    pillar functions (_ml_direction_score, _technical_score, _news_score,
    _volume_score, _momentum_score) directly — this is reuse, not
    reimplementation. The exact same code that produced the original
    signal is called a second time, purely to recover the intermediate
    values for display. No scoring formula, weight, or threshold is
    duplicated or reimplemented anywhere in this file.

Constraints honoured:
    - Deterministic, rule-based only — no LLM, no external API calls.
    - Never raises — every public function has a safe fallback path.
    - Never modifies recommendation generation; purely additive.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from config import (
    W_ML_DIR, W_ML_CONF, W_TECH, W_NEWS,
    W_VOLUME, W_REGIME, W_TIMEFRAME, W_MOMENTUM,
    BUY_MIN, STRONG_BUY_MIN, HOLD_MIN, SELL_MIN,
)
from utils.decision_engine import (
    _ml_direction_score,
    _ml_confidence_score,
    _technical_score,
    _news_score,
    _volume_score,
    _momentum_score,
)
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Safe fallback — returned whenever anything in this module fails
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_EXPLANATION: dict = {
    "summary": "Explanation is not available for this recommendation.",
    "signal_explanation": "Explanation is not available for this recommendation.",
    "strengths": [],
    "weaknesses": [],
    "watch_points": [],
    "pillar_breakdown": [],
    "risk_summary": "Risk data unavailable.",
    "confidence_note": "",
    "final_interpretation": (
        "Recommendation generated successfully, but explanation details "
        "could not be prepared."
    ),
}


def _safe_fallback() -> dict:
    """Return a fresh copy of the fallback explanation (avoid shared mutable state)."""
    return {
        "summary": _FALLBACK_EXPLANATION["summary"],
        "signal_explanation": _FALLBACK_EXPLANATION["signal_explanation"],
        "strengths": [],
        "weaknesses": [],
        "watch_points": [],
        "pillar_breakdown": [],
        "risk_summary": _FALLBACK_EXPLANATION["risk_summary"],
        "confidence_note": _FALLBACK_EXPLANATION["confidence_note"],
        "final_interpretation": _FALLBACK_EXPLANATION["final_interpretation"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Pillar classification & text helpers
# ══════════════════════════════════════════════════════════════════════════════

def classify_impact(score: float) -> str:
    """
    Classify a pillar score into positive / neutral / negative.

    Rules (fixed, per spec):
        score >= 0.25  -> "positive"
        score <= -0.25 -> "negative"
        otherwise      -> "neutral"
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "neutral"

    if score >= 0.25:
        return "positive"
    if score <= -0.25:
        return "negative"
    return "neutral"


_PILLAR_TEXT: dict[str, dict[str, str]] = {
    "ML Direction": {
        "positive": "The ML model expects upward movement.",
        "negative": "The ML model expects downside risk.",
        "neutral":  "The ML model does not show a strong directional edge.",
    },
    "ML Confidence": {
        "positive": "The model is confident in its prediction.",
        "negative": "The model's confidence is working against the signal direction.",
        "neutral":  "Model confidence is moderate.",
    },
    "Technical Analysis": {
        "positive": "Price is above key moving averages and momentum indicators are supportive.",
        "negative": "Technical indicators show bearish pressure (RSI, MACD, or moving averages).",
        "neutral":  "Technical indicators are mixed, with no clear directional bias.",
    },
    "News Sentiment": {
        "positive": "Recent news sentiment is supportive.",
        "negative": "Recent news sentiment is negative.",
        "neutral":  "News sentiment is mostly neutral.",
    },
    "Volume": {
        "positive": "Volume confirms the price move with above-average participation.",
        "negative": "Volume is weak, suggesting the move lacks conviction.",
        "neutral":  "Volume is at typical levels, neither confirming nor denying the move.",
    },
    "Market Regime": {
        "positive": "The broader market regime supports bullish trades.",
        "negative": "The broader market regime is unfavorable.",
        "neutral":  "Market regime is mixed or neutral.",
    },
    "Multi-Timeframe": {
        "positive": "Weekly and daily trends are aligned in the same bullish direction.",
        "negative": "Weekly and daily trends are aligned in the same bearish direction.",
        "neutral":  "Weekly and daily trends are mixed or not strongly aligned.",
    },
    "Momentum": {
        "positive": "Momentum is positive.",
        "negative": "Momentum is weak or negative over the recent period.",
        "neutral":  "Momentum is roughly flat.",
    },
}


def pillar_to_text(pillar_name: str, score: float, impact: str) -> str:
    """
    Return a human-readable one-line explanation for a pillar given its
    score and classified impact.

    Falls back to a generic sentence for any pillar name not in the
    lookup table, so this never raises on an unexpected pillar.
    """
    table = _PILLAR_TEXT.get(pillar_name)
    if table is None:
        if impact == "positive":
            return f"{pillar_name} is contributing positively to this recommendation."
        if impact == "negative":
            return f"{pillar_name} is working against this recommendation."
        return f"{pillar_name} is roughly neutral for this recommendation."
    return table.get(impact, table.get("neutral", f"{pillar_name}: no data."))


# ══════════════════════════════════════════════════════════════════════════════
# Pillar score recomputation (reuses decision_engine's own private functions)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pillar_scores(
    prediction: int,
    confidence: float,
    news_score: float,
    timeframe_score: float,
    data: Optional[pd.DataFrame],
    regime_info: Optional[dict],
) -> dict[str, float]:
    """
    Recompute the eight raw pillar scores using decision_engine's own
    pillar functions — the same functions generate_signal() calls
    internally. This guarantees the explanation reflects exactly the
    same numbers that produced the signal, without modifying
    decision_engine.py or duplicating its formulas.

    Returns:
        dict[str, float] keyed by pillar name (matches the names used in
        pillar_to_text / _PILLAR_TEXT), each in [-1.0, +1.0].
    """
    ml_dir = _ml_direction_score(prediction)
    ml_conf = _ml_confidence_score(confidence) * ml_dir

    tech_score = 0.0
    if data is not None and not data.empty:
        tech_score, _ = _technical_score(data)

    news_s, _ = _news_score(news_score)

    vol_s = 0.0
    if data is not None and not data.empty:
        vol_s, _ = _volume_score(data)

    regime_s = 0.0
    if regime_info:
        regime_s = float(regime_info.get("regime_score", 0.0))

    tf_s = max(-1.0, min(1.0, float(timeframe_score or 0.0)))

    momentum_s = 0.0
    if data is not None and not data.empty:
        momentum_s, _ = _momentum_score(data)

    return {
        "ML Direction":       ml_dir,
        "ML Confidence":      ml_conf,
        "Technical Analysis": tech_score,
        "News Sentiment":     news_s,
        "Volume":             vol_s,
        "Market Regime":      regime_s,
        "Multi-Timeframe":    tf_s,
        "Momentum":           momentum_s,
    }


def compute_pillar_scores(
    prediction:      int,
    confidence:      float,
    news_score:      float,
    timeframe_score: float,
    data,
    regime_info:     Optional[dict],
) -> dict[str, float]:
    """
    Public entry point for recomputing the eight raw pillar scores —
    used by recommendation persistence call sites (scanner/engine.py,
    api/services.py, app.py) so a recommendation can be saved with its
    full pillar breakdown for later analysis by the Recommendation
    Intelligence Engine (analytics/recommendation_intelligence.py).

    Thin public wrapper around _compute_pillar_scores() — see that
    function's docstring and the module docstring for why this reuses
    decision_engine's own pillar functions instead of duplicating them.
    """
    return _compute_pillar_scores(
        prediction=prediction,
        confidence=confidence,
        news_score=news_score,
        timeframe_score=timeframe_score,
        data=data,
        regime_info=regime_info,
    )


_PILLAR_WEIGHTS: dict[str, float] = {
    "ML Direction":       W_ML_DIR,
    "ML Confidence":      W_ML_CONF,
    "Technical Analysis": W_TECH,
    "News Sentiment":     W_NEWS,
    "Volume":             W_VOLUME,
    "Market Regime":      W_REGIME,
    "Multi-Timeframe":    W_TIMEFRAME,
    "Momentum":           W_MOMENTUM,
}


def build_pillar_breakdown(pillar_scores: dict[str, float]) -> list[dict]:
    """
    Convert raw pillar scores into the structured breakdown list.

    Each entry:
        {
          "pillar": str,
          "score": float,
          "impact": "positive" | "neutral" | "negative",
          "weight": float,              # from config.py — never hardcoded
          "weighted_contribution": float,
          "explanation": str,
        }
    """
    breakdown: list[dict] = []
    for pillar_name, weight in _PILLAR_WEIGHTS.items():
        score = float(pillar_scores.get(pillar_name, 0.0) or 0.0)
        impact = classify_impact(score)
        breakdown.append({
            "pillar": pillar_name,
            "score": round(score, 4),
            "impact": impact,
            "weight": weight,
            "weighted_contribution": round(score * weight, 4),
            "explanation": pillar_to_text(pillar_name, score, impact),
        })
    return breakdown


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Strengths and weaknesses
# ══════════════════════════════════════════════════════════════════════════════

def _build_strengths(pillar_scores: dict[str, float], confidence: float) -> list[str]:
    strengths: list[str] = []

    tech = pillar_scores.get("Technical Analysis", 0.0)
    if tech > 0.25:
        strengths.append("Technical setup is supportive.")

    momentum = pillar_scores.get("Momentum", 0.0)
    if momentum > 0.25:
        strengths.append("Momentum is positive.")

    news = pillar_scores.get("News Sentiment", 0.0)
    if news > 0.10:
        strengths.append("News sentiment is slightly positive.")

    regime = pillar_scores.get("Market Regime", 0.0)
    if regime > 0.25:
        strengths.append("Market regime supports the recommendation.")

    if confidence is not None and confidence >= 75:
        strengths.append("ML confidence is strong.")

    volume = pillar_scores.get("Volume", 0.0)
    if volume > 0.25:
        strengths.append("Volume confirms the move with strong participation.")

    timeframe = pillar_scores.get("Multi-Timeframe", 0.0)
    if timeframe > 0.25:
        strengths.append("Higher timeframe trend is supportive.")

    return strengths


def _build_weaknesses(pillar_scores: dict[str, float], confidence: float) -> list[str]:
    weaknesses: list[str] = []

    ml_dir = pillar_scores.get("ML Direction", 0.0)
    if ml_dir < -0.25:
        weaknesses.append("ML direction is bearish.")

    volume = pillar_scores.get("Volume", 0.0)
    if volume < -0.25:
        weaknesses.append("Volume confirmation is weak.")

    timeframe = pillar_scores.get("Multi-Timeframe", 0.0)
    if timeframe < -0.25:
        weaknesses.append("Higher timeframe trend is not supportive.")

    news = pillar_scores.get("News Sentiment", 0.0)
    if news < -0.10:
        weaknesses.append("Recent news sentiment is negative.")

    if confidence is not None and confidence < 60:
        weaknesses.append("ML confidence is moderate/weak.")

    tech = pillar_scores.get("Technical Analysis", 0.0)
    if tech < -0.25:
        weaknesses.append("Technical indicators are bearish.")

    regime = pillar_scores.get("Market Regime", 0.0)
    if regime < -0.25:
        weaknesses.append("Market regime is unfavorable.")

    return weaknesses


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Signal-type explanation
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_EXPLANATIONS: dict[str, str] = {
    "STRONG BUY": (
        "STRONG BUY means the recommendation has very strong alignment "
        "across ML, technicals, sentiment, and trend pillars."
    ),
    "BUY": (
        "BUY means multiple pillars are supportive and the confluence "
        "score is above the buy threshold."
    ),
    "HOLD": (
        "HOLD means the stock has mixed signals. Some factors are "
        "positive, but not enough pillars agree to justify a BUY."
    ),
    "SELL": (
        "SELL means downside risk is elevated based on the current "
        "combination of ML, technical, trend, and sentiment factors."
    ),
    "STRONG SELL": (
        "STRONG SELL means multiple pillars show strong bearish alignment."
    ),
}


def explain_signal_type(signal: str) -> str:
    """
    Return the fixed, signal-specific explanation sentence.
    Unknown signals fall back to a generic, safe sentence.
    """
    if not signal:
        return "Signal type is unavailable for this recommendation."
    return _SIGNAL_EXPLANATIONS.get(
        signal.strip().upper(),
        f"'{signal}' is not a recognised signal type — no specific explanation available.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Why HOLD instead of BUY
# ══════════════════════════════════════════════════════════════════════════════

def explain_hold_reason(
    score: float,
    pillar_scores: dict[str, float],
    confidence: float,
) -> list[str]:
    """
    Explain why a recommendation landed on HOLD rather than BUY, using the
    actual pillar values and the real BUY_MIN threshold from config.py.

    Returns a list of reason strings; empty list if nothing notable is
    pulling the score down (shouldn't normally happen for a true HOLD,
    but handled safely).
    """
    reasons: list[str] = []

    try:
        # score is on the 0-1 scale (as returned by generate_signal());
        # BUY_MIN is on the 0-100 scale — compare on a common scale.
        score_100 = float(score) * 100.0 if score is not None else 0.0
        if score_100 < BUY_MIN:
            reasons.append(
                f"Confluence score ({score_100:.0f}/100) is below the BUY "
                f"threshold ({BUY_MIN}/100)."
            )
    except (TypeError, ValueError):
        pass

    ml_dir = pillar_scores.get("ML Direction", 0.0)
    if ml_dir < 0:
        reasons.append("ML direction is bearish.")

    volume = pillar_scores.get("Volume", 0.0)
    if volume < 0.25:
        reasons.append("Volume confirmation is weak.")

    news = pillar_scores.get("News Sentiment", 0.0)
    if -0.25 < news < 0.25:
        reasons.append("News sentiment is neutral.")

    timeframe = pillar_scores.get("Multi-Timeframe", 0.0)
    if -0.25 < timeframe < 0.25:
        reasons.append("Multi-timeframe score is mixed.")

    momentum = pillar_scores.get("Momentum", 0.0)
    if momentum < 0.25:
        reasons.append("Momentum is not strong enough.")

    if confidence is not None and confidence < 70:
        reasons.append(f"ML confidence ({confidence:.0f}%) is not high enough to offset other weak pillars.")

    return reasons


# ══════════════════════════════════════════════════════════════════════════════
# Part 6 — Risk summary
# ══════════════════════════════════════════════════════════════════════════════

def build_risk_summary(risk: Optional[dict]) -> str:
    """
    Build a one-line risk summary from the risk dict produced by
    utils.risk.calculate_risk(). Safe against missing/partial data.
    """
    if not risk:
        return "Risk data is not available for this recommendation."

    close     = risk.get("close")
    stop_loss = risk.get("stop_loss")
    target    = risk.get("target")
    rr_ratio  = risk.get("rr_ratio")

    if close is None or stop_loss is None or target is None:
        return "Risk data is not available for this recommendation."

    rr_text = f"{rr_ratio:.1f}" if isinstance(rr_ratio, (int, float)) else "N/A"
    return (
        f"Risk setup: CMP ₹{close:,.2f}, stop-loss ₹{stop_loss:,.2f}, "
        f"target ₹{target:,.2f}, RR ratio {rr_text}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part 7 — Watch points
# ══════════════════════════════════════════════════════════════════════════════

def _build_watch_points(
    signal: str,
    pillar_scores: dict[str, float],
    weaknesses: list[str],
) -> list[str]:
    watch_points: list[str] = []

    volume = pillar_scores.get("Volume", 0.0)
    if volume < 0.25:
        watch_points.append("Watch whether volume improves.")

    tech = pillar_scores.get("Technical Analysis", 0.0)
    if tech < 0.25:
        watch_points.append("Watch if price stays above EMA20.")

    news = pillar_scores.get("News Sentiment", 0.0)
    if news < 0:
        watch_points.append("Watch for negative news flow.")

    regime = pillar_scores.get("Market Regime", 0.0)
    if regime < 0.25:
        watch_points.append("Watch if market regime turns bearish.")

    if signal and signal.strip().upper() == "HOLD":
        watch_points.append("Watch whether confluence score improves above BUY threshold.")

    signal_upper = (signal or "").strip().upper()
    if signal_upper in ("BUY", "STRONG BUY"):
        watch_points.append("Track target and stop-loss closely.")
    elif signal_upper in ("SELL", "STRONG SELL"):
        watch_points.append("Avoid fresh entry unless signal improves.")
    elif signal_upper == "HOLD":
        watch_points.append("Wait for stronger confirmation before taking action.")

    return watch_points


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_recommendation_explanation(
    symbol:          str,
    stock_name:      str,
    signal:          str,
    score:           float,
    confidence:      float,
    accuracy:        float,
    prediction:      int,
    news_score:      float,
    timeframe_score: float,
    regime_info:     Optional[dict],
    factors:         Optional[list],
    pillar_scores:   Optional[dict] = None,
    risk:            Optional[dict] = None,
    data:            Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build a complete, structured, human-readable explanation for one
    recommendation. Deterministic and rule-based — no LLM, no external
    API calls, no randomness.

    Args:
        symbol, stock_name, signal, score, confidence, accuracy,
        prediction, news_score, timeframe_score: same values used to
            produce the recommendation (see decision_engine.generate_signal).
        regime_info: output of utils.regime.detect_regime(), or None.
        factors: the factor list already returned by generate_signal()
            (kept for completeness / potential future use — not required
            to build the explanation, since pillar_scores covers the
            same ground in structured form).
        pillar_scores: optional pre-computed dict of the 8 raw pillar
            scores (see _compute_pillar_scores). If not provided, this
            function recomputes them internally using `data`,
            `prediction`, `confidence`, `news_score`, `timeframe_score`,
            and `regime_info` — reusing decision_engine's own pillar
            functions (see module docstring).
        risk: output of utils.risk.calculate_risk(), or None.
        data: feature-engineered OHLCV DataFrame — only needed if
            pillar_scores is not already provided.

    Returns:
        dict with keys: summary, signal_explanation, strengths,
        weaknesses, watch_points, pillar_breakdown, risk_summary,
        confidence_note, final_interpretation.

        On any internal failure, returns a safe fallback dict (see
        Part 11 / _safe_fallback()) rather than raising — explanation
        failures must never break recommendation generation.
    """
    try:
        signal_clean = (signal or "HOLD").strip().upper()
        confidence_val = float(confidence) if confidence is not None else 0.0
        score_val = float(score) if score is not None else 0.0

        # ── Resolve pillar scores ──────────────────────────────────────────────
        if pillar_scores is None:
            pillar_scores = _compute_pillar_scores(
                prediction=int(prediction) if prediction is not None else 0,
                confidence=confidence_val,
                news_score=float(news_score) if news_score is not None else 0.0,
                timeframe_score=float(timeframe_score) if timeframe_score is not None else 0.0,
                data=data,
                regime_info=regime_info,
            )
        else:
            # Defensive: ensure every expected key exists, default 0.0
            pillar_scores = {
                name: float(pillar_scores.get(name, 0.0) or 0.0)
                for name in _PILLAR_WEIGHTS
            }

        # ── Pillar breakdown table ───────────────────────────────────────────
        pillar_breakdown = build_pillar_breakdown(pillar_scores)

        # ── Strengths / weaknesses ───────────────────────────────────────────
        strengths  = _build_strengths(pillar_scores, confidence_val)
        weaknesses = _build_weaknesses(pillar_scores, confidence_val)

        # ── Signal explanation ───────────────────────────────────────────────
        signal_explanation = explain_signal_type(signal_clean)

        # ── HOLD-specific reasoning ──────────────────────────────────────────
        if signal_clean == "HOLD":
            hold_reasons = explain_hold_reason(score_val, pillar_scores, confidence_val)
            if hold_reasons:
                weaknesses = weaknesses + [
                    r for r in hold_reasons if r not in weaknesses
                ]

        # ── Watch points ──────────────────────────────────────────────────────
        watch_points = _build_watch_points(signal_clean, pillar_scores, weaknesses)

        # ── Risk summary ─────────────────────────────────────────────────────
        risk_summary = build_risk_summary(risk)

        # ── Confidence note ──────────────────────────────────────────────────
        if confidence_val >= 75:
            confidence_note = f"ML confidence is high ({confidence_val:.0f}%)."
        elif confidence_val >= 60:
            confidence_note = f"ML confidence is moderate ({confidence_val:.0f}%)."
        else:
            confidence_note = f"ML confidence is low ({confidence_val:.0f}%) — treat with caution."

        # ── Summary & final interpretation ───────────────────────────────────
        score_100 = round(score_val * 100)
        summary = (
            f"{stock_name or symbol} — {signal_clean} "
            f"(confluence {score_100}/100, ML confidence {confidence_val:.0f}%)."
        )

        strength_count  = len(strengths)
        weakness_count  = len(weaknesses)
        if signal_clean in ("STRONG BUY", "BUY"):
            final_interpretation = (
                f"{strength_count} supportive factor(s) outweigh "
                f"{weakness_count} concern(s) — confluence is above the BUY threshold."
            )
        elif signal_clean in ("STRONG SELL", "SELL"):
            final_interpretation = (
                f"{weakness_count} bearish factor(s) outweigh "
                f"{strength_count} supportive one(s) — downside risk is elevated."
            )
        elif signal_clean == "HOLD":
            final_interpretation = (
                "Signals are mixed — some pillars are supportive, but not "
                "enough align to justify a directional trade right now."
            )
        else:
            final_interpretation = (
                "Recommendation generated, but the signal type is unrecognised."
            )

        explanation = {
            "summary": summary,
            "signal_explanation": signal_explanation,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "watch_points": watch_points,
            "pillar_breakdown": pillar_breakdown,
            "risk_summary": risk_summary,
            "confidence_note": confidence_note,
            "final_interpretation": final_interpretation,
        }

        logger.info(
            "EXPLANATION_BUILT | %s | %s | %.4f",
            symbol or "?", signal_clean, score_val,
        )
        return explanation

    except Exception as e:
        log_exception(logger, f"EXPLANATION_FAILED | {symbol or '?'}", e)
        logger.error("EXPLANATION_FAILED | %s | %s", symbol or "?", e)
        return _safe_fallback()


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight card summary — for Top Picks cards (no re-analysis required)
# ══════════════════════════════════════════════════════════════════════════════

def build_card_summary(rec: dict) -> dict:
    """
    Build a lightweight explanation for a Top Picks scan result card,
    using only the fields already present in the cached scan output
    (scanner/engine.py's _scan_one() return dict) — no re-fetch, no
    re-training, no re-computation of pillar scores.

    This is intentionally simpler than build_recommendation_explanation():
    the scanner cache does not retain the raw prediction, news_score,
    timeframe_score, or full regime_info dict needed for a true 8-pillar
    breakdown, and re-running the full analysis for every cached card
    (potentially 50+ per category) would be wasteful. Instead this reuses
    the `reason` and `factors` the scanner already computed and persisted,
    formatted for a compact card/expander view.

    Args:
        rec: one row from scanner.cache.load_category_cache(), e.g. with
             keys: symbol, stock, signal, score, confidence, accuracy,
             reason, factors, regime, weekly_trend, daily_trend, close,
             target, stop_loss, rr_ratio.

    Returns:
        dict with keys: signal_explanation, reason, factors, risk_summary,
        watch_points. Never raises — returns a safe minimal dict on failure.
    """
    try:
        signal = (rec.get("signal") or "HOLD").strip().upper()

        risk_summary = build_risk_summary({
            "close":     rec.get("close"),
            "stop_loss": rec.get("stop_loss"),
            "target":    rec.get("target"),
            "rr_ratio":  rec.get("rr_ratio"),
        })

        watch_points: list[str] = []
        if signal in ("BUY", "STRONG BUY"):
            watch_points.append("Track target and stop-loss closely.")
        elif signal in ("SELL", "STRONG SELL"):
            watch_points.append("Avoid fresh entry unless signal improves.")
        elif signal == "HOLD":
            watch_points.append("Wait for stronger confirmation before taking action.")

        return {
            "signal_explanation": explain_signal_type(signal),
            "reason":             rec.get("reason", ""),
            "factors":            rec.get("factors", []) or [],
            "risk_summary":       risk_summary,
            "watch_points":       watch_points,
        }
    except Exception as e:
        log_exception(logger, f"build_card_summary failed for {rec.get('symbol', '?')}", e)
        return {
            "signal_explanation": "Explanation is not available for this recommendation.",
            "reason": "",
            "factors": [],
            "risk_summary": "Risk data unavailable.",
            "watch_points": [],
        }


def compute_weighted_score(pillar_scores: dict[str, float]) -> float:
    """
    Compute the weighted confluence score [-1, +1] from raw pillar scores,
    using the same weights generate_signal() uses internally (imported
    from config.py — never hardcoded).

    This is the same value decision_engine.generate_signal() computes as
    its internal `weighted` variable before mapping to the final 0-100
    score, recovered here for persistence/analysis purposes only.
    """
    total = 0.0
    for pillar_name, weight in _PILLAR_WEIGHTS.items():
        total += float(pillar_scores.get(pillar_name, 0.0) or 0.0) * weight
    return round(total, 4)