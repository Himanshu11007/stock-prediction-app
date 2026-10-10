"""
prediction_v2/performance.py — aggregate evaluated outcomes.

Per (setup, horizon): sample size, hit rate with a 95% Wilson interval,
mean cost-adjusted return with a normal-approximation 95% interval, mean
excess return vs NIFTY, mean MFE/MAE, and coverage (directional calls /
all predictions). No probability or "accuracy" is reported for samples
below MIN_REPORTABLE; promotion requires PROMOTION_MIN_EVENTS holdout events
(provisional gate, reviewed by a person - never automatic).
"""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Iterable, Optional

from sqlmodel import Session, select

from db.models.prediction import Prediction, PredictionOutcome, PredictionRun

MIN_REPORTABLE = 20
PROMOTION_MIN_EVENTS = 200
Z = 1.96


def wilson(hits: int, n: int) -> Optional[tuple[float, float]]:
    if n == 0:
        return None
    p = hits / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs: list[float]) -> Optional[tuple[float, float, float]]:
    if not xs:
        return None
    m = mean(xs)
    if len(xs) < 2:
        return (m, m, m)
    half = Z * stdev(xs) / math.sqrt(len(xs))
    return (m, m - half, m + half)


def _mean(xs: list) -> Optional[float]:
    return mean(xs) if xs else None


def summarize(rows: Iterable[tuple[Prediction, PredictionOutcome]]) -> dict[str, Any]:
    rows = list(rows)
    directional = [(p, o) for p, o in rows if o.outcome_status == "EVALUATED"]
    n = len(directional)
    hits = sum(1 for _, o in directional if o.hit)
    car = [o.cost_adjusted_return for _, o in directional if o.cost_adjusted_return is not None]
    exc = [(1 if p.direction == "UP" else -1) * o.excess_return_nifty for p, o in directional
           if o.excess_return_nifty is not None]
    ci_ret = mean_ci(car)
    out: dict[str, Any] = {
        "predictions": len(rows), "directional_calls": n,
        "coverage": (n / len(rows)) if rows else None,
        "no_call_rate": (sum(1 for p, _ in rows if p.direction == "NO_CALL") / len(rows)) if rows else None,
        "hits": hits,
        "hit_rate": (hits / n) if n >= MIN_REPORTABLE else None,
        "hit_rate_ci95": wilson(hits, n) if n >= MIN_REPORTABLE else None,
        "mean_cost_adjusted_return": ci_ret[0] if ci_ret and n >= MIN_REPORTABLE else None,
        "cost_adjusted_return_ci95": ci_ret[1:] if ci_ret and n >= MIN_REPORTABLE else None,
        "mean_directional_excess_vs_nifty": mean(exc) if exc and n >= MIN_REPORTABLE else None,
        "mean_mfe": _mean([o.mfe for _, o in directional if o.mfe is not None]),
        "mean_mae": _mean([o.mae for _, o in directional if o.mae is not None]),
        "sample_note": (None if n >= MIN_REPORTABLE else
                        f"only {n} evaluated directional calls; statistics are shown from {MIN_REPORTABLE}"),
        "promotion_gate_met": bool(n >= PROMOTION_MIN_EVENTS and ci_ret and ci_ret[1] > 0),
    }
    return out


def performance(session: Session, engine_version: Optional[str] = None, horizon: Optional[int] = None,
                run_type: Optional[str] = None) -> dict[str, Any]:
    stmt = (select(Prediction, PredictionOutcome, PredictionRun)
            .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.prediction_id)
            .join(PredictionRun, PredictionRun.run_id == Prediction.run_id))
    if engine_version:
        stmt = stmt.where(PredictionRun.engine_version == engine_version)
    if horizon:
        stmt = stmt.where(PredictionOutcome.horizon_sessions == horizon)
    if run_type:
        stmt = stmt.where(PredictionRun.run_type == run_type)
    rows = session.exec(stmt).all()
    groups: dict[tuple[str, int], list] = {}
    for p, o, _ in rows:
        groups.setdefault((p.setup_type, o.horizon_sessions), []).append((p, o))
    return {
        "overall": {h: summarize([(p, o) for p, o, _ in rows if o.horizon_sessions == h])
                    for h in sorted({o.horizon_sessions for _, o, _ in rows})},
        "by_setup": [{"setup_type": k[0], "horizon_sessions": k[1], **summarize(v)} for k, v in sorted(groups.items())],
        "evaluated_outcomes": len(rows),
        "gates": {"min_reportable_sample": MIN_REPORTABLE, "promotion_min_holdout_events": PROMOTION_MIN_EVENTS,
                  "shadow_weeks_required": "4-8", "note": "Promotion is a human decision after review."},
    }
