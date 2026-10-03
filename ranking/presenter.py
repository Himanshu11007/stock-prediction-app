"""
ranking/presenter.py — API representations of stored analysis results.

The single place that turns StockAnalysisResult rows into client payloads,
so /top-picks, /stocks/{symbol}/analysis, /fqvf and /ranking stay consistent.
No values are computed here beyond formatting; everything comes from the
stored engine output.
"""
from __future__ import annotations

from datetime import timezone
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.market import EngineRun, MarketRegimeSnapshot, MarketSnapshot, StockAnalysisResult
from db.models.stock import Company
from fqvf import CHECKS, THRESHOLDS
from ranking.service import COMPONENT_LABELS, DEFAULT_WEIGHTS

DISCLAIMER_KEY = "app.disclaimer"


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def regime_payload(regime: Optional[MarketRegimeSnapshot]) -> Optional[dict]:
    if regime is None:
        return None
    return {"index": regime.index_symbol, "status": regime.status, "regime": regime.regime,
            "regime_score": regime.regime_score, "reason": regime.reason, "as_of_date": regime.as_of_date,
            "computed_at": _iso(regime.computed_at)}


def ranking_payload(r: StockAnalysisResult) -> dict:
    return {
        "stockai_score": r.stockai_score,
        "score_coverage": r.score_coverage,
        "rank": r.rank,
        "eligible_for_top_picks": r.eligible,
        "ineligible_reasons": r.ineligible_reasons or [],
        "components": [{"key": k, **v} for k, v in (r.components or {}).items()],
        "positives": r.positives or [],
        "risks": r.risks or [],
        "engine_version": r.engine_version,
        "run_id": r.run_id,
        "computed_at": _iso(r.computed_at),
    }


def fqvf_payload(r: StockAnalysisResult) -> dict:
    return {**(r.fqvf or {}), "run_id": r.run_id}


def candidate_payload(r: StockAnalysisResult, company: Optional[Company]) -> dict:
    return {
        "rank": r.rank,
        "symbol": r.symbol,
        "name": company.name if company else r.symbol,
        "sector": company.sector if company else None,
        "industry": company.industry if company else None,
        "stockai_score": r.stockai_score,
        "score_coverage": r.score_coverage,
        "fqvf_score": (r.fqvf or {}).get("score"),
        "fqvf_summary": (r.fqvf or {}).get("summary"),
        "fqvf_counts": (r.fqvf or {}).get("counts"),
        "positives": (r.positives or [])[:5],
        "risks": (r.risks or [])[:5],
        "freshness": r.freshness or {},
        "engine_version": r.engine_version,
        "computed_at": _iso(r.computed_at),
    }


def market_payload(snap: Optional[MarketSnapshot]) -> Optional[dict]:
    if snap is None:
        return None
    t = snap.technical or {}
    return {
        "status": snap.status, "as_of_date": snap.as_of_date, "close": snap.close, "volume": snap.volume,
        "fetched_at": _iso(snap.fetched_at), "issues": snap.issues or [],
        "technical": {k: t.get(k) for k in (
            "return_20d", "return_60d", "return_250d", "volatility_annual", "max_drawdown_1y", "atr_pct",
            "rsi", "avg_volume_20d", "trend_daily", "trend_weekly", "trend_score", "regime", "regime_score",
            "regime_reason")},
    }


def analysis_payload(session: Session, r: StockAnalysisResult, company: Company) -> dict:
    run = session.exec(select(EngineRun).where(EngineRun.run_id == r.run_id)).first()
    snap = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == r.symbol,
                                                     MarketSnapshot.fetched_at <= r.computed_at)
                        .order_by(MarketSnapshot.fetched_at.desc())).first()
    return {
        "symbol": company.symbol, "name": company.name, "exchange": company.exchange,
        "sector": company.sector, "industry": company.industry,
        "data_status": company.data_status, "data_status_reason": company.data_status_reason,
        "ranking": ranking_payload(r),
        "fqvf": fqvf_payload(r),
        "market": market_payload(snap),
        "ml_signal": r.ml_signal,
        "freshness": r.freshness or {},
        "engine": {"ranking": r.engine_version, "fqvf": r.fqvf_version, "run_id": r.run_id,
                   "run_kind": run.kind if run else None},
    }


def fqvf_reference() -> dict:
    return {
        "name": "Fundamental Quality & Value Framework",
        "short_name": "FQVF",
        "checks": [{"id": i, "name": n, "scored": s} for i, n, s in CHECKS],
        "statuses": {
            "PASS": "meets the preferred threshold",
            "WARNING": "tolerable/acceptable band or mixed result (see grade)",
            "FAIL": "does not meet the threshold",
            "NOT_AVAILABLE": "required data is missing; not a failure",
        },
        "thresholds": THRESHOLDS,
        "score": "100 x (PASS + 0.5 x WARNING) / evaluated scored checks; checks 3 and 4 are informational",
        "documentation": "docs/FQVF.md",
    }


def ranking_reference(weights: dict, rules: dict) -> dict:
    return {
        "components": [{"key": k, "label": COMPONENT_LABELS[k], "weight": weights.get(k),
                        "default_weight": DEFAULT_WEIGHTS[k]} for k in DEFAULT_WEIGHTS],
        "rules": rules,
        "documentation": "docs/RANKING_METHODOLOGY.md",
    }


def to_jsonable(obj: Any) -> Any:
    """Convert datetimes inside model_dump() output for JSON responses."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return _iso(obj) if hasattr(obj, "tzinfo") else obj.isoformat()
    return obj
