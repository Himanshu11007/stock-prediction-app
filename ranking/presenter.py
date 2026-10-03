"""
ranking/presenter.py — API representations of stored analysis results.

The single place that turns StockAnalysisResult rows into client payloads,
so /top-picks, /stocks/{symbol}/analysis, /fqvf and /ranking stay consistent.
No values are computed here beyond formatting; everything comes from the
stored engine output.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.market import EngineRun, MarketRegimeSnapshot, MarketSnapshot, StockAnalysisResult
from db.models.stock import Company
from config import MARKET_DATA_STALE_DAYS
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


# ── analytical labels, explanation and freshness ─────────────────────────────
# Labels describe the stored component scores; they are analytical
# descriptions, never instructions (no "buy"), and only appear when the data
# supports them.
STRONG, WEAK = 70.0, 30.0
ANALYSIS_STALE_DAYS = 3
SHORT_NAMES = {"quality": "Quality", "valuation": "Valuation", "financial_health": "Financial Health",
               "technical_trend": "Trend", "momentum": "Momentum", "risk": "Risk",
               "sector_outlook": "Sector Outlook", "market_regime": "Price Regime"}
KEY_COMPONENTS = ("quality", "valuation", "financial_health", "momentum", "risk", "market_regime")


def _comp(r: StockAnalysisResult, key: str) -> Optional[float]:
    return ((r.components or {}).get(key) or {}).get("score")


def labels(r: StockAnalysisResult, top_limit: Optional[int], fresh: Optional[dict] = None) -> list[dict]:
    out = []
    if r.eligible and r.rank is not None and top_limit and r.rank <= top_limit:
        out.append({"key": "top_candidate", "label": "Top Candidate", "tone": "positive"})
    for key, text in (("quality", "Strong Quality"), ("valuation", "Attractive Valuation"),
                      ("financial_health", "Strong Financial Health"), ("momentum", "Positive Momentum")):
        v = _comp(r, key)
        if v is not None and v >= STRONG:
            out.append({"key": key, "label": text, "tone": "positive"})
    risk = _comp(r, "risk")
    if risk is not None and risk <= WEAK:
        out.append({"key": "high_risk", "label": "High Risk", "tone": "negative"})
    review = (r.score_coverage is not None and r.score_coverage < 0.8) or \
        ((r.fqvf or {}).get("coverage") or 0) < 0.6 or (fresh or {}).get("status") in ("STALE", "UNAVAILABLE")
    if review:
        out.append({"key": "needs_review", "label": "Needs Review", "tone": "warning"})
    return out


def key_components(r: StockAnalysisResult) -> list[dict]:
    return [{"key": k, "label": SHORT_NAMES[k], "score": _comp(r, k)} for k in KEY_COMPONENTS]


def _age_days(iso: Optional[str], today: date) -> Optional[int]:
    if not iso:
        return None
    try:
        return (today - date.fromisoformat(str(iso)[:10])).days
    except ValueError:
        return None


def freshness_status(r: StockAnalysisResult, now: Optional[datetime] = None) -> dict:
    """OK / STALE / UNAVAILABLE for the market data and the analysis itself,
    with the exact timestamps, so clients never present stale data as current."""
    now = now or datetime.now(timezone.utc)
    f = r.freshness or {}
    market_as_of = f.get("market_data_as_of")
    market_age = _age_days(market_as_of, now.date())
    computed = r.computed_at
    if computed is not None and computed.tzinfo is None:
        computed = computed.replace(tzinfo=timezone.utc)
    analysis_age = (now - computed).days if computed else None
    issues = []
    if market_as_of is None:
        status = "UNAVAILABLE"
        issues.append("Market data unavailable")
    else:
        status = "OK"
        if market_age is not None and market_age > MARKET_DATA_STALE_DAYS:
            status = "STALE"
            issues.append(f"Market data may be stale (last update {market_as_of})")
    if analysis_age is not None and analysis_age > ANALYSIS_STALE_DAYS:
        status = "STALE" if status == "OK" else status
        issues.append(f"Analysis is {analysis_age} days old")
    return {"status": status, "market_data_as_of": market_as_of, "market_data_age_days": market_age,
            "analysis_computed_at": _iso(r.computed_at), "analysis_age_days": analysis_age,
            "fundamentals_fetched_at": f.get("fundamentals_fetched_at"),
            "fiscal_period_end": f.get("fiscal_period_end"), "issues": issues}


def explanation(r: StockAnalysisResult, eligible_total: Optional[int], top_limit: Optional[int]) -> dict:
    """Plain-language "why is this stock ranked here", built only from the
    stored component scores and eligibility."""
    comps = [(k, c) for k, c in (r.components or {}).items()
             if k != "ml_signal" and (c or {}).get("weight") and c.get("score") is not None]
    strengths = sorted((x for x in comps if x[1]["score"] >= STRONG), key=lambda x: -x[1]["score"])
    weaknesses = sorted((x for x in comps if x[1]["score"] <= WEAK), key=lambda x: x[1]["score"])
    missing = [SHORT_NAMES.get(k, k) for k, c in (r.components or {}).items()
               if k != "ml_signal" and (c or {}).get("weight") and (c or {}).get("score") is None]
    if r.eligible and r.rank is not None:
        where = f"Ranked {r.rank}" + (f" of {eligible_total} eligible stocks" if eligible_total else "")
        if top_limit and r.rank <= top_limit:
            where += f", inside the Top {top_limit} Investment Candidates."
        else:
            where += "." + (f" The Top Candidates list shows the top {top_limit}." if top_limit else "")
    else:
        where = "Not ranked: " + ("; ".join(r.ineligible_reasons or []) or "eligibility rules not met") + "."
    lines = [where]
    if r.stockai_score is not None:
        lines.append(f"StockLens Score {r.stockai_score:.1f}/100 is the weighted average of the components below "
                     f"(data coverage {(r.score_coverage or 0):.0%}).")
    if strengths:
        lines.append("Strongest: " + ", ".join(f"{SHORT_NAMES.get(k, k)} {c['score']:.0f}"
                                              for k, c in strengths[:3]) + ".")
    if weaknesses:
        lines.append("Weakest: " + ", ".join(f"{SHORT_NAMES.get(k, k)} {c['score']:.0f}"
                                            for k, c in weaknesses[:3]) + ".")
    if missing:
        lines.append("Not available (excluded, not counted as zero): " + ", ".join(missing) + ".")
    return {"summary": " ".join(lines), "strengths": [SHORT_NAMES.get(k, k) for k, _ in strengths],
            "weaknesses": [SHORT_NAMES.get(k, k) for k, _ in weaknesses], "missing": missing,
            "note": "The StockLens Score is an analytical ranking of stocks against each other on the data "
                    "shown; it is not a forecast or a guarantee of returns."}


def candidate_payload(r: StockAnalysisResult, company: Optional[Company], top_limit: Optional[int] = None) -> dict:
    fresh = freshness_status(r)
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
        "freshness_status": fresh,
        "labels": labels(r, top_limit, fresh),
        "components": key_components(r),
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


def analysis_payload(session: Session, r: StockAnalysisResult, company: Company,
                     top_limit: Optional[int] = None) -> dict:
    run = session.exec(select(EngineRun).where(EngineRun.run_id == r.run_id)).first()
    eligible_total = None
    if r.eligible:
        eligible_total = len(session.exec(select(StockAnalysisResult.id).where(
            StockAnalysisResult.run_id == r.run_id, StockAnalysisResult.eligible == True)).all())  # noqa: E712
    regime = session.exec(select(MarketRegimeSnapshot).where(MarketRegimeSnapshot.run_id == r.run_id)).first()
    if regime is None:
        regime = session.exec(select(MarketRegimeSnapshot).where(MarketRegimeSnapshot.computed_at <= r.computed_at)
                              .order_by(MarketRegimeSnapshot.computed_at.desc())).first()
    fresh = freshness_status(r)
    snap = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == r.symbol,
                                                     MarketSnapshot.fetched_at <= r.computed_at)
                        .order_by(MarketSnapshot.fetched_at.desc())).first()
    ranking = ranking_payload(r)
    expl_source = r
    if run is not None and run.kind == "SINGLE":
        # An on-demand analysis scores this one stock; its "rank" within that
        # run (always 1) is meaningless. Show the position from the latest
        # full ranking run instead, labelled as such.
        full = session.exec(select(StockAnalysisResult, EngineRun)
                            .join(EngineRun, EngineRun.run_id == StockAnalysisResult.run_id)
                            .where(StockAnalysisResult.symbol == r.symbol, EngineRun.kind == "RANKING",
                                   EngineRun.status.in_(("COMPLETED", "COMPLETED_WITH_ERRORS")))
                            .order_by(StockAnalysisResult.computed_at.desc())).first()
        ranking["rank"] = None
        ranking["universe_rank"] = None
        eligible_total = None
        if full is not None:
            fr, frun = full
            total = len(session.exec(select(StockAnalysisResult.id).where(
                StockAnalysisResult.run_id == fr.run_id, StockAnalysisResult.eligible == True)).all())  # noqa: E712
            ranking["universe_rank"] = {"rank": fr.rank, "eligible_total": total, "stockai_score": fr.stockai_score,
                                        "run_id": fr.run_id, "computed_at": _iso(fr.computed_at)}
            ranking["rank"] = fr.rank
            eligible_total = total
        expl = explanation(r, eligible_total, top_limit)
        if ranking["rank"] is not None:
            expl["summary"] = (f"On-demand analysis of this stock. In the last full ranking run "
                               f"({_iso(full[0].computed_at)[:10]}) it ranked {ranking['rank']} of {eligible_total} "
                               f"eligible stocks. " + expl["summary"].split(". ", 1)[-1])
        else:
            expl["summary"] = ("On-demand analysis of this stock only; it has no position in a full ranking run "
                               "yet. " + expl["summary"].split(". ", 1)[-1])
    else:
        expl = explanation(expl_source, eligible_total, top_limit)
    return {
        "symbol": company.symbol, "name": company.name, "exchange": company.exchange,
        "sector": company.sector, "industry": company.industry,
        "data_status": company.data_status, "data_status_reason": company.data_status_reason,
        "ranking": ranking,
        "fqvf": fqvf_payload(r),
        "market": market_payload(snap),
        "ml_signal": r.ml_signal,
        "freshness": r.freshness or {},
        "freshness_status": fresh,
        "labels": labels(r, top_limit, fresh),
        "explanation": expl,
        "market_regime": regime_payload(regime),
        "reference_price": {"close": snap.close if snap else None, "as_of_date": snap.as_of_date if snap else None,
                            "note": "Last daily close used by the analysis (not a live quote)"},
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
