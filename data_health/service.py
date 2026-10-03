"""
data_health/service.py — data-health and API-health reporting for admins.

Reads the stored snapshots only (never fetches, never fills). Every finding
names the stock, the category and the reason.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlmodel import Session, func, select

from config import (APP_ENV, FQVF_ENGINE_VERSION, FUNDAMENTALS_STALE_DAYS, MARKET_DATA_STALE_DAYS,
                    RANKING_ENGINE_VERSION, RECOMMENDATION_ENGINE_VERSION)
from db.models.market import EngineRun, FundamentalSnapshot, MarketSnapshot
from db.models.stock import Company, StockUniverseMember
from engine_runs.service import latest_completed_run, latest_market_regime

CATEGORIES = (
    "missing_market_data", "stale_market_data", "invalid_market_values", "zero_volume",
    "missing_history", "missing_fundamentals", "stale_fundamentals", "partial_fundamentals",
    "invalid_fundamental_values", "provider_failures", "inactive_stocks", "non_tradable_stocks",
)

NEWS_TIMESTAMP_STATUS = {
    "status": "UNAVAILABLE",
    "reason": ("The news source (Google News RSS via news/api.py) is used for live headlines only; "
               "publication timestamps are not captured, so historical news cannot be time-aligned. "
               "News is not an input to FQVF or the StockAI Score."),
}


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    return None if dt is None else (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))


def _latest_by_symbol(session: Session, model) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in session.exec(select(model).order_by(model.fetched_at)).all():
        latest[row.symbol] = row
    return latest


def data_health(session: Session, universe_only: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    stmt = select(Company)
    if universe_only:
        stmt = stmt.where(Company.symbol.in_(select(StockUniverseMember.symbol)))
    companies = list(session.exec(stmt).all())
    markets = _latest_by_symbol(session, MarketSnapshot)
    funds = _latest_by_symbol(session, FundamentalSnapshot)
    findings: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    def add(cat, symbol, reason):
        findings[cat].append({"symbol": symbol, "reason": reason})

    never_checked = 0
    for c in companies:
        if not c.active:
            add("inactive_stocks", c.symbol, "inactive in the Stock Master")
            continue
        if not c.tradable:
            add("non_tradable_stocks", c.symbol, "marked non-tradable")
        m, f = markets.get(c.symbol), funds.get(c.symbol)
        if m is None and f is None:
            never_checked += 1
            continue
        if m is None or m.status in ("UNAVAILABLE",):
            add("missing_market_data", c.symbol, (m.error if m else None) or "no market snapshot")
        elif m.status == "ERROR":
            add("provider_failures", c.symbol, m.error or "market data error")
        else:
            age = (now.date() - date.fromisoformat(m.as_of_date)).days if m.as_of_date else None
            if age is not None and age > MARKET_DATA_STALE_DAYS:
                add("stale_market_data", c.symbol, f"last bar {m.as_of_date} ({age} days old)")
            for issue in m.issues or []:
                if "zero volume" in issue:
                    add("zero_volume", c.symbol, issue)
                elif "gap" in issue:
                    add("missing_history", c.symbol, issue)
                else:
                    add("invalid_market_values", c.symbol, issue)
            if m.bars is not None and m.bars < 250:
                add("missing_history", c.symbol, f"only {m.bars} daily bars in the 2-year window")
        if f is None or f.status == "UNAVAILABLE":
            add("missing_fundamentals", c.symbol, (f.error if f else None) or "no fundamentals snapshot")
        elif f.status == "ERROR":
            add("provider_failures", c.symbol, f.error or "fundamentals error")
        else:
            if f.status == "PARTIAL":
                add("partial_fundamentals", c.symbol, f.error or "partial")
            if now - _aware(f.fetched_at) > timedelta(days=FUNDAMENTALS_STALE_DAYS):
                add("stale_fundamentals", c.symbol, f"fetched {_aware(f.fetched_at):%Y-%m-%d}")
            for issue in (f.data or {}).get("issues", []):
                add("invalid_fundamental_values", c.symbol, issue)

    run = latest_completed_run(session)
    return {
        "generated_at": now.isoformat(),
        "scope": "Large/Mid/Small Cap universe" if universe_only else "all stocks",
        "stocks_in_scope": len(companies),
        "never_checked": never_checked,
        "thresholds": {"market_data_stale_days": MARKET_DATA_STALE_DAYS,
                       "fundamentals_stale_days": FUNDAMENTALS_STALE_DAYS},
        "summary": {k: len(v) for k, v in findings.items()},
        "findings": findings,
        "news_timestamps": NEWS_TIMESTAMP_STATUS,
        "latest_run": {"run_id": run.run_id, "status": run.status, "finished_at": run.finished_at,
                       "errors": len(run.errors or [])} if run else None,
    }


def api_health(session: Session) -> dict:
    checks: dict[str, Any] = {}
    try:
        session.exec(text("SELECT 1")).one()
        checks["database"] = {"status": "OK"}
    except Exception as e:  # pragma: no cover - exercised only on DB failure
        checks["database"] = {"status": "ERROR", "error": type(e).__name__}
    run = latest_completed_run(session)
    running = session.exec(select(func.count()).select_from(EngineRun).where(EngineRun.status == "RUNNING")).one()
    last_failed = session.exec(select(EngineRun).where(EngineRun.status == "FAILED")
                               .order_by(EngineRun.started_at.desc())).first()
    if run is None:
        checks["analysis_engine"] = {"status": "NO_DATA", "detail": "no completed ranking run yet"}
    else:
        age_h = (datetime.now(timezone.utc) - _aware(run.finished_at)).total_seconds() / 3600
        checks["analysis_engine"] = {
            "status": "STALE" if age_h > 48 else "OK", "latest_run": run.run_id,
            "finished_at": run.finished_at, "age_hours": round(age_h, 1), "run_status": run.status,
            "running_now": running,
            "last_failed_run": last_failed.run_id if last_failed else None}
    regime = latest_market_regime(session)
    checks["market_regime"] = ({"status": regime.status, "as_of_date": regime.as_of_date}
                               if regime else {"status": "NO_DATA"})
    overall = "OK" if all(c.get("status") == "OK" for c in checks.values()) else "DEGRADED"
    return {"status": overall, "environment": APP_ENV, "checks": checks, "versions": engine_versions()}


def engine_versions() -> dict:
    from models.trainer import ENSEMBLE_WEIGHTS
    return {
        "api": "1.0.0",
        "ranking_engine": RANKING_ENGINE_VERSION,
        "fqvf": FQVF_ENGINE_VERSION,
        "recommendation_engine": RECOMMENDATION_ENGINE_VERSION,
        "ml_model": {"type": "LR/RF/XGBoost blend", "weights": ENSEMBLE_WEIGHTS,
                     "status": "informational (no demonstrated predictive discrimination)"},
    }
