"""
analytics/performance_overview.py — the user-facing Performance page.

Keeps incompatible methodologies apart instead of pooling them into one
number:

  ranking_v1_prospective   realised outcomes of production Ranking Engine
                           v1.0 snapshots (ranking/tracking.py), Top 10 /
                           Top 20 / all eligible vs NIFTY 50 and the eligible
                           universe; empty until horizons have elapsed
  ranking_v1_validation    the frozen point-in-time historical validation
                           (docs/RANKING_VALIDATION_V1.md), quoted as published
  post_fix_signals         legacy BUY/SELL/HOLD signals recorded after the
                           temporal-integrity fix (engine v1.1)
  legacy_signals           legacy signals recorded before that fix (engine
                           NULL / v1.0) - retired methodology, shown for
                           transparency only
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from sqlmodel import Session, func, select

from config import PRE_TEMPORAL_FIX_ENGINE_VERSIONS, RANKING_ENGINE_VERSION, TRACKER_DB
from db.models.market import RankingOutcome, RankingSnapshot
from ranking import tracking

# Published results of the frozen validation (docs/RANKING_VALIDATION_V1.md,
# sections 11-14 and 24). Static: the validation is complete and frozen.
RANKING_VALIDATION = {
    "title": "Ranking Engine v1.0 — historical point-in-time validation",
    "methodology": "Production StockLens Score recomputed at 39 month-ends (Jul 2023 – Sep 2026) using only data "
                   "available at each date; development period Jul 2023 – Dec 2024, final test period "
                   "Jan 2025 – Sep 2026 used once.",
    "benchmark": "NIFTY 50 and the equal-weight eligible universe",
    "horizon": "3 months (primary); 1, 6 and 12 months also reported",
    "observations": {"ranking_dates": 39, "stocks_per_date": "277–282", "eligible_per_date": "144–190"},
    "results": [
        {"metric": "Top 10 vs eligible universe, 3 months (final test period)", "value": "+3.3% (not statistically significant)"},
        {"metric": "Top 10 vs eligible universe, 3 months (development period)", "value": "+0.2%"},
        {"metric": "Rank correlation of score with 3-month return (final / development)", "value": "-0.004 / 0.004"},
    ],
    "conclusion": "The validation did not establish a statistically reliable stock-selection edge. The StockLens "
                  "Score is provided as a transparent analytical ranking; its live results are being tracked "
                  "prospectively below.",
    "limitations": ["About 1.5 years per test period in one market cycle",
                    "Universe uses current index membership (survivorship bias)",
                    "Price returns only; no transaction costs"],
    "document": "docs/RANKING_VALIDATION_V1.md",
}


def _prospective(session: Session) -> dict[str, Any]:
    snaps = session.exec(select(func.count(), func.min(RankingSnapshot.ranked_at), func.max(RankingSnapshot.ranked_at),
                                func.count(func.distinct(RankingSnapshot.run_id)))
                         .where(RankingSnapshot.engine_version == RANKING_ENGINE_VERSION)).one()
    outcomes = session.exec(select(func.count()).select_from(RankingOutcome)).one()
    perf = [p for p in tracking.summarise_outcomes(session) if p["engine_version"] == RANKING_ENGINE_VERSION]
    horizon_order = {"1M": 0, "3M": 1, "6M": 2, "12M": 3}
    perf.sort(key=lambda p: (horizon_order.get(p["horizon"], 9), p["portfolio"]))
    return {
        "title": f"Ranking Engine {RANKING_ENGINE_VERSION} — live (prospective) tracking",
        "methodology": "Every production ranking run is frozen when it completes; realised returns are recorded "
                       "only after each horizon (1, 3, 6, 12 months) has fully elapsed and are never revised.",
        "benchmark": "NIFTY 50 and the run's equal-weight eligible universe",
        "ranking_runs_tracked": int(snaps[3] or 0),
        "stock_snapshots": int(snaps[0] or 0),
        "first_snapshot": snaps[1].isoformat() if snaps[1] else None,
        "latest_snapshot": snaps[2].isoformat() if snaps[2] else None,
        "outcomes_recorded": int(outcomes or 0),
        "performance": perf,
        "status": "REPORTING" if perf else "COLLECTING",
        "message": None if perf else "Performance tracking will appear after enough observations are available.",
        "limitations": ["Each ranking run counts as one observation; early figures rest on few runs",
                        "Outcomes use provider-adjusted closes; no transaction costs"],
    }


def _signal_block(rows: list[sqlite3.Row], title: str, methodology: str, caveat: Optional[str]) -> dict[str, Any]:
    by_signal: dict[str, list] = {}
    for r in rows:
        by_signal.setdefault(r["signal"], []).append(r)
    groups = []
    for sig, rs in sorted(by_signal.items()):
        rets = [x["return_pct"] for x in rs if x["return_pct"] is not None]
        groups.append({"signal": sig, "count": len(rs),
                       "success_rate": round(100 * sum(1 for x in rs if x["success"]) / len(rs), 1),
                       "avg_return_pct": round(sum(rets) / len(rets), 2) if rets else None})
    days = [r["days"] for r in rows if r["days"] is not None]
    return {
        "title": title, "methodology": methodology, "count": len(rows),
        "period_start": min((r["saved_date"] for r in rows), default=None),
        "period_end": max((r["saved_date"] for r in rows), default=None),
        "horizon": (f"Validated after at least 5 trading days; actual holding {int(min(days))}–{int(max(days))} "
                    f"days (average {sum(days) / len(days):.0f})") if days else None,
        "benchmark": None,
        "by_signal": groups,
        "success_definition": "BUY: price rose; SELL: price fell; HOLD: moved 3% or less. Definitions differ "
                              "by signal, so a single pooled success rate is not meaningful and is not shown.",
        "caveat": caveat,
    }


def _legacy() -> tuple[dict, dict]:
    rows: list[sqlite3.Row] = []
    try:
        con = sqlite3.connect(str(TRACKER_DB))
        con.row_factory = sqlite3.Row
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_validation'").fetchone():
            rows = con.execute("""SELECT saved_date, signal, return_pct, success, engine_version,
                                         julianday(validation_date) - julianday(saved_date) AS days
                                  FROM recommendation_validation WHERE is_validated = 1""").fetchall()
        con.close()
    except sqlite3.Error:
        rows = []
    pre = [r for r in rows if r["engine_version"] in PRE_TEMPORAL_FIX_ENGINE_VERSIONS]
    post = [r for r in rows if r["engine_version"] not in PRE_TEMPORAL_FIX_ENGINE_VERSIONS]
    legacy = _signal_block(
        pre, "Legacy signals (retired methodology)",
        "BUY/SELL/HOLD signals from the earlier ML-and-technical engine, recorded before the temporal-integrity fix.",
        "Retired. These signals were generated with a look-ahead defect in the ML direction input "
        "(docs/PRODUCTION_TEMPORAL_INTEGRITY.md) and are not comparable to the current StockLens Score. "
        "Shown for transparency only; they are not evidence of skill.")
    post_fix = _signal_block(
        post, "Legacy signals after the temporal-integrity fix",
        "Same legacy signal engine (v1.1), recorded after the fix.",
        None if post else "No validated signals have been recorded with the corrected engine yet.")
    return legacy, post_fix


def overview(session: Session) -> dict[str, Any]:
    legacy, post_fix = _legacy()
    return {
        "sections": {
            "ranking_v1_prospective": _prospective(session),
            "ranking_v1_validation": RANKING_VALIDATION,
            "post_fix_signals": post_fix,
            "legacy_signals": legacy,
        },
        "order": ["ranking_v1_prospective", "ranking_v1_validation", "post_fix_signals", "legacy_signals"],
        "note": "Results from different methodologies are reported separately and never combined. Past results "
                "do not indicate future returns.",
    }
