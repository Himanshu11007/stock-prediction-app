"""
catalysts/candidates.py — explainable news-driven stock screen at a cutoff
(rule candidates-v0.1, SHADOW / RESEARCH: never a live alert).

For every stock it returns exactly one call:
  UP / DOWN   catalysts.impact found decisive, credible, non-contradicted
              evidence, the news is still fresh for its category's horizon and
              price data exists
  NEUTRAL     news exists but is not decisive (conflicting reports, priced in,
              no credible directional source, below the score threshold, or
              stale): the reasons say which
  NO_CALL     no material news before the cutoff, or no usable price data

Scores are reported separately and are NOT probabilities:
  novelty      NEW 1.0 / FOLLOW_UP 0.7 / REPEAT 0.3 (best piece of evidence)
  credibility  PRIMARY 1.0, TIER1 0.9, TIER2 0.75, TIER3 0.4, UNKNOWN 0.2
  materiality  classification materiality (0..1)
  freshness    1 - age / (category horizon), floored at 0; 0 = stale
  surprise     UNEXPECTED / EXPECTED / UNKNOWN as stated by the source text
  sentiment    tone of the text (-1..1), deliberately distinct from the call:
               positive wording is not a forecast of positive returns
               (PRICED_IN, conflicts and relation weights decide that)

Each row carries its evidence (titles, source links, publication /
availability / ingestion times, how the stock was mapped), affected
sectors, likely horizon, risk factors and, for NEUTRAL / NO_CALL, the
reasons the stock was excluded.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Iterable, Optional

from sqlmodel import Session, select

from catalysts import impact
from db.models.news import EventEntity

CANDIDATES_VERSION = "candidates-v0.1"
CREDIBILITY = {"PRIMARY": 1.0, "TIER1": 0.9, "TIER2": 0.75, "TIER3": 0.4, "UNKNOWN": 0.2}
SESSION_HOURS = 24.0


def _hours(a: dt.datetime, b: dt.datetime) -> float:
    return (a - b).total_seconds() / 3600.0


def screen(session: Session, as_of: dt.datetime, companies: Iterable[tuple[str, Optional[str]]],
           price_move: Optional[Callable[[str], Optional[Callable]]] = None,
           has_prices: Optional[Callable[[str], bool]] = None) -> dict[str, Any]:
    """`companies`: (symbol, sector). `price_move(symbol)` returns the
    impact.assess_stock price-move callable for that stock, or None when the
    priced-in check cannot be performed. `has_prices(symbol)` False -> NO_CALL."""
    as_of = impact._utc(as_of)
    comps = list(companies)
    sector_of = dict(comps)
    rows = []
    for sym, sector in comps:
        pm = price_move(sym) if price_move else None
        a = impact.assess_stock(session, sym, as_of, pm)
        rows.append(_row(session, a, sym, sector, sector_of, as_of, pm is not None,
                         has_prices(sym) if has_prices else True))
    counts = {c: sum(1 for r in rows if r["call"] == c) for c in ("UP", "DOWN", "NEUTRAL", "NO_CALL")}
    order = {"UP": 0, "DOWN": 0, "NEUTRAL": 1, "NO_CALL": 2}
    rows.sort(key=lambda r: (order[r["call"]], -abs(r["score"]), r["symbol"]))
    return {"as_of": as_of.isoformat(), "version": CANDIDATES_VERSION, "mode": "SHADOW", "counts": counts,
            "candidates": rows,
            "note": "evidence strength, not a probability; research / shadow only, never a live alert"}


def _row(session: Session, a: impact.Assessment, sym: str, sector: Optional[str], sector_of: dict, as_of: dt.datetime,
         price_checked: bool, prices_ok: bool) -> dict[str, Any]:
    ev = [e for e in a.evidence if e.weight > 0] or a.evidence
    risks: list[str] = []
    excluded: list[str] = []
    horizon = max((e.impact_sessions or 1 for e in ev), default=None)
    fresh = None
    if ev:
        newest = max(dt.datetime.fromisoformat(e.available_at) for e in ev)
        age = max(0.0, _hours(as_of, newest))
        fresh = round(max(0.0, 1 - age / (SESSION_HOURS * (horizon or 1))), 3)
    flags = {f for e in a.evidence for f in e.flags}
    if "CONFLICTING_REPORTS" in flags or "CONTRADICTED" in flags:
        risks.append("conflicting or contradicted reports")
    if "PRICED_IN" in flags:
        risks.append("price already moved in the news' direction (priced in)")
    if ev and all(e.inferred for e in ev):
        risks.append("only inferred exposure (transmission hypothesis), no direct report")
    if ev and all(e.relation == "MENTIONED" for e in ev):
        risks.append("company only mentioned, not the subject")
    if ev and max(CREDIBILITY.get(e.source_tier, 0.2) for e in ev) < 0.75:
        risks.append("no credible (primary / tier 1-2) source")
    if not price_checked and ev:
        risks.append("priced-in check not performed (no price data supplied)")
    if any(e.expectedness == "EXPECTED" for e in ev):
        risks.append("the source describes the outcome as expected")
    sectors = sorted({s for e in ev for s in _event_sectors(session, e.event_id, sector_of)} | ({sector} if sector else set())) if ev else []

    if not prices_ok:
        call, excluded = "NO_CALL", ["no usable price data for this stock"]
    elif a.status == "NO_NEWS":
        call, excluded = "NO_CALL", a.reasons or ["no material news before the cutoff"]
    elif a.status == "DIRECTIONAL" and fresh == 0.0:
        call, excluded = "NEUTRAL", [f"news is stale for its {horizon}-session horizon"]
    elif a.status == "DIRECTIONAL":
        call = a.direction
    else:
        call, excluded = "NEUTRAL", a.reasons
    d = a.as_dict()
    best = max(ev, key=lambda e: e.weight) if ev else None
    return {"symbol": sym, "sector": sector, "call": call, "score": round(a.score, 3),
            "expected_return_direction": call if call in ("UP", "DOWN") else None,
            "news_sentiment": d["news_sentiment"], "surprise": d["surprise"],
            "scores": {"novelty": impact.NOVELTY.get(best.novelty or "NEW") if best else None,
                       "credibility": CREDIBILITY.get(best.source_tier, 0.2) if best else None,
                       "materiality": best.materiality if best else None, "freshness": fresh},
            "horizon_sessions": horizon, "affected_sectors": sectors, "risks": risks,
            "exclusion_reasons": excluded, "uncertainty": risks + ([] if call in ("UP", "DOWN") else excluded),
            "evidence": [{"event_id": e.event_id, "title": e.title, "category": e.category, "relation": e.relation,
                          "inferred": e.inferred, "sign": e.sign, "weight": e.weight, "flags": e.flags,
                          "mechanism": e.mechanism, "source_tier": e.source_tier, "available_at": e.available_at,
                          "mapping": e.mapping, "articles": e.articles} for e in a.evidence],
            "confidence": None}


def _event_sectors(session: Session, event_id: int, sector_of: dict) -> set[str]:
    """Sectors of every company the event is linked to (subject, mentioned or exposed)."""
    syms = session.exec(select(EventEntity.symbol).where(EventEntity.event_id == event_id,
                                                         EventEntity.symbol.is_not(None))).all()
    return {sector_of[s] for s in syms if sector_of.get(s)}
