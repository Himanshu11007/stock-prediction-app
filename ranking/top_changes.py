"""
ranking/top_changes.py — "What changed since the previous ranking?" for the
Top Candidates page: stocks that entered the list, stocks that left it (and
why), and big rank moves inside it. Read-only; compares two stored full
ranking runs and never recalculates anything.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.market import EngineRun, StockAnalysisResult
from db.models.stock import Company
from notifications.detector import _results, previous_full_run, top_candidates
from ranking import presenter

BIG_MOVE = 3          # places, inside the list


def _date_label(d: Optional[str]) -> str:
    try:
        return dt.date.fromisoformat(d[:10]).strftime("%d %b %Y") if d else "the previous ranking"
    except ValueError:
        return d


def top_changes(session: Session, run: EngineRun, limit: int) -> Optional[dict[str, Any]]:
    prev_run = previous_full_run(session, run)
    if prev_run is None:
        return None
    active = set(session.exec(select(Company.symbol).where(Company.active == True)).all())  # noqa: E712
    cur, prev = _results(session, run.run_id), _results(session, prev_run.run_id)
    cur_top, prev_top = top_candidates(cur, active, limit), top_candidates(prev, active, limit)
    names = {c.symbol: c.name for c in session.exec(
        select(Company).where(Company.symbol.in_(set(cur_top) | set(prev_top)))).all()}

    def entered_reason(sym: str) -> str:
        p: Optional[StockAnalysisResult] = prev.get(sym)
        if p is None:
            return "newly added to the stocks StockLens ranks"
        if not p.eligible:
            return "now meets the data and liquidity checks"
        return f"up from rank {p.rank}" if p.rank else "now ranked"

    def left_reason(sym: str) -> str:
        c: Optional[StockAnalysisResult] = cur.get(sym)
        if c is None:
            return "not in the latest ranking"
        if not c.eligible:
            why = (c.ineligible_reasons or [None])[0]
            return f"no longer eligible{': ' + why if why else ''}"
        return f"now rank {c.rank}" if c.rank else "no longer ranked"

    entered = [{"symbol": s, "name": names.get(s), "rank": cur[s].rank,
                "previous_rank": prev[s].rank if s in prev else None, "reason": entered_reason(s)}
               for s in cur_top if s not in prev_top]
    left = [{"symbol": s, "name": names.get(s), "previous_rank": prev[s].rank,
             "rank": cur[s].rank if s in cur and cur[s].eligible else None, "reason": left_reason(s)}
            for s in prev_top if s not in cur_top]
    moved = sorted(({"symbol": s, "name": names.get(s), "rank": cur[s].rank, "previous_rank": prev[s].rank}
                    for s in cur_top if s in prev_top and abs(cur[s].rank - prev[s].rank) >= BIG_MOVE),
                   key=lambda m: m["rank"])
    since = presenter.ranking_date(session, prev_run)
    parts = [f"{len(entered)} new", f"{len(left)} dropped out"] + ([f"{len(moved)} moved {BIG_MOVE}+ places"]
                                                                   if moved else [])
    summary = (f"No changes to the Top {limit} since {_date_label(since)}." if not (entered or left or moved)
               else f"Since {_date_label(since)}: " + ", ".join(parts) + ".")
    return {"previous_ranking_date": since, "previous_run_id": prev_run.run_id, "limit": limit,
            "entered": entered, "left": left, "moved": moved, "summary": summary}
