"""
analytics/track_record.py — "Were the top gainers already in StockLens's list?"

For each trading session D, take the Top Candidates list StockLens published
the evening before (the latest Ranking Engine v1.0 run whose reference close
is the previous session and which was published before the market opened on
D, 09:15 IST), and the day's TOP_GAINERS biggest gainers in the ranked
universe (stocks whose price rose). Report how many gainers were in the list, next to what picking the
same number of stocks at random would give on average:

    expected by chance = TOP_GAINERS x list size / universe size

The same is done over the next WEEK_SESSIONS sessions ("next week").

Point in time only: a list published after the open is not counted for that
day, so the record can never use hindsight. Returns are close-to-close
prices from the stored ranking snapshots (no dividends or costs); a move
larger than MAX_ABS_MOVE is treated as a likely split / bonus adjustment and
excluded rather than counted as a gain. Read-only: nothing is stored and
the ranking is unchanged.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Optional

from sqlmodel import Session, select

import masters.service as masters
from config import RANKING_ENGINE_VERSION
from db.models.market import EngineRun, RankingSnapshot
from db.models.stock import Company
from utils.market_session import IST, NSE_OPEN, is_trading_day

TOP_GAINERS = 10
WEEK_SESSIONS = 5
LOOKBACK_DAYS = 60
MAX_DAYS_SHOWN = 20
MAX_ABS_MOVE = 0.35

NOTE = ("This shows how past lists turned out. It is not a prediction: a stock being in the list does not mean "
        "it will rise, and past results do not indicate future returns.")


def _utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _next_session(day: dt.date, holidays: list[str], n: int = 1) -> dt.date:
    while n:
        day += dt.timedelta(days=1)
        if is_trading_day(day, holidays):
            n -= 1
    return day


def _runs(session: Session, since: str) -> dict[str, list[dict[str, Any]]]:
    """reference_date -> completed runs (published_at = run finished, prices,
    list ranks), oldest first."""
    rows = session.exec(select(RankingSnapshot).where(RankingSnapshot.engine_version == RANKING_ENGINE_VERSION,
                                                      RankingSnapshot.reference_date >= since)).all()
    by_run: dict[str, dict[str, Any]] = {}
    for r in rows:
        run = by_run.setdefault(r.run_id, {"run_id": r.run_id, "dates": defaultdict(int), "published_at": None,
                                           "prices": {}, "ranks": {}})
        if r.reference_date:
            run["dates"][r.reference_date] += 1
        if r.reference_price and r.reference_price > 0:
            run["prices"][r.symbol] = r.reference_price
        if r.eligible and r.rank is not None:
            run["ranks"][r.symbol] = r.rank
    finished = {e.run_id: e.finished_at for e in session.exec(
        select(EngineRun).where(EngineRun.run_id.in_(list(by_run)), EngineRun.status == "COMPLETED")).all()
        if e.finished_at is not None} if by_run else {}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in by_run.values():
        if run["run_id"] in finished:
            run["published_at"] = _utc(finished[run["run_id"]])
        if run["dates"] and run["published_at"] is not None:
            out[max(run["dates"], key=run["dates"].get)].append(run)        # the run's main reference close
    for runs in out.values():
        runs.sort(key=lambda x: x["published_at"])
    return out


def _evaluate(start_run: dict, end_prices: dict[str, float], list_size: int,
              names: dict[str, str]) -> Optional[dict[str, Any]]:
    moves = {}
    for sym, p0 in start_run["prices"].items():
        p1 = end_prices.get(sym)
        if p1:
            r = p1 / p0 - 1
            if abs(r) <= MAX_ABS_MOVE:
                moves[sym] = r
    in_list = {s: k for s, k in start_run["ranks"].items() if k <= list_size and s in moves}
    if len(moves) < TOP_GAINERS * 2 or not in_list:
        return None
    gainers = [s for s in sorted(moves, key=lambda s: (-moves[s], s)) if moves[s] > 0][:TOP_GAINERS]
    lst = list(in_list)
    return {
        "universe": len(moves), "list_count": len(lst), "gainers": len(gainers),
        "hits": sum(1 for s in gainers if s in in_list),
        "expected_by_chance": round(len(gainers) * len(lst) / len(moves), 2),
        "list_avg_return": round(sum(moves[s] for s in lst) / len(lst), 4),
        "universe_avg_return": round(sum(moves.values()) / len(moves), 4),
        "top_gainers": [{"symbol": s, "name": names.get(s), "return": round(moves[s], 4),
                         "in_list": s in in_list, "list_rank": in_list.get(s)} for s in gainers],
    }


def _horizon(runs: dict, holidays: list[str], list_size: int, names: dict, sessions: int,
             key: str, label: str) -> dict[str, Any]:
    days = []
    for d0 in sorted(runs):
        start_day = dt.date.fromisoformat(d0)
        first = _next_session(start_day, holidays)
        opens = dt.datetime.combine(first, NSE_OPEN, tzinfo=IST)
        published = [r for r in runs[d0] if r["published_at"] < opens]
        end_day = _next_session(start_day, holidays, sessions).isoformat()
        if not published or end_day not in runs:
            continue
        start = published[-1]
        res = _evaluate(start, runs[end_day][-1]["prices"], list_size, names)
        if res is not None:
            days.append({"date": first.isoformat(), "end_date": end_day, "list_date": d0,
                         "list_published_at": start["published_at"].isoformat(), **res})
    days.sort(key=lambda x: x["date"], reverse=True)
    n = len(days)
    summary = None
    if n:
        summary = {
            "periods": n, "hits": sum(x["hits"] for x in days), "possible": sum(x["gainers"] for x in days),
            "avg_hits": round(sum(x["hits"] for x in days) / n, 2),
            "avg_expected_by_chance": round(sum(x["expected_by_chance"] for x in days) / n, 2),
            "list_avg_return": round(sum(x["list_avg_return"] for x in days) / n, 4),
            "universe_avg_return": round(sum(x["universe_avg_return"] for x in days) / n, 4),
            "periods_list_beat_universe": sum(1 for x in days if x["list_avg_return"] > x["universe_avg_return"]),
        }
    return {"key": key, "label": label, "sessions": sessions, "summary": summary, "periods": days[:MAX_DAYS_SHOWN]}


def track_record(session: Session, now: Optional[dt.datetime] = None) -> dict[str, Any]:
    now = now or dt.datetime.now(dt.timezone.utc)
    holidays = masters.get_config(session, "market.holidays") or []
    list_size = int(masters.get_config(session, "top_picks.limit") or 20)
    since = (now.date() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    runs = _runs(session, since)
    names = {c.symbol: c.name for c in session.exec(select(Company)).all()} if runs else {}
    horizons = [_horizon(runs, holidays, list_size, names, 1, "1D", "Next trading day"),
                _horizon(runs, holidays, list_size, names, WEEK_SESSIONS, "1W", "Next 5 trading days")]
    return {
        "title": "Were the top gainers already in StockLens's list?",
        "methodology": (f"For each trading day, the {TOP_GAINERS} biggest gainers among the ranked stocks are "
                        f"compared with the Top {list_size} list StockLens published before that day's market "
                        "open. 'By chance' is what picking the same number of stocks at random would give on "
                        "average."),
        "top_gainers": TOP_GAINERS, "list_size": list_size, "engine_version": RANKING_ENGINE_VERSION,
        "horizons": horizons,
        "message": None if any(h["summary"] for h in horizons) else
        "Not enough daily rankings yet: results appear after two consecutive trading days are ranked.",
        "limitations": ["Close-to-close price change only; no dividends or trading costs",
                        f"Moves larger than {MAX_ABS_MOVE:.0%} are excluded as likely split or bonus adjustments",
                        "A short record says little; results vary a lot from day to day"],
        "note": NOTE,
    }
