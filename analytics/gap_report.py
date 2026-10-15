"""
analytics/gap_report.py — "Missed gainers": where the ranking falls short.

For each trading day D (same point-in-time pairing as analytics/track_record:
the Top list published before D's 09:15 IST open), take the day's biggest
gainers among the ranked stocks that were NOT in the list and explain why:

  * excluded by the data / liquidity checks (with the stored reason), or
  * ranked too low: the components that cost it the most points compared with
    the average stock in the list (weight x score gap), and its distance to
    the last place in the list;
  * whether StockLens had any news about it before the list was published,
    or only between publication and that day's close.

It also lists "false alarms" (list stocks that fell most that day) and adds
everything up, so recurring gaps (a component, missing news, exclusions) show
up as numbers instead of anecdotes. Read-only; nothing is stored or changed.
Gainers outside the ranked universe cannot be seen here.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from statistics import median
from typing import Any, Optional

from sqlmodel import Session, select

import masters.service as masters
from analytics.track_record import (LOOKBACK_DAYS, MAX_ABS_MOVE, TOP_GAINERS, _next_session, _runs, _utc)
from db.models.market import StockAnalysisResult
from db.models.news import EventEntity
from db.models.prediction import MarketEvent
from db.models.stock import Company
from ranking.service import COMPONENT_LABELS, DEFAULT_WEIGHTS
from utils.market_session import IST, NSE_CLOSE, NSE_OPEN

NEWS_LOOKBACK = dt.timedelta(hours=72)
NEAR_MISS_RANKS = 20          # ranks list_size+1 .. list_size+NEAR_MISS_RANKS count as near misses
MAX_DAYS_SHOWN = 20
SHORT = {"quality": "Business quality", "valuation": "Valuation", "financial_health": "Financial health",
         "sector_outlook": "Sector outlook", "technical_trend": "Trend", "momentum": "Momentum",
         "market_regime": "Stock price regime", "risk": "Risk"}


def _news_counts(session: Session, symbol: str, published: dt.datetime, close: dt.datetime) -> tuple[int, int]:
    """(events about the stock available in the 72 h before the list was
    published, events available between publication and the day's close)."""
    rows = session.exec(select(MarketEvent.effective_available_at).join(
        EventEntity, EventEntity.event_id == MarketEvent.id).where(
        EventEntity.symbol == symbol, EventEntity.relation == "SUBJECT")).all()
    times = [_utc(t) for t in rows if t is not None]
    before = sum(1 for t in times if published - NEWS_LOOKBACK <= t <= published)
    after = sum(1 for t in times if published < t <= close)
    return before, after


def _gaps(comps: dict, top_avg: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for k, w in DEFAULT_WEIGHTS.items():
        if w <= 0 or k not in top_avg:
            continue
        v = (comps or {}).get(k)
        if v is None:
            continue
        pts = (top_avg[k] - v) * w / 100
        if pts > 0:
            out.append({"component": k, "label": SHORT.get(k, COMPONENT_LABELS.get(k, k)), "score": round(v, 1),
                        "list_average": round(top_avg[k], 1), "points": round(pts, 1)})
    return sorted(out, key=lambda g: -g["points"])


def gap_report(session: Session, now: Optional[dt.datetime] = None) -> dict[str, Any]:
    now = now or dt.datetime.now(dt.timezone.utc)
    holidays = masters.get_config(session, "market.holidays") or []
    list_size = int(masters.get_config(session, "top_picks.limit") or 20)
    runs = _runs(session, (now.date() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat())
    names = {c.symbol: c.name for c in session.exec(select(Company)).all()} if runs else {}
    days = []
    for d0 in sorted(runs):
        first = _next_session(dt.date.fromisoformat(d0), holidays)
        opens = dt.datetime.combine(first, NSE_OPEN, tzinfo=IST)
        published = [r for r in runs[d0] if r["published_at"] < opens]
        end_day = first.isoformat()
        if not published or end_day not in runs:
            continue
        start, end_prices = published[-1], runs[end_day][-1]["prices"]
        moves = {s: end_prices[s] / p0 - 1 for s, p0 in start["prices"].items()
                 if end_prices.get(s) and abs(end_prices[s] / p0 - 1) <= MAX_ABS_MOVE}
        in_list = {s: k for s, k in start["ranks"].items() if k <= list_size and s in moves}
        if len(moves) < TOP_GAINERS * 2 or not in_list:
            continue
        rows = start["rows"]
        top_avg = {}
        for k in DEFAULT_WEIGHTS:
            vals = [(rows[s].component_scores or {}).get(k) for s in in_list]
            vals = [v for v in vals if v is not None]
            if vals:
                top_avg[k] = sum(vals) / len(vals)
        cutoff = next((r.stockai_score for s, r in rows.items() if r.eligible and r.rank == list_size), None)
        close = dt.datetime.combine(first, NSE_CLOSE, tzinfo=IST)
        gainers = [s for s in sorted(moves, key=lambda s: (-moves[s], s)) if moves[s] > 0][:TOP_GAINERS]
        misses = []
        for s in gainers:
            if s in in_list:
                continue
            row = rows.get(s)
            before, after = _news_counts(session, s, start["published_at"], close)
            item = {"symbol": s, "name": names.get(s), "return": round(moves[s], 4),
                    "rank": row.rank if row is not None and row.eligible else None,
                    "score": round(row.stockai_score, 1) if row is not None and row.stockai_score is not None else None,
                    "points_below_list": (round(cutoff - row.stockai_score, 1) if cutoff is not None and row is not None
                                          and row.eligible and row.stockai_score is not None else None),
                    "excluded_reason": None, "gaps": [], "main_gap": None,
                    "news_before_list": before, "news_after_list": after}
            if row is None or not row.eligible:
                res = session.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == start["run_id"],
                                                                     StockAnalysisResult.symbol == s)).first()
                item["excluded_reason"] = ((res.ineligible_reasons or [None])[0] if res is not None else None) \
                    or "not eligible in that ranking"
            else:
                item["gaps"] = _gaps(row.component_scores, top_avg)[:3]
                item["main_gap"] = item["gaps"][0]["label"] if item["gaps"] else None
            misses.append(item)
        falls = sorted((s for s in in_list if moves[s] < 0), key=lambda s: moves[s])[:3]
        days.append({"date": end_day, "list_date": d0, "gainers": len(gainers), "hits": len(gainers) - len(misses),
                     "misses": misses,
                     "false_alarms": [{"symbol": s, "name": names.get(s), "return": round(moves[s], 4),
                                       "rank": in_list[s]} for s in falls]})
    days.sort(key=lambda d: d["date"], reverse=True)
    allm = [m for d in days for m in d["misses"]]
    ranked = [m for m in allm if m["rank"] is not None]
    summary = None
    if allm:
        summary = {
            "days": len(days), "gainers": sum(d["gainers"] for d in days), "missed": len(allm),
            "excluded": sum(1 for m in allm if m["excluded_reason"]),
            "excluded_reasons": dict(Counter(m["excluded_reason"] for m in allm if m["excluded_reason"]).most_common(5)),
            "main_gap": dict(Counter(m["main_gap"] for m in ranked if m["main_gap"]).most_common()),
            "near_misses": sum(1 for m in ranked if m["rank"] <= list_size + NEAR_MISS_RANKS),
            "median_rank": median(m["rank"] for m in ranked) if ranked else None,
            "with_news_before_list": sum(1 for m in allm if m["news_before_list"]),
            "news_only_after_list": sum(1 for m in allm if not m["news_before_list"] and m["news_after_list"]),
            "no_news": sum(1 for m in allm if not m["news_before_list"] and not m["news_after_list"]),
        }
    return {
        "title": "Missed gainers: where the ranking falls short",
        "methodology": (f"Each trading day, the {TOP_GAINERS} biggest gainers among the ranked stocks are compared "
                        f"with the Top {list_size} list published before that day's open. For each gainer that was "
                        "not in the list: was it excluded (and why), or which parts of the score held it back "
                        "compared with the average stock in the list, and did StockLens have any news about it."),
        "list_size": list_size, "summary": summary, "days": days[:MAX_DAYS_SHOWN],
        "message": None if days else "Not enough daily rankings yet: the report needs a list published before "
                                     "the open and the next day's ranking.",
        "limitations": ["Only stocks StockLens ranks are checked; gainers outside that list are not visible here",
                        "One day's gainers are noisy; look at the totals over several weeks",
                        "News counts include only the sources StockLens collects (mostly official feeds)",
                        "Read-only research: it never changes the ranking"],
    }
