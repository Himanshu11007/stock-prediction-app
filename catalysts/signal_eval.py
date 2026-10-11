"""
catalysts/signal_eval.py — does news improve predictions? Chronological,
point-in-time comparison of strategies on the SAME stock-sessions (research
only; Prediction Engine v2 stays in shadow; Ranking Engine v1.0 untouched).

Arms (all evaluated close(t) -> close(t+1), direction-adjusted, after costs):
  v1_rank            UP for Ranking v1.0 top-20 members (month-end snapshot <= t)
  v2                 Prediction Engine v2 market / technical rules
  news_<provider>    catalysts.impact on that provider's archive alone,
                     replayed point in time (catalysts.replay)
  news_combined      every available provider archive together
  v2_plus_news       the news call where one exists, else the v2 call
  baselines          always_up, previous_day, sector, random_matched

Per split (development / validation / holdout, embargoed) and arm: calls,
coverage (calls / evaluated stock-sessions), direction accuracy (hit rate,
Wilson 95%), mean net return (session-bootstrap 95%), excess over NIFTY,
total net, maximum drawdown of the equal-weight daily book, and - for news
arms - results by event category. An arm with fewer than MIN_CALLS scored
calls in a split is INSUFFICIENT_SAMPLE there: its numbers are shown, not
interpreted. Nothing is tuned: rule thresholds are fixed in advance.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, Optional

from prediction_v2 import backtest
from prediction_v2.backtest import Trade

MIN_CALLS = 100


def trades_from_replay(calls: Iterable[Any], strategy: str, split_of: dict[dt.date, str],
                       sectors: dict[str, Optional[str]]) -> list[Trade]:
    out = []
    for c in calls:
        if c.ret is None or c.cutoff not in split_of:
            continue
        out.append(Trade(strategy=strategy, symbol=c.symbol, cutoff=c.cutoff, direction=c.direction, ret=c.ret,
                         split=split_of[c.cutoff], setup=getattr(c, "category", "") or "", excess=c.excess,
                         sector=sectors.get(c.symbol)))
    return out


def overlay(base: list[Trade], news: list[Trade], strategy: str) -> list[Trade]:
    """News call where one exists for the stock-session, else the base call."""
    by = {(t.symbol, t.cutoff): t for t in base}
    for t in news:
        by[(t.symbol, t.cutoff)] = t
    return [Trade(**{**t.__dict__, "strategy": strategy}) for t in by.values()]


def max_drawdown(trades: list[Trade], cost_bps: float) -> dict[str, Any]:
    """Equal-weight book: each session's P&L is the mean net return of that
    session's calls; drawdown is the largest peak-to-trough fall of the
    cumulative sum."""
    by: dict[dt.date, list[float]] = {}
    for t in trades:
        by.setdefault(t.cutoff, []).append(t.ret - cost_bps / 1e4)
    cum = peak = dd = 0.0
    for d in sorted(by):
        cum += sum(by[d]) / len(by[d])
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {"sessions_traded": len(by), "cumulative_net": round(cum, 5), "max_drawdown": round(dd, 5)}


def compare(trades: list[Trade], splits: dict[str, list[dt.date]], evaluated: dict[str, int], cost_bps: float,
            min_calls: int = MIN_CALLS, category_arms: tuple[str, ...] = ()) -> dict[str, Any]:
    arms = sorted({t.strategy for t in trades})
    out: dict[str, Any] = {"cost_bps": cost_bps, "min_calls": min_calls, "splits": {}}
    for split, days in splits.items():
        rows = {}
        for arm in arms:
            tr = [t for t in trades if t.strategy == arm and t.split == split]
            r = backtest.summarize_trades(tr, cost_bps)
            r["coverage"] = round(len(tr) / evaluated[split], 4) if evaluated.get(split) else None
            r.update(max_drawdown(tr, cost_bps))
            r["status"] = "OK" if len(tr) >= min_calls else "INSUFFICIENT_SAMPLE"
            rows[arm] = r
        out["splits"][split] = {"sessions": len(days), "first": days[0].isoformat() if days else None,
                                "last": days[-1].isoformat() if days else None,
                                "evaluated_stock_sessions": evaluated.get(split, 0), "arms": rows}
    out["by_category"] = {}
    for arm in category_arms:
        groups: dict[str, list[Trade]] = {}
        for t in trades:
            if t.strategy == arm:
                groups.setdefault(t.setup or "UNKNOWN", []).append(t)
        out["by_category"][arm] = {k: {**backtest.summarize_trades(v, cost_bps, bootstrap=False),
                                       "status": "OK" if len(v) >= min_calls else "INSUFFICIENT_SAMPLE"}
                                   for k, v in sorted(groups.items())}
    return out


def verdict(cmp: dict[str, Any], arm: str, against: str, split: str = "holdout") -> dict[str, Any]:
    """Does `arm` beat `against` on the holdout? Only claimed when both have
    enough calls and the arm's session-bootstrap interval of the mean net
    return excludes zero and lies above the comparator's mean."""
    rows = cmp["splits"][split]["arms"]
    a, b = rows.get(arm), rows.get(against)
    if not a or not b:
        return {"arm": arm, "against": against, "result": "NOT_AVAILABLE"}
    if a["status"] != "OK" or b["status"] != "OK":
        return {"arm": arm, "against": against, "result": "INSUFFICIENT_SAMPLE", "calls": a["calls"],
                "comparator_calls": b["calls"]}
    ci = a.get("mean_net_return_ci95_clustered") or a.get("mean_net_return_ci95")
    better = bool(ci and ci[0] > 0 and a["mean_net_return"] is not None and b["mean_net_return"] is not None
                  and ci[0] > b["mean_net_return"])
    return {"arm": arm, "against": against, "result": "IMPROVES" if better else "NO_EVIDENCE_OF_IMPROVEMENT",
            "mean_net": a["mean_net_return"], "mean_net_ci95": ci, "comparator_mean_net": b["mean_net_return"],
            "hit_rate": a["hit_rate"], "comparator_hit_rate": b["hit_rate"]}
