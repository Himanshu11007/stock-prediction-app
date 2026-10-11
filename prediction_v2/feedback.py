"""
prediction_v2/feedback.py — daily prediction-quality report (internal).

For one target session it reads the frozen predictions made for it and the
evaluated 1-session outcomes, and reports:

  outcomes        correct / incorrect directional calls, NEUTRAL and NO_CALL,
                  pending (not evaluated yet), net of the stored cost
  by_setup / by_sector / by_event_category
  missed_news     material news that was available before the cutoff and
                  linked to a stock (subject, or a SUPPORTED channel) where
                  StockLens made no directional call but the stock moved
                  >= MISSED_MOVE vs NIFTY in the session
  false_positive_catalysts   NEWS_CATALYST calls that lost money
  mapping_errors  inferred exposures whose sign was contradicted by a move
                  >= MAPPING_MOVE vs NIFTY (per hypothesis)
  data            ingestion latency (ingested - published), failed / rate-
                  limited provider runs, failed or missing snapshots
  recommendations evidence-based suggestions for RESEARCH REVIEW. Nothing
                  here changes rules: rules change only through a new,
                  versioned and validated rule set.

Small samples are labelled; one day proves nothing.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.news import NewsArticle, NewsIngestionRun
from db.models.prediction import Prediction, PredictionOutcome, PredictionRun
from db.models.stock import Company

MISSED_MOVE = 0.03
MAPPING_MOVE = 0.02
MIN_SAMPLE = 20


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def _bucket(stats: dict, key: str, outcome: PredictionOutcome) -> None:
    b = stats.setdefault(key or "UNKNOWN", {"calls": 0, "correct": 0, "net": []})
    b["calls"] += 1
    b["correct"] += int(bool(outcome.hit))
    if outcome.cost_adjusted_return is not None:
        b["net"].append(outcome.cost_adjusted_return)


def _finish(stats: dict) -> dict:
    return {k: {"calls": v["calls"], "correct": v["correct"],
                "hit_rate": v["correct"] / v["calls"] if v["calls"] else None,
                "mean_net": sum(v["net"]) / len(v["net"]) if v["net"] else None,
                "small_sample": v["calls"] < MIN_SAMPLE} for k, v in sorted(stats.items())}


def daily_report(session: Session, target_session: dt.date) -> dict[str, Any]:
    day = target_session.isoformat()
    runs = session.exec(select(PredictionRun).where(PredictionRun.target_session_date == day)).all()
    completed = [r for r in runs if r.status == "COMPLETED"]
    sectors = {c.symbol: c.sector for c in session.exec(select(Company)).all()}
    out: dict[str, Any] = {"target_session": day, "runs": [{"run_id": r.run_id, "run_type": r.run_type,
                                                             "status": r.status, "failure_reason": r.failure_reason}
                                                            for r in runs]}
    counts = Counter()
    by_setup, by_sector, by_cat = {}, {}, {}
    false_pos, missed, mapping = [], [], defaultdict(list)
    for run in completed:
        preds = session.exec(select(Prediction).where(Prediction.run_id == run.run_id)).all()
        for p in preds:
            o = session.exec(select(PredictionOutcome).where(PredictionOutcome.prediction_id == p.prediction_id,
                                                             PredictionOutcome.horizon_sessions == 1)).first()
            if o is None or o.outcome_status != "EVALUATED":
                counts["pending" if o is None else o.outcome_status.lower()] += 1
                continue
            news = (p.features or {}).get("news") or {}
            if p.direction in ("UP", "DOWN"):
                counts["correct" if o.hit else "incorrect"] += 1
                _bucket(by_setup, p.setup_type, o)
                _bucket(by_sector, sectors.get(p.symbol) or "UNKNOWN", o)
                for e in news.get("evidence", [])[:1]:
                    _bucket(by_cat, e.get("category") or "UNKNOWN", o)
                if p.setup_type == "NEWS_CATALYST" and (o.cost_adjusted_return or 0) < 0:
                    false_pos.append({"symbol": p.symbol, "run_type": run.run_type, "direction": p.direction,
                                      "net": o.cost_adjusted_return, "catalyst": (news.get("evidence") or [{}])[0].get("title")})
            else:
                counts[p.direction.lower()] += 1
                if o.excess_return_nifty is not None and abs(o.excess_return_nifty) >= MISSED_MOVE:
                    material = [e for e in news.get("evidence", []) if e.get("weight", 0) >= 0.25]
                    if material:
                        missed.append({"symbol": p.symbol, "run_type": run.run_type, "excess_move": o.excess_return_nifty,
                                       "decision": p.direction, "why_no_call": news.get("reasons"),
                                       "events": [e.get("title") for e in material]})
            excess = o.excess_return_nifty
            for e in news.get("evidence", []):
                if e.get("inferred") and e.get("sign") and excess is not None and e["sign"] * excess <= -MAPPING_MOVE:
                    mapping[e.get("hypothesis_id") or "UNKNOWN"].append({"symbol": p.symbol, "excess_move": excess,
                                                                         "expected_sign": e["sign"]})
    # news available before the cutoff but never linked to a prediction: data/coverage view
    lo = dt.datetime.combine(target_session - dt.timedelta(days=1), dt.time(0), tzinfo=dt.timezone.utc)
    hi = dt.datetime.combine(target_session, dt.time(23, 59), tzinfo=dt.timezone.utc)
    arts = session.exec(select(NewsArticle).where(NewsArticle.ingested_at >= lo, NewsArticle.ingested_at <= hi)).all()
    lags = [(_utc(a.ingested_at) - _utc(a.published_at)).total_seconds() / 60 for a in arts]
    ing = session.exec(select(NewsIngestionRun).where(NewsIngestionRun.started_at >= lo,
                                                      NewsIngestionRun.started_at <= hi)).all()
    directional = counts["correct"] + counts["incorrect"]
    out.update({
        "outcomes": {**dict(counts), "directional": directional,
                     "hit_rate": counts["correct"] / directional if directional else None,
                     "small_sample": directional < MIN_SAMPLE},
        "by_setup": _finish(by_setup), "by_sector": _finish(by_sector), "by_event_category": _finish(by_cat),
        "missed_news": missed, "false_positive_catalysts": false_pos,
        "mapping_errors": {k: {"count": len(v), "examples": v[:5]} for k, v in sorted(mapping.items())},
        "data": {"articles_ingested": len(arts),
                 "ingestion_latency_minutes": {"median": median(lags) if lags else None,
                                               "max": max(lags) if lags else None,
                                               "over_60_min": sum(1 for x in lags if x > 60)},
                 "provider_runs": dict(Counter(r.status for r in ing)),
                 "failed_snapshots": [r.run_type for r in runs if r.status == "FAILED"],
                 "missing_snapshots": sorted({"TODAY_PREOPEN", "TOMORROW_EOD"} - {r.run_type for r in completed})},
    })
    out["recommendations"] = _recommend(out)
    out["notice"] = ("Internal quality report. Recommendations are for research review only; rules change only "
                     "through a new versioned rule set validated on development and validation data.")
    return out


def _recommend(r: dict) -> list[str]:
    rec = []
    for hid, m in r["mapping_errors"].items():
        if m["count"] >= 3:
            rec.append(f"Review hypothesis {hid}: {m['count']} exposed stocks moved >= {MAPPING_MOVE:.0%} against "
                       "the expected sign today (check over a longer window before changing anything).")
    if r["missed_news"]:
        rec.append(f"{len(r['missed_news'])} material-news stock(s) moved >= {MISSED_MOVE:.0%} without a directional "
                   "call; inspect the 'why_no_call' reasons (threshold, credibility, contradiction, priced-in).")
    if r["false_positive_catalysts"]:
        rec.append(f"{len(r['false_positive_catalysts'])} news-driven call(s) lost money; log the catalysts for the "
                   "category-level false-positive review.")
    d = r["data"]
    if d["provider_runs"].get("FAILED") or d["provider_runs"].get("RATE_LIMITED"):
        rec.append(f"News provider problems: {d['provider_runs']}. Check provider status and credentials.")
    if d["ingestion_latency_minutes"]["over_60_min"]:
        rec.append(f"{d['ingestion_latency_minutes']['over_60_min']} article(s) arrived more than 60 minutes after "
                   "publication; consider a more frequent or lower-latency source.")
    if d["missing_snapshots"] or d["failed_snapshots"]:
        rec.append(f"Snapshots missing {d['missing_snapshots']} or failed {d['failed_snapshots']}.")
    if r["outcomes"]["small_sample"]:
        rec.append("Fewer than 20 directional calls: today's hit rate is not evidence of anything.")
    return rec
