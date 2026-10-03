"""
ranking/tracking.py — prospective tracking of the frozen ranking engine
(docs/RANKING_VALIDATION_V1.md, "Prospective tracking").

  record_run_snapshots  called when a RANKING run completes: freezes one
                        RankingSnapshot per stock (score, rank, FQVF summary,
                        component scores, freshness, engine version, market
                        regime, reference price). Never modified afterwards.
  record_outcomes       for every snapshot whose horizon (1M/3M/6M/12M =
                        21/63/126/252 NIFTY 50 sessions after the reference
                        date) has fully elapsed, inserts the realised return
                        and the NIFTY 50 return once. Existing outcomes are
                        never recomputed or overwritten; horizons that have
                        not elapsed, or stocks without a close near the exit
                        session, are left pending (nothing is estimated).
  summarise_outcomes    realised performance by engine version, horizon and
                        portfolio (Top 10 / Top 20 / all eligible).

Returns use one price series fetched at outcome time for both endpoints
(provider adjusted closes, so splits and dividends between the dates cannot
distort the return); the stored reference_price is the close the run
actually used and is kept for audit.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

import pandas as pd
from sqlmodel import Session, select

from config import RANKING_ENGINE_VERSION
from db.models.market import RankingOutcome, RankingSnapshot, StockAnalysisResult
from ranking.service import DEFAULT_RULES, DEFAULT_WEIGHTS, validate_rules, validate_weights

BENCHMARK = "^NSEI"
HORIZON_SESSIONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
MAX_EXIT_GAP_DAYS = 5


def fqvf_summary(fqvf: Optional[dict]) -> Optional[dict]:
    if not fqvf:
        return None
    return {"score": fqvf.get("score"), "coverage": fqvf.get("coverage"), "counts": fqvf.get("counts"),
            "version": fqvf.get("version")}


def tracked_engine_version(weights: Optional[dict] = None, rules: Optional[dict] = None) -> str:
    """The frozen version label, or "<version>+custom-config" when a run used
    administrator-overridden weights/rules, so prospective results of the
    frozen v1.0 configuration are never mixed with other configurations."""
    frozen = (validate_weights(weights or {}) == validate_weights(DEFAULT_WEIGHTS)
              and validate_rules(rules or {}) == validate_rules(DEFAULT_RULES))
    return RANKING_ENGINE_VERSION if frozen else f"{RANKING_ENGINE_VERSION}+custom-config"


def record_run_snapshots(session: Session, run_id: str, references: dict[str, tuple[Optional[str], Optional[float]]],
                         market_regime: Optional[str], benchmark_price: Optional[float],
                         weights: Optional[dict] = None, rules: Optional[dict] = None) -> int:
    """Freeze the run's results. `references[symbol] = (as_of_date, close)`
    of the market data the run used. Idempotent: existing snapshots are kept."""
    version = tracked_engine_version(weights, rules)
    existing = set(session.exec(select(RankingSnapshot.symbol).where(RankingSnapshot.run_id == run_id)).all())
    n = 0
    for r in session.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == run_id)).all():
        if r.symbol in existing:
            continue
        ref_date, ref_price = references.get(r.symbol, (None, None))
        session.add(RankingSnapshot(
            run_id=run_id, symbol=r.symbol, ranked_at=r.computed_at, engine_version=version,
            fqvf_version=r.fqvf_version, stockai_score=r.stockai_score, score_coverage=r.score_coverage,
            eligible=r.eligible, rank=r.rank, fqvf_summary=fqvf_summary(r.fqvf),
            component_scores={k: (v or {}).get("score") for k, v in (r.components or {}).items()},
            freshness=r.freshness, market_regime=market_regime, reference_date=ref_date,
            reference_price=ref_price, benchmark_symbol=BENCHMARK, benchmark_reference_price=benchmark_price))
        n += 1
    return n


def realised_window(close: pd.Series, calendar: pd.DatetimeIndex, reference_date: str,
                    sessions: int) -> Optional[tuple[str, float, str, float]]:
    """(start_date, start_price, exit_date, exit_price) for the window that
    ends `sessions` benchmark sessions after `reference_date`, or None if it
    has not fully elapsed or the stock has no close near the exit session."""
    ref = pd.Timestamp(reference_date)
    close = close.dropna()
    start = close.loc[close.index <= ref]
    pos = calendar.searchsorted(ref, side="right") - 1
    if start.empty or pos < 0 or pos + sessions >= len(calendar):
        return None
    exit_session = calendar[pos + sessions]
    end = close.loc[close.index <= exit_session]
    if end.empty or end.index[-1] <= ref or (exit_session - end.index[-1]).days > MAX_EXIT_GAP_DAYS:
        return None
    return (start.index[-1].date().isoformat(), float(start.iloc[-1]),
            end.index[-1].date().isoformat(), float(end.iloc[-1]))


def record_outcomes(session: Session, fetch_prices: Optional[Callable[..., dict]] = None) -> dict[str, int]:
    """Insert outcomes for every elapsed, not-yet-recorded (snapshot, horizon)."""
    if fetch_prices is None:
        from fundamentals.provider import fetch_price_history as fetch_prices
    done = {(o.snapshot_id, o.horizon) for o in session.exec(select(RankingOutcome)).all()}
    pending = [s for s in session.exec(select(RankingSnapshot).where(RankingSnapshot.reference_date.is_not(None))).all()
               if any((s.id, h) not in done for h in HORIZON_SESSIONS)]
    if not pending:
        return {"inserted": 0, "pending": 0}
    oldest = min(date.fromisoformat(s.reference_date) for s in pending)
    years = max(2, (date.today() - oldest).days // 365 + 2)
    prices = fetch_prices(sorted({s.symbol for s in pending}) + [BENCHMARK], period=f"{years}y")
    bench = prices.get(BENCHMARK)
    if bench is None or bench.empty:
        return {"inserted": 0, "pending": len(pending), "error": "benchmark prices unavailable"}
    calendar = bench.index
    inserted = still_pending = 0
    for s in pending:
        df = prices.get(s.symbol)
        for h, n in HORIZON_SESSIONS.items():
            if (s.id, h) in done:
                continue
            w = realised_window(df["Close"], calendar, s.reference_date, n) if df is not None else None
            b = realised_window(bench["Close"], calendar, s.reference_date, n)
            if w is None or b is None:
                still_pending += 1
                continue
            ret = w[3] / w[1] - 1
            bret = b[3] / b[1] - 1
            session.add(RankingOutcome(snapshot_id=s.id, horizon=h, start_date=w[0], start_price=w[1],
                                       outcome_date=w[2], outcome_price=w[3], stock_return=ret,
                                       benchmark_return=bret, excess_return=ret - bret))
            inserted += 1
    session.commit()
    return {"inserted": inserted, "pending": still_pending}


def summarise_outcomes(session: Session) -> list[dict[str, Any]]:
    """Realised performance per engine version / horizon / portfolio. Each
    run is one observation (equal-weight portfolio of its members)."""
    rows = session.exec(select(RankingSnapshot, RankingOutcome)
                        .where(RankingOutcome.snapshot_id == RankingSnapshot.id)).all()
    if not rows:
        return []
    df = pd.DataFrame([{"run_id": s.run_id, "engine_version": s.engine_version, "eligible": s.eligible,
                        "rank": s.rank, "horizon": o.horizon, "ret": o.stock_return,
                        "excess": o.excess_return} for s, o in rows])
    out = []
    for (version, h), g in df.groupby(["engine_version", "horizon"]):
        elig = g[g["eligible"]]
        universe = elig.groupby("run_id")["ret"].mean()
        for name, sel in (("Top 10", elig[elig["rank"] <= 10]), ("Top 20", elig[elig["rank"] <= 20]),
                          ("All eligible", elig)):
            if sel.empty:
                continue
            per_run = sel.groupby("run_id").agg(ret=("ret", "mean"), excess=("excess", "mean"))
            vs_universe = per_run["ret"] - universe.reindex(per_run.index)
            out.append({"engine_version": version, "horizon": h, "portfolio": name, "runs": int(len(per_run)),
                        "stock_outcomes": int(len(sel)), "mean_return": float(per_run["ret"].mean()),
                        "mean_excess_vs_nifty": float(per_run["excess"].mean()),
                        "mean_excess_vs_eligible_universe": float(vs_universe.mean()),
                        "hit_rate_vs_nifty": float((per_run["excess"] > 0).mean())})
    return out
