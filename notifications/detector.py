"""
notifications/detector.py — what changed between two ranking runs.

Compares the stored results of a completed, full-universe RANKING run with
the previous one. Pure read: nothing is written here. Only material changes
are reported (thresholds from notifications.settings):

  entered_top / left_top   membership of the Top Investment Candidates
                           (eligible, rank <= top_picks.limit, active company)
  score                    |StockLens Score change| >= score_change_threshold
  fqvf                     |change in FQVF checks passed| >= fqvf_change_min_checks
  rank                     |rank change| >= rank_change_threshold (both ranked)
  status                   eligibility gained/lost, or the risk component
                           crossing the high-risk line (<= 30)
  regime                   NIFTY 50 regime label changed and the regime score
                           moved by >= regime_min_score_change
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sqlmodel import Session, select

from db.models.market import EngineRun, MarketRegimeSnapshot, StockAnalysisResult
from db.models.stock import Company

COMPLETED = ("COMPLETED", "COMPLETED_WITH_ERRORS")
HIGH_RISK_SCORE = 30.0


@dataclass
class StockChange:
    symbol: str
    name: str
    entered_top: Optional[dict] = None
    left_top: Optional[dict] = None
    score: Optional[dict] = None
    fqvf: Optional[dict] = None
    rank: Optional[dict] = None
    status: Optional[dict] = None

    def any(self) -> bool:
        return any((self.entered_top, self.left_top, self.score, self.fqvf, self.rank, self.status))


@dataclass
class RunComparison:
    run: EngineRun
    previous: EngineRun
    top_limit: int
    current_top: list[str] = field(default_factory=list)
    previous_top: list[str] = field(default_factory=list)
    changes: dict[str, StockChange] = field(default_factory=dict)
    regime: Optional[dict] = None


def is_full_universe(run: EngineRun) -> bool:
    cfg = run.config or {}
    return run.kind == "RANKING" and not cfg.get("symbols") and not cfg.get("limit")


def previous_full_run(session: Session, run: EngineRun) -> Optional[EngineRun]:
    rows = session.exec(select(EngineRun).where(EngineRun.kind == "RANKING", EngineRun.status.in_(COMPLETED),
                                                EngineRun.started_at < run.started_at)
                        .order_by(EngineRun.started_at.desc())).all()
    return next((r for r in rows if is_full_universe(r)), None)


def latest_full_run(session: Session) -> Optional[EngineRun]:
    rows = session.exec(select(EngineRun).where(EngineRun.kind == "RANKING", EngineRun.status.in_(COMPLETED))
                        .order_by(EngineRun.started_at.desc())).all()
    return next((r for r in rows if is_full_universe(r)), None)


def _results(session: Session, run_id: str) -> dict[str, StockAnalysisResult]:
    return {r.symbol: r for r in session.exec(
        select(StockAnalysisResult).where(StockAnalysisResult.run_id == run_id)).all()}


def top_candidates(results: dict[str, StockAnalysisResult], active: set[str], limit: int) -> list[str]:
    rows = sorted((r for r in results.values() if r.eligible and r.rank is not None and r.symbol in active),
                  key=lambda r: r.rank)
    return [r.symbol for r in rows[:limit]]


def passes(r: Optional[StockAnalysisResult]) -> Optional[int]:
    counts = (r.fqvf or {}).get("counts") if r is not None else None
    return counts.get("PASS") if counts else None


def _component(r: StockAnalysisResult, key: str) -> Optional[float]:
    return ((r.components or {}).get(key) or {}).get("score")


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def removal_reason(prev: StockAnalysisResult, cur: Optional[StockAnalysisResult], active: bool) -> str:
    if cur is None:
        return "it was not analysed in the latest run."
    if not active:
        return "the stock is no longer active in the stock master."
    if not cur.eligible:
        reasons = "; ".join(cur.ineligible_reasons or []) or "eligibility rules not met"
        return f"data quality / eligibility changed ({reasons})."
    parts = []
    if prev.stockai_score is not None and cur.stockai_score is not None and cur.stockai_score < prev.stockai_score:
        parts.append(f"StockLens Score decreased from {_fmt(prev.stockai_score)} to {_fmt(cur.stockai_score)}")
    pr, cr = _component(prev, "risk"), _component(cur, "risk")
    if pr is not None and cr is not None and cr <= pr - 10:
        parts.append("risk increased")
    if not parts:
        parts.append(f"other stocks ranked higher (now rank {cur.rank})")
    return "; ".join(parts) + "."


def compare(session: Session, run: EngineRun, previous: EngineRun, settings: dict, top_limit: int) -> RunComparison:
    cur, prev = _results(session, run.run_id), _results(session, previous.run_id)
    names = {c.symbol: (c.name, c.active) for c in session.exec(select(Company)).all()}
    active = {s for s, (_, a) in names.items() if a}
    out = RunComparison(run=run, previous=previous, top_limit=top_limit,
                        current_top=top_candidates(cur, active, top_limit),
                        previous_top=top_candidates(prev, active, top_limit))
    ct, pt = set(out.current_top), set(out.previous_top)

    def change(sym: str) -> StockChange:
        if sym not in out.changes:
            out.changes[sym] = StockChange(symbol=sym, name=names.get(sym, (sym, False))[0] or sym)
        return out.changes[sym]

    for sym in out.current_top:
        if sym not in pt:
            r = cur[sym]
            change(sym).entered_top = {"rank": r.rank, "score": r.stockai_score}
    for sym in out.previous_top:
        if sym not in ct:
            p, c = prev[sym], cur.get(sym)
            change(sym).left_top = {"reason": removal_reason(p, c, sym in active), "old_rank": p.rank,
                                    "new_rank": c.rank if c else None, "old_score": p.stockai_score,
                                    "new_score": c.stockai_score if c else None}

    for sym, c in cur.items():
        p = prev.get(sym)
        if p is None or sym not in active:
            continue
        if c.stockai_score is not None and p.stockai_score is not None and \
                abs(c.stockai_score - p.stockai_score) >= settings["score_change_threshold"]:
            change(sym).score = {"old": p.stockai_score, "new": c.stockai_score,
                                 "delta": round(c.stockai_score - p.stockai_score, 1)}
        op, np_ = passes(p), passes(c)
        if op is not None and np_ is not None and abs(np_ - op) >= settings["fqvf_change_min_checks"]:
            change(sym).fqvf = {"old_passes": op, "new_passes": np_}
        if c.rank is not None and p.rank is not None and abs(c.rank - p.rank) >= settings["rank_change_threshold"]:
            change(sym).rank = {"old": p.rank, "new": c.rank}
        status: dict[str, Any] = {}
        if p.eligible != c.eligible:
            status["eligible"] = c.eligible
            if not c.eligible:
                status["reasons"] = c.ineligible_reasons or []
        pr, cr = _component(p, "risk"), _component(c, "risk")
        if pr is not None and cr is not None and (pr <= HIGH_RISK_SCORE) != (cr <= HIGH_RISK_SCORE):
            status["high_risk"] = cr <= HIGH_RISK_SCORE
        if status:
            change(sym).status = status

    out.changes = {s: c for s, c in out.changes.items() if c.any()}

    regimes = {r.run_id: r for r in session.exec(select(MarketRegimeSnapshot).where(
        MarketRegimeSnapshot.run_id.in_([run.run_id, previous.run_id]))).all()}
    a, b = regimes.get(previous.run_id), regimes.get(run.run_id)
    if a and b and a.regime and b.regime and a.regime != b.regime and a.regime_score is not None \
            and b.regime_score is not None and abs(b.regime_score - a.regime_score) >= settings["regime_min_score_change"]:
        out.regime = {"old": a.regime, "new": b.regime, "old_score": a.regime_score, "new_score": b.regime_score,
                      "reason": b.reason, "as_of_date": b.as_of_date}
    return out
