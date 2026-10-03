"""
engine_runs/service.py — controlled execution of the analysis engine.

One run:
  1. resolves the universe (active, tradable, analysis-enabled stocks; by
     default the Large/Mid/Small Cap universe, or an explicit symbol list)
  2. fetches 2 years of daily prices (batched) and the NIFTY 50 regime
  3. fetches fundamentals (re-using snapshots younger than
     FUNDAMENTALS_TTL_HOURS unless refresh is requested)
  4. computes technical/risk metrics and, optionally, the ML signal
  5. derives industry median PE from real peer snapshots
  6. evaluates FQVF and the StockAI Score; stores one StockAnalysisResult per stock
  7. records counts, errors and the final status on the EngineRun

Failure isolation: every per-stock stage is wrapped; a failure is recorded in
EngineRun.errors and the stock is marked failed, the run continues. Only an
infrastructure failure (e.g. database) fails the whole run.

Concurrency: at most one run at a time (in-process lock + a RUNNING row
check). A RUNNING row older than RUN_STALE_AFTER is treated as abandoned.
"""
from __future__ import annotations

import statistics
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlmodel import Session, select

from config import (ENGINE_RUN_MAX_WORKERS, FQVF_ENGINE_VERSION, FUNDAMENTALS_STALE_DAYS,
                    FUNDAMENTALS_TTL_HOURS, MARKET_DATA_STALE_DAYS, RANKING_ENGINE_VERSION)
from db.models.market import (EngineRun, FundamentalSnapshot, MarketRegimeSnapshot, MarketSnapshot,
                              Sector, StockAnalysisResult)
from db.models.stock import Company, StockUniverseMember
from fqvf import FQVFInputs, evaluate
from fundamentals import provider
from masters.service import ranking_config, upsert_classification
from ranking.service import RankingInput, StockRankingService
from ranking.technical import ml_signal, price_issues, technical_snapshot
from utils.logger import get_logger

logger = get_logger(__name__)

RUN_STALE_AFTER = timedelta(hours=3)
MARKET_INDEX = "^NSEI"
_lock = threading.Lock()                      # one RANKING run at a time
_single_slots = threading.BoundedSemaphore(2)  # concurrent single-stock refreshes
_PRICE_BATCH = 50


class RunInProgressError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _age_days(as_of: Optional[str], today: Optional[date] = None) -> Optional[int]:
    if not as_of:
        return None
    return ((today or _now().date()) - date.fromisoformat(as_of)).days


# ── Universe ─────────────────────────────────────────────────────────────────

def resolve_universe(session: Session, symbols: Optional[list[str]] = None,
                     limit: Optional[int] = None) -> list[Company]:
    stmt = select(Company).where(Company.active == True, Company.analysis_enabled == True,  # noqa: E712
                                 Company.tradable == True)  # noqa: E712
    if symbols:
        wanted = sorted({s.strip().upper() for s in symbols if s.strip()})
        stmt = stmt.where(Company.symbol.in_(wanted))
    else:
        members = select(StockUniverseMember.symbol)
        stmt = stmt.where(Company.symbol.in_(members))
    companies = list(session.exec(stmt.order_by(Company.symbol)).all())
    return companies[:limit] if limit else companies


# ── Run lifecycle ────────────────────────────────────────────────────────────

def _running(session: Session, kind: str = "RANKING") -> Optional[EngineRun]:
    run = session.exec(select(EngineRun).where(EngineRun.status == "RUNNING", EngineRun.kind == kind)
                       .order_by(EngineRun.started_at.desc())).first()
    if run and _now() - _aware(run.started_at) > RUN_STALE_AFTER:
        run.status, run.finished_at = "FAILED", _now()
        run.errors = (run.errors or []) + [{"symbol": None, "stage": "run",
                                            "error": "abandoned (no completion within 3 hours)"}]
        session.add(run)
        session.commit()
        return None
    return run


def create_run(session: Session, *, kind: str, triggered_by: Optional[int], config: dict) -> EngineRun:
    if kind == "RANKING" and _running(session, kind) is not None:
        raise RunInProgressError("An engine run is already in progress")
    run = EngineRun(run_id=f"{kind}-{_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}", kind=kind,
                    triggered_by=triggered_by, engine_version=RANKING_ENGINE_VERSION,
                    fqvf_version=FQVF_ENGINE_VERSION, config=config, errors=[])
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def start_run_in_background(engine, *, triggered_by: Optional[int], symbols: Optional[list[str]] = None,
                            limit: Optional[int] = None, include_ml: bool = True,
                            refresh_fundamentals: bool = False) -> EngineRun:
    if not _lock.acquire(blocking=False):
        raise RunInProgressError("An engine run is already in progress")
    try:
        with Session(engine) as session:
            weights, rules = ranking_config(session)
            run = create_run(session, kind="RANKING", triggered_by=triggered_by, config={
                "symbols": symbols, "limit": limit, "include_ml": include_ml,
                "refresh_fundamentals": refresh_fundamentals, "weights": weights, "rules": rules})
            run_id = run.run_id
            session.expunge(run)
    except Exception:
        _lock.release()
        raise

    def _target():
        try:
            execute_run(engine, run_id)
        finally:
            _lock.release()

    threading.Thread(target=_target, name=f"engine-run-{run_id}", daemon=True).start()
    return run


# ── Per-stock stages ─────────────────────────────────────────────────────────

def _latest_fundamentals(session: Session, symbol: str) -> Optional[FundamentalSnapshot]:
    return session.exec(select(FundamentalSnapshot).where(FundamentalSnapshot.symbol == symbol)
                        .order_by(FundamentalSnapshot.fetched_at.desc())).first()


def _build_market(symbol: str, df, run_errors: list, include_ml: bool) -> MarketSnapshot:
    """Compute (not persist) a market snapshot. Pure CPU work, safe in a thread."""
    if df is None or df.empty:
        return MarketSnapshot(symbol=symbol, status="UNAVAILABLE", error="provider returned no price history")
    issues = price_issues(df)
    try:
        tech = technical_snapshot(df)
    except Exception as e:
        run_errors.append({"symbol": symbol, "stage": "technical", "error": f"{type(e).__name__}: {e}"[:300]})
        tech = None
    if tech is not None and include_ml:
        try:
            tech["ml_signal"] = ml_signal(df)
        except Exception as e:
            tech["ml_signal"] = {"available": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    as_of = df.index[-1].date().isoformat()
    age = _age_days(as_of)
    status = "STALE" if age is not None and age > MARKET_DATA_STALE_DAYS else "OK"
    close = provider.clean_number(df["Close"].iloc[-1])
    if close is None or close <= 0:
        status, issues = "ERROR", issues + ["last close is not a valid positive number"]
    return MarketSnapshot(symbol=symbol, status=status, as_of_date=as_of, close=close,
                          volume=provider.clean_number(df["Volume"].iloc[-1]), bars=int(len(df)),
                          technical=tech, issues=issues,
                          error=f"last bar is {age} days old" if status == "STALE" else None)


def _industry_pe_medians(session: Session) -> dict[str, list[tuple[str, float]]]:
    """(symbol, trailing PE) per industry, from the latest fresh snapshot of
    each company that reports a positive PE."""
    cutoff = _now() - timedelta(days=FUNDAMENTALS_STALE_DAYS)
    latest: dict[str, FundamentalSnapshot] = {}
    for snap in session.exec(select(FundamentalSnapshot).where(FundamentalSnapshot.status.in_(("OK", "PARTIAL")))
                             .order_by(FundamentalSnapshot.fetched_at)).all():
        latest[snap.symbol] = snap
    by_industry: dict[str, list[tuple[str, float]]] = {}
    for snap in latest.values():
        d = snap.data or {}
        pe, ind = d.get("trailing_pe"), d.get("industry")
        if _aware(snap.fetched_at) >= cutoff and ind and pe is not None and pe > 0:
            by_industry.setdefault(ind, []).append((snap.symbol, pe))
    return by_industry


def build_fqvf_inputs(fund: Optional[FundamentalSnapshot], market: Optional[MarketSnapshot],
                      sector: Optional[Sector], industry_pe: dict[str, list[tuple[str, float]]],
                      symbol: str) -> FQVFInputs:
    d = (fund.data or {}) if fund and fund.status in ("OK", "PARTIAL") else {}
    industry = d.get("industry")
    # Peer median excludes the stock itself.
    peer_pes = [pe for sym, pe in industry_pe.get(industry, []) if sym != symbol] if industry else []
    med, peers = (statistics.median(peer_pes), len(peer_pes)) if peer_pes else (None, 0)
    return FQVFInputs(
        price=market.close if market and market.status in ("OK", "STALE") else None,
        price_as_of=market.as_of_date if market else None,
        price_age_days=_age_days(market.as_of_date) if market else None,
        sector=d.get("sector"), industry=industry, annual=d.get("annual") or [],
        trailing_pe=d.get("trailing_pe"), trailing_eps=d.get("trailing_eps"),
        price_to_book=d.get("price_to_book"), book_value_per_share=d.get("book_value_per_share"),
        price_to_sales=d.get("price_to_sales"), debt_to_equity=d.get("debt_to_equity"),
        provider_peg=d.get("provider_peg"), dividend_yield=d.get("dividend_yield"),
        payout_ratio=d.get("payout_ratio"),
        industry_pe_median=med, industry_peer_count=peers,
        sector_outlook=sector.outlook if sector else None,
        sector_outlook_notes=sector.outlook_notes if sector else None,
        sector_outlook_updated_at=_aware(sector.outlook_updated_at).isoformat()
        if sector and sector.outlook_updated_at else None,
        fundamentals_fetched_at=_aware(fund.fetched_at).isoformat() if fund else None,
        fiscal_period_end=fund.fiscal_period_end if fund else None,
    )


# ── Run execution ────────────────────────────────────────────────────────────

def execute_run(engine, run_id: str) -> None:
    errors: list[dict[str, Any]] = []
    try:
        with Session(engine) as session:
            run = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()
            cfg = run.config or {}
            weights, rules = cfg.get("weights"), cfg.get("rules")
            companies = resolve_universe(session, cfg.get("symbols"), cfg.get("limit"))
            run.total = len(companies)
            session.add(run)
            session.commit()
            symbols = [c.symbol for c in companies]
            logger.info("ENGINE_RUN_START | %s | %d stocks", run_id, len(symbols))

            # 1. market regime (NIFTY 50)
            try:
                idx = provider.fetch_price_history([MARKET_INDEX]).get(MARKET_INDEX)
                if idx is None:
                    session.add(MarketRegimeSnapshot(run_id=run_id, status="UNAVAILABLE",
                                                     reason="provider returned no index history"))
                else:
                    t = technical_snapshot(idx)
                    session.add(MarketRegimeSnapshot(run_id=run_id, as_of_date=t["as_of_date"], regime=t["regime"],
                                                     regime_score=t["regime_score"], reason=t["regime_reason"]))
            except Exception as e:
                errors.append({"symbol": MARKET_INDEX, "stage": "market_regime", "error": f"{type(e).__name__}: {e}"[:300]})
            session.commit()

            # 2. prices (batched, failure isolated per batch)
            prices: dict[str, Any] = {}
            for i in range(0, len(symbols), _PRICE_BATCH):
                batch = symbols[i:i + _PRICE_BATCH]
                try:
                    prices.update(provider.fetch_price_history(batch))
                except Exception as e:
                    errors.append({"symbol": None, "stage": "prices",
                                   "error": f"batch {batch[0]}..{batch[-1]}: {type(e).__name__}: {e}"[:300]})

            # 3. fundamentals (fresh snapshot reused unless refresh requested)
            ttl_cutoff = _now() - timedelta(hours=FUNDAMENTALS_TTL_HOURS)
            to_fetch = []
            for s in symbols:
                snap = _latest_fundamentals(session, s)
                if cfg.get("refresh_fundamentals") or snap is None or _aware(snap.fetched_at) < ttl_cutoff \
                        or snap.status in ("ERROR", "UNAVAILABLE"):
                    to_fetch.append(s)
            def _safe_fetch(sym):
                # The provider adapter reports errors as status ERROR, but any
                # unexpected exception must stay confined to this one stock.
                try:
                    return provider.fetch_fundamentals(sym)
                except Exception as e:
                    return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:500],
                            "fiscal_period_end": None, "data": None, "fetched_at": _now()}

            with ThreadPoolExecutor(max_workers=ENGINE_RUN_MAX_WORKERS) as pool:
                fetched = dict(zip(to_fetch, pool.map(_safe_fetch, to_fetch)))
            for s, f in fetched.items():
                session.add(FundamentalSnapshot(symbol=s, fetched_at=f["fetched_at"], status=f["status"],
                                                error=f["error"], fiscal_period_end=f["fiscal_period_end"],
                                                data=f["data"]))
                if f["status"] == "ERROR":
                    errors.append({"symbol": s, "stage": "fundamentals", "error": f["error"]})
                d = f["data"] or {}
                try:
                    upsert_classification(session, d.get("sector"), d.get("industry"))
                    company = session.get(Company, s)
                    if d.get("sector") and not company.sector:
                        company.sector = d["sector"]
                    if d.get("industry") and not company.industry:
                        company.industry = d["industry"]
                    session.add(company)
                except Exception as e:
                    errors.append({"symbol": s, "stage": "classification", "error": f"{type(e).__name__}: {e}"[:300]})
            session.commit()

            # 4. technical + ML (CPU; threads keep the provider frames in memory)
            def _market(sym):
                local_errors: list = []
                try:
                    return sym, _build_market(sym, prices.get(sym), local_errors, cfg.get("include_ml", True)), local_errors
                except Exception as e:
                    return sym, None, local_errors + [{"symbol": sym, "stage": "market",
                                                       "error": f"{type(e).__name__}: {e}"[:300]}]

            # Workers compute; all writes happen here, sequentially (SQLite has one writer).
            markets: dict[str, Optional[MarketSnapshot]] = {}
            with ThreadPoolExecutor(max_workers=ENGINE_RUN_MAX_WORKERS) as pool:
                for sym, snap, errs in pool.map(_market, symbols):
                    errors.extend(errs)
                    if snap is not None:
                        session.add(snap)
                    markets[sym] = snap
            session.commit()

            # 5-6. FQVF + ranking
            industry_pe = _industry_pe_medians(session)
            sectors = {s.name: s for s in session.exec(select(Sector)).all()}
            inputs, fqvfs, failed = [], {}, set()
            for c in companies:
                try:
                    fund = _latest_fundamentals(session, c.symbol)
                    market = markets.get(c.symbol)
                    sector_name = (fund.data or {}).get("sector") if fund and fund.data else c.sector
                    fq = evaluate(build_fqvf_inputs(fund, market, sectors.get(sector_name), industry_pe, c.symbol))
                    fqvfs[c.symbol] = fq.to_dict()
                    tech = (market.technical if market else None) or {}
                    inputs.append(RankingInput(
                        symbol=c.symbol, name=c.name, sector=sector_name, fqvf=fqvfs[c.symbol],
                        technical=tech, ml=tech.get("ml_signal"),
                        market_status=market.status if market else "UNAVAILABLE",
                        market_age_days=_age_days(market.as_of_date) if market else None,
                        fundamentals_status=fund.status if fund else "UNAVAILABLE",
                        company_active=c.active, company_tradable=c.tradable,
                        freshness={
                            "fundamentals_fetched_at": _aware(fund.fetched_at).isoformat() if fund else None,
                            "fiscal_period_end": fund.fiscal_period_end if fund else None,
                            "market_data_as_of": market.as_of_date if market else None,
                            "technical_computed_at": _aware(market.fetched_at).isoformat() if market else None,
                            "sector_outlook_updated_at": fqvfs[c.symbol]["checks"][17]["source_timestamp"],
                        }))
                except Exception as e:
                    failed.add(c.symbol)
                    errors.append({"symbol": c.symbol, "stage": "fqvf", "error": f"{type(e).__name__}: {e}"[:300]})

            reference = _reference_technicals(session) if run.kind == "SINGLE" else None
            ranked = StockRankingService(weights, rules).rank(inputs, reference=reference)
            now = _now()
            for r in ranked:
                fund_status = next(i.fundamentals_status for i in inputs if i.symbol == r["symbol"])
                mkt = markets.get(r["symbol"])
                session.add(StockAnalysisResult(
                    run_id=run_id, symbol=r["symbol"], computed_at=now, stockai_score=r["stockai_score"],
                    score_coverage=r["score_coverage"], eligible=r["eligible"],
                    ineligible_reasons=r["ineligible_reasons"], rank=r["rank"], components=r["components"],
                    fqvf=fqvfs[r["symbol"]], positives=r["positives"], risks=r["risks"],
                    freshness=r["freshness"], ml_signal=(mkt.technical or {}).get("ml_signal") if mkt else None,
                    engine_version=RANKING_ENGINE_VERSION, fqvf_version=FQVF_ENGINE_VERSION))
                company = session.get(Company, r["symbol"])
                company.data_status, company.data_status_reason = _company_data_status(mkt, fund_status, r)
                company.data_checked_at = now
                session.add(company)

            # Outcome per stock: failed = a processing stage raised; succeeded =
            # a StockAI Score was produced; skipped = processed but not
            # scorable (missing data), never fabricated.
            failed |= {e["symbol"] for e in errors if e.get("stage") in ("market", "technical") and e.get("symbol")}
            failed &= set(symbols)
            run = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()
            run.processed = len(companies)
            run.failed = len(failed)
            run.succeeded = sum(1 for r in ranked if r["stockai_score"] is not None and r["symbol"] not in failed)
            run.skipped = run.processed - run.succeeded - run.failed
            run.errors = errors
            run.status = "COMPLETED_WITH_ERRORS" if errors else "COMPLETED"
            run.finished_at = _now()
            session.add(run)
            session.commit()
            logger.info("ENGINE_RUN_END | %s | %s | ok=%d skipped=%d failed=%d errors=%d",
                        run_id, run.status, run.succeeded, run.skipped, run.failed, len(errors))
    except Exception as e:
        logger.exception("ENGINE_RUN_FAILED | %s", run_id)
        with Session(engine) as session:
            run = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).first()
            if run is not None:
                run.status, run.finished_at = "FAILED", _now()
                run.errors = errors + [{"symbol": None, "stage": "run", "error": f"{type(e).__name__}: {e}"[:500],
                                        "trace": traceback.format_exc()[-1500:]}]
                session.add(run)
                session.commit()


def _company_data_status(market: Optional[MarketSnapshot], fund_status: str, result: dict) -> tuple[str, str]:
    problems = []
    if market is None or market.status in ("UNAVAILABLE", "ERROR"):
        problems.append("market data unavailable")
    elif market.status == "STALE":
        problems.append(market.error or "market data stale")
    if fund_status in ("UNAVAILABLE", "ERROR"):
        problems.append(f"fundamentals {fund_status.lower()}")
    elif fund_status == "PARTIAL":
        problems.append("fundamentals partial")
    if market is None or market.status in ("UNAVAILABLE", "ERROR"):
        status = "UNAVAILABLE" if fund_status in ("UNAVAILABLE", "ERROR") else "PARTIAL"
    elif market.status == "STALE":
        status = "STALE"
    elif fund_status in ("UNAVAILABLE", "ERROR", "PARTIAL"):
        status = "PARTIAL"
    else:
        status = "OK"
    return status, "; ".join(problems) or "all inputs available"


def _reference_technicals(session: Session) -> dict[str, dict]:
    """Technical metrics of the latest full run's universe, so a single-stock
    refresh is ranked on momentum/risk against the same peers."""
    run = latest_completed_run(session)
    if run is None:
        return {}
    syms = [r.symbol for r in session.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == run.run_id)).all()]
    out = {}
    for sym in syms:
        snap = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == sym, MarketSnapshot.fetched_at <= run.finished_at)
                            .order_by(MarketSnapshot.fetched_at.desc())).first()
        if snap and snap.technical:
            out[sym] = snap.technical
    return out


def run_single_stock(engine, symbol: str, triggered_by: Optional[int], include_ml: bool = True) -> str:
    """Synchronously analyse one active stock; returns the run id."""
    symbol = symbol.strip().upper()
    with Session(engine) as session:
        company = session.get(Company, symbol)
        if company is None or not company.active:
            raise KeyError(f"Stock {symbol} not found")
        if not company.analysis_enabled or not company.tradable:
            raise ValueError(f"Analysis is disabled for {symbol}")
    if not _single_slots.acquire(timeout=30):
        raise RunInProgressError("Too many single-stock analyses in progress; try again shortly")
    try:
        with Session(engine) as session:
            weights, rules = ranking_config(session)
            run = create_run(session, kind="SINGLE", triggered_by=triggered_by, config={
                "symbols": [symbol], "limit": None, "include_ml": include_ml,
                "refresh_fundamentals": False, "weights": weights, "rules": rules})
            run_id = run.run_id
        execute_run(engine, run_id)
        return run_id
    finally:
        _single_slots.release()


# ── Queries ──────────────────────────────────────────────────────────────────

COMPLETED = ("COMPLETED", "COMPLETED_WITH_ERRORS")


def latest_completed_run(session: Session) -> Optional[EngineRun]:
    return session.exec(select(EngineRun).where(EngineRun.status.in_(COMPLETED), EngineRun.kind == "RANKING")
                        .order_by(EngineRun.finished_at.desc())).first()


def latest_result(session: Session, symbol: str) -> Optional[StockAnalysisResult]:
    stmt = (select(StockAnalysisResult)
            .join(EngineRun, EngineRun.run_id == StockAnalysisResult.run_id)
            .where(StockAnalysisResult.symbol == symbol.upper(), EngineRun.status.in_(COMPLETED))
            .order_by(StockAnalysisResult.computed_at.desc()))
    return session.exec(stmt).first()


def latest_market_regime(session: Session) -> Optional[MarketRegimeSnapshot]:
    return session.exec(select(MarketRegimeSnapshot).order_by(MarketRegimeSnapshot.computed_at.desc())).first()
