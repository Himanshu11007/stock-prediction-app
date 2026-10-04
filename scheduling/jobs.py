"""
scheduling/jobs.py — the scheduled jobs, shared by the cron entry point
(scripts/scheduled_jobs.py, run by Render Cron / cron / Task Scheduler) and
the admin "run the daily ranking job now" action. Nothing here loops: each
call does one bounded unit of work and returns.

Daily ranking job (once per trading day, after the close):
  1. calendar: weekends and the administrator's NSE holiday list are skipped
     (recorded as SKIPPED; the previous ranking stays current)
  2. not before 16:00 IST (closing prices settled)
  3. slot lock: scheduled_job_runs has UNIQUE(job, slot=trading date). Only
     one trigger can hold today's slot; a duplicate trigger finds it RUNNING
     or COMPLETED and returns without doing anything. A FAILED/SKIPPED slot
     may be retried by a later trigger (max 3 attempts; a RUNNING slot older
     than 3 hours is treated as abandoned)
  4. data availability: the NIFTY 50 bar for today must exist
  5. the EXISTING engine run (engine_runs.service.create_run/execute_run:
     FQVF + StockLens Score, frozen methodology) creates a new engine run;
     previous runs are never modified
  6. the notification engine compares it with the previous full run
     (idempotent per ranking run id)
  7. current prices of the universe are refreshed (prices/service.py)

Prices job (every 15 minutes during the session, trading days only):
refreshes the current price of every active universe stock.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select, update

import engine_runs.service as runs
import masters.service as masters
from db.models.market import EngineRun, ScheduledJobRun
from notifications import detector
from utils.logger import get_logger
from utils.market_session import DAILY_BAR_FINAL_AFTER, IST, is_trading_day, market_status, now_ist

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
STALE_RUNNING = dt.timedelta(hours=3)
COMPLETED_RUN = ("COMPLETED", "COMPLETED_WITH_ERRORS")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ── slot ledger (cross-process lock) ─────────────────────────────────────────

def acquire_slot(session: Session, job: str, slot: str, now: Optional[dt.datetime] = None) -> Optional[ScheduledJobRun]:
    """Claim (job, slot). Returns the RUNNING row if this caller owns it, or
    None if another trigger is running it or it already completed."""
    now = now or _now()
    session.add(ScheduledJobRun(job=job, slot=slot, status="RUNNING", started_at=now))
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        stale_before = now - STALE_RUNNING
        # Atomic conditional re-claim: only one concurrent caller can match.
        res = session.exec(update(ScheduledJobRun)
                           .where(ScheduledJobRun.job == job, ScheduledJobRun.slot == slot,
                                  ScheduledJobRun.attempts < MAX_ATTEMPTS,
                                  ((ScheduledJobRun.status.in_(("FAILED", "SKIPPED")))
                                   | ((ScheduledJobRun.status == "RUNNING")
                                      & (ScheduledJobRun.started_at < stale_before))))
                           .values(status="RUNNING", started_at=now, finished_at=None,
                                   attempts=ScheduledJobRun.attempts + 1))
        session.commit()
        if res.rowcount != 1:
            return None
    return session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == job,
                                                      ScheduledJobRun.slot == slot)).one()


def finish_slot(session: Session, row: ScheduledJobRun, status: str, result: dict,
                run_id: Optional[str] = None) -> None:
    row = session.get(ScheduledJobRun, row.id)
    row.status, row.result, row.finished_at = status, result, _now()
    if run_id:
        row.run_id = run_id
    session.add(row)
    session.commit()


def record_skip(session: Session, job: str, slot: str, reason: str) -> None:
    """Record a calendar skip once per slot (no duplicate rows)."""
    row = session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == job,
                                                     ScheduledJobRun.slot == slot)).first()
    if row is None:
        session.add(ScheduledJobRun(job=job, slot=slot, status="SKIPPED", finished_at=_now(),
                                    result={"status": "SKIPPED", "reason": reason}))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()


# ── daily ranking ────────────────────────────────────────────────────────────

def ranking_job(engine, now: Optional[dt.datetime] = None, fetch_index: Optional[Callable] = None,
                refresh_prices: bool = True) -> dict:
    local = now_ist(now)
    today = local.date()
    slot = today.isoformat()
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays")
        if not is_trading_day(today, holidays):
            reason = f"{today} is not a trading day"
            record_skip(session, "ranking", slot, reason)
            return {"status": "SKIPPED", "reason": reason}
        if local.time() < DAILY_BAR_FINAL_AFTER:
            return {"status": "SKIPPED", "reason": f"before {DAILY_BAR_FINAL_AFTER:%H:%M} IST; closing data not final"}
        latest = detector.latest_full_run(session)
        if latest is not None and latest.started_at is not None:
            started = latest.started_at if latest.started_at.tzinfo else latest.started_at.replace(tzinfo=dt.timezone.utc)
            if started.astimezone(IST).date() == today and started.astimezone(IST).time() >= DAILY_BAR_FINAL_AFTER:
                return {"status": "SKIPPED", "reason": f"today's ranking run {latest.run_id} already completed"}
        claim = acquire_slot(session, "ranking", slot, now)
        if claim is None:
            return {"status": "SKIPPED", "reason": f"today's ranking job ({slot}) is already running or done"}
        weights, rules = masters.ranking_config(session)

    def done(status: str, result: dict, run_id: Optional[str] = None) -> dict:
        with Session(engine) as s:
            finish_slot(s, claim, status, result, run_id)
        return result

    try:
        if fetch_index is None:
            from fundamentals.provider import fetch_price_history
            fetch_index = lambda: fetch_price_history([runs.MARKET_INDEX], period="5d").get(runs.MARKET_INDEX)  # noqa: E731
        idx = fetch_index()
        if idx is None or idx.empty or idx.index[-1].date() != today:
            return done("SKIPPED", {"status": "SKIPPED", "reason": "NIFTY 50 data for today is not available yet"})
        try:
            with Session(engine) as session:
                run = runs.create_run(session, kind="RANKING", triggered_by=None, config={
                    "symbols": None, "limit": None, "include_ml": True, "refresh_fundamentals": False,
                    "weights": weights, "rules": rules, "scheduled": True, "ranking_slot": slot})
                run_id = run.run_id
        except runs.RunInProgressError:
            return done("SKIPPED", {"status": "SKIPPED", "reason": "a ranking run is already in progress"})
        runs.execute_run(engine, run_id)
        with Session(engine) as session:
            r = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()
            outcome = {"status": r.status, "run_id": run_id, "processed": r.processed, "succeeded": r.succeeded,
                       "skipped": r.skipped, "failed": r.failed}
        if outcome["status"] not in COMPLETED_RUN:
            # Not published: the previous completed ranking stays current and
            # no ranking notifications are generated.
            return done("FAILED", outcome, run_id)
        runs.notify_after_run(engine, run_id)
        if refresh_prices:
            outcome["prices"] = prices_job(engine, now, force=True)
        return done("COMPLETED", outcome, run_id)
    except Exception as e:
        logger.exception("SCHEDULED_RANKING_FAILED | %s", slot)
        return done("FAILED", {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]})


# ── current prices ───────────────────────────────────────────────────────────

def prices_job(engine, now: Optional[dt.datetime] = None, fetch: Optional[Callable] = None,
               force: bool = False) -> dict:
    from prices.service import refresh_quotes, universe_symbols
    now = now or _now()
    with Session(engine) as session:
        status = market_status(now, masters.get_config(session, "market.holidays"))
        if not force and status["status"] in ("WEEKEND", "HOLIDAY"):
            return {"status": "SKIPPED", "reason": f"market closed ({status['label']})"}
        symbols = universe_symbols(session)
        totals = {"updated": 0, "failed": 0}
        for i in range(0, len(symbols), 50):
            res = refresh_quotes(session, symbols[i:i + 50], now, fetch)
            totals = {k: totals[k] + res[k] for k in totals}
    return {"status": "DONE", "market_status": status["status"], **totals}
