"""
scheduling/remote.py — the authenticated job interface used by an external
scheduler (GitHub Actions; docs/SCHEDULER.md). The scheduler only *triggers*
work: every calculation runs here, in the backend, through the same
scheduling/jobs.py functions the CLI uses, so there is one implementation of
calendars, slot locks, retries and snapshots.

Job types (strict; anything else is rejected before any work):

  TODAY_PREOPEN       v2 snapshot before 09:15 IST      -> jobs.prediction_job
  TODAY_CONFIRMED     v2 intraday snapshot, 09:45-15:30 -> jobs.prediction_job
                      (refused unless PREDICTION_V2_INTRADAY_ENABLED is on)
  TOMORROW_EOD        v2 snapshot after the close       -> jobs.prediction_job
  OUTCOME_EVALUATION  1/3/5-session outcomes + exits    -> jobs.prediction_outcomes_job
  SNAPSHOT_MONITOR    missed / failed snapshot check    -> jobs.prediction_monitor_job

Lifecycle of a trigger:
  1. preflight (synchronous, no slot used): holiday calendar configured for
     the current year (otherwise FAILED - an empty calendar would let a
     pre-open snapshot be published for a market holiday), trading day,
     timing window, data feed availability;
  2. long jobs are started in a background thread and the caller receives
     ACCEPTED with the slot to poll; SNAPSHOT_MONITOR runs inline;
  3. the database slot ledger (scheduled_job_runs, UNIQUE(job, slot)) makes
     duplicate triggers no-ops, caps attempts per slot at jobs.MAX_ATTEMPTS
     and reclaims a RUNNING slot abandoned for jobs.STALE_RUNNING;
  4. status() reports the slot row: RUNNING, COMPLETED, SKIPPED or FAILED
     (an exhausted slot is reported as FAILED, never as success).
"""
from __future__ import annotations

import datetime as dt
import threading
from enum import Enum
from typing import Any, Callable, Optional

from sqlmodel import Session, select

import masters.service as masters
from db.models.market import ScheduledJobRun
from scheduling import jobs
from utils.logger import get_logger
from utils.market_session import is_daily_bar_complete, is_trading_day, now_ist

logger = get_logger(__name__)


class JobType(str, Enum):
    TODAY_PREOPEN = "TODAY_PREOPEN"
    TODAY_CONFIRMED = "TODAY_CONFIRMED"
    TOMORROW_EOD = "TOMORROW_EOD"
    OUTCOME_EVALUATION = "OUTCOME_EVALUATION"
    SNAPSHOT_MONITOR = "SNAPSHOT_MONITOR"


# Slot-ledger job names (shared with the CLI, scripts/scheduled_jobs.py).
SLOT_JOB = {JobType.TODAY_PREOPEN: "predict_preopen", JobType.TODAY_CONFIRMED: "predict_confirmed",
            JobType.TOMORROW_EOD: "predict_eod", JobType.OUTCOME_EVALUATION: "prediction_outcomes",
            JobType.SNAPSHOT_MONITOR: "prediction_monitor"}
PREDICTION_TYPES = (JobType.TODAY_PREOPEN, JobType.TODAY_CONFIRMED, JobType.TOMORROW_EOD)

# In-process results of triggers that ended without a slot row (e.g. a job
# that decided to skip before claiming). Status falls back to these.
_last: dict[tuple[str, str], dict] = {}
_last_lock = threading.Lock()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def calendar_configured(holidays: list[str], year: int) -> bool:
    """True when the administrator's NSE holiday list covers `year` (at least
    one date in it). NSE has holidays every year, so an empty year means the
    calendar was never entered - not that there are no holidays."""
    return any(str(d).startswith(f"{year}-") for d in holidays or [])


def slot_for(job_type: JobType, now: dt.datetime) -> str:
    return now_ist(now).date().isoformat()


def preflight(engine, job_type: JobType, now: dt.datetime) -> Optional[dict[str, Any]]:
    """None when the job may start now; otherwise the final result
    (SKIPPED for an expected non-run, FAILED for a configuration error)."""
    from config import PREDICTION_V2_INTRADAY_ENABLED
    from prediction_v2 import service as v2
    local = now_ist(now)
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays") or []
    if not calendar_configured(holidays, local.year):
        return {"status": "FAILED", "code": "CALENDAR_NOT_CONFIGURED",
                "reason": f"market.holidays has no NSE holidays for {local.year}; scheduled jobs refuse to run "
                          "until the official calendar is entered (Admin > Configuration)"}
    if job_type == JobType.SNAPSHOT_MONITOR:
        return None
    if job_type == JobType.OUTCOME_EVALUATION:
        if is_trading_day(local.date(), holidays) and not is_daily_bar_complete(local.date(), now):
            return {"status": "SKIPPED", "code": "DATA_NOT_FINAL", "reason": "today's daily bars are not final yet"}
        return None
    if job_type == JobType.TODAY_CONFIRMED and not PREDICTION_V2_INTRADAY_ENABLED:
        return {"status": "SKIPPED", "code": "INTRADAY_FEED_DISABLED",
                "reason": "no approved intraday data feed (PREDICTION_V2_INTRADAY_ENABLED is off)"}
    try:
        v2.plan(job_type.value, now, holidays)
    except v2.RunSkipped as e:
        code = "NOT_A_TRADING_DAY" if "not a trading day" in str(e) else "OUTSIDE_WINDOW"
        return {"status": "SKIPPED", "code": code, "reason": str(e)}
    return None


def _execute(engine, job_type: JobType, now: Optional[dt.datetime]) -> dict[str, Any]:
    if job_type in PREDICTION_TYPES:
        return jobs.prediction_job(engine, job_type.value, now)
    if job_type == JobType.OUTCOME_EVALUATION:
        return jobs.prediction_outcomes_job(engine, now)
    return jobs.prediction_monitor_job(engine, now)


def _run_and_remember(engine, job_type: JobType, now: Optional[dt.datetime], slot: str) -> dict[str, Any]:
    try:
        out = _execute(engine, job_type, now)
    except Exception as e:                       # jobs already record their own failures; this is a last resort
        logger.exception("SCHEDULER_JOB_CRASHED | %s | %s", job_type.value, slot)
        out = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]}
    with _last_lock:
        _last[(SLOT_JOB[job_type], slot)] = out
    logger.info("SCHEDULER_JOB | %s | %s | %s", job_type.value, slot, out.get("status"))
    return out


def spawn(fn: Callable[[], Any]) -> None:
    """Background execution (tests replace this to run inline)."""
    threading.Thread(target=fn, name="stocklens-scheduler-job", daemon=True).start()


def trigger(engine, job_type: JobType, now: Optional[dt.datetime] = None) -> dict[str, Any]:
    """Start (or report) one job. Never raises for job-level problems."""
    clock = now or _now()
    slot = slot_for(job_type, clock)
    pre = preflight(engine, job_type, clock)
    if pre is not None:
        logger.info("SCHEDULER_PREFLIGHT | %s | %s | %s", job_type.value, pre["status"], pre.get("code"))
        return {"job_type": job_type.value, "slot": slot, **pre}
    if job_type == JobType.SNAPSHOT_MONITOR:
        out = _run_and_remember(engine, job_type, clock, slot)
        return {"job_type": job_type.value, "slot": slot, **out}
    current = status(engine, job_type, slot)
    if current["status"] == "COMPLETED":
        return {**current, "duplicate": True}
    if current["status"] == "RUNNING" and not current.get("stale"):
        return {**current, "duplicate": True}
    if current["status"] == "FAILED" and current.get("code") == "ATTEMPTS_EXHAUSTED":
        return current
    spawn(lambda: _run_and_remember(engine, job_type, clock, slot))
    return {"job_type": job_type.value, "slot": slot, "status": "ACCEPTED",
            "poll": f"/api/v1/scheduler/jobs/{job_type.value}/runs/{slot}"}


def status(engine, job_type: JobType, slot: str) -> dict[str, Any]:
    """The slot's state from the database ledger (authoritative), else the
    in-process result of a trigger that never claimed the slot."""
    job = SLOT_JOB[job_type]
    with Session(engine) as session:
        if job_type == JobType.SNAPSHOT_MONITOR:
            row = session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == job,
                                                             ScheduledJobRun.slot.startswith(slot))
                               .order_by(ScheduledJobRun.slot.desc())).first()
        else:
            row = session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == job,
                                                             ScheduledJobRun.slot == slot)).first()
    base = {"job_type": job_type.value, "slot": slot}
    if row is not None:
        out = {**base, "status": row.status, "attempts": row.attempts, "run_id": row.run_id,
               "started_at": row.started_at.isoformat() if row.started_at else None,
               "finished_at": row.finished_at.isoformat() if row.finished_at else None, "result": row.result}
        started = row.started_at.replace(tzinfo=row.started_at.tzinfo or dt.timezone.utc) if row.started_at else None
        if row.status == "RUNNING" and started and _now() - started > jobs.STALE_RUNNING:
            out["stale"] = True                               # the next trigger reclaims it
        if row.status in ("FAILED", "SKIPPED") and row.attempts >= jobs.MAX_ATTEMPTS:
            if row.status == "FAILED":
                out["code"] = "ATTEMPTS_EXHAUSTED"
        return out
    with _last_lock:
        mem = _last.get((job, slot))
    if mem is not None:
        return {**base, "status": mem.get("status", "UNKNOWN"), "result": mem}
    return {**base, "status": "NOT_FOUND"}
