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
from utils.market_session import (DAILY_BAR_FINAL_AFTER, IST, is_daily_bar_complete, is_trading_day,
                                  market_status, now_ist)

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


# ── Prediction Engine v2 (shadow) ────────────────────────────────────────────
# Same slot ledger as the ranking job, one slot per (job, trading date).
# Timing and calendar rules live in prediction_v2.service.plan; a trigger
# that is too early returns SKIPPED without recording (a later trigger the
# same day still runs), a non-trading day is recorded once.

PREDICTION_JOBS = {"TODAY_PREOPEN": "predict_preopen", "TODAY_CONFIRMED": "predict_confirmed",
                   "TOMORROW_EOD": "predict_eod"}


def prediction_job(engine, run_type: str, now: Optional[dt.datetime] = None, fetch: Optional[Callable] = None,
                   fetch_intraday: Optional[Callable] = None) -> dict:
    from prediction_v2 import service as v2
    now = now or _now()
    job = PREDICTION_JOBS[run_type]
    slot = now_ist(now).date().isoformat()
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays") or []
        try:
            v2.plan(run_type, now, holidays)
        except v2.RunSkipped as e:
            if "not a trading day" in str(e):
                record_skip(session, job, slot, str(e))
            return {"status": "SKIPPED", "reason": str(e)}
        claim = acquire_slot(session, job, slot, now)
        if claim is None:
            return {"status": "SKIPPED", "reason": f"{job} for {slot} is already running or done"}
    started = _now()
    try:
        out = v2.run_predictions(engine, run_type, now, fetch=fetch, fetch_intraday=fetch_intraday)
    except Exception as e:                                   # never leave the slot RUNNING
        logger.exception("PREDICTION_JOB_FAILED | %s | %s", job, slot)
        out = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]}
    out["duration_s"] = round((_now() - started).total_seconds(), 1)
    status = {"COMPLETED": "COMPLETED", "FAILED": "FAILED"}.get(out["status"], "SKIPPED")
    with Session(engine) as s:
        finish_slot(s, claim, status, out, out.get("run_id"))
    return out


def prediction_outcomes_job(engine, now: Optional[dt.datetime] = None, fetch: Optional[Callable] = None) -> dict:
    """Daily after the close: 1/3/5-session outcomes and shadow exit states.
    Both steps are idempotent, so a retry is safe."""
    from prediction_v2 import outcomes
    now = now or _now()
    slot = now_ist(now).date().isoformat()
    with Session(engine) as session:
        if not is_daily_bar_complete(now_ist(now).date(), now) and is_trading_day(
                now_ist(now).date(), masters.get_config(session, "market.holidays")):
            return {"status": "SKIPPED", "reason": "today's daily bars are not final yet"}
        claim = acquire_slot(session, "prediction_outcomes", slot, now)
        if claim is None:
            return {"status": "SKIPPED", "reason": f"prediction outcomes for {slot} already running or done"}
    try:
        out = {"status": "DONE", "outcomes": outcomes.evaluate_due(engine, now, fetch),
               "exits": outcomes.update_exit_states(engine, now, fetch)}
        status = "COMPLETED"
    except Exception as e:
        logger.exception("PREDICTION_OUTCOMES_FAILED | %s", slot)
        out, status = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]}, "FAILED"
    with Session(engine) as s:
        finish_slot(s, claim, status, out)
    return out


# ── News ingestion (catalysts/) ──────────────────────────────────────────────
# Hourly, every day (weekend and holiday news matters for the next pre-open).
# One slot per IST hour; each provider resumes from the end of its last
# successful window (at most NEWS_MAX_LOOKBACK back) so nothing is skipped
# after an outage, and the overlap is harmless because ingestion deduplicates.

NEWS_MAX_LOOKBACK = dt.timedelta(days=3)
NEWS_DEFAULT_LOOKBACK = dt.timedelta(hours=24)
NEWS_STALE_AFTER = dt.timedelta(hours=3)


def news_providers() -> list:
    """Configured providers (NEWS_PROVIDERS, comma-separated; default 'rss').
    newsapi needs NEWSAPI_KEY; gdelt is research-only and unreliable."""
    import os

    from catalysts import providers as p
    names = [n.strip().lower() for n in os.environ.get("NEWS_PROVIDERS", "rss").split(",") if n.strip()]
    factory = {"rss": p.RssProvider, "newsapi": p.NewsApiProvider, "gdelt": p.GdeltProvider}
    return [factory[n]() for n in names if n in factory]


def news_ingestion_job(engine, now: Optional[dt.datetime] = None, providers: Optional[list] = None) -> dict:
    from catalysts import pipeline
    from db.models.news import NewsIngestionRun
    now = now or _now()
    slot = f"{now_ist(now):%Y-%m-%dT%H}"
    with Session(engine) as session:
        claim = acquire_slot(session, "news_ingestion", slot, now)
        if claim is None:
            return {"status": "SKIPPED", "reason": f"news ingestion for {slot} already running or done"}
    results: list[dict] = []
    try:
        provs = providers if providers is not None else news_providers()
        for prov in provs:
            with Session(engine) as session:
                last = session.exec(select(NewsIngestionRun).where(
                    NewsIngestionRun.provider == prov.name, NewsIngestionRun.status.in_(("COMPLETED", "PARTIAL")))
                    .order_by(NewsIngestionRun.window_until.desc())).first()
                since = (last.window_until.replace(tzinfo=last.window_until.tzinfo or dt.timezone.utc)
                         if last and last.window_until else now - NEWS_DEFAULT_LOOKBACK)
                since = max(since - dt.timedelta(minutes=30), now - NEWS_MAX_LOOKBACK)
                run = pipeline.ingest(session, prov, since, now, now=now)
                results.append({"provider": run.provider, "status": run.status, "fetched": run.fetched,
                                "inserted": run.inserted, "new_events": run.new_events,
                                "duplicates": run.duplicates, "error": run.error})
        ok = sum(r["status"] in ("COMPLETED", "PARTIAL") for r in results)
        if not results:
            status, out = "FAILED", {"status": "FAILED", "error": "no news provider configured (NEWS_PROVIDERS)"}
        elif ok == 0:
            status, out = "FAILED", {"status": "FAILED", "error": "every provider failed", "providers": results}
        else:
            status, out = "COMPLETED", {"status": "DONE", "providers": results,
                                        "degraded": ok < len(results)}
    except Exception as e:
        logger.exception("NEWS_INGESTION_FAILED | %s", slot)
        status, out = "FAILED", {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500], "providers": results}
    with Session(engine) as s:
        finish_slot(s, claim, status, out)
    return out


# Snapshots that must exist once a trading day reaches the given IST time.
EXPECTED_SNAPSHOTS = (("TODAY_PREOPEN", dt.time(9, 30), "today"), ("TOMORROW_EOD", dt.time(20, 30), "next"))


def prediction_monitor_job(engine, now: Optional[dt.datetime] = None) -> dict:
    """Missed-snapshot check. Returns status ALERT (the CLI exits 1, so the
    scheduler reports a failed job) when an expected snapshot is missing,
    failed or still running long after its window. TODAY_CONFIRMED is only
    expected when an intraday feed is enabled."""
    from config import PREDICTION_V2_INTRADAY_ENABLED
    from db.models.prediction import PredictionRun
    from prediction_v2 import calendar as cal
    now = now or _now()
    local = now_ist(now)
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays") or []
        if not is_trading_day(local.date(), holidays):
            return {"status": "OK", "reason": "not a trading day"}
        expected = list(EXPECTED_SNAPSHOTS)
        if PREDICTION_V2_INTRADAY_ENABLED:
            expected.append(("TODAY_CONFIRMED", dt.time(10, 30), "today"))
        problems = []
        for run_type, due, _ in expected:
            if local.time() < due:
                continue
            day_runs = session.exec(select(PredictionRun).where(
                PredictionRun.run_type == run_type, PredictionRun.trading_date == local.date().isoformat())
                .order_by(PredictionRun.started_at.desc())).all()
            if any(r.status == "COMPLETED" for r in day_runs):
                continue
            if not day_runs:
                problems.append({"run_type": run_type, "problem": "MISSING"})
            else:
                problems.append({"run_type": run_type, "problem": day_runs[0].status, "run_id": day_runs[0].run_id,
                                 "reason": day_runs[0].failure_reason})
        # News freshness: once ingestion has ever run, the latest successful run
        # must be recent (news is the primary v2 input).
        from db.models.news import NewsIngestionRun
        news_runs = session.exec(select(NewsIngestionRun).order_by(NewsIngestionRun.started_at.desc())).all()
        if news_runs:
            done = [r.finished_at.replace(tzinfo=r.finished_at.tzinfo or dt.timezone.utc) for r in news_runs
                    if r.status in ("COMPLETED", "PARTIAL") and r.finished_at]
            latest = max(done, default=None)
            if latest is None or now - latest > NEWS_STALE_AFTER:
                problems.append({"run_type": "NEWS_INGESTION", "problem": "STALE",
                                 "reason": f"last successful ingestion: {latest.isoformat() if latest else 'never'}",
                                 "latest_status": news_runs[0].status})
        # Outcome evaluation has no snapshot of its own: report its slot when it
        # failed, or has been RUNNING past the stale limit (an interrupted worker).
        oc = session.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == "prediction_outcomes",
                                                        ScheduledJobRun.slot == local.date().isoformat())).first()
        if oc is not None:
            started = oc.started_at.replace(tzinfo=oc.started_at.tzinfo or dt.timezone.utc) if oc.started_at else None
            if oc.status == "FAILED":
                problems.append({"run_type": "OUTCOME_EVALUATION", "problem": "FAILED",
                                 "reason": (oc.result or {}).get("error"), "attempts": oc.attempts})
            elif oc.status == "RUNNING" and started and now - started > STALE_RUNNING:
                problems.append({"run_type": "OUTCOME_EVALUATION", "problem": "STALE_RUNNING",
                                 "reason": f"running since {started.isoformat()}"})
        out = {"status": "ALERT" if problems else "OK", "checked_at": now.isoformat(), "problems": problems,
               "next_trading_day": cal.next_trading_day(local.date(), holidays).isoformat()}
        session.add(ScheduledJobRun(job="prediction_monitor", slot=f"{local:%Y-%m-%dT%H:%M}", status=(
            "FAILED" if problems else "COMPLETED"), finished_at=now, result=out))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
    if problems:
        logger.error("PREDICTION_SNAPSHOT_MISSING | %s", problems)
    return out
