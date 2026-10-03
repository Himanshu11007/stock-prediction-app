"""
scripts/scheduled_jobs.py — production scheduled jobs, run by the operating
system scheduler (cron / systemd timer / Windows Task Scheduler), never by a
loop inside the FastAPI web process. Every job is idempotent and safe to run
more often than needed. See docs/SCHEDULING.md.

  python scripts/scheduled_jobs.py ranking        daily market-close ranking run
  python scripts/scheduled_jobs.py notifications  daily summaries + queued pushes (every 15 min)
  python scripts/scheduled_jobs.py outcomes       prospective tracking outcomes (daily)

ranking job rules (Indian market calendar):
  - skip weekends and the administrator-maintained NSE holiday list
  - only after 16:00 IST (closing prices settled; utils.market_session)
  - only when the NIFTY 50 daily bar for today exists (data availability)
  - skip if today's full ranking run already completed (idempotent)
  - then: ranking run -> notification engine -> exit status

Exit code 0 for "done" and for an expected skip (reason printed as JSON);
1 for a failure, so the scheduler can alert.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select  # noqa: E402

import engine_runs.service as runs  # noqa: E402
import masters.service as masters  # noqa: E402
from db.models.market import EngineRun  # noqa: E402
from db.session import engine  # noqa: E402
from notifications import detector  # noqa: E402
from notifications.service import dispatch_pending, send_daily_summaries  # noqa: E402
from utils.market_session import DAILY_BAR_FINAL_AFTER, IST, is_trading_day, now_ist  # noqa: E402


def ranking_job(now: dt.datetime | None = None, fetch_index=None) -> dict:
    local = now_ist(now)
    today = local.date()
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays")
        if not is_trading_day(today, holidays):
            return {"status": "SKIPPED", "reason": f"{today} is not a trading day"}
        if local.time() < DAILY_BAR_FINAL_AFTER:
            return {"status": "SKIPPED", "reason": f"before {DAILY_BAR_FINAL_AFTER:%H:%M} IST; closing data not final"}
        latest = detector.latest_full_run(session)
        if latest is not None and latest.started_at is not None:
            started = latest.started_at if latest.started_at.tzinfo else latest.started_at.replace(tzinfo=dt.timezone.utc)
            if started.astimezone(IST).date() == today and started.astimezone(IST).time() >= DAILY_BAR_FINAL_AFTER:
                return {"status": "SKIPPED", "reason": f"today's ranking run {latest.run_id} already completed"}
        weights, rules = masters.ranking_config(session)
    if fetch_index is None:
        from fundamentals.provider import fetch_price_history
        fetch_index = lambda: fetch_price_history([runs.MARKET_INDEX], period="5d").get(runs.MARKET_INDEX)  # noqa: E731
    idx = fetch_index()
    if idx is None or idx.empty or idx.index[-1].date() != today:
        return {"status": "SKIPPED", "reason": "NIFTY 50 data for today is not available yet"}
    try:
        with Session(engine) as session:
            run = runs.create_run(session, kind="RANKING", triggered_by=None, config={
                "symbols": None, "limit": None, "include_ml": True, "refresh_fundamentals": False,
                "weights": weights, "rules": rules, "scheduled": True})
            run_id = run.run_id
    except runs.RunInProgressError:
        return {"status": "SKIPPED", "reason": "a ranking run is already in progress"}
    runs.execute_run(engine, run_id)
    runs.notify_after_run(engine, run_id)
    with Session(engine) as session:
        r = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()
        return {"status": r.status, "run_id": run_id, "succeeded": r.succeeded, "skipped": r.skipped,
                "failed": r.failed}


def notifications_job(now: dt.datetime | None = None) -> dict:
    with Session(engine) as session:
        summary = send_daily_summaries(session, now)
        queued = dispatch_pending(session, now)
    return {"status": "DONE", "daily_summary": summary, "queued_dispatch": queued}


def outcomes_job() -> dict:
    from ranking.tracking import record_outcomes
    with Session(engine) as session:
        return {"status": "DONE", **record_outcomes(session)}


JOBS = {"ranking": ranking_job, "notifications": notifications_job, "outcomes": outcomes_job}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=sorted(JOBS))
    args = ap.parse_args(argv)
    try:
        result = JOBS[args.job]()
    except Exception as e:  # reported to the scheduler via the exit code
        print(json.dumps({"job": args.job, "status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]}))
        return 1
    print(json.dumps({"job": args.job, **result}, default=str))
    return 1 if result.get("status") == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
