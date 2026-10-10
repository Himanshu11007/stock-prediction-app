"""
scripts/scheduled_jobs.py — production scheduled jobs, run by the operating
system scheduler (cron / systemd timer / Windows Task Scheduler), never by a
loop inside the FastAPI web process. Every job is idempotent and safe to run
more often than needed. See docs/SCHEDULING.md.

  python scripts/scheduled_jobs.py ranking        daily market-close ranking run (scheduling/jobs.py)
  python scripts/scheduled_jobs.py prices         current prices of the universe (every 15 min in session)
  python scripts/scheduled_jobs.py notifications  daily summaries + queued pushes (every 15 min)
  python scripts/scheduled_jobs.py outcomes       prospective tracking outcomes (daily)
  python scripts/scheduled_jobs.py predict_preopen      v2 TODAY_PREOPEN snapshot (before 09:15 IST)
  python scripts/scheduled_jobs.py predict_confirmed    v2 TODAY_CONFIRMED (needs an approved intraday feed)
  python scripts/scheduled_jobs.py predict_eod          v2 TOMORROW_EOD snapshot (after 16:00 IST)
  python scripts/scheduled_jobs.py prediction_outcomes  v2 1/3/5-session outcomes + shadow exit states
  python scripts/scheduled_jobs.py prediction_monitor   v2 missed-snapshot check (exit 1 = alert)

ranking job rules (Indian market calendar):
  - skip weekends and the administrator-maintained NSE holiday list
  - only after 16:00 IST (closing prices settled; utils.market_session)
  - only when the NIFTY 50 daily bar for today exists (data availability)
  - skip if today's full ranking run already completed; a per-day slot lock
    (scheduled_job_runs) makes duplicate triggers no-ops
  - then: ranking run -> notification engine -> current prices -> exit status

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

import engine_runs.service as runs  # noqa: E402,F401  (patched by tests)
from db.session import engine  # noqa: E402
from notifications.service import dispatch_pending, send_daily_summaries  # noqa: E402
from scheduling import jobs  # noqa: E402


def ranking_job(now: dt.datetime | None = None, fetch_index=None) -> dict:
    return jobs.ranking_job(engine, now, fetch_index)


def prices_job(now: dt.datetime | None = None) -> dict:
    return jobs.prices_job(engine, now)


def notifications_job(now: dt.datetime | None = None) -> dict:
    with Session(engine) as session:
        summary = send_daily_summaries(session, now)
        queued = dispatch_pending(session, now)
    return {"status": "DONE", "daily_summary": summary, "queued_dispatch": queued}


def outcomes_job() -> dict:
    from ranking.tracking import record_outcomes
    with Session(engine) as session:
        return {"status": "DONE", **record_outcomes(session)}


# Prediction Engine v2 (shadow): snapshots, outcomes/exit states, monitor.
def predict_preopen_job(now: dt.datetime | None = None) -> dict:
    return jobs.prediction_job(engine, "TODAY_PREOPEN", now)


def predict_confirmed_job(now: dt.datetime | None = None) -> dict:
    return jobs.prediction_job(engine, "TODAY_CONFIRMED", now)


def predict_eod_job(now: dt.datetime | None = None) -> dict:
    return jobs.prediction_job(engine, "TOMORROW_EOD", now)


def prediction_outcomes_job(now: dt.datetime | None = None) -> dict:
    return jobs.prediction_outcomes_job(engine, now)


def prediction_monitor_job(now: dt.datetime | None = None) -> dict:
    return jobs.prediction_monitor_job(engine, now)


JOBS = {"ranking": ranking_job, "prices": prices_job, "notifications": notifications_job,
        "outcomes": outcomes_job, "predict_preopen": predict_preopen_job, "predict_confirmed": predict_confirmed_job,
        "predict_eod": predict_eod_job, "prediction_outcomes": prediction_outcomes_job,
        "prediction_monitor": prediction_monitor_job}


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
    return 1 if result.get("status") in ("FAILED", "ALERT") else 0


if __name__ == "__main__":
    sys.exit(main())
