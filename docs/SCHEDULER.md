# StockLens scheduler (GitHub Actions → backend job interface)

**Status: implemented and tested, NOT ACTIVE.** Turning it on needs
approval and the owner steps in [Activation](#activation-requires-approval).
Nothing here is enabled by pushing a branch.

## Design

GitHub Actions only **triggers** jobs. Everything else runs in the backend,
in `scheduling/jobs.py` and `prediction_v2/`, the same code the CLI
(`scripts/scheduled_jobs.py`) uses:
- calendars;
- data checks;
- prediction calculations;
- persistence;
- locking.

```
GitHub Actions (cron, UTC)              StockLens API (Render)
  scripts/scheduler_client.py  ──POST /api/v1/scheduler/jobs/{job_type}──►  scheduling/remote.py
    wake API (GET /health)                Bearer <scheduler token>                preflight → slot lock → job
    poll  ◄──GET /api/v1/scheduler/jobs/{job_type}/runs/{YYYY-MM-DD}──           scheduled_job_runs (DB)
    exit 0 / 1                                                                    prediction_runs (DB)
```

- **Dedicated credential.** The API stores only the SHA-256 digest of the
  scheduler token (`SCHEDULER_TOKEN_SHA256`). The token itself is the
  GitHub secret `STOCKLENS_SCHEDULER_TOKEN`.
- **Accepted credentials.** User and administrator logins are rejected on
  this interface. The digest is compared in constant time, and the token is
  never logged or printed.
- **Interface off by default.** It is disabled (404) until
  `SCHEDULER_TOKEN_SHA256` is set.
- **Strict job types.** `NEWS_INGESTION`, `TODAY_PREOPEN`, `TODAY_CONFIRMED`, `TOMORROW_EOD`,
  `OUTCOME_EVALUATION`, `SNAPSHOT_MONITOR`, `DAILY_RANKING`. Anything else gets 422 before
  any work happens.
- **Preflight** (no slot is used):
  - **Holiday calendar:** the job **fails** when `market.holidays` has no
    dates for the current year. With an empty calendar, a pre-open snapshot
    could be published for an NSE holiday.
  - **Calendar and timing:** weekend or holiday gives `SKIPPED` with
    `NOT_A_TRADING_DAY`; outside the job's time window gives `SKIPPED` with
    `OUTSIDE_WINDOW`.
  - **Intraday feed:** `TODAY_CONFIRMED` needs it.
  - **Final bars:** outcomes need today's final bars.
- **Idempotency and duplicates.** `scheduled_job_runs` has
  `UNIQUE(job, slot = IST date)`, and each snapshot has a unique
  `prediction_runs.idempotency_key`. A partial unique index allows one
  RUNNING run per type. A second trigger returns the existing state
  (`duplicate: true`).
- **Bounded retries:**
  - The client makes at most 3 POST attempts, on 502/503/504 and connection
    errors only.
  - Each workflow has 2 scheduled triggers a day.
  - The backend allows at most `MAX_ATTEMPTS = 3` per slot. After that the
    status is `FAILED` with `ATTEMPTS_EXHAUSTED`.
- **Failure status.** Exceptions are recorded as `FAILED` on the slot and
  the run. The client exits 1 on `FAILED`, `ALERT`, a timeout or an HTTP
  error, so a failed job is never a green workflow.
- **Stale-run recovery.**
  - A slot that has been RUNNING for more than 3 h (`jobs.STALE_RUNNING`)
    is reported `stale` and reclaimed by the next trigger.
  - A prediction run that has been RUNNING for more than 2 h is marked
    FAILED and releases its key.
- **Closing data for TOMORROW_EOD:**
  - It runs only after 16:00 IST.
  - Today's NIFTY 50 bar must exist; otherwise `SKIPPED`, nothing is
    published, and the 19:30 retry tries again.
  - Each stock without today's bar is flagged `STALE_PRICE` and gets
    NO_CALL.
  - The snapshot is published only if at least 50% of the v2 universe has
    usable data. It is written in one transaction.
- **Monitoring** (`SNAPSHOT_MONITOR`) reports `ALERT` for stale news ingestion (no successful run in 3 h, once ingestion has started) and:
  - a missing, failed or still-running TODAY_PREOPEN after 09:30 IST;
  - the same for TOMORROW_EOD after 20:30 IST (its last retry is at 19:30);
  - no completed v1 daily ranking (from any trigger, the admin button
    included) after 20:30 IST on a trading day. This check starts once the
    scheduler has run `DAILY_RANKING` at least once, and a ranking that is
    still running normally is not an alert;
  - the same for TODAY_CONFIRMED after 10:30 IST, only when intraday is
    enabled;
  - a failed or stale OUTCOME_EVALUATION.

  The monitor's own row is stored as `FAILED`, and the workflow run fails.

## Schedules

GitHub cron is in **UTC**. IST = UTC+05:30. Prediction, outcome and
monitor schedules run Monday to Friday, and the backend skips NSE holidays
from the configured calendar. News ingestion runs **every day**, because
weekend and holiday news feeds the next pre-open snapshot.

| Workflow | Job type | UTC cron | IST | Purpose |
|---|---|---|---|---|
| `stocklens-news-ingestion.yml` | `NEWS_INGESTION` | `40 0-16 * * *` | 06:10, then hourly to 22:10 | official feeds (and any configured provider); runs before the 08:30 and 16:15 snapshots |
| `stocklens-today-preopen.yml` | `TODAY_PREOPEN` | `0 3 * * 1-5` | 08:30 | snapshot from the previous close plus news ingested by then (must start before 09:15) |
| | | `30 3 * * 1-5` | 09:00 | retry; no-op if done |
| `stocklens-today-confirmed.yml` | `TODAY_CONFIRMED` | `15 4 * * 1-5` | 09:45 | intraday confirmation (**disabled**, see below) |
| | | `45 4 * * 1-5` | 10:15 | retry |
| `stocklens-snapshot-monitor.yml` | `SNAPSHOT_MONITOR` | `30 4 * * 1-5` | 10:00 | checks TODAY_PREOPEN and news freshness |
| | | `15 11 * * 1-5` | 16:45 | checks failed outcomes and news freshness |
| | | `15 15 * * 1-5` | 20:45 | checks TOMORROW_EOD after its retry, outcomes, news |
| `stocklens-tomorrow-eod.yml` | `TOMORROW_EOD` | `45 10 * * 1-5` | 16:15 | next-session snapshot once today's validated closing data exists |
| | | `0 14 * * 1-5` | 19:30 | retry for late or revised provider data |
| `stocklens-outcome-evaluation.yml` | `OUTCOME_EVALUATION` | `0 11 * * 1-5` | 16:30 | 1/3/5-session outcomes and shadow exit states |
| | | `30 15 * * 1-5` | 21:00 | retry |
| `stocklens-daily-ranking.yml` | `DAILY_RANKING` | `30 11 * * 1-5` | 17:00 | **v1 daily ranking (Top Candidates)**, then ranking notifications and current prices |
| | | `0 13 * * 1-5` | 18:30 | retry; no-op if done |
| | | `30 14 * * 1-5` | 20:00 | last retry before the 20:45 monitor check |

**Data readiness at 16:15:** Yahoo's daily bar for today usually exists soon
after the 15:30 close, but it can still be revised in the following hours.
The 16:15 run publishes only if today's NIFTY bar exists and at least 50% of
the universe has validated bars for today; otherwise it is SKIPPED and the
19:30 run tries again. A revision after publication does not change a
published snapshot; outcomes are evaluated from final bars.

GitHub starts scheduled runs late under load, often by 5–30 minutes and
sometimes more. The pre-open triggers start early enough for that. A late
TODAY_PREOPEN trigger after 09:15 IST is skipped rather than producing a
snapshot after the open.

## Pushing a branch does not run anything

| Check | Result (10 Oct 2026) |
|---|---|
| Workflow triggers | `schedule` and `workflow_dispatch` only, so a push or pull request never runs them |
| Scheduled runs | GitHub runs `schedule` workflows **only from the default branch** (`main`). Review branches never run on a schedule |
| Repository gate | **every** run, scheduled or manual (`workflow_dispatch`), needs the repository variable `STOCKLENS_SCHEDULER_ENABLED = true`, which is not set. A manual run cannot bypass it (fixed 11 Oct 2026: the job condition used to accept any manual run). TODAY_CONFIRMED also needs `STOCKLENS_TODAY_CONFIRMED_ENABLED = true` |
| Workflows registered on GitHub | 0 (public API; the backend repository is public) |
| Render | both blueprints (`render.yaml`, `deploy/render-free/render.yaml`) deploy `branch: main` only; no `previews:` section. The review branch does **not** change either blueprint, and the earlier v2 Render cron templates were removed |
| Backend credential | `SCHEDULER_TOKEN_SHA256` is not set in production, so the interface returns 404 even if it is deployed |
| Not verifiable here | repository Actions permission settings and the Render dashboard's preview setting (no authenticated access). The owner should confirm both |

## Activation (requires approval)

1. **Seed the official NSE holiday calendar** for the current and next year
   (Admin > Configuration > `market.holidays`). The jobs refuse to run
   without it.
2. **Create the credential** on a trusted machine, without putting the token
   in shell history or files:
   ```python
   import secrets, hashlib
   token = secrets.token_urlsafe(48)
   print("GitHub secret STOCKLENS_SCHEDULER_TOKEN:", token)
   print("Render env SCHEDULER_TOKEN_SHA256:", hashlib.sha256(token.encode()).hexdigest())
   ```
3. **Render:** add the environment variable `SCHEDULER_TOKEN_SHA256` to
   `stocklens-api`. This is a production configuration change and redeploys
   the service.
4. **GitHub** (Settings > Environments):
   - create the environment `stocklens-scheduler`, optionally restricted to
     the `main` branch;
   - add the secret `STOCKLENS_SCHEDULER_TOKEN` to it;
   - add the repository variable `STOCKLENS_API_URL`, for example
     `https://stocklens-api.onrender.com`.
5. Merge the reviewed branch into `main`.
6. **Dry run without GitHub:** from a trusted machine, call each job once
   with `python scripts/scheduler_client.py --job <JOB>` (it reads
   `STOCKLENS_API_URL` and `STOCKLENS_SCHEDULER_TOKEN` from the
   environment) and check the result. Manual GitHub runs are refused while
   the switch is off.
7. **Enable:** set the repository variable `STOCKLENS_SCHEDULER_ENABLED`
   to `true`. From then on scheduled runs start, and *Run workflow* works
   for recovery. Setting it back to anything else stops both.

**TODAY_CONFIRMED** stays off until an approved intraday feed exists. Then
set both `PREDICTION_V2_INTRADAY_ENABLED=true` on the API and
`STOCKLENS_TODAY_CONFIRMED_ENABLED=true` in GitHub. Without the API flag the
job returns `SKIPPED` / `INTRADAY_FEED_DISABLED` even if the workflow runs.

## Disable

| Scope | How |
|---|---|
| All scheduled runs | set `STOCKLENS_SCHEDULER_ENABLED` to `false`, or delete it |
| One workflow | Actions > the workflow > ⋯ > *Disable workflow* |
| Backend (hard stop) | unset `SCHEDULER_TOKEN_SHA256` on Render; every call then gets 404 |
| Rotate the credential | generate a new token and digest, update the GitHub secret and the Render variable; the old token stops working immediately |

## Manual recovery

Each job is safe to repeat; a repeat is a no-op once the slot is COMPLETED.

| Situation | What to do |
|---|---|
| A run failed (red workflow) | read the job summary (`result.error`), fix the cause, then *Run workflow* manually. At most 3 attempts per job per day |
| `ATTEMPTS_EXHAUSTED` | the day's slot is closed. Investigate first. TODAY_PREOPEN and TOMORROW_EOD can't be backfilled later: their cutoff and time window have passed, so record the gap |
| `CALENDAR_NOT_CONFIGURED` | enter the year's NSE holidays, then re-run |
| A worker died mid-run (RUNNING, no progress) | after 3 h the slot shows `stale: true` and the next trigger reclaims it. Prediction runs are marked FAILED after 2 h |
| The monitor shows `MISSING` | check the job's own workflow run. If the market was open and data is available, re-run the job within its window |
| From a terminal (equivalent) | `STOCKLENS_API_URL=… STOCKLENS_SCHEDULER_TOKEN=… python scripts/scheduler_client.py --job TOMORROW_EOD`, or on a host with database access `python scripts/scheduled_jobs.py predict_eod` |

## Limitations and costs (review before activation)

- **GitHub Actions cost:** the backend repository is public, so standard
  runners are free. If it becomes private, the free tier is 2,000 minutes a
  month. Estimated use: 17 hourly news runs a day (about 1 minute each),
  about 11 v2 and monitor runs a trading day of 1–15 minutes while polling,
  and the daily ranking (one run of up to 40 minutes while polling, its two
  retries about 1 minute each): roughly 1,300–2,200 minutes a month. Free on
  a public repository; if the repository became private this could exceed
  the 2,000-minute free tier.
- **Timing:** scheduled runs can start late or, rarely, be dropped at busy
  times. GitHub also **disables scheduled workflows after 60 days without
  repository activity** in public repositories. The monitor and the retry
  triggers reduce the effect; they don't remove it.
- **Render Free:**
  - The API instance sleeps after about 15 minutes idle, so the client
    wakes it first (cold start up to a few minutes).
  - The job runs inside the web process: 512 MB, shared with user traffic.
  - A deploy or restart during a job interrupts it; stale recovery applies.
  - The free PostgreSQL instance expires; see OPERATIONS_AND_SECURITY §2.
  - A paid instance is recommended before relying on the scheduler.
- **Data:** Yahoo Finance daily bars are delayed and unlicensed. The
  closing bar can change shortly after 16:00, which is why EOD runs at
  19:30. There is no intraday feed, so TODAY_CONFIRMED is disabled.
- **Shadow mode:**
  - Scheduled snapshots are stored for evaluation only.
  - Predictions stay admin-only (`PREDICTION_V2_PUBLIC=false`).
  - No exit alerts or notifications are sent.
  - The backtest shows no edge (`docs/research/PREDICTION_V2_BACKTEST_2026-10.md`).
- **v1 daily ranking (`DAILY_RANKING`):** the scheduler calls the *same*
  job as the Admin Console button "Run the daily ranking job now" and
  `scripts/scheduled_jobs.py ranking` (`scheduling/jobs.ranking_job`). All
  three share one slot per trading day (`scheduled_job_runs`, job
  `ranking`), so a scheduled run after a manual one, or the reverse, is a
  no-op. The methodology is unchanged.
  - It runs only on trading days after 16:00 IST, and only once today's
    NIFTY 50 bar exists; otherwise `SKIPPED`, and the next trigger tries
    again (at most 3 attempts a day).
  - The times (17:00, 18:30, 20:00) avoid the v2 jobs because on Render Free
    every job runs inside the API process: a full ranking peaks at about
    370 MB of the instance's 512 MB. The 17:00 run can still overlap a slow
    16:15 or 16:30 v2 job; a paid instance removes that risk.
  - A full run can take longer than the workflow's 40-minute wait on a slow
    instance. The workflow then reports a failure while the backend
    finishes; the retries see the slot RUNNING or COMPLETED, and the 20:45
    monitor reports the outcome.
  - The ranking job also refreshes current prices when it completes. During
    the session, prices of the stocks on screen are refreshed on demand when
    a client opens a page (`CURRENT_PRICE_ON_DEMAND`), so no separate price
    job is scheduled here.
