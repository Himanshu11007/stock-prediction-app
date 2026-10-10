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
- **Strict job types.** `TODAY_PREOPEN`, `TODAY_CONFIRMED`, `TOMORROW_EOD`,
  `OUTCOME_EVALUATION`, `SNAPSHOT_MONITOR`. Anything else gets 422 before
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
    published, and the 20:30 retry tries again.
  - Each stock without today's bar is flagged `STALE_PRICE` and gets
    NO_CALL.
  - The snapshot is published only if at least 50% of the v2 universe has
    usable data. It is written in one transaction.
- **Monitoring** (`SNAPSHOT_MONITOR`) reports `ALERT` for:
  - a missing, failed or still-running TODAY_PREOPEN after 09:30 IST;
  - the same for TOMORROW_EOD after 20:30 IST;
  - the same for TODAY_CONFIRMED after 10:30 IST, only when intraday is
    enabled;
  - a failed or stale OUTCOME_EVALUATION.

  The monitor's own row is stored as `FAILED`, and the workflow run fails.

## Schedules

GitHub cron is in **UTC**. IST = UTC+05:30. Every schedule is Monday to
Friday; the backend skips NSE holidays from the configured calendar.

| Workflow | Job type | UTC cron | IST | Purpose |
|---|---|---|---|---|
| `stocklens-today-preopen.yml` | `TODAY_PREOPEN` | `15 2 * * 1-5` | 07:45 | snapshot from the previous close (must start before 09:15) |
| | | `0 3 * * 1-5` | 08:30 | retry; no-op if done |
| `stocklens-today-confirmed.yml` | `TODAY_CONFIRMED` | `20 4 * * 1-5` | 09:50 | intraday confirmation (**disabled**, see below) |
| | | `50 4 * * 1-5` | 10:20 | retry |
| `stocklens-outcome-evaluation.yml` | `OUTCOME_EVALUATION` | `0 13 * * 1-5` | 18:30 | 1/3/5-session outcomes and shadow exit states |
| | | `30 15 * * 1-5` | 21:00 | retry |
| `stocklens-tomorrow-eod.yml` | `TOMORROW_EOD` | `0 14 * * 1-5` | 19:30 | next-session snapshot after final daily bars |
| | | `0 15 * * 1-5` | 20:30 | retry (also covers late provider data) |
| `stocklens-snapshot-monitor.yml` | `SNAPSHOT_MONITOR` | `15 4 * * 1-5` | 09:45 | checks TODAY_PREOPEN |
| | | `15 16 * * 1-5` | 21:45 | checks TOMORROW_EOD and outcomes |

GitHub starts scheduled runs late under load, often by 5–30 minutes and
sometimes more. The pre-open triggers start early enough for that. A late
TODAY_PREOPEN trigger after 09:15 IST is skipped rather than producing a
snapshot after the open.

## Pushing a branch does not run anything

| Check | Result (10 Oct 2026) |
|---|---|
| Workflow triggers | `schedule` and `workflow_dispatch` only, so a push or pull request never runs them |
| Scheduled runs | GitHub runs `schedule` workflows **only from the default branch** (`main`). Review branches never run on a schedule |
| Repository gate | scheduled runs also need the repository variable `STOCKLENS_SCHEDULER_ENABLED = true`, which is not set. TODAY_CONFIRMED also needs `STOCKLENS_TODAY_CONFIRMED_ENABLED = true` |
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
6. **Dry run:** start each workflow manually (Actions > workflow > Run
   workflow) and check the job summary.
7. **Enable:** set the repository variable `STOCKLENS_SCHEDULER_ENABLED`
   to `true`.

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
  month. Estimated use is about 10 runs a trading day, mostly 1–15 minutes
  each while polling, so about 300–700 minutes a month.
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
- **v1 is unaffected:** the v1 daily ranking keeps its own job
  (`scripts/scheduled_jobs.py ranking`). It is not part of this scheduler.
