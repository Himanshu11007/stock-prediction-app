# Scheduled Jobs (production)

Production users need a fresh ranking every trading day without an
administrator pressing a button. Scheduled work runs as **separate
processes** started by a scheduler: Render Cron, cron or Windows Task
Scheduler. It never runs as a loop inside the FastAPI web process. The web
process can then restart, scale to several workers or crash without
double-running or losing jobs.

The jobs are in `scheduling/jobs.py`; `scripts/scheduled_jobs.py` is the
command-line entry point. Every job is idempotent and safe to run more often
than needed.

| Job | Command | Schedule (IST) | What it does |
|---|---|---|---|
| Daily ranking | `python scripts/scheduled_jobs.py ranking` | Every 15 minutes, 15:30–20:45, Monday–Friday | The existing engine run (FQVF + StockLens Score, unchanged methodology), then notifications, then current prices |
| Current prices | `python scripts/scheduled_jobs.py prices` | Every 15 minutes, 08:30–16:15, Monday–Friday | Refreshes the current price of every universe stock |
| Notifications | `python scripts/scheduled_jobs.py notifications` | Every 15 minutes, all day | Daily summaries once each user's time has passed (trading days, new ranking only), then pushes queued for quiet hours |
| Prospective outcomes | `python scripts/scheduled_jobs.py outcomes` | Daily, 21:00 | Records realised 1M/3M/6M/12M outcomes for frozen ranking snapshots |

## Daily ranking job

Steps, in order:

1. **Trading day.** Weekends and the administrator's NSE holiday list
   (`app_config market.holidays`) are skipped and recorded as `SKIPPED`. No
   ranking is created; the previous ranking stays current.
2. **After the close.** Before 16:00 IST the job exits without doing
   anything, so the earlier triggers on a trading day are harmless.
3. **Per-day lock.** `scheduled_job_runs` has a unique (job, slot) row per
   trading date:
   - Only one trigger can hold today's slot. A duplicate trigger, whether a
     cron retry, an overlapping run or the admin button, finds the slot
     `RUNNING` or `COMPLETED` and returns.
   - A `FAILED` or `SKIPPED` slot is retried by the next trigger, up to 3
     attempts.
   - A `RUNNING` slot older than 3 hours is treated as abandoned, for
     example after a crashed instance.
4. **Data available.** Today's NIFTY 50 bar must be available from the
   provider. Otherwise the slot is marked `SKIPPED` and the next trigger
   tries again.
5. **Engine run.** `engine_runs.service.create_run` and `execute_run`
   create a **new** run, with config `scheduled: true` and
   `ranking_slot: <date>`. Earlier runs, results and ranking snapshots are
   never modified.
6. **Publish gate.** If fewer than `ENGINE_RUN_MIN_SCORED_RATIO` (default
   0.5) of the universe could be scored with market data, for example
   because the provider failed:
   - the run is marked `FAILED` and no ranking snapshots are written;
   - Top Picks keeps showing the previous completed ranking;
   - no ranking notifications are created.
7. **Notifications.** The notification engine compares the new run with the
   previous full run. This is idempotent per ranking run id, so a retry
   creates nothing new.
8. **Current prices.** All universe prices are refreshed.

The slot row records the following, visible in the admin console under
**Engine Runs → Daily schedule**:
- status, start and finish time, and attempts;
- the run id;
- the processed/succeeded/skipped/failed counts;
- the error, if any.

The run itself records engine version, regime and data freshness.

The ranking job's command prints its result as JSON. It exits with code 0
for success or a skip, and code 1 for a failure, so the scheduler can alert.

## Prices: reference price vs current price

Top Picks and Stock Analysis show two different prices:

| Field | Meaning | Changes when |
|---|---|---|
| `reference_price`, `reference_price_as_of` | The daily close the ranking run used. Frozen in `ranking_snapshots` | Only with a new ranking run |
| `current_price`, `current_price_as_of`, `current_price_status` | Latest available market price, from `price_quotes` | Prices job, or on-demand refresh |

`current_price_status`:

| Status | Meaning |
|---|---|
| `LAST_CLOSE` | The day's final close (after 16:00 IST); `as_of` is 15:30 IST of that day |
| `DELAYED_INTRADAY` | Today's price during the session. Yahoo Finance NSE data is delayed, so it is never called "live"; `as_of` is the fetch time |
| `STALE` | The newest bar is older than `MARKET_DATA_STALE_DAYS` |
| `NOT_AVAILABLE` | No valid price has been received. `current_price` is null and the app shows "Not available" |

Prices are never invented:
- A provider failure keeps the previous price with its original timestamp
  and records the error.
- The reference price is never used as the current price.

**On-demand refresh.** Prices also refresh when a user opens a screen.
When a client opens Top Picks or a stock, the API refreshes stale prices of
just those stocks:
- during the session, prices older than `CURRENT_PRICE_TTL_MINUTES`
  (default 15);
- outside the session, prices older than 6 hours, or the intraday price
  once the close is final.

The refresh has one in flight at a time. Set `CURRENT_PRICE_ON_DEMAND=false`
to disable it. This keeps prices fresh on hosts without cron.

## Render Free cannot run these jobs

Render runs cron jobs only on paid instance types.
`deploy/render-free/render.yaml` therefore has **no cron jobs**. On the
free plan:

- **The daily ranking does not run automatically.** An administrator starts
  it each trading day after 16:00 IST from the StockLens Streamlit app
  (https://ai-stock-predict-dashboard.streamlit.app/) → **Administration** →
  **Admin Console** → **Engine Runs** → **Daily schedule** → **Run the daily
  ranking job now**. This runs the same calendar-aware job: trading-day and
  16:00 checks, lock, notifications and prices. The free server sleeps after
  ~15 minutes idle; the Engine Runs page refreshes every minute while a run
  is in progress, so keep it open until the run finishes.
- **Daily summaries and queued pushes are not sent on a schedule.** Ranking
  notifications are still generated when a ranking run completes.
- **Current prices still update.** The on-demand refresh needs no cron.

Automation options:

| | Scheduler | Status |
|---|---|---|
| Now | Manual: Admin Console button (Render Free) | in use |
| Later | Render paid Cron (`render.yaml`, all four jobs) | ready, needs a paid plan |
| Later | An external scheduler such as GitHub Actions running `python scripts/scheduled_jobs.py ranking` against the database | not enabled |

All of them run the same job; only the trigger differs.

## Examples (self-hosted)

**cron (Linux).** The server clock must be in UTC; IST is UTC+05:30.

```cron
*/15 10-14 * * 1-5  cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py ranking       >> logs/jobs.log 2>&1
*/15 3-10  * * 1-5  cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py prices        >> logs/jobs.log 2>&1
*/15 *     * * *    cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py notifications >> logs/jobs.log 2>&1
30   15    * * *    cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py outcomes      >> logs/jobs.log 2>&1
```

**Windows Task Scheduler.** Create one task per job:

- Action: `venv\Scripts\python.exe scripts\scheduled_jobs.py ranking`
- Start in: the repository folder
- Trigger: repeat every 15 minutes

The jobs use the same `DATABASE_URL` and environment variables as the API.

**Resources.** Measured on a laptop with the 299-stock universe:
- full ranking run: ~370 MB peak (ML off, 2 threads);
- prices job: ~330 MB peak process size, ~15–20 s.

**Requirement.** Keep the NSE holiday list current, for example each
December for the coming year. Without it, holidays are treated as trading
days. The ranking job then finds no new NIFTY bar and skips harmlessly, but
Home cannot say "holiday".

## Prediction Engine v2 jobs (shadow)

| Job | When | Purpose |
|---|---|---|
| `news_ingestion` | hourly 06:10–22:10 IST, every day | official news feeds (catalysts/) |
| `predict_preopen` | 08:30 IST, retry 09:00 | TODAY_PREOPEN snapshot |
| `predict_confirmed` | 09:45 IST, retry 10:15 | TODAY_CONFIRMED, only with an approved intraday feed (disabled) |
| `prediction_outcomes` | 16:30 IST, retry 21:00 | 1/3/5-session outcomes and shadow exit states |
| `predict_eod` | 16:15 IST, retry 19:30 | TOMORROW_EOD for the next trading day |
| `prediction_monitor` | 10:00, 16:45 and 20:45 IST | Missed or failed snapshot check; fails the run on a gap |

They use the same slot ledger, calendar and idempotency as the ranking job.
The proposed scheduler is **GitHub Actions calling the backend's authenticated
job interface**: workflows `.github/workflows/stocklens-*.yml`, client
`scripts/scheduler_client.py`, interface `scheduling/remote.py`. It is
inactive until approved; see docs/SCHEDULER.md for UTC and IST times,
enabling, disabling and recovery. Render is not changed: there are no v2
cron entries in either blueprint.
