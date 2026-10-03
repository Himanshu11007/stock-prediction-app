# Scheduled Jobs (production)

Production users need recommendations generated without an administrator
pressing a button. Scheduled work runs as **separate processes** started by
the operating system scheduler, not as a loop inside the FastAPI web
process. The web process can then restart, scale to several workers or
crash without double-running or losing jobs.

Every job in `scripts/scheduled_jobs.py` is idempotent and safe to run more
often than needed.

| Job | Command | Suggested schedule (IST) | What it does |
|---|---|---|---|
| Daily ranking | `python scripts/scheduled_jobs.py ranking` | Every 15 minutes, 16:00–20:00, Monday–Friday | Ranking run, then the notification engine (see rules below) |
| Notifications | `python scripts/scheduled_jobs.py notifications` | Every 15 minutes, all day | Daily summaries once each user's time has passed (trading days), then pushes queued for quiet hours |
| Prospective outcomes | `python scripts/scheduled_jobs.py outcomes` | Daily, 21:00 | Records realised 1M/3M/6M/12M outcomes for frozen ranking snapshots |

## Rules for the ranking job

It runs only when all of the following hold:

- **Trading day:** today is not a weekend and not in the administrator's
  NSE holiday list (`app_config market.holidays`, set in Application
  Configuration).
- **After the close:** it is 16:00 IST or later, so closing prices have
  settled (`utils.market_session.DAILY_BAR_FINAL_AFTER`).
- **Data available:** the NIFTY 50 daily bar for today is available from
  the provider.
- **Not already done:** no full ranking run started today after 16:00.

Each skip prints its reason as JSON and exits with code 0. A failure exits
with code 1, so the scheduler can alert.

There is no pre-market analysis job. Fundamentals and closing prices only
change after the close, so a morning run would repeat the evening's result.

## Examples

**cron (Linux).** The server clock must be in UTC; IST is UTC+05:30.

```cron
*/15 10-14 * * 1-5  cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py ranking       >> logs/jobs.log 2>&1
*/15 *     * * *    cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py notifications >> logs/jobs.log 2>&1
30   15    * * *    cd /srv/stocklens && venv/bin/python scripts/scheduled_jobs.py outcomes      >> logs/jobs.log 2>&1
```

**Windows Task Scheduler.** Create one task per job:

- Action: `venv\Scripts\python.exe scripts\scheduled_jobs.py ranking`
- Start in: the repository folder
- Trigger: repeat every 15 minutes

The jobs use the same `DATABASE_URL` and environment variables as the API.

**Requirement.** Keep the NSE holiday list current, for example each
December for the coming year. Without it, holidays are treated as trading
days. The ranking job then finds no new NIFTY bar and skips harmlessly, but
Home cannot say "holiday".
