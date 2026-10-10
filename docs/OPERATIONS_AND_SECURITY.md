# Operations and security (StockLens backend)

Production today:
- **API:** Render Free web service `stocklens-api`.
- **Database:** Render Free PostgreSQL `stocklens-db`.
- **Web:** Render Static Site `stocklens-web`.
- **Streamlit:** Community Cloud (dashboard plus Admin Console).

Every step marked **(approval)** changes production or costs money. It is
not done automatically.

## 1. Secrets

| Secret | Where | Status / action |
|---|---|---|
| `DATABASE_URL` / PostgreSQL password | Render (linked from `stocklens-db`) | **Exposed** in a conversation screenshot (October 2026). Rotate it: see 1.1 **(approval)** |
| `JWT_SECRET_KEY` | Render, generated | Never committed. Rotating it signs every user out |
| `GOOGLE_OAUTH_CLIENT_ID` | Render, web `appsettings.json` | Public identifier, not a secret |
| NewsAPI key | Git history | See 1.2 |
| FCM / APNs credentials | not configured | Push notifications are recorded as `PROVIDER_NOT_CONFIGURED` |

Rules:
- Secrets live only in the hosting provider's environment settings.
- `config.py` reads them from the environment, and nothing logs them. The
  scheduled jobs print results only.
- The API refuses to start in production without `JWT_SECRET_KEY`,
  `DATABASE_URL` and explicit CORS origins (`api/main.py`).

### 1.1 Rotating the database password (approval)

1. Render → `stocklens-db` → **Info**, and check whether the page offers
   credential management for the current plan.
2. If it does:
   1. Create a new credential.
   2. Set `stocklens-api` → Environment → `DATABASE_URL` to the new
      **internal** URL; Render redeploys.
   3. Confirm `GET /api/v1/health` and a sign-in.
   4. Delete the old credential.
3. If the Free plan doesn't offer rotation:
   - upgrade the plan (section 3), or
   - create a new database, then `pg_dump` / `pg_restore` (section 2), then
     switch `DATABASE_URL`, then delete the old database.
   Either way this is a planned change with a short write freeze.
4. Restrict external access (Render → `stocklens-db` → **Access Control**)
   to the IPs that need it: your laptop, plus GitHub's runners if GitHub
   Actions becomes the scheduler.

### 1.2 NewsAPI key: four separate steps

| Step | What it means | Status |
|---|---|---|
| a. Revocation | Regenerate or revoke the key at newsapi.org. **This is the only step that removes the risk**, because the key stays in public history | **Owner action, open.** Current code does not use the key (news comes from Google News RSS) |
| b. Replacement configuration | Not needed: no code reads `API_KEY`. If news is ever re-enabled, the key goes in Streamlit Cloud → app → Settings → Secrets, never in the repository | Nothing to do |
| c. Local tracked-file cleanup | `git rm --cached .streamlit/secrets.toml`. The file is already in `.gitignore` and stays on the laptop | Prepared; **after (a)**, with approval |
| d. History cleanup | The key is in commits `bcf4da6` (news/api.py) and `c9ca6e2` onwards (`.streamlit/secrets.toml`). Removing it needs a history rewrite (git filter-repo) **and a force-push**, which invalidates every clone | **Not done.** After (a) it isn't necessary; only on explicit request |

## 2. Database persistence, backup and recovery

- **Free plan:** the database **expires 30 days after creation** (created
  around 3–4 October 2026; the exact date is on Render → `stocklens-db` →
  Info). There are **no backups and no point-in-time recovery**.
- **Manual backup, today:** run from a laptop with the PostgreSQL client
  tools. It reads only:
  ```bash
  pg_dump --format=custom --no-owner --file=stocklens-$(date +%Y%m%d).dump "$EXTERNAL_DATABASE_URL"
  ```
  Restore into a new, empty database:
  ```bash
  pg_restore --no-owner --dbname="$NEW_DATABASE_URL" stocklens-YYYYMMDD.dump
  ```
  Then `alembic upgrade head`, then point `DATABASE_URL` at it.
- **Recommendation (approval, cost):** upgrade `stocklens-db` to a paid plan
  before expiry. Paid plans include point-in-time recovery; check retention
  and price on Render's plan page. Take a `pg_dump` first, and plan a short
  maintenance window.

## 3. Migrations: order and rollback

| Revision | Change | Downgrade |
|---|---|---|
| `66e61a7dffc5` | price quotes, scheduled job runs (deployed 4 Oct) | drops both tables |
| `4e6773a9e7db` | partial unique index: one RUNNING RANKING run | drops the index only |
| `b78f20585037` | Prediction v2 tables (additive) | **drops v2 tables and their data**; take a `pg_dump` first |

- **Deploy:** the free blueprint's start command runs `alembic upgrade head`.
  A failed migration stops the new deploy, and Render keeps serving the
  previous version.
- **`4e6773a9e7db` pre-check:** it stops without changing data if two
  RANKING runs are RUNNING (`docs/DATA_HEALTH_AND_ENGINE_RUNS.md`,
  "Overlapping runs").
- **Before applying in production (approval), run read-only:**
  ```sql
  SELECT run_id, started_at FROM engine_runs WHERE status = 'RUNNING' AND kind = 'RANKING';
  ```
- **Rollback:** deploy the previous commit, then
  `alembic downgrade <revision>` from a laptop with `DATABASE_URL` set.
- **Branch conflict:** the remote branch
  `claude/stocklens-auth-security-hardening-qczhnr` has its own migration
  `c3f9d8e0a4b2`. It creates the same index, and it **auto-marks duplicate
  runs FAILED**. Integrate only one guard (`docs/FINDINGS_LEDGER.md`, B-01).
  `4e6773a9e7db` is idempotent if the index already exists.

## 4. Scheduling

Every job is idempotent, calendar-aware, and protected by `scheduled_job_runs`
`UNIQUE(job, slot)` plus the database indexes. Extra triggers are no-ops, and
none depends on a laptop or `StockLens Admin.bat`. The Admin Console buttons
remain as manual recovery.

| Option | Cost | Reliability | Status |
|---|---|---|---|
| **GitHub Actions → backend job interface (proposed for v2)** | Free while the repository is public (about 300–700 minutes a month otherwise) | Scheduled starts can be 5–30+ minutes late; two triggers per job plus the monitor; the backend does all work. Needs only a dedicated scheduler token (`STOCKLENS_SCHEDULER_TOKEN`), **not** `DATABASE_URL` | Implemented and tested, **inactive**: docs/SCHEDULER.md |
| Render paid cron jobs (`render.yaml`: the 4 v1 jobs on `main`) | Billed by run time per job; check Render pricing **(approval)** | On time; secrets stay in Render | Not active. The v2 cron templates were removed in favour of GitHub Actions |
| Render Free | — | **No cron jobs on the Free plan** | — |

**Activation (approval):** pick one option, then:
1. Set secrets and access control.
2. Seed `market.holidays` from the official NSE circular.
3. Watch the first week through **Engine Runs** and the
   `prediction_monitor` job.

**Failure recovery:**
- **Failed or stale runs:**
  - a ranking slot is re-claimable after FAILED or SKIPPED (3 attempts);
  - a RUNNING ranking older than 3 hours is treated as abandoned;
  - a v2 RUNNING snapshot older than 2 hours is marked FAILED and releases
    its key.
- **Missed runs:** the monitor job exits with code 1 and logs
  `PREDICTION_SNAPSHOT_MISSING`, so the scheduler shows a failed job.

## 5. Monitoring and logging

- **Jobs:** status, duration and results are in `scheduled_job_runs`. Admin
  sees them in **Engine Runs → Daily schedule** and
  `GET /api/v1/admin/prediction-runs`.
- **Data health:** `GET /api/v1/admin/universe-health` lists v1 symbols
  without market data (18 renamed or delisted symbols in October 2026) and
  companies outside the v1 universe.
- **Logs:** structured `PREDICTION_RUN`, `ENGINE_RUN_*` and
  `SCHEDULED_RANKING_*` lines. No tokens or connection strings are logged.

## 6. Data providers and licensing

| Provider | Use | Limits |
|---|---|---|
| Yahoo Finance (`yfinance`) | daily and intraday bars, fundamentals | Unofficial and delayed. Rate limits are not published; the jobs batch 50 symbols per request. Redistribution terms are not granted, so it's acceptable for internal shadow evaluation. A licensed feed is needed before public v2 predictions or `TODAY_CONFIRMED` |
| Google News RSS | legacy dashboard headlines | headlines only, no publication timestamps used |
| Event or filings source | none configured | see `docs/PREDICTION_V2.md`, "Event sources" |

## 7. API hardening (Phase 0)

- `/docs`, `/redoc` and `/openapi.json` are off in production;
  `API_DOCS_ENABLED=true` re-enables them.
- Security headers are sent on every response: HSTS, `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy`.
- Admin routes return 401 or 403 without an admin token. CORS is limited to
  the listed origins, with no wildcard.

## 8. Production deployment checklist (every item needs approval)

1. Review and merge the branches (`docs/FINDINGS_LEDGER.md`, branch
   inventory), resolving B-01 first.
2. Take a `pg_dump` backup of production (section 2).
3. Run the read-only RUNNING-runs query (section 3).
4. Push to `main`. Render deploys and the migrations run; watch the deploy log.
5. Smoke test:
   - `/api/v1/health`;
   - `/docs` returns 404;
   - security headers are present;
   - mobile and web sign-in;
   - Top Picks shows `ranking_freshness`.
6. Seed the v2 universe (`POST /api/v1/admin/v2-universe`, or the seed
   function) and `market.holidays`.
7. Choose and activate a scheduler (section 4). Keep `PREDICTION_V2_PUBLIC`
   and `PREDICTION_V2_INTRADAY_ENABLED` false.
8. Shadow period: 4–8 weeks, reviewed against the promotion gates.
