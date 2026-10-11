# Independent review package (11 Oct 2026)

Everything here is on **local** branches named `review/independent-review-prep`
in both repositories. **Nothing has been pushed.** Nothing was merged into
`main`, deployed, migrated in production, or activated (schedulers, alerts,
intraday jobs). No production credential, plan or configuration was changed,
and no shared history was rewritten.

## 1. Branches

| Repository | Local branch | Based on | Base commit | `main` |
|---|---|---|---|---|
| Backend `Himanshu11007/stock-prediction-app` | `review/independent-review-prep` | `origin/review/stocklens-complete` | `24be1c7` | `373be2f` (Alembic head `66e61a7dffc5`) |
| Web `Himanshu11007/StockLens.Web` | `review/independent-review-prep` | `origin/review/stocklens-complete` | `231de50` | `478c0e7` |

**Backend: new commits on top of `24be1c7`** (37 files, +1501 / −52):

| Commit | Change |
|---|---|
| `baa442c` | Scheduler: manual dispatch can no longer bypass `STOCKLENS_SCHEDULER_ENABLED` |
| `071c491` | Research scripts: portable, validated database path |
| `a06a521` | Security: secrets file untracked, secret scanner, remediation plan |
| `fdaee2a` | PostgreSQL: migration tests, opt-in PostgreSQL test runs, CI |
| `7c7c78a` | Tests: authorization of every API route |
| `ddf34cf` | Tests: provider network errors, GDELT parsing, news API error paths |
| `da17d05` | Full suite on PostgreSQL; prediction evidence; prediction contract fixtures |
| (this file) | Review package |

The base review branch itself is 39 commits ahead of `main` (merge base
`373be2f`); those commits are described in `FINDINGS_LEDGER.md` and
`REVIEW_CHECKLIST.md`.

**Web: new commits on top of `231de50`** (10 files, +830 / −1):

| Commit | Change |
|---|---|
| `0886c40` | Merge `origin/main` (forgot / reset password pages) into the review branch: **divergence resolved**, 0 behind and 10 ahead of `main` |
| `e8daf76` | Tests: v2 prediction pages render real backend responses |

## 2. Files changed and why

### Backend

| File(s) | Why |
|---|---|
| `.github/workflows/stocklens-*.yml` (6) | The job condition `github.event_name == 'workflow_dispatch' \|\| vars.STOCKLENS_SCHEDULER_ENABLED == 'true'` let any manual run bypass the global switch. It is now `vars.STOCKLENS_SCHEDULER_ENABLED == 'true'`, plus `vars.STOCKLENS_TODAY_CONFIRMED_ENABLED == 'true'` for TODAY_CONFIRMED. Crons, concurrency, permissions and authentication are unchanged |
| `docs/SCHEDULER.md` | Every run needs the switch; the dry run now uses `scripts/scheduler_client.py` from a trusted machine |
| `tests/test_scheduler.py` | Each job condition is **evaluated** for scheduled and manual events across unset / false / mis-cased / true switch values. Covers switches defaulting to off, the backend's intraday flag and job credential defaulting to off, and the old condition demonstrably allowing the bypass. 13 of these tests fail against the old workflows |
| `scripts/research/research_db.py` (new) and 4 research scripts | Removed the hardcoded `C:\Python\...\app.db` path from `news_signal_backtest.py`, `news_provider_benchmark.py`, `factor_shock_replay.py` and `transmission_validation.py`. Path order is `--db PATH`, then `$STOCKLENS_RESEARCH_DB`, then `storage/app.db`, opened read-only with tables and columns checked. Missing data exits with code 2 before any network request; a missing StockLens archive is an error (or `UNAVAILABLE` with `--allow-missing-archive`), never zero |
| `tests/test_research_db.py` (new) | Resolution order; missing file, table, column and non-database; error before network; empty archive reported as `EMPTY` |
| `.streamlit/secrets.toml` | **Untracked** (`git rm --cached`; the local copy stays and is ignored). It held only `API_KEY`, the exposed NewsAPI key, and no code reads it |
| `.gitignore` | Also ignores `.env*` (except `.env.example`), `*.pem`, `*.p12`, `*.pfx`, `client_secret*.json` |
| `scripts/security/scan_secrets.py`, `tests/test_secret_scan.py` (new) | Fails on tracked secret files or credential-shaped values. Prints file, line and rule only. `--history` scans every commit |
| `docs/SECRET_REMEDIATION.md` (new) | Verified key status, affected commits and branches, owner steps, and a prepared but **not executed** history rewrite |
| `tests/test_postgres_migrations.py` (new) | Single head and linear chain from the production head. On PostgreSQL: empty → head, `alembic check`, round trip; upgrade from the production schema with data, then rollback |
| `tests/conftest.py` | `STOCKLENS_TEST_ON_POSTGRES=1` with `TEST_DATABASE_URL` runs the unchanged suite on a disposable PostgreSQL |
| `tests/test_prediction_v2.py`, `test_prediction_api.py`, `test_prediction_feedback.py`, `test_product_api.py`, `test_daily_ranking.py`, `test_notifications.py` | Fixtures wrote child rows before parents in one flush. SQLite does not enforce foreign keys; PostgreSQL rejected them. Parents are now written first. **Test-only:** no assertion was weakened and no production path failed |
| `.github/workflows/ci.yml` (new) | Secret scan; full suite on SQLite; migrations plus full suite on a PostgreSQL 16 service container. No secrets, no deploy |
| `tests/test_route_authorization.py` (new) | Every route from the app's OpenAPI schema: anonymous → 401 unless in an explicit public allowlist; ordinary user → 403 on admin and shadow v2 / news routes |
| `tests/test_news_providers.py`, `tests/test_news_api.py` | Previously untested error paths: network retries, HTTP error status, GDELT parsing and non-JSON, invalid companion report, filter, ingestion runs |
| `tests/test_news_e2e.py`, `tests/contracts/*.json`, `scripts/export_web_news_fixtures.py` | The v2 snapshot and detail responses are also contract fixtures shared with the web |
| `docs/PREDICTION_EVIDENCE.md` (new) | Coverage vs directional accuracy; chronological results after costs; safeguards; no-edge conclusion |

### Web

| File(s) | Why |
|---|---|
| Merge of `origin/main` | Adds `ForgotPassword.razor`, `ResetPassword.razor`, the `Login.razor` link, `PasswordResetPageTests.cs` and `docs/DEPLOY.md`, and takes `main`'s mobile submodule pointer `d0e0614`, which provides `IAuthService.RequestPasswordResetAsync` / `ResetPasswordAsync`. Contract checked against the backend: `POST /auth/forgot-password {email}`, `POST /auth/reset-password {token, new_password}`; the emailed link `FRONTEND_BASE_URL/reset-password?token=` matches the page route |
| `tests/.../PredictionContractTests.cs`, `TestData/predictions_tomorrow.json`, `TestData/prediction_detail.json` | Prediction pages render real backend responses |

## 3. Tests actually executed

| Run | Result |
|---|---|
| Backend full suite, SQLite (final branch state) | **1247 passed, 2 skipped** (the 2 PostgreSQL-only migration tests) |
| Backend full suite, PostgreSQL 16.15 (`STOCKLENS_TEST_ON_POSTGRES=1`) | **1264 passed, 0 failed.** The first full attempt had 45 failed and 11 errors, all foreign-key ordering in test fixtures; fixed before this run |
| Scope you named, PostgreSQL: authentication, password reset, SSO, OTP, authorization, concurrency, news ingestion and API, end to end, catalysts, replay, predictions, prediction jobs, feedback, scheduler | 415 passed (inside the full run as well) |
| `tests/test_postgres_migrations.py` on PostgreSQL | 3 passed |
| Final PostgreSQL spot check on the final branch state (end to end with prediction contracts, migrations, notifications, route authorization) | 264 passed |
| Route authorization (`test_route_authorization.py`) | 219 passed; a mutated allowlist makes it fail |
| Scheduler tests | 54 passed; the new switch tests fail against the old workflows (13 failures) |
| Secret scan, backend tracked files | 0 findings (`.streamlit/secrets.toml` reported before untracking) |
| Secret scan, web tracked files (103) and full web history | 0 findings |
| Ranking v1 golden and validation tests | 22 passed |
| Web suite (merged branch) | **104 passed** |

**Not run:**
- `.github/workflows/ci.yml` on GitHub: nothing pushed.
- A real browser end-to-end run: the web is tested with bUnit against real backend response fixtures.
- Mobile tests: mobile not modified.
- Tests against production: forbidden.

## 4. PostgreSQL migration evidence

- **Disposable container:** `postgres:16` (16.15, matching production's
  `postgresMajorVersion: "16"`), bound to `127.0.0.1:55432`, random local
  password. **Never connected to production.**
- **Revision graph:** exactly one head (`3c1f2a9d7e44`), one base, no branch
  or merge points.
- **Chain:** `main`'s head `66e61a7dffc5` is an ancestor. Six new
  migrations follow it:

  | Revision | Change |
  |---|---|
  | `4e6773a9e7db` | engine-run guard |
  | `b78f20585037` | Prediction v2 tables |
  | `b7e4c2a91d35` | password reset, auth throttle, `users.token_valid_after` |
  | `8d7b9e6c90dd` | news tables |
  | `806b95b85360` | news provenance |
  | `3c1f2a9d7e44` | original timestamp, link evidence |

- **No existing migration was modified:** the review branch only adds
  migration files (+605 lines, 0 deleted).
- **Empty database:** upgrade to head, then `alembic check` ("No new
  upgrade operations detected"), then downgrade to base, then upgrade to
  head: all succeed.
- **Upgrade from the production schema built by `main`'s own code**
  (worktree of `373be2f`, 11 migrations):
  - one representative row in each of the **33 tables**;
  - upgrade by the review branch: **every row preserved**, check clean;
  - downgrade to `66e61a7dffc5`: **every row preserved**;
  - re-upgrade: OK.
- **Not done:** an upgrade of a copy of the real production data. That
  needs your approval and a backup.

## 5. Remaining security and integration risks

1. **The exposed NewsAPI key is still active** (HTTP 200 on 11 Oct 2026).
   - It is in 7 commits and in the history of every remote branch of a
     public repository.
   - Untracking does not fix this; only revocation does.
2. **Production database password rotation** is still pending.
3. **Third-party headlines in a public repository:** the committed research
   reports and web test fixtures contain news headlines and links from
   Google News and other publishers (benchmark events, audit rows, false
   matches). Their licensing is not cleared.
4. **SQLite does not enforce foreign keys in tests by default.** PostgreSQL
   found only fixture issues this time; CI now runs the full suite on
   PostgreSQL to keep it that way.
5. **Ranking v1 response additions:** `ranking/presenter.py` adds
   `reference_price_source`, `freshness` and FQVF display labels. They are
   additive and the golden test passes, but older mobile or web clients
   must ignore unknown fields.
6. **Web merge:** it takes `main`'s mobile submodule pointer (`d0e0614`).
   Pages built against the older pointer `d24eaf5` would not compile with
   the password-reset pages.
7. **Research-only providers:** Google News, Marketaux, Alpha Vantage, GDELT
   and publisher RSS stay research-only. NSE / BSE collection refuses to run
   without recorded written permission.
8. **Coverage gaps remaining:** `catalysts/transmission.py` 81%,
   `catalysts/providers.py` about 85% (fixture-only provider, rare parse
   branches).
9. **No predictive edge** (`PREDICTION_EVIDENCE.md`). Promoting v2 or news
   calls out of shadow mode is not supported by the evidence.

## 6. Actions requiring your approval

1. **Push** the two local branches (`review/independent-review-prep`) to
   GitHub, after which CI runs.
2. **Revoke or regenerate the NewsAPI key** at newsapi.org, then verify
   with `python scripts/security/check_exposed_newsapi_key.py`, which must
   exit 0.
3. **History rewrite** to remove the key from all commits: a force-push to
   a public repository, only after revocation (`SECRET_REMEDIATION.md` §4).
4. **Rotate the production database password.**
5. **Merging into `main`** (backend and web) and the resulting Render
   deploys.
6. **Production migrations** (6 revisions): back up first; the rollback
   path to `66e61a7dffc5` is tested.
7. **Scheduler activation:** `SCHEDULER_TOKEN_SHA256` on Render, the GitHub
   environment and secret, then `STOCKLENS_SCHEDULER_ENABLED=true`.
   TODAY_CONFIRMED stays off.
8. **Licensing:** NSE / BSE written permission or licence; any paid news
   plan (Marketaux, Alpha Vantage); and whether third-party headlines may
   stay in the public repository's research outputs.
9. **Render configuration** for password reset (`FRONTEND_BASE_URL` /
   `PASSWORD_RESET_URL`) and CORS for the web origin, at deploy time.
