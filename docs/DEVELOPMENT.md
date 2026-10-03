# Development, Production Configuration and Testing

## Local development

```bash
pip install -r requirements.txt
alembic upgrade head
python scripts/seed_admin.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
streamlit run admin_console/app.py          # admin console
```

Development defaults: SQLite `storage/app.db`, CORS `*`, an ephemeral JWT key
per process (tokens invalid after restart; a warning is logged).

## Configuration (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `production` enables the startup guard below |
| `DATABASE_URL` | SQLite `storage/app.db` | any SQLAlchemy URL (e.g. PostgreSQL) |
| `JWT_SECRET_KEY` | ephemeral | **required** in production |
| `CORS_ALLOWED_ORIGINS` | `*` (dev) / empty (prod) | comma-separated explicit origins in production |
| `FUNDAMENTALS_TTL_HOURS` | 24 | reuse window for fundamentals |
| `MARKET_DATA_STALE_DAYS` | 4 | stale market data threshold |
| `FUNDAMENTALS_STALE_DAYS` | 7 | stale fundamentals threshold (data health, peer medians) |
| `ENGINE_RUN_MAX_WORKERS` | 8 | provider / compute threads per run |
| `FCM_PROJECT_ID`, `FCM_SERVICE_ACCOUNT_FILE` | — | Android push (docs/NOTIFICATIONS.md) |
| `APNS_KEY_FILE`, `APNS_KEY_ID`, `APNS_TEAM_ID`, `APNS_BUNDLE_ID`, `APNS_USE_SANDBOX` | — | iOS push |
| Google / Apple / OTP settings | — | see docs/AUTHENTICATION.md |

With `APP_ENV=production` the API **refuses to start** unless
`JWT_SECRET_KEY`, explicit `CORS_ALLOWED_ORIGINS` and `DATABASE_URL` are set.

Production deployment: run `alembic upgrade head`, serve with
`uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers N` behind an HTTPS
reverse proxy, schedule a daily `POST /api/v1/admin/engine-runs` after market
close, and never commit secrets (`.streamlit/secrets.toml`, `.env` are ignored).
Logs never contain tokens, passwords or OTP codes.

## Testing

| Suite | Command |
|---|---|
| Backend unit/integration (FastAPI TestClient, in-memory DB) | `python -m pytest -q` |
| Admin console (Streamlit AppTest against the real API) | included above (`tests/test_admin_console.py`) |
| Backend E2E against a running server | `python scripts/e2e_backend.py --user … --user-password … --admin … --admin-password …` |
| Mobile Core library | `dotnet test StockAIPro.Mobile.Tests/StockAIPro.Mobile.Tests.csproj` (mobile repo) |

Use a copy of the database for manual E2E so real users are not affected:
`copy storage\app.db storage\e2e_app.db` and run the server with
`DATABASE_URL=sqlite:///storage/e2e_app.db`.

Mobile contract fixtures: `python scripts/export_mobile_contract_fixtures.py` regenerates the
real API responses the mobile tests deserialize.

Key test files: `test_notifications.py` (detection, dedup, rate limits, quiet hours, devices,
providers, preferences, reports, scheduler calendar), `test_fqvf.py` (18 checks, NOT_AVAILABLE ≠ FAIL, provider
normalisation), `test_ranking.py` (score, eligibility, engine runs, failure
isolation, concurrency), `test_product_api.py` (Top Picks, analysis,
authorization of every admin route, config validation, no error leakage,
production guard), `test_admin_console.py`, plus the existing auth, admin,
watchlist, temporal-integrity and research suites.
