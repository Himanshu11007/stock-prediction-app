# StockAI Pro — Architecture

```
Python data/analysis engine ──► FastAPI backend (/api/v1, JWT) ──► .NET MAUI Blazor Hybrid app (Android, iOS)
                                         └──────────────────────► Streamlit admin console (ADMIN role)
```

The **backend is the single source of truth**. The mobile app requests and
displays data and sends user actions; it contains no ranking, FQVF or
scoring logic. Feature flags, the disclaimer and the announcement come from
`GET /api/v1/app/config`.

## Backend modules

| Area | Modules | Notes |
|---|---|---|
| API | `api/main.py`, `api/routes/*.py`, `api/schemas*.py` | `/api/v1`; success envelope `{success, data, message}` for business routes; centralised 400/404/500 handlers (500 returns a reference id, never exception text) |
| Auth | `auth/`, `api/routes/auth*.py` | JWT access (15 min) + rotating refresh tokens, Google/Apple, OTP, device sessions, PIN-enabled devices. See AUTHENTICATION.md |
| Database | `db/models/*.py`, `db/session.py`, `alembic/` | SQLModel; SQLite in development, any SQLAlchemy URL in production (`DATABASE_URL`) |
| Master data | `db/models/stock.py` (Company, StockUniverseMember), `db/models/market.py` (Sector, Industry, AppConfig), `masters/service.py` | Stock Master, Sector/Industry masters, application configuration |
| Data ingestion | `fundamentals/provider.py` | Yahoo Finance; values are provider data or NULL with recorded issues |
| Snapshots | `FundamentalSnapshot`, `MarketSnapshot`, `MarketRegimeSnapshot` | append-only history |
| FQVF | `fqvf/service.py` | fixed 18 checks (docs/FQVF.md) |
| Ranking | `ranking/service.py`, `ranking/technical.py`, `ranking/presenter.py` | StockAI Score (docs/RANKING_METHODOLOGY.md); presenter = single API representation |
| Engine runs | `engine_runs/service.py`, `EngineRun`, `StockAnalysisResult` | controlled runs, lock, failure isolation (docs/DATA_HEALTH_AND_ENGINE_RUNS.md) |
| Data health | `data_health/service.py` | data/API health reports for admins |
| Admin | `admin/service.py`, `admin/audit.py`, `api/routes/admin*.py`, `admin_console/` | every mutation audited in the same transaction |
| Legacy engine | `scanner/`, `utils/decision_engine.py`, `models/trainer.py`, `storage/tracker.py` | ML + confluence signal engine (`/analyze-stock`, `/top-picks/start`), temporal-integrity fixed (v1.1); retained for compatibility and the Performance/Intelligence history |
| Research | `evaluation/`, `scripts/research/`, `scripts/audit/` | walk-forward benchmark and experiment registry |

## Data flow of an analysis run

1. Resolve universe: active, tradable, analysis-enabled stocks in the
   Large/Mid/Small Cap universe (or an explicit list).
2. NIFTY 50 regime → `MarketRegimeSnapshot`.
3. Batched 2-year daily prices.
4. Fundamentals (reused if younger than `FUNDAMENTALS_TTL_HOURS`) →
   `FundamentalSnapshot`; sector/industry masters filled from provider classification.
5. Technical/risk metrics (+ informational ML signal) → `MarketSnapshot`
   (computed in worker threads, written sequentially).
6. Industry PE medians from real peer snapshots.
7. FQVF + StockAI Score → `StockAnalysisResult`; `Company.data_status` updated.
8. `EngineRun` finalised with counts and errors.

## Databases

- `storage/app.db` (or `DATABASE_URL`): users, auth, masters, snapshots,
  engine runs, analysis results, migrated recommendation history, audit log.
  Schema managed by Alembic (`alembic upgrade head`).
- `storage/tracker.db`: legacy recommendation-validation store used by the
  legacy engine, Performance and Intelligence. Rows are tagged by
  `engine_version`; `NULL`/`v1.0` rows predate the temporal-integrity fix.

## Mobile app

`StockAIPro.Mobile.Core` (net10.0 library: models, API clients, auth/session
services, configuration — unit tested) and `StockAIPro.Mobile` (MAUI Blazor
Hybrid UI, platform services such as SecureStorage token store). See
docs/MOBILE_SETUP.md.
