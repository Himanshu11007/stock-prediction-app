# API Reference (summary)

Full, generated specification: `GET /openapi.json`, interactive at `/docs`
and `/redoc`. Base path: `/api/v1`. Authentication: `Authorization: Bearer
<access token>` from `/auth/login` (OAuth2 password form), `/auth/google`,
`/auth/apple` or `/auth/otp/verify`; refresh via `/auth/refresh` (rotating).

Response shapes: business routes return `{"success": true, "data": …,
"message": "…"}`; errors return `{"success": false, "error": …, "details": …}`
(400/404/500; 500 carries only a reference id) or FastAPI's `{"detail": …}`
(401/403/404/409 raised by routes).

## Mobile endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness |
| GET | `/app/config` | none | feature flags, disclaimer, announcement, Top Picks limit, engine versions |
| POST | `/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout` | — | password auth, refresh rotation |
| POST | `/auth/google`, `/auth/apple`, `/auth/otp/request`, `/auth/otp/verify` | — | SSO / OTP |
| GET | `/auth/me` | user | profile and roles |
| GET/POST | `/auth/sessions`, `/auth/sessions/revoke`, `/auth/sessions/revoke-all`, `/auth/devices/pin-enabled` | user | device sessions |
| GET | `/stocks?search=&limit=&offset=` | user | Stock Master search (active stocks) |
| GET | `/stocks/{symbol}` | user | one stock |
| GET | `/stocks/{symbol}/analysis` | user | StockAI Score + components + reasons, 18 FQVF checks, market/technical, ML (informational), freshness, engine |
| GET | `/stocks/{symbol}/fqvf` | user | FQVF only |
| GET | `/stocks/{symbol}/ranking` | user | score, components, positives, risks |
| POST | `/stocks/{symbol}/analysis/refresh` | user | on-demand analysis (10–30 s); returns existing result if < 60 min old; 409 if busy |
| GET | `/top-picks?limit=` | user | Top Investment Candidates (latest completed run); empty list before the first run |
| GET | `/market/regime` | user | NIFTY 50 regime |
| GET | `/fqvf/reference`, `/ranking/reference` | user | framework / methodology reference |
| GET/POST/DELETE | `/watchlist`, `/watchlist/{id}` | user | the caller's own watchlist only |
| GET | `/performance/*`, `/intelligence/report` | user | historical analytics of the legacy recommendation engine |
| POST | `/analyze-stock`, `/top-picks/start`, `GET /top-picks/status/{id}`, `/top-picks/result/{id}` | user | legacy ML/confluence signal engine (kept for compatibility) |

## Admin endpoints (ADMIN role)

`/admin/dashboard`, `/admin/users*`, `/admin/roles`, `/admin/stocks*`
(GET/PATCH/POST), `/admin/sectors*`, `/admin/industries*`,
`/admin/fundamentals`, `/admin/market-data`, `/admin/technical`,
`/admin/valuation`, `/admin/market-regime`, `/admin/fqvf/reference`,
`/admin/config` (GET), `/admin/config/{key}` (PUT), `/admin/ranking/config`,
`/admin/analysis-results`, `/admin/engine-runs` (GET/POST),
`/admin/engine-runs/{run_id}`, `/admin/data-health`, `/admin/api-health`,
`/admin/engine-versions`, `/admin/stock-master/summary`,
`/admin/ranking-tracking/snapshots`, `/admin/ranking-tracking/summary`,
`POST /admin/ranking-tracking/outcomes` (prospective tracking, docs/RANKING_VALIDATION_V1.md),
`/admin/recommendations`, `/admin/watchlist`, `/admin/audit-logs`,
`/logs/latest`, `DELETE /logs/clear`, `POST /tracker/save`, `POST /tracker/validate-old`.

Authorization is covered by `tests/test_product_api.py`,
`tests/test_admin.py` and `tests/test_business_endpoint_auth.py`.
