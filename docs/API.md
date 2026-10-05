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
| POST | `/auth/forgot-password` `{email}` → 202 `{message}` (identical for every email) | — | emails a reset link to the StockLens web app |
| POST | `/auth/reset-password` `{token, new_password}` → 200 `{message}`; 400 invalid/expired/used link, 422 password policy, 429 | — | sets the password, ends every session (docs/AUTH_HARDENING.md) |
| GET | `/auth/me` | user | profile and roles |
| GET/POST | `/auth/sessions`, `/auth/sessions/revoke`, `/auth/sessions/revoke-all`, `/auth/devices/pin-enabled` | user | device sessions |
| GET | `/stocks?search=&limit=&offset=` | user | Stock Master search (active stocks) |
| GET | `/stocks/{symbol}` | user | one stock |
| GET | `/stocks/{symbol}/analysis` | user | StockLens Score + components + reasons, 18 FQVF checks, market/technical, ML (informational), freshness, engine; `reference_price` (close the analysis used), `current_price` block (latest available price, see below) and `ranking_date` |
| GET | `/stocks/{symbol}/fqvf` | user | FQVF only |
| GET | `/stocks/{symbol}/ranking` | user | score, components, positives, risks |
| POST | `/stocks/{symbol}/analysis/refresh` | user | on-demand analysis (10–30 s); returns existing result if < 60 min old; 409 if busy |
| GET | `/top-picks?limit=` | user | Top Investment Candidates (latest completed run, read from the database, never recalculated); empty list before the first run. Fields below |
| GET | `/market/regime` | user | NIFTY 50 regime |
| GET | `/fqvf/reference`, `/ranking/reference` | user | framework / methodology reference |
| GET/POST/DELETE | `/watchlist`, `/watchlist/{id}` | user | the caller's own watchlist only |
| GET | `/performance/*`, `/intelligence/report` | user | historical analytics of the legacy recommendation engine |
| POST | `/analyze-stock`, `/top-picks/start`, `GET /top-picks/status/{id}`, `/top-picks/result/{id}` | user | legacy ML/confluence signal engine (kept for compatibility) |

### Top Picks fields

Response `data`:
- `items`, `total_eligible`, `limit`;
- `run`: `run_id`, `status`, `finished_at`, `stocks_analysed`,
  `engine_version`, `fqvf_version`, `ranking_date`;
- `ranking_date`, `market_regime`;
- `market_status`: the `/market/status` object;
- `disclaimer`.

Each item has these fields (fields are only ever added, never renamed):

| Field | Meaning |
|---|---|
| `rank`, `symbol`, `name`, `sector`, `industry` | Candidate |
| `stockai_score`, `stocklens_score` | StockLens Score (same value; `stockai_score` kept for older clients) |
| `fqvf_score`, `fqvf_summary`, `fqvf_counts` | FQVF result (18 checks) |
| `positives`, `risks`, `labels`, `components`, `score_coverage` | Explanation |
| `engine_version`, `computed_at`, `freshness`, `freshness_status` | Provenance of the analysis |
| `ranking_date` | Trading date of the ranking run's market data (NIFTY 50 bar) |
| `reference_price`, `reference_price_as_of` | Close the ranking used; frozen with the run (`ranking_snapshots`) |
| `current_price`, `current_price_as_of`, `current_price_status`, `current_price_source`, `current_price_date` | Latest available market price: `LAST_CLOSE`, `DELAYED_INTRADAY` (delayed, never "live"), `STALE` or `NOT_AVAILABLE` (price null). Never the reference price |
| `market_status` | NSE session status code at the time of the request |

Opening Top Picks never recalculates the ranking. It may refresh the
current prices of the displayed stocks if they are stale (docs/SCHEDULING.md).

## Notifications, devices, reports (authenticated user)

`GET /notifications` (`unread_only`, `limit`, `offset`), `GET /notifications/unread-count`,
`POST /notifications/{id}/read`, `POST /notifications/read-all`,
`GET|PUT /notifications/preferences`, `GET|POST /devices`, `DELETE /devices/{device_id}`,
`GET|POST /feedback`, `POST /account/deletion-request`,
`GET /watchlist/overview`, `PUT /watchlist/{item_id}/alerts`,
`GET /market/status`, `GET /performance/overview`, `GET /intelligence/overview`.
Notification payloads carry `route` (in-app deep link, e.g. `/stock/TCS.NS`,
`/top-picks`). See docs/NOTIFICATIONS.md.

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
`/admin/scheduled-jobs` (GET: daily job history, latest published ranking, its notifications, current-price freshness),
`POST /admin/scheduled-jobs/ranking/run` (start the calendar-aware daily ranking job now; 202),
`POST /admin/ranking-tracking/outcomes` (prospective tracking, docs/RANKING_VALIDATION_V1.md),
`/admin/recommendations`, `/admin/watchlist`, `/admin/audit-logs`,
`/admin/notifications/stats|runs|recent` (GET), `/admin/notifications/process-run|daily-summary|dispatch|test` (POST),
`/admin/feedback` (GET), `/admin/feedback/{id}` (PATCH),
`/logs/latest`, `DELETE /logs/clear`, `POST /tracker/save`, `POST /tracker/validate-old`.

Authorization is covered by `tests/test_product_api.py`,
`tests/test_admin.py` and `tests/test_business_endpoint_auth.py`.
