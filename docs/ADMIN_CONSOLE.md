# Admin / Master Control Console

## Where to open it

**In the deployed StockLens Streamlit app** (no separate deployment):

https://ai-stock-predict-dashboard.streamlit.app/ → sidebar **Administration**
→ **Admin Console** → sign in with an ADMIN account → admin pages appear in
the sidebar. Daily ranking: **Engine Runs** → **Daily schedule** → **Run the
daily ranking job now (calendar-aware)**. Direct link to the sign-in page:
https://ai-stock-predict-dashboard.streamlit.app/admin

How it fits together:

```
Streamlit app (app.py, st.navigation)
  ├─ StockLens: Dashboard                (the existing dashboard tabs)
  └─ Administration                      (admin_console/embedded.py)
       ├─ Admin Console  (sign-in, shown to everyone)
       └─ after an ADMIN sign-in: the 25 console pages + Sign out
            ↓ REST, the admin's token (admin_console/client.py)
  FastAPI backend (Render) → require_admin on every /admin/* request
            ↓
  PostgreSQL, ranking engine, daily ranking job, notifications
```

- Visitors who are not signed in see only the **Admin Console** sign-in
  entry. The admin page URLs are not registered for them.
- Sign-in is the backend's own (`POST /auth/login`, `GET /auth/me`), accepted
  only for the **ADMIN** role. Every admin page then calls the API with that
  user's token; the backend authorizes and audits every request.
- The Streamlit app holds no database URL, JWT secret or provider
  credential. Its only setting is the API address: `STOCKAI_API_URL`
  (Streamlit Community Cloud: app → **Settings** → **Secrets**,
  `STOCKAI_API_URL = "https://stocklens-api-otrc.onrender.com/api/v1"`).
  Without it the production API above is used. The sign-in form does not let
  users change it.
- While a run is in progress, Engine Runs refreshes every 60 s. This also
  keeps the Render Free backend awake until the run finishes.
- Tested in `tests/test_streamlit_admin_navigation.py`.

## Standalone (local / any API)

`admin_console/app.py` also runs on its own (e.g. against a local API):

```bash
set STOCKAI_API_URL=http://127.0.0.1:8000/api/v1     # optional (default shown)
streamlit run admin_console/app.py
```

Sign in with an account that has the **ADMIN** role (`python
scripts/seed_admin.py` creates the first one). The console talks **only** to
the REST API with the admin's JWT (`admin_console/client.py`): it has no
database access, so authorization (`require_admin`) and audit logging are
enforced by the backend for every action. Tokens are kept in memory and
refreshed on 401. Tested end to end in `tests/test_admin_console.py`.

## Pages

| Page | API | Actions |
|---|---|---|
| Admin Dashboard | `/admin/dashboard`, `/admin/api-health` | counts, system status |
| Stock Master | `/admin/stocks`, `/admin/stock-master/summary` | search; edit active / analysis-enabled / **tradable** / name / sector / industry; create stock |
| Sector Master | `/admin/sectors` | set/clear **sector outlook** (FQVF check 18) with notes; active flag |
| Industry Master | `/admin/industries` | active flag (industries are created from provider classifications) |
| Fundamental Data | `/admin/fundamentals` | latest snapshot per stock, or one stock's history incl. annual series |
| Market Data | `/admin/market-data` | latest price snapshot and data issues |
| Valuation Data | `/admin/valuation` | FQVF valuation checks (6–11, 14) for the latest run |
| Technical / Market Signals | `/admin/technical` | returns, volatility, drawdown, RSI, trend, regime, ML signal (informational) |
| Market Regime | `/admin/market-regime` | NIFTY 50 regime history |
| FQVF Reference | `/admin/fqvf/reference` | read-only: the framework is fixed |
| Ranking Configuration | `/admin/ranking/config`, `PUT /admin/config/ranking.weights` | component weights (validated, audited) |
| Top Picks | `/admin/analysis-results` | every stock in a run incl. ineligible ones with reasons |
| Users | `/admin/users`, `/activate`, `/deactivate` | cannot deactivate yourself or the last admin |
| Roles | `/admin/roles`, `/admin/users/{id}/roles` | assign / remove |
| Watchlist Administration | `/admin/watchlist` | read-only (admins cannot edit users' watchlists) |
| Recommendation History | `/admin/recommendations` | read-only; engine version shown |
| Data Validation | `/admin/data-health` | findings per stock and category |
| Data Health | `/admin/data-health` | summary counts, news-timestamp status, latest run |
| API Health | `/admin/api-health` | database, engine freshness, regime, versions |
| Engine Runs | `/admin/scheduled-jobs`, `/admin/engine-runs` | **Daily schedule**: latest published ranking (ranking date, counts, engine version), its notifications, current-price freshness, scheduled job history, and "Run the daily ranking job now". **Manual run**: start a run (symbols / limit / ML / refresh), list runs, errors per run |
| Audit Logs | `/admin/audit-logs` | every admin mutation |
| Engine Versions | `/admin/engine-versions` | current versions, results per version, recommendation rows per engine version |
| Application Configuration | `/admin/config` | `app.features`, `app.disclaimer`, `app.announcement`, `top_picks.limit`, `ranking.rules` (validated, audited) |

## What administrators control on the mobile app

Active/tradable stocks and analysis eligibility, sector outlooks (FQVF 18),
ranking weights and Top Picks rules, the number of candidates shown, feature
visibility (`app.features`), the disclaimer and a home-screen announcement.
