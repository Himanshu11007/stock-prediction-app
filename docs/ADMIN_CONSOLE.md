# Admin / Master Control Console

`admin_console/app.py` (Streamlit) — run with:

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
| Engine Runs | `/admin/engine-runs` | start a run (symbols / limit / ML / refresh), list runs, errors per run |
| Audit Logs | `/admin/audit-logs` | every admin mutation |
| Engine Versions | `/admin/engine-versions` | current versions, results per version, recommendation rows per engine version |
| Application Configuration | `/admin/config` | `app.features`, `app.disclaimer`, `app.announcement`, `top_picks.limit`, `ranking.rules` (validated, audited) |

## What administrators control on the mobile app

Active/tradable stocks and analysis eligibility, sector outlooks (FQVF 18),
ranking weights and Top Picks rules, the number of candidates shown, feature
visibility (`app.features`), the disclaimer and a home-screen announcement.
