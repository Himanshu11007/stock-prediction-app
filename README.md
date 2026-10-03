# StockAI Pro — Backend

StockAI Pro is a transparent stock-analysis and ranking system for NSE-listed
stocks. The backend (this repository) is the single source of truth: it
ingests market and fundamental data, evaluates every stock with the
**Fundamental Quality & Value Framework (FQVF)**, computes the **StockAI
Score**, publishes the **Top Investment Candidates**, and serves everything to
the .NET MAUI mobile app ([StockAIPro.Mobile](https://github.com/Himanshu11007/StockAIPro.Mobile))
and the Streamlit admin console.

> StockAI Pro provides research and analysis, not investment advice. Scores
> rank stocks on the data available; they are not predictions or guarantees of
> returns. The ML direction model showed **no demonstrated out-of-sample
> predictive skill** (docs/ML_EXPERIMENT_REGISTRY.md) and is informational only.

```
Yahoo Finance ──► fundamentals/provider.py ──► FundamentalSnapshot / MarketSnapshot
                                                         │
             engine_runs/service.py (controlled runs, failure isolation)
                    │                     │                    │
                fqvf/ (18 checks)   ranking/ (StockAI Score)  data_health/
                    └─────────► StockAnalysisResult ◄─────────┘
                                         │
                          FastAPI /api/v1 (api/main.py)
                         ┌───────────────┴───────────────┐
              MAUI mobile app (Android/iOS)     admin_console/ (Streamlit, ADMIN)
```

## Quick start (local development)

Requirements: Python 3.11, internet access (Yahoo Finance).

```bash
python -m venv venv && venv\Scripts\activate          # Windows (source venv/bin/activate elsewhere)
pip install -r requirements.txt
alembic upgrade head                                   # create / migrate storage/app.db
python scripts/seed_admin.py                           # create the first ADMIN user (prompts)
uvicorn api.main:app --host 0.0.0.0 --port 8000        # API: http://127.0.0.1:8000/api/v1
```

- OpenAPI: http://127.0.0.1:8000/docs and http://127.0.0.1:8000/redoc
- Admin console: `streamlit run admin_console/app.py` (sign in as an ADMIN user)
- First analysis run: Admin console → **Engine Runs** → *Start run* (or
  `POST /api/v1/admin/engine-runs`). A full run over the 301-stock
  Large/Mid/Small Cap universe takes several minutes; Top Picks are empty
  (not an error) until one completes.
- Legacy Streamlit UI (technical/ML signal tools): `streamlit run app.py`

## Documentation

| Topic | Document |
|---|---|
| System architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Fundamental Quality & Value Framework (18 checks, thresholds, formulas) | [docs/FQVF.md](docs/FQVF.md) |
| StockAI Score / Top Investment Candidates methodology | [docs/RANKING_METHODOLOGY.md](docs/RANKING_METHODOLOGY.md) |
| Admin / master controls | [docs/ADMIN_CONSOLE.md](docs/ADMIN_CONSOLE.md) |
| API reference (mobile + admin endpoints) | [docs/API.md](docs/API.md) |
| Data health and engine runs | [docs/DATA_HEALTH_AND_ENGINE_RUNS.md](docs/DATA_HEALTH_AND_ENGINE_RUNS.md) |
| Mobile app setup (emulator, device, production URL, builds) | [docs/MOBILE_SETUP.md](docs/MOBILE_SETUP.md) |
| Local development, production configuration, testing | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Authentication (JWT, refresh rotation, Google/Apple, OTP, PIN) | [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md) |
| ML research history (what the model can and cannot do) | [docs/ML_EXPERIMENT_REGISTRY.md](docs/ML_EXPERIMENT_REGISTRY.md) |
| Temporal integrity of predictions | [docs/PRODUCTION_TEMPORAL_INTEGRITY.md](docs/PRODUCTION_TEMPORAL_INTEGRITY.md) |

## Tests

```bash
python -m pytest -q                                    # full backend suite
python scripts/e2e_backend.py --user U --user-password P --admin A --admin-password Q   # against a running server
```

## Known limitations (summary)

- Fundamentals come from Yahoo Finance, which supplies ~4 fiscal years of
  statements for most NSE stocks: "5-Year EPS Progression" is evaluated on 4
  years (and says so), "Historical Average PE (5–7 years)" is usually
  NOT_AVAILABLE, and EPS history is not restated for splits/bonus issues
  (detected and reported, not guessed).
- Sector Outlook (FQVF check 18) has no data source: an administrator sets it
  per sector in the Sector Master; until then it is NOT_AVAILABLE.
- News has no point-in-time timestamps and is not an input to FQVF or the score.
- The ML direction signal has no demonstrated skill and carries weight 0.
- See each document's limitations section for details.
