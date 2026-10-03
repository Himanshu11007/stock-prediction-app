# Data Health and Engine Runs

## Engine runs (`engine_runs/service.py`)

An **engine run** fetches data and computes FQVF + StockAI Score for a set of
stocks. Two kinds:

| Kind | Started by | Scope | Concurrency |
|---|---|---|---|
| `RANKING` | Admin (`POST /admin/engine-runs`, Engine Runs page) | Large/Mid/Small Cap universe of active, tradable, analysis-enabled stocks (or a symbol list / limit) | one at a time: in-process lock + RUNNING-row check → 409 |
| `SINGLE` | Any user (`POST /stocks/{symbol}/analysis/refresh`) | one stock, ranked against the latest full run's universe | up to 2 concurrently; not blocked by a RANKING run |

Every run records `run_id`, `kind`, `status` (`RUNNING`, `COMPLETED`,
`COMPLETED_WITH_ERRORS`, `FAILED`), `started_at`, `finished_at`,
`triggered_by`, `engine_version`, `fqvf_version`, the configuration used
(weights, rules, options), and counts:

- `total` / `processed` — stocks in scope;
- `succeeded` — a StockAI Score was produced;
- `skipped` — processed but not scorable (missing data; never fabricated);
- `failed` — a processing stage raised for that stock;
- `errors` — list of `{symbol, stage, error}`.

**Failure isolation:** each per-stock stage (fundamentals, classification,
technical, market, FQVF) is wrapped; a failure is recorded and the run
continues. A provider batch failure loses only that batch's prices. Only an
infrastructure failure marks the whole run `FAILED` (with the error).
Workers compute; all database writes happen sequentially in one session.
A `RUNNING` row older than 3 hours is marked `FAILED` (abandoned).

Fundamentals younger than `FUNDAMENTALS_TTL_HOURS` (24) are reused unless
`refresh_fundamentals` is set.

Scheduling: production runs are started by `scripts/scheduled_jobs.py ranking`
from the operating-system scheduler (calendar- and data-aware, idempotent);
see docs/SCHEDULING.md. Administrators can still start a run manually.
After every background ranking run the notification engine compares it with
the previous full run (docs/NOTIFICATIONS.md).

Prospective tracking: every completed RANKING run also writes an append-only
`ranking_snapshots` row per stock (score, rank, FQVF summary, components,
freshness, regime, reference price). Call `POST /api/v1/admin/ranking-tracking/outcomes`
periodically (e.g. weekly) to insert realised 1M/3M/6M/12M outcomes once each
horizon has elapsed; see docs/RANKING_VALIDATION_V1.md section 28.

## Data health (`data_health/service.py`, `GET /admin/data-health`)

Reads stored snapshots only; never fetches or fills. Findings per stock:

| Category | Rule |
|---|---|
| `missing_market_data` | no snapshot, or provider returned no history |
| `stale_market_data` | last bar older than `MARKET_DATA_STALE_DAYS` (4) |
| `invalid_market_values` | NaN/∞ OHLC, non-positive close |
| `zero_volume` | trading sessions (holiday placeholders excluded) with zero volume |
| `missing_history` | gaps > 7 days between sessions, or < 250 daily bars in 2 years |
| `missing_fundamentals` | no snapshot or provider returned nothing |
| `stale_fundamentals` | fetched more than `FUNDAMENTALS_STALE_DAYS` (7) ago |
| `partial_fundamentals` | summary or statements missing |
| `invalid_fundamental_values` | provider values that were not finite numbers (recorded in `issues`) |
| `provider_failures` | fetch errors |
| `inactive_stocks`, `non_tradable_stocks` | Stock Master flags |

Plus `news_timestamps: UNAVAILABLE` (news headlines have no publication
timestamps; news is not an input to FQVF or the score) and the latest run.
`Company.data_status` (`OK`, `PARTIAL`, `STALE`, `UNAVAILABLE`) and
`data_status_reason` are updated by every run.

`GET /admin/api-health`: database connectivity, latest run age (STALE after
48 h), running runs, last failed run, market regime, versions.

Invalid data is never turned into 0, 50%, a fake price, a fake fundamental, a
fake prediction or a fake recommendation: it is stored as NULL with a reason
and surfaces as `NOT_AVAILABLE`, an unavailable component or an
ineligibility reason.
