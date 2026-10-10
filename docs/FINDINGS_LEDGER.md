# StockLens findings ledger (October 2026)

Statuses:
- VERIFIED: the defect or fact is confirmed;
- IMPLEMENTED / TESTED: a fix exists and is covered by tests;
- BLOCKED: needs data, access or a contract;
- DEFERRED: postponed;
- REQUIRES_APPROVAL: production, cost or v1 change;
- NOT_REPRODUCIBLE.

**Branches:** backend `feature/phase0-safety` (P0) and `feature/prediction-v2`
(stacked on it); web `feature/prediction-v2-ui`. Everything is local and
**not pushed**.

Test IDs:
- backend: `tests/<file>`;
- web: `StockLens.Web.Tests/<file>`.

## Data, universe and ranking (v1 is frozen)

| ID | Finding | Evidence | Status | Fix / test | Remaining |
|---|---|---|---|---|---|
| F-01 | Companies created in Admin are never ranked in full v1 runs (`admin/service.py create_stock` writes only `companies`; `engine_runs.resolve_universe` needs `stock_universe`) | code | VERIFIED; IMPLEMENTED (visibility, v2 path); TESTED | `GET /admin/universe-health` lists active companies outside v1; v2 universe accepts them (`test_prediction_api::test_v2_universe_admin_changes_never_touch_v1`) | Adding to the **v1** universe changes frozen v1: REQUIRES_APPROVAL |
| F-02 | CPCL and MRPL are active companies outside the v1 universe; Kanohar and Moneyview are not in the master list | local DB, `data/nse_stocks.csv` | VERIFIED; IMPLEMENTED (v2 add) | as F-01 | NIFTY 500 / F&O constituent lists: BLOCKED (no licensed list source); import via `POST /admin/v2-universe` |
| F-03 | 18 of 301 v1 universe symbols return no Yahoo data (renamed or delisted: TATAMOTORS, ZOMATO, LTIM, ISEC, GUJGASLTD, …) | yfinance errors; 3 Oct run skipped 17 | VERIFIED; IMPLEMENTED (report); TESTED | `universe-health.v1_without_market_data` | Renaming or removing v1 members: REQUIRES_APPROVAL |
| F-04 | Two processes could start overlapping RANKING runs (check-then-insert; process-local lock only) | race test: **19/20** overlaps | IMPLEMENTED; TESTED | migration `4e6773a9e7db` + `create_run` mapping; **0/20** in 20 real two-process trials (`test_engine_run_concurrency`), migration clean / duplicate / idempotent (`test_migration_engine_run_guard`) | Production deploy: REQUIRES_APPROVAL; see B-01 |
| F-05 | No timestamped news or corporate events (`news/api.py` returns headline strings only) | code | IMPLEMENTED (foundation); TESTED | `events/ingest.py`, `market_events`, `event_classifications` (`test_prediction_v2::test_event_*`) | Provider: BLOCKED (licensing) |
| F-06 | No 1/3/5-day returns, gap, abnormal volume, close location or relative strength | `ranking/technical.py` | IMPLEMENTED; TESTED | `prediction_v2/features.py` (formula, cutoff and missing-data tests) | — |
| F-07 | `market.holidays` is empty; NSE was closed on 2 Oct 2026 | local DB, prices | VERIFIED; IMPLEMENTED (calendar-aware jobs, flag on every v2 snapshot); TESTED | `test_prediction_jobs::test_configured_holiday_is_respected` | Seeding production config: REQUIRES_APPROVAL (dates from the official NSE circular) |
| F-08 | Sector outlook unset for every stock (FQVF check 18 NOT_AVAILABLE) | validation CSV | VERIFIED | v1 unchanged (missing is never favourable); v2 uses data-driven sector relative strength instead | Admin data entry: DEFERRED |
| F-09 | The 3 Oct live v1 run has no `ranking_snapshots` rows in the local DB, so the reference price falls back to market snapshots. Cause not established | local DB | VERIFIED (local); IMPLEMENTED (labelling); TESTED | `reference_price_source` in Top Picks (`test_top_picks_freshness`) | Production state: BLOCKED (no read-only access) |
| F-10 | Intraday data (Yahoo 1/5/15-minute bars) is delayed and unlicensed | provider | VERIFIED; IMPLEMENTED (gate) | `TODAY_CONFIRMED` SKIPPED unless `PREDICTION_V2_INTRADAY_ENABLED` (`test_confirmed_snapshot_*`) | Licensed feed: BLOCKED |
| F-11 | FMCG index (`^CNXFMCG`) has no usable Yahoo history | download | VERIFIED; IMPLEMENTED | sector peer baskets (≥3 peers) for `sector_rs_5` and outcome sector returns | — |
| F-12 | The legacy Streamlit "Top Picks" tab is a different engine (legacy ML scanner) and can be confused with v1 Top Candidates | `app.py` | VERIFIED | — | Relabel: DEFERRED (Streamlit change, needs approval) |
| F-13 | Production database not inspectable (no read-only credentials) | — | BLOCKED | — | Read-only role or dashboard screenshots |
| X-01 | FINOPB showed a 242× volume spike (probable data error) | universe scan | VERIFIED; IMPLEMENTED | `SUSPECT_VOLUME_SPIKE` gives NO_CALL (`test_stale_suspect_volume_and_price_band_flags`) | — |

## Security and operations

| ID | Finding | Status | Fix / evidence | Remaining |
|---|---|---|---|---|
| S-01 | PostgreSQL password exposed in a conversation screenshot | REQUIRES_APPROVAL | Rotation procedure: `docs/OPERATIONS_AND_SECURITY.md` §1.1 | Owner action |
| S-02 | NewsAPI key in public Git history (`bcf4da6`, `c9ca6e2`…) | REQUIRES_APPROVAL | Revoke / replace / untrack / history: §1.2 | Owner revokes; untracking after that, with approval; no history rewrite |
| S-03 | `/docs`, `/redoc`, `/openapi.json` public in production | IMPLEMENTED; TESTED | `API_DOCS_ENABLED`; 404 in production-mode subprocess tests (`test_api_security`) | Deploy: REQUIRES_APPROVAL |
| S-04 | No HSTS / nosniff / frame / referrer headers on the API | IMPLEMENTED; TESTED | middleware (`test_api_security`) | Deploy |
| S-05 | Free PostgreSQL: expires about 30 days after creation, no backups | VERIFIED (blueprint); REQUIRES_APPROVAL | `pg_dump` procedure §2; upgrade advice | Cost decision |
| O-01 | No scheduler is active (Render Free has no cron; 0 GitHub workflows) | VERIFIED; IMPLEMENTED (templates); TESTED (jobs) | `render.yaml` 9 cron templates; `deploy/github-actions/prediction-jobs.yml` (inactive, actionlint clean) | Activation: REQUIRES_APPROVAL |
| O-02 | Missed runs were invisible | IMPLEMENTED; TESTED | `prediction_monitor` (exit 1), `GET /admin/prediction-runs` (`test_prediction_jobs::test_monitor_*`) | — |

## Prediction Engine v2 (shadow)

| ID | Item | Status | Tests |
|---|---|---|---|
| V-01 | Additive v2 schema (10 tables, uniqueness for immutability and idempotency) | IMPLEMENTED; TESTED | upgrade / downgrade / upgrade + `alembic check` clean |
| V-02 | `TODAY_PREOPEN` / `TODAY_CONFIRMED` / `TOMORROW_EOD` snapshots (cutoff, idempotency, one RUNNING per type, quality gate, immutability, retry after failure) | IMPLEMENTED; TESTED | `test_prediction_v2`, `test_prediction_jobs` |
| V-03 | Baseline rules (UP/DOWN/NEUTRAL/NO_CALL, levels, confidence null) | IMPLEMENTED; TESTED | `test_rules_*` |
| V-04 | 1/3/5-session outcomes (direction- and cost-adjusted, MFE/MAE, idempotent, waits for final bars) | IMPLEMENTED; TESTED | `test_outcomes_*` |
| V-05 | Shadow exit engine (11 reason codes, legal transitions, gaps, missing bars, no notifications) | IMPLEMENTED; TESTED | `test_exit_*` |
| V-06 | Performance aggregation (Wilson intervals, small-sample suppression, promotion gate) | IMPLEMENTED; TESTED | `test_wilson_*`, web `Performance_hides_statistics_for_small_samples` |
| V-07 | Chronological backtest (splits, embargo, costs, 5 baselines, leak test) | IMPLEMENTED; TESTED | `test_backtest_*`; real-data run: `docs/research/PREDICTION_V2_BACKTEST_2026-10.md` |
| V-08 | v2 API (admin-only shadow, one snapshot per response, freshness) | IMPLEMENTED; TESTED | `test_prediction_api` (32) |
| V-09 | Web Labs views (Today, Tomorrow, detail, exits, performance) | IMPLEMENTED; TESTED | web `PredictionPagesTests` (15) |
| V-10 | Event-aware and catalyst setups (re-rating, surprise, business update) | BLOCKED | needs an event provider and historical consensus |
| V-11 | Calibrated confidence | DEFERRED | after sufficient holdout outcomes |

## Top Candidates consistency

| ID | Finding | Status | Evidence |
|---|---|---|---|
| T-01 | Ranking score versus refreshed current price: already separated in `85c7eab` (frozen reference price, separate current price and timestamps) | VERIFIED (no defect) | existing tests |
| T-02 | No signal when the stored ranking is stale, and no label when the reference price is a fallback | IMPLEMENTED; TESTED | backend `ranking_freshness`, `reference_price_source`, `run_id` (additive; mobile contract still met); web banner + note (`Top_candidates_flags_a_stale_ranking_*`) |

## October 2026 case review (`docs/research/WEEKLY_CASES_2026-10.md`)

| ID | Case | Status |
|---|---|---|
| C-01 | Canara Bank: rank 1 of 156, score 78.3 and all components match the 30 Sep reproduction; the live 3 Oct run also ranked it 1 (77.0). Top-10 in **24 of 39** snapshots (not 24/28). No qualifying short-term move within 5 sessions | VERIFIED (data, history); production snapshot BLOCKED |
| C-02 | ITC, Trent, Titan, CPCL: price and volume moves verified; catalyst publication and ingestion times unavailable | Prices VERIFIED; timing BLOCKED |
| C-03 | Formal directional false positives and false negatives for the week cannot be computed (no frozen directional predictions existed) | VERIFIED limitation |

## Branches and integration

| ID | Finding | Status |
|---|---|---|
| B-01 | Remote `origin/claude/stocklens-auth-security-hardening-qczhnr` (d8d3c6e, 5 Oct, 31 files) adds an **equivalent** engine-run guard (`c3f9d8e0a4b2`, after `b7e4c2a91d35`) whose migration **automatically marks duplicate RUNNING runs FAILED** (data change without approval). Integrating both branches gives two Alembic heads | VERIFIED; integration decision REQUIRES_APPROVAL. Recommendation: keep the non-destructive `4e6773a9e7db` (idempotent if the index exists); drop or rewrite `c3f9d8e0a4b2` on that branch; add an Alembic merge revision |
| B-02 | Local `agents/*` branches (5) are fully contained in `main`; 2 of their worktrees have 1 uncommitted file each | VERIFIED; untouched (no deletion) |

## Discovered and fixed during implementation

| ID | Issue | Fix |
|---|---|---|
| D-01 | A blanket "one RUNNING run" index would have blocked concurrent SINGLE-stock analyses | Index scoped to `kind = 'RANKING'` |
| D-02 | A FAILED v2 snapshot blocked the same day's retry | The failed run releases its key; attempts capped by the slot |
| D-03 | Exit engine lacked the event-day levels; `from_state` was stale in replay | Fixed (`exits`, `outcomes`) |
| D-04 | `summarize()` crashed on empty MFE lists; `sector_rs_5` key missing below 3 peers | Fixed (tests) |
| D-05 | ORM instance expired after an audit commit, so the API returned `{}` | Serialize before commit (`test_admin_event_review_keeps_the_original`) |
