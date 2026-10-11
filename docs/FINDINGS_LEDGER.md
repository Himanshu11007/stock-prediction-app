# StockLens findings ledger (October 2026, review edition)

**Statuses:**

| Status | Meaning |
|---|---|
| VERIFIED | the defect or fact is confirmed |
| IMPLEMENTED | code exists |
| TESTED | automated tests cover it |
| BLOCKED | needs data, access or a contract that does not exist |
| DEFERRED | postponed on purpose |
| REQUIRES_APPROVAL | a production, cost or v1 change, or an owner action |
| NOT_REPRODUCIBLE | original evidence is unavailable |

"Interface only" means a schema or extension point exists **without** a
working end-to-end feature; such items are never marked complete.

**Review branches** (backend):
- `review/phase0-safety`: Phase 0 only.
- `review/prediction-v2`: Phase 0 + v2 + scheduler + research.
- `review/stocklens-integration`: `review/prediction-v2` plus the reconciled
  auth-hardening commit.

**Web:** `review/prediction-v2-ui`. **Mobile:** unchanged (`master`
`d24eaf5`). The requirement-by-requirement checklist is
`docs/REVIEW_CHECKLIST.md`.

## Data, universe and ranking (v1 is frozen)

| ID | Finding | Evidence | Status | Fix / test | Remaining |
|---|---|---|---|---|---|
| F-01 | Companies created in Admin are never ranked in full v1 runs (`admin/service.py create_stock` writes only `companies`; `engine_runs.resolve_universe` needs `stock_universe`) | code | VERIFIED; IMPLEMENTED (visibility, v2 path); TESTED | `GET /admin/universe-health`; v2 universe accepts them (`test_prediction_api::test_v2_universe_admin_changes_never_touch_v1`) | Adding to the **v1** universe changes frozen v1: REQUIRES_APPROVAL |
| F-02 | CPCL and MRPL are active companies outside the v1 universe; Kanohar and Moneyview are not in the master list | local DB, `data/nse_stocks.csv` | VERIFIED; IMPLEMENTED (v2 add) | as F-01 | Official NIFTY 500 / F&O list: BLOCKED (no licensed source) |
| F-03 | 17–18 v1 universe symbols return no Yahoo data (renamed or delisted: TATAMOTORS, ZOMATO, LTIM, ISEC, GUJGASLTD, …) | yfinance; `case_evidence_2026_10.json` lists 17 | VERIFIED; IMPLEMENTED (report); TESTED | `universe-health.v1_without_market_data` | Renaming or removing v1 members: REQUIRES_APPROVAL |
| F-04 | Two processes could start overlapping RANKING runs | race test: 19/20 overlaps before | IMPLEMENTED; TESTED | migration `4e6773a9e7db` + `create_run`; 0/20 in real two-process trials (`test_engine_run_concurrency`); migration clean / duplicate-abort / idempotent (`test_migration_engine_run_guard`) | Production migration: REQUIRES_APPROVAL. PostgreSQL run not done locally (P-01) |
| F-05 | No timestamped news or corporate events | code | IMPLEMENTED (**foundation only**); TESTED | `events/ingest.py`, `market_events`, `event_classifications` (`test_prediction_v2::test_event_*`) | **No provider connected**: BLOCKED (licensing). Catalyst setups do not exist |
| F-06 | No 1/3/5-day returns, gap, abnormal volume, close location or relative strength | code | IMPLEMENTED; TESTED | `prediction_v2/features.py` (formula, cutoff, missing-data tests) | Breadth, peer momentum and factor betas: DEFERRED (F-14) |
| F-07 | `market.holidays` is empty; NSE was closed on 2 Oct 2026 | local DB, prices | VERIFIED; IMPLEMENTED; TESTED | jobs are calendar-aware; the **scheduler refuses to run without the current year's calendar** (`test_scheduler::test_missing_holiday_calendar_fails_instead_of_running`) | Seeding: REQUIRES_APPROVAL (official NSE circular) |
| F-08 | Sector outlook unset for every stock (FQVF check 18 NOT_AVAILABLE) | validation CSV, live run | VERIFIED | v1 unchanged; v2 uses peer-relative strength | Admin data entry: DEFERRED |
| F-09 | The 3 Oct live run has 0 `ranking_snapshots` rows locally, so the reference price falls back to market snapshots; cause not established | local DB | VERIFIED (local); IMPLEMENTED (labelling); TESTED | `reference_price_source` (`test_top_picks_freshness`); web note | Production state: BLOCKED (F-13) |
| F-10 | Intraday data (Yahoo 1/5/15-minute) is delayed and unlicensed; no intraday adapter exists | provider, code | VERIFIED; IMPLEMENTED (gate); TESTED | `TODAY_CONFIRMED` refused unless `PREDICTION_V2_INTRADAY_ENABLED` **and** `vars.STOCKLENS_TODAY_CONFIRMED_ENABLED` (`test_scheduler::test_today_confirmed_*`, `test_confirmed_snapshot_*`) | Licensed feed + adapter: BLOCKED |
| F-11 | FMCG index (`^CNXFMCG`) has no usable Yahoo history | download | VERIFIED; IMPLEMENTED | sector peer baskets (≥3 peers) | — |
| F-12 | The legacy Streamlit "Top Picks" tab is a different engine (legacy ML scanner) | `app.py` | VERIFIED | — | Relabel: DEFERRED (Streamlit change) |
| F-13 | Production database not inspectable (no read-only credentials) | — | BLOCKED | — | Read-only role or dashboard screenshots (owner) |
| F-14 | Commodity / factor observations (Brent, cracks) have a table (`factor_observations`) but **no writer**; past-only betas not implemented | code | VERIFIED; **interface only** | — | DEFERRED: needs a factor data source decision |
| F-15 | The v2 universe must be seeded from v1 by an administrator (`POST /admin/v2-universe`, `universe.seed_from_v1`); no automatic seeding | code | VERIFIED; IMPLEMENTED (manual) | `test_universe_seed_is_a_copy_and_admin_adds_do_not_touch_v1` | Production seeding: REQUIRES_APPROVAL |
| X-01 | FINOPB showed a 242× volume spike (probable data error) | universe scan | VERIFIED; IMPLEMENTED; TESTED | `SUSPECT_VOLUME_SPIKE` → NO_CALL | — |

## Security and operations

| ID | Finding | Status | Fix / evidence | Owner action |
|---|---|---|---|---|
| S-01 | PostgreSQL password exposed in a conversation screenshot | REQUIRES_APPROVAL (owner) | Rotation procedure: `docs/OPERATIONS_AND_SECURITY.md` §1.1. Never used by this work | **Rotate the database password** (Render dashboard) |
| S-02 | NewsAPI key in public Git history (`bcf4da6`, `c9ca6e2`, …) | REQUIRES_APPROVAL (owner) | §1.2: revoke → replace → untrack `.streamlit/secrets.toml` → **no history rewrite** | **Revoke the key at newsapi.org**; approve untracking afterwards |
| S-03 | `/docs`, `/redoc`, `/openapi.json` public in production | IMPLEMENTED; TESTED | `API_DOCS_ENABLED`; 404 in production-mode subprocess tests (`test_api_security`) | Deploy: REQUIRES_APPROVAL |
| S-04 | No HSTS / nosniff / frame / referrer headers | IMPLEMENTED; TESTED | middleware (`test_api_security`) | Deploy |
| S-05 | Free PostgreSQL expires about 30 days after creation, no backups | VERIFIED (blueprint); REQUIRES_APPROVAL | `pg_dump` procedure §2; upgrade advice | **Decide plan / cost before expiry**; take a `pg_dump` now |
| S-06 | The scheduler needs a credential that is not a user login and not the database URL | IMPLEMENTED; TESTED | dedicated bearer token, SHA-256 digest on the server (`SCHEDULER_TOKEN_SHA256`), constant-time compare, user/admin JWTs rejected, 404 when unset, never printed (`test_scheduler::test_authentication_*`, `test_client_never_prints_the_token`) | Create token and secret: REQUIRES_APPROVAL |
| O-01 | No scheduler is active (Render Free has no cron; 0 registered GitHub workflows) | VERIFIED; IMPLEMENTED; TESTED; **inactive** | GitHub Actions → backend job interface: `.github/workflows/stocklens-*.yml`, `scheduling/remote.py`, `api/routes/scheduler.py`, `scripts/scheduler_client.py`, `docs/SCHEDULER.md` (`test_scheduler`, 34 tests; actionlint clean) | Activation: REQUIRES_APPROVAL (calendar, token, Render env var, merge, repository variable) |
| O-02 | Missed runs were invisible | IMPLEMENTED; TESTED | `SNAPSHOT_MONITOR` (ALERT = failed run, never success), also flags failed or stale outcome evaluation (`test_prediction_jobs::test_monitor_*`, `test_scheduler::test_monitor_*`) | — |
| O-03 | The v2 commit had added 5 **paid** Render cron entries to `render.yaml`; a blueprint sync after merge would have created billed services | VERIFIED; fixed | `render.yaml` restored to `main` (commit `c87c431`); superseded `deploy/github-actions/prediction-jobs.yml` removed. Test: every CLI job has exactly one scheduler (`test_cloud_deploy`) | — |
| O-04 | Free Render instance sleeps (cold start), 512 MB, restarts interrupt jobs | VERIFIED (plan) | client wakes the API, bounded retries; stale-run recovery after 3 h | Paid instance recommended before relying on the scheduler (cost) |

## Prediction Engine v2 (shadow)

| ID | Item | Status | Tests / evidence |
|---|---|---|---|
| V-01 | Additive v2 schema (10 tables, uniqueness for immutability and idempotency) | IMPLEMENTED; TESTED | upgrade / downgrade / upgrade + `alembic check` clean (SQLite) |
| V-02 | `TODAY_PREOPEN` / `TODAY_CONFIRMED` / `TOMORROW_EOD` snapshots: cutoff, idempotency key, one RUNNING per type, quality gate (≥50% usable), immutability, a FAILED run releases its key | IMPLEMENTED; TESTED | `test_prediction_v2`, `test_prediction_jobs`, `test_scheduler` |
| V-03 | Baseline rules (UP / DOWN / NEUTRAL / NO_CALL, ATR levels, confidence null) | IMPLEMENTED; TESTED | `test_rules_*` |
| V-04 | 1/3/5-session outcomes (direction- and cost-adjusted at 15 + 5 bps, NIFTY and sector benchmarks, MFE/MAE, idempotent, waits for final bars) | IMPLEMENTED; TESTED | `test_outcomes_*`, `test_outcomes_job_waits_for_final_bars_and_is_idempotent` |
| V-05 | Shadow exit engine (11 reason codes, legal transitions, gap and missing-bar handling, no notifications) | IMPLEMENTED; TESTED (**all 11 codes** since `f219acc`; previously 9) | `test_exit_*`, `test_catalyst_exhaustion_*`, `test_distribution_*` |
| V-06 | Performance aggregation (Wilson intervals, small-sample suppression, promotion gate of 200 events) | IMPLEMENTED; TESTED | `test_wilson_*`; web `Performance_hides_statistics_for_small_samples` |
| V-07 | Chronological backtest (60/20/20 splits, embargo, costs, 6 baselines, leak test) | IMPLEMENTED; TESTED | `test_backtest_*` |
| V-08 | v2 API (admin-only shadow, one snapshot per response, freshness) | IMPLEMENTED; TESTED | `test_prediction_api` (32) |
| V-09 | Web Labs views (Today, Tomorrow, detail, exits, performance) | IMPLEMENTED; TESTED | web `PredictionPagesTests` (15) |
| V-10 | Event-aware and catalyst setups (re-rating, surprise, business update) | BLOCKED | needs an event provider and historical consensus |
| V-11 | Calibrated confidence | DEFERRED | after enough holdout outcomes; `confidence` stays null |
| V-12 | **Negative backtest** (see P2 below) | VERIFIED; v2 **stays in shadow**; promotion gate not met | `docs/research/PREDICTION_V2_BACKTEST_2026-10.md` |
| V-13 | Scheduling of v2 jobs | IMPLEMENTED; TESTED; inactive | O-01 |

## Backtest investigation (P2)

| ID | Finding | Status | Evidence |
|---|---|---|---|
| P2-01 | **Horizon misalignment:** `fwd()` stepped along each stock's own bars, so a missing bar stretched a "1-session" outcome over 2+ market sessions | VERIFIED; IMPLEMENTED; TESTED | `68640a0`; `test_backtest_horizon_follows_the_market_calendar_not_the_stocks_own_bars`; counts in `backtest_analysis.json` `alignment_audit` |
| P2-02 | Look-ahead | VERIFIED **absent**: synthetic future-shock test plus a real-data recompute of sampled decisions from truncated data | `test_backtest_is_chronological_embargoed_and_leak_free`; `backtest_analysis.json` `lookahead_check` |
| P2-03 | Naive intervals treat same-day trades as independent | IMPLEMENTED; TESTED | session-clustered bootstrap (`test_backtest_regime_breakdown_and_clustered_interval`) |
| P2-04 | Results by setup, direction, sector, regime, liquidity and 1/3/5 horizon, against baselines | IMPLEMENTED | `scripts/research/backtest_v2_analysis.py`, `backtest_analysis.json`, `docs/research/PREDICTION_V2_BACKTEST_2026-10.md` |
| P2-05 | Conclusion | see the backtest document: **no edge after costs at any horizon; not promoted** | — |

## Top Candidates consistency

| ID | Finding | Status | Evidence |
|---|---|---|---|
| T-01 | Ranking score vs refreshed current price: already separated in `85c7eab` | VERIFIED (no defect) | existing tests |
| T-02 | No signal when the stored ranking is stale; no label for a fallback reference price | IMPLEMENTED; TESTED | backend `ranking_freshness`, `reference_price_source`, `run_id` (additive; mobile fixture still served: `test_mobile_contract_fixture_fields_are_still_served`); web banner and note |
| T-03 | In the week of 5–9 Oct, Top Candidates served the 1 Oct ranking for 5 sessions (BEHIND, then STALE) with no indication | VERIFIED (local) | `case_evidence_2026_10.json` `audit.sessions` |

## October 2026 case review (`docs/research/WEEKLY_CASES_2026-10.md`)

| ID | Case | Status |
|---|---|---|
| C-01 | Canara Bank: live run 3 Oct 14:50 IST, rank 1 of 157, score 77.0 (quality 100, valuation 83.3, financial health 100, trend 8.3). Fallback reference ₹118.36. +1.04% in 5 sessions vs Bank Nifty +1.48%; no qualifying move. Class: **coincidental positive outcome**; an early-signal success is impossible for v1. Top 10 in 24 of 39 month-end snapshots | VERIFIED (local DB, Yahoo); production snapshot BLOCKED |
| C-02 | Trent (gapped in at the 6 Oct open), Titan (7 Oct open), ITC (built during 5 Oct), CPCL (built during 7 Oct; reversal 8 Oct): timing from 5-minute bars | Market timing VERIFIED; publication and ingestion times BLOCKED |
| C-03 | Formal false positives / negatives for the week | NOT_REPRODUCIBLE: no frozen directional predictions existed. A v2 simulation is reported separately and labelled as such |
| C-04 | 47 material stock-days across 37 stocks; 1 in v1 top 20; 17 v1-ineligible; 8 outside the universe. Revises the earlier ad-hoc "38 / 33" | VERIFIED (reproducible script) |
| C-05 | Original 5–9 Oct experiment reports and their exact claims | NOT_REPRODUCIBLE (not available to this review) |

## Branches and integration

| ID | Finding | Status |
|---|---|---|
| B-01 | Remote `origin/claude/stocklens-auth-security-hardening-qczhnr` (`d8d3c6e`) adds an equivalent engine-run guard (`c3f9d8e0a4b2`) that **auto-marks duplicate RUNNING runs FAILED**. Combining it with v2 gave **two Alembic heads**, so `alembic upgrade head` would fail on deploy | VERIFIED; **reconciled on `review/stocklens-integration`**: the auth commit is cherry-picked as `e32a7d8` with `-x` (original author kept); `c3f9d8e0a4b2` is dropped, `b7e4c2a91d35` re-parented onto `b78f20585037`, giving one head (`0af4988`); upgrade / downgrade / re-upgrade / `alembic check` clean; full suite passing. The remote branch itself is untouched. Merging: REQUIRES_APPROVAL |
| B-02 | Local `agents/*` branches (5) are contained in `main`; 2 of their worktrees have 1 uncommitted file each | VERIFIED; untouched |
| B-03 | The auth commit adds `GOOGLE_ALLOWED_AUDIENCES` and `FRONTEND_BASE_URL` (both `sync: false`, no values) to `render.yaml` | VERIFIED; kept as part of the security fix; takes effect only on blueprint sync after merge: REQUIRES_APPROVAL |
| B-04 | Commit `c87c431` (test isolation) also contains the `render.yaml` restore and the template removal; both were already staged when it was made | VERIFIED; documented here. Not amended (no history rewrite) |
| B-05 | The user's draft `.github/workflows/daily-ranking.yml` and `tests/test_github_actions_workflow.py` stay **untracked and uncommitted** | VERIFIED |

## Testing and environment

| ID | Finding | Status |
|---|---|---|
| E-1 | `test_branding::test_app_config_and_onboarding_use_the_brand` read the developer's `storage/app.db` (failed on every clean checkout, including `main`) | IMPLEMENTED; TESTED: isolated in-memory database (`c87c431`) |
| E-2 | Streamlit navigation tests started a real background market scan (prices, news, FinBERT download) in threads outliving the test run | IMPLEMENTED; TESTED: scan starter stubbed (`c87c431`) |
| E-3 | Mobile contract fixture test skips when the mobile repository is not checked out beside the backend | VERIFIED (by design); it runs in this workspace layout |
| P-01 | Migrations have not been run on PostgreSQL in this work (Docker Desktop not running; no local server). The auth session reported PostgreSQL 16 verification of its own migrations | BLOCKED locally; the `postgresql_where` and `sqlite_where` clauses are identical. Recommend a PostgreSQL run before any production migration |

## Discovered and fixed during implementation

| ID | Issue | Fix |
|---|---|---|
| D-01 | A blanket "one RUNNING run" index would have blocked concurrent SINGLE-stock analyses | Index scoped to `kind = 'RANKING'` |
| D-02 | A FAILED v2 snapshot blocked the same day's retry | The failed run releases its key; attempts capped by the slot |
| D-03 | Exit engine lacked the event-day levels; `from_state` was stale in replay | Fixed (`exits`, `outcomes`) |
| D-04 | `summarize()` crashed on empty MFE lists; `sector_rs_5` key missing below 3 peers | Fixed (tests) |
| D-05 | ORM instance expired after an audit commit, so the API returned `{}` | Serialize before commit |
| D-06 | Backtest horizon misalignment | P2-01 |
| D-07 | Two exit reason codes untested | V-05 |

## News-first engine, FQVF and Phase 2 (`review/stocklens-complete`)

| ID | Item | Status | Evidence |
|---|---|---|---|
| N-01 | News reception | Official RBI / SEBI / Fed RSS **verified live** (29 articles → 24 events, 10 Oct). **Not running automatically**: scheduler inactive, migration not applied in production (REQUIRES_APPROVAL) | `docs/NEWS_ENGINE.md` §1–2 |
| N-02 | International, geopolitical, tariff, commodity-news and company-filings coverage | **BLOCKED**: NewsAPI key revoked or not configured (new key: owner decision); GDELT refused from this network; NSE/BSE automated access restricted; licensed wires, broker ratings and consensus need contracts | §2 |
| N-03 | Dedup, syndication, canonical ids, credibility, timestamps (published / ingested / corrected / effective), classification, materiality, entities, contradiction, expiry, untrusted text, failure states | IMPLEMENTED AND TESTED | `catalysts/`, `tests/test_catalysts.py` |
| N-04 | Defects found by the live run: sign-0 exposure noise; RBI feed encoding | IMPLEMENTED AND TESTED (`9401771`) | — |
| N-05 | Classification and entity links were dated by wall clock (would break replays) | IMPLEMENTED AND TESTED: ingestion clock | `test_assessment_uses_only_what_was_available_before_the_cutoff` |
| N-06 | Transmission hypotheses (13) with mechanism, sign, validation status; facts vs inference stored separately | IMPLEMENTED AND TESTED | `catalysts/transmission.py`, `event_entities.inferred` |
| N-07 | Price-based validation of factor hypotheses | IMPLEMENTED AND TESTED: 3 SUPPORTED, 6 INCONCLUSIVE, 4 UNVALIDATED (event history) | `catalysts/transmission_validation.json` |
| N-08 | News is primary in v2 snapshots (`NEWS_CATALYST`, `NEWS_CONFLICT`); sentiment / surprise / novelty / direction / confidence (null, uncalibrated) / data completeness kept separate | IMPLEMENTED AND TESTED | `test_snapshot_uses_news_and_freezes_the_evidence` |
| N-09 | Historical news replay engine (point-in-time, late ingestion, corrections, costs, baselines, unscored) | IMPLEMENTED AND TESTED | `catalysts/replay.py`, `tests/test_news_replay.py` |
| N-10 | Historical news replay **results** | **BLOCKED**: no accessible point-in-time archive; StockLens' own archive is 29 articles (`INSUFFICIENT_SAMPLE`). Next: approve scheduled ingestion (archive grows about 5 sessions a week) or license a provider with history | `scripts/research/news_replay.py` |
| N-11 | Factor-shock replay of the macro channels | IMPLEMENTED AND TESTED: **no tradable edge after the open** (holdout +0.02% net, CI includes 0); effect is in the overnight gap | `factor_shock_replay.json` |
| N-12 | Event-category analysis (tariffs, earnings vs expectations, broker actions, absorption speed) | Macro absorption: answered (gap). Others **BLOCKED** (no dated event history or consensus) | §6 |
| N-13 | Daily prediction-quality report (outcomes, missed news, false-positive catalysts, mapping errors, latency, provider failures; advisory recommendations) | IMPLEMENTED AND TESTED; runs with OUTCOME_EVALUATION once the scheduler is active | `prediction_v2/feedback.py`, `test_prediction_feedback.py` |
| N-14 | Stock-selection combination with market, sector, volume and fundamentals; position sizing | DEFERRED: requires a validated news signal first (N-10); exits stay shadow | — |
| N-15 | Calibrated confidence | DEFERRED: no validated probability model; `confidence` stays null | — |
| N-16 | NEWS_INGESTION job: hourly, every day, resumable windows, stale-news monitoring; scheduler throttling of failed auth and triggers; audit log | IMPLEMENTED AND TESTED; activation REQUIRES_APPROVAL | `test_scheduler.py` (40) |
| N-17 | News API, market-catalyst summary, current vs reference price | IMPLEMENTED AND TESTED | `test_news_api.py`; web `NewsUiTests` |
| N-18 | NLP quality (precision / recall of categories, sentiment, entities on real news) | NOT measured: needs a labelled sample (DEFERRED) | — |
| Q-01 | FQVF verified: **Fundamental Quality & Value Framework** (`fqvf-v1.0`, 18 fixed checks; score = 100 × (pass + ½ warning) / evaluated; with coverage). Display name **"Fundamental Quality & Value Score"** | IMPLEMENTED AND TESTED (web: `TerminologyTests`; backend: `test_terminology.py`) | `docs/FQVF.md`; web `971f843`; backend `ecc35c5` |
| Q-02 | Mobile app shows bare "FQVF" (Stock detail metric and tab, Top Picks, Watchlist, Welcome, Notification settings) | REQUIRES_APPROVAL: mobile is out of scope under the standing "do not modify mobile" instruction | mobile `Components/Pages/*.razor` |
| Q-03 | Frozen Ranking v1 explanation strings contain "FQVF" (component basis, risks, ineligible reasons) | Unchanged (golden test); the web rewords them for display only | `QualityValue.Plain` |
| Q-04 | Stored admin overrides of notification templates or onboarding in a database keep their old text | REQUIRES_APPROVAL (admin edit in production) | — |
