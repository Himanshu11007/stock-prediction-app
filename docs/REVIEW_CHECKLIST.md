# StockLens review checklist: every requirement and its status

The checklist covers every requirement from the master engineering review,
the phase approvals and Priorities 1–5, the review-bundle and master-backlog
requests, and the scheduler request.

**Status** is one of IMPLEMENTED · TESTED · BLOCKED · DEFERRED ·
REQUIRES_APPROVAL · NOT_REPRODUCIBLE. "TESTED" implies implemented. Several
statuses can apply, for example *TESTED; activation REQUIRES_APPROVAL*.

**Evidence** names a file, a test (`tests/<file>::<test>`, or web
`PredictionPagesTests` and similar), a commit, or a research artefact
(`scripts/research/output/prediction_v2/*.json`). Findings IDs refer to
`docs/FINDINGS_LEDGER.md`.

## A. Constraints (all phases)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| A1 | Ranking Engine v1.0 scoring, eligibility, history and API contracts unchanged | TESTED | `tests/test_ranking_v1_golden.py` (freeze `37bd87a`, fixture `tests/data/ranking_v1_golden.json`); Top Picks changes are additive only (`test_top_picks_freshness::test_top_picks_keeps_every_existing_field_*`, mobile fixture test) |
| A2 | v2 separate from v1 (versions, persistence, universe, flags) | TESTED | `prediction_v2/`, `db/models/prediction.py`, `test_universe_seed_is_a_copy_and_admin_adds_do_not_touch_v1` |
| A3 | No stock-specific rules; no tuning to hand-picked cases | IMPLEMENTED (verified by inspection) | `prediction_v2/rules.py` thresholds are fixed a priori (`baseline-v0.1`). Case scripts only read data |
| A4 | No push, merge, deploy, force-push or history rewrite (until this request); no branch deletion | IMPLEMENTED | Only `review/*` branches are pushed, after the deployment check (section J). Nothing is merged or deployed. B-04 left unamended |
| A5 | No production migrations, scheduling, credentials or user-facing predictions or alerts | IMPLEMENTED | Scheduler inactive (`docs/SCHEDULER.md`); `PREDICTION_V2_PUBLIC=false`; exit engine has no notifications (`test_exit_states_are_persisted_in_shadow_mode_only`) |
| A6 | No destructive DB operations; stop if duplicate RUNNING rows exist | TESTED | `4e6773a9e7db` aborts without changes (`test_migration_engine_run_guard`); auth `c3f9d8e0a4b2` (auto-FAILED) dropped in integration (B-01) |
| A7 | Never commit secrets; do not use the exposed DB URL | IMPLEMENTED | Credential scan of every pushed diff (section J); S-01 URL never used |
| A8 | Preserve the untracked `.github/workflows/daily-ranking.yml` and `tests/test_github_actions_workflow.py`; do not modify mobile | IMPLEMENTED | B-05; mobile `master` `d24eaf5` unchanged |
| A9 | Never treat missing values as favourable or zero | TESTED | `features.py` returns None plus a flag; `test_missing_history_gives_none_and_flags_never_zero`, `test_rules_no_call_when_a_required_feature_is_missing`, `test_rules_return_no_call_for_every_blocking_flag` |
| A10 | Do not weaken or delete tests | IMPLEMENTED | Two test expectations changed with the design, both documented in their commits: `test_cloud_deploy` (one scheduler per job, still strict) and `test_status_slot_must_be_a_date` (also accepts 405). No test removed |

## B. Phase 0: safety and verification

| # | Requirement | Status | Evidence |
|---|---|---|---|
| B1 | Verify production PostgreSQL persistence, expiry, backup and recovery | BLOCKED (no production access, F-13); procedure IMPLEMENTED; plan decision REQUIRES_APPROVAL | `docs/OPERATIONS_AND_SECURITY.md` §2; S-05 |
| B2 | Address exposed credentials; verify access restrictions | REQUIRES_APPROVAL (owner rotates and revokes) | S-01, S-02; §1.1–1.2; docs routes off in production (S-03, `test_api_security`) |
| B3 | Database-level guard against overlapping runs, with concurrency tests | TESTED | F-04 (`test_engine_run_concurrency` 20 real two-process trials, `test_migration_engine_run_guard`) |
| B4 | Confirm repository commits; v1 unchanged | TESTED | A1; branch inventory in the review report |
| B5 | Verify deployment and scheduler limitations | IMPLEMENTED | O-01, O-04; `docs/SCHEDULER.md` "Limitations and costs" |
| B6 | NewsAPI: distinguish local cleanup, revocation, replacement, history cleanup; no history rewrite | IMPLEMENTED (procedure); REQUIRES_APPROVAL (owner revokes) | §1.2 |
| B7 | Verify docs routes in production mode without exposing secrets | TESTED | `test_api_security` (production-mode subprocess: 404) |
| B8 | No DB credential rotation or Postgres upgrade without a cost, downtime, backup and recovery review | IMPLEMENTED (not done) | §2 |

## C. Phase 1: automated prediction infrastructure

| # | Requirement | Status | Evidence |
|---|---|---|---|
| C1 | Additive v2 data model and migrations | TESTED | V-01; migration `b78f20585037` |
| C2 | Immutable prediction runs and snapshots | TESTED | V-02 (`test_eod_snapshot_is_created_frozen_and_complete`, `test_retry_and_duplicate_triggers_never_change_a_snapshot`, idempotency key UNIQUE) |
| C3 | Automated TODAY_PREOPEN and TOMORROW_EOD scheduling | TESTED; activation REQUIRES_APPROVAL | O-01: workflows, `scheduling/remote.py`, `test_scheduler` |
| C4 | TODAY_CONFIRMED only with a reliable intraday feed | TESTED (gated off); feed BLOCKED | F-10; `test_scheduler::test_today_confirmed_stays_disabled_without_an_intraday_feed` |
| C5 | Idempotency, holiday handling, retries, stale-run recovery, failure monitoring | TESTED | `test_prediction_jobs`, `test_scheduler` (duplicates, holidays, missing calendar, bounded retries, stale reclaim, monitor) |
| C6 | Short-horizon feature pipeline | TESTED | F-06 |
| C7 | Prediction outcome tracking | TESTED | V-04 |
| C8 | v2 universe separate from v1 | TESTED | A2; F-15 (seeding manual) |

## D. Phase 2: evidence and evaluation

| # | Requirement | Status | Evidence |
|---|---|---|---|
| D1 | Canara Bank production snapshot | BLOCKED (production); local live run VERIFIED | C-01 |
| D2 | Canara: original selection timestamp, score, components, rank history, reference price, subsequent move, early-signal verdict | TESTED (reproducible script); VERIFIED | `scripts/research/case_evidence_2026_10.py` → `case_evidence_2026_10.json` `canara`; `docs/research/WEEKLY_CASES_2026-10.md` §1. Verdict: coincidental positive outcome; early-signal success impossible for v1 |
| D3 | Canara: drivers; technical, momentum, sector and regime inputs; event check | VERIFIED | same; F-08 (sector outlook unset); no event source (F-05) |
| D4 | Canara: first qualifying move; 1/3/5 outcomes from the frozen reference; NIFTY, Bank Nifty and baseline comparison | VERIFIED | same (rule fixed in advance; same-run top-10 and momentum baselines) |
| D5 | ITC, Trent, CPCL, Titan, HCL, Kanohar, Moneyview against timestamped evidence | Prices and market timing VERIFIED; catalyst publication and ingestion BLOCKED | `WEEKLY_CASES_2026-10.md` §2; `intraday_reaction` in the JSON (5-minute bars) |
| D6 | Trent announcement timing | Market side VERIFIED (fully gapped in at the 6 Oct 09:15 open, so public before then); filing timestamp BLOCKED | same |
| D7 | ITC re-rating | Price path VERIFIED (built through 5 Oct); broker note time and target REPORTED, BLOCKED | same |
| D8 | CPCL commodity exposure and reversal | Prices and timing VERIFIED; universe omission VERIFIED (F-02); crude and crack observations BLOCKED (F-14) | same |
| D9 | Titan negative surprise | Price timing VERIFIED (7 Oct open); expectations BLOCKED (no historical consensus) | same |
| D10 | Timestamped corporate-event ingestion from an approved source | IMPLEMENTED (foundation, **interface only for providers**); provider BLOCKED | F-05 |
| D11 | Chronological backtest and performance reporting | TESTED | V-07, V-06, P2-01..04 |
| D12 | Exit engine in shadow mode | TESTED | V-05 |

## E. Master review: detailed requirements

| # | Requirement | Status | Evidence |
|---|---|---|---|
| E1 | Case ledger fields per case (timestamps, snapshot, components, features, cutoff, catalysts, OHLCV, benchmarks, 1/3/5, MFE/MAE, actionable, verified or not, improvement) | IMPLEMENTED where data exists; catalysts BLOCKED; original reports NOT_REPRODUCIBLE | `weekly_case_ledger.json`, `case_evidence_2026_10.json`, `WEEKLY_CASES_2026-10.md` |
| E2 | Missed movers, formal false positives and negatives, per-session counts, omissions, exclusions, stale snapshots, failed jobs | Movers and counts VERIFIED; formal FP/FN **NOT_REPRODUCIBLE** (no frozen predictions); job history BLOCKED | `WEEKLY_CASES_2026-10.md` §3; C-03, C-04, T-03 |
| E3 | Performance by setup, sector, liquidity and horizon (live) | DEFERRED until shadow snapshots accumulate; backtest equivalent IMPLEMENTED | P2-04 |
| E4 | Top Candidates: immutable runs, separate current price and timestamp, stale indicator, never mutate, new run per recalculation, frontend tests | TESTED | T-01, T-02; web `TopCandidatesTests`, `Top_candidates_flags_a_stale_ranking_*` |
| E5 | Prediction output fields (direction, setup, horizon, cutoff, version, reference, null confidence, factors, catalysts, entry, stop, target, trailing, invalidation, flags) | TESTED | `db/models/prediction.py`; `test_rules_*`; `test_prediction_api` |
| E6 | Entities: run, prediction, outcome, event (+ classification), features, exit state and transitions | TESTED | V-01 |
| E7 | API endpoints (predictions today/tomorrow, detail, performance, stock events, exit signals, admin runs GET/POST, admin events GET/PATCH) | TESTED | `api/routes/predictions.py` (14 routes); `test_prediction_api` |
| E8 | Scheduler: Asia/Kolkata, NSE calendar, no stale holiday snapshots, idempotent retries, recovery, no overlap, status and duration, alerts, no duplicates, history | TESTED | C3–C5; durations in `scheduled_job_runs.result.duration_s` |
| E9 | Manual BAT not an operational dependency | IMPLEMENTED | scheduler; manual paths are recovery tools only |
| E10 | Feature inventory (21 features) | Partly: returns 1/3/5/20, gap, abnormal volume, close location, ATR %, volatility, RS vs NIFTY and sector, distance from highs and lows, liquidity, data flags TESTED. Breadth, sector rotation, trend acceleration, breakout confirmation as a feature, spread proxy, event timing and reaction DEFERRED or BLOCKED | `features.py` docstring; F-14 |
| E11 | Catalyst taxonomy, raw vs derived, LLM experimental and versioned | IMPLEMENTED (schema, review and correction, versioned classifications); LLM classification DEFERRED; provider BLOCKED | `events/ingest.py`, `test_admin_event_review_keeps_the_original` |
| E12 | Sector and macro hierarchy; past-only factor betas | Sector peer RS TESTED; factor betas DEFERRED (F-14) | — |
| E13 | Exit state machine, 11 reason codes, per-setup definitions, gap and circuit handling, persistence, duplicate prevention, audit | TESTED (shadow); circuit handling via `POSSIBLE_PRICE_BAND` NO_CALL | V-05 |
| E14 | Backtest: 2–3 years, splits, 5-session embargo, 5 baselines, metrics, costs 0.1–0.2% plus slippage, promotion gate, 4–8 weeks shadow | TESTED; Brier and calibration N/A (no probabilities); exit-vs-hold DEFERRED (live shadow); shadow weeks not started (REQUIRES_APPROVAL to schedule) | `PREDICTION_V2_BACKTEST_2026-10.md` |
| E15 | Missing-parameter inventory and behaviour | IMPLEMENTED | `WEEKLY_CASES_2026-10.md` §4; master review §7; flags in `features.py` and `rules.BLOCKING_FLAGS` |
| E16 | Frontend separation: ranking, today, tomorrow, exits, performance; no uncalibrated percentages | TESTED | V-09 (`Shadow_notice_is_shown_and_no_probability_is_claimed`) |
| E17 | Tests: unit, integration, regression, time and data, failure recovery | TESTED (SQLite); PostgreSQL integration BLOCKED locally (P-01) | suites listed in the review report |
| E18 | Do not log credentials, tokens or connection strings | IMPLEMENTED | scheduler logs never include the token (`test_client_never_prints_the_token`) |

## F. Priorities 1–5 (continuation request)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| F1 | Reconcile with the remote auth branch; one coherent Alembic graph; graph and clean-database tests | TESTED | B-01; `review/stocklens-integration` (`e32a7d8` cherry-pick of `d8d3c6e`, `0af4988` reconciliation): `alembic heads` = `b7e4c2a91d35` only; upgrade from empty, downgrade to base, re-upgrade, `alembic check` clean (SQLite) |
| F2 | Backtest config, samples, dates, costs, slippage, benchmarks, metrics | IMPLEMENTED | `backtest_analysis.json`; backtest document |
| F3 | No look-ahead or misalignment | TESTED (misalignment found and fixed) | P2-01, P2-02 |
| F4 | Breakdowns by setup, direction, sector, regime, 1/3/5 | IMPLEMENTED | P2-04 |
| F5 | Statistical meaning of the lack of edge | IMPLEMENTED | backtest document "Is the result meaningful?" |
| F6 | Do not tune; keep shadow; do not promote | IMPLEMENTED | rules unchanged (`baseline-v0.1`); V-12 |
| F7 | Real disposition for every finding; interfaces not marked complete | IMPLEMENTED | `FINDINGS_LEDGER.md` ("interface only" entries: F-05, F-14, providers) |
| F8 | Re-run tests; report commands, skips, limitations | TESTED | review report; E-1..E-3, P-01 |
| F9 | Review package per repository and branch; ordered integration recommendation | IMPLEMENTED | review report |

## G. Review bundle request

| # | Requirement | Status | Evidence |
|---|---|---|---|
| G1 | Read-only review bundle archive | Superseded at your request: no ZIP; GitHub review branches instead. The earlier archive (`Downloads/StockLens-review-bundle-2026-10-10.zip`) remains and is now outdated | — |

## H. Master-backlog request (items 1–9)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| H1 | Canara reconciliation | VERIFIED | D2–D4 |
| H2 | Trent, ITC, CPCL, Titan with timestamped evidence only | VERIFIED (market timing) / BLOCKED (publication) | D5–D9 |
| H3 | Missed predictions, FP, FN, parameters, catalyst gaps, refresh inconsistencies; why formal classification is impossible | VERIFIED / NOT_REPRODUCIBLE (formal) | E2, E15, T-03 |
| H4 | Snapshot semantics, immutability, cutoff, scheduling, retries, holidays, monitoring, recovery | TESTED; activation REQUIRES_APPROVAL | `docs/PREDICTION_V2.md`, `docs/SCHEDULER.md`, C2–C5 |
| H5 | 1/3/5 outcomes, costs, benchmarks, negative holdout, promotion conditions | TESTED | V-04, V-12, backtest document |
| H6 | Exit requirements, all reason codes, shadow testing, no live alerts | TESTED | V-05; A5 |
| H7 | Backend, web, mobile compatibility, API, migration, concurrency, security, regression findings | TESTED (SQLite) | ledger; P-01 for PostgreSQL |
| H8 | DB credential, NewsAPI, secret cleanup, backup and recovery, DB expiry, hosting and scheduler costs; owner actions | REQUIRES_APPROVAL (owner) | S-01, S-02, S-05, O-04; `docs/SCHEDULER.md` costs |
| H9 | Every branch; conflicts; uncommitted files; integration order | IMPLEMENTED | B-01..B-05; review report |
| H10 | Push only dedicated review branches after checking deployment implications | IMPLEMENTED | section J |

## I. Scheduler request (items 1–10)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| I1 | Separate workflows: TODAY_PREOPEN, TODAY_CONFIRMED, TOMORROW_EOD, OUTCOME_EVALUATION, SNAPSHOT_MONITOR | TESTED | `.github/workflows/stocklens-*.yml`; `test_scheduler::test_workflow_is_inactive_by_default_*` |
| I2 | UTC schedules with documented IST equivalents; trading days, weekends, holidays, data readiness | TESTED | `docs/SCHEDULER.md` table (checked by `test_documentation_lists_every_schedule_in_utc_and_ist`); backend preflight |
| I3 | GitHub Actions secret for a dedicated credential; never exposed | TESTED; creating the secret REQUIRES_APPROVAL | S-06 |
| I4 | Authenticated backend job interface; strict job types; authorization | TESTED | `api/routes/scheduler.py`; `test_job_type_is_strictly_validated`, `test_authentication_is_the_dedicated_token_only` |
| I5 | DB-backed idempotency, duplicate protection, bounded retries, failure status, stale recovery | TESTED | `test_eod_publishes_once_and_duplicates_are_no_ops`, `test_failed_job_is_reported_failed_and_retries_are_bounded`, `test_interrupted_run_is_recovered_after_the_stale_limit` |
| I6 | TOMORROW_EOD only after validated closing data | TESTED | `test_eod_waits_for_validated_closing_data` |
| I7 | TODAY_CONFIRMED disabled until an approved feed | TESTED | C4 |
| I8 | Monitoring of missing snapshots; failures never reported as success | TESTED | `test_monitor_reports_missing_snapshot_as_alert_never_success`, client exit codes |
| I9 | Tests of workflow config, auth, duplicates, retries, holidays, missing data, recovery | TESTED | `tests/test_scheduler.py` (34) |
| I10 | Enable, disable and manual recovery documented | IMPLEMENTED | `docs/SCHEDULER.md` |
| I11 | Check Actions permissions, usage limits, deployment implications | Public repository (free minutes) and 0 registered workflows VERIFIED; Actions permission settings and the Render preview setting BLOCKED (no authenticated access; owner to confirm) | `docs/SCHEDULER.md` "Pushing a branch does not run anything" |
| I12 | No activation, production secrets, Render changes or deploys | IMPLEMENTED | `render.yaml` equals `main` on `review/prediction-v2`; the auth env keys on the integration branch are B-03 |

## J. Push and deployment check

| Check | Result |
|---|---|
| Render services | every service in both blueprints is pinned to `branch: main` (`autoDeploy` from `main` only); no `previews:` |
| GitHub Actions | workflows trigger only on `schedule` (default branch only) and `workflow_dispatch`, and are gated by unset repository variables. 0 workflows registered today |
| Web (Render static site) | `branch: main` only |
| Secrets | credential-pattern scan of `git diff main...<branch>` for every pushed branch: see the review report |

## Incomplete requirements (summary)

| Item | Why | Status |
|---|---|---|
| Event / filings provider, catalyst setups, broker re-ratings, historical consensus | no licensed source chosen | BLOCKED |
| Intraday feed and adapter (TODAY_CONFIRMED live) | no licensed feed | BLOCKED |
| Factor observations writer, past-only betas, breadth and rotation features | needs a data source decision | DEFERRED |
| Production DB verification (snapshots, run history, expiry, backups) | no read-only access | BLOCKED |
| Credential rotation, NewsAPI revocation, DB plan | owner actions | REQUIRES_APPROVAL |
| Holiday calendar seeding, v2 universe seeding, scheduler activation, Render env var, merge | production changes | REQUIRES_APPROVAL |
| PostgreSQL migration run in this work | no local server | BLOCKED locally |
| 4–8 weeks shadow operation; live per-setup performance; calibration; exit-vs-hold | needs the scheduler running | DEFERRED |
| Formal FP / FN for 5–9 Oct; original experiment reports | no frozen predictions; reports unavailable | NOT_REPRODUCIBLE |
| v2 promotion | negative backtest; gate not met | not promoted (by evidence) |
| Mobile v2 screens; legacy Streamlit tab relabel; sector outlook data | out of scope or approval needed | DEFERRED |

## K. News-first backlog and Phase 2

| # | Requirement | Status | Evidence |
|---|---|---|---|
| K1 | Inspect branches, workflows, Render; dedicated review branch | IMPLEMENTED | `review/stocklens-complete` (backend, web); push check in section J |
| K2 | Preserve v1 (golden), v2 separate and additive | TESTED | `test_ranking_v1_golden`; additive migration `8d7b9e6c90dd` |
| K3 | News-first chain (event → verification → surprise and context → transmission → exposure → direction → risk → outcome) | IMPLEMENTED AND TESTED (shadow); surprise vs consensus BLOCKED | `docs/NEWS_ENGINE.md` §3–4 |
| K4 | Global, domestic, geopolitical, company coverage | PARTLY: Indian monetary and regulatory and US Fed, from primary sources (working). Everything else BLOCKED (providers) | N-01, N-02 |
| K5 | Ingestion capabilities (dedup, ids, provenance, timestamps, language, entities, classification, novelty, materiality, sentiment, surprise, contradiction, links, expiry, failures, untrusted text) | IMPLEMENTED AND TESTED | N-03 |
| K6 | Transmission engine with testable relationships; facts vs inference | IMPLEMENTED AND TESTED; validated where price data allows | N-06, N-07 |
| K7 | TODAY / TOMORROW snapshots with news ids and availability, immutable | IMPLEMENTED AND TESTED | N-08 |
| K8 | Scheduling via GitHub Actions + authenticated backend (incl. news) | IMPLEMENTED AND TESTED; **not operational** (REQUIRES_APPROVAL) | N-16, `docs/SCHEDULER.md` |
| K9 | Data model and APIs | IMPLEMENTED AND TESTED | N-17 |
| K10 | Outcomes 1/3/5, baselines, chronological splits, embargo, costs; review the 47.3% / −0.12% claim | IMPLEMENTED AND TESTED. The claim **reproduces exactly**: 1,604 calls, hit 47.3%, −0.12% at 10 bps. Unchanged by the alignment fix (P2-01) | `docs/research/PREDICTION_V2_BACKTEST_2026-10.md` |
| K11 | Exit engine shadow, 11 codes; exit vs hold backtest | 11 codes TESTED. Exit-vs-hold DEFERRED (live shadow outcomes) | V-05 |
| K12 | FQVF terminology | IMPLEMENTED AND TESTED (web, backend). Mobile REQUIRES_APPROVAL | Q-01..Q-04 |
| K13 | Frontend explanation (direction, horizon, news, sources, times, mechanism, conflicts, confidence meaning, current vs reference price, risks, coverage) and a daily market summary | IMPLEMENTED AND TESTED (web Labs) | N-17 |
| K14 | Diagnostic cases (Canara, ITC, Trent, CPCL, Titan, HCL, Kanohar, Moneyview) | TESTED / VERIFIED; formal classification NOT_REPRODUCIBLE (no frozen predictions) | `docs/research/WEEKLY_CASES_2026-10.md` |
| K15 | Security: job endpoint auth, throttling, audit; secrets; RUNNING recovery; branch conflicts; Alembic heads; backups; deterministic tests | IMPLEMENTED AND TESTED; credential rotation and backups REQUIRES_APPROVAL (owner) | S-01..S-06, B-01, E-1, E-2 |
| K16 | Historical event replay with point-in-time availability | Engine TESTED; news results BLOCKED (archive); macro factor replay TESTED (no edge) | N-09..N-11 |
| K17 | Daily prediction-quality feedback loop | IMPLEMENTED AND TESTED | N-13 |
| K18 | Event-specific performance analysis | PARTLY (macro: absorbed at the open); the rest BLOCKED | N-12 |
| K19 | Stock selection and risk controls after news validation | DEFERRED | N-14 |
| K20 | Calibrated confidence | DEFERRED | N-15 |
| K21 | Production readiness (schedules, calendar, idempotency, retries, alerts, backups, migrations, security, regression) | Code IMPLEMENTED AND TESTED. **Not operational**: scheduler, migrations, holiday calendar, credentials, database plan all REQUIRE APPROVAL; PostgreSQL run BLOCKED locally (P-01) | sections B, C, I |
