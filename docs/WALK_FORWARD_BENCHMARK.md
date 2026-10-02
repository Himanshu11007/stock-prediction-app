# Walk-Forward Benchmark (Phase 10)

Status: evaluation infrastructure only. No model, threshold, feature, target,
scoring, Top Picks, mobile or authentication behaviour was changed to produce
anything in this document. This document gives no overall score, no
good/bad verdict and no "best model".

Companion to [RECOMMENDATION_QUALITY_AUDIT.md](RECOMMENDATION_QUALITY_AUDIT.md) (Phase 9)
and [PRODUCTION_TEMPORAL_INTEGRITY.md](PRODUCTION_TEMPORAL_INTEGRITY.md) (Phase 11A).

> **Phase 11A update (post-fix).** The section 2 defect was fixed in Phase
> 11A: production now trains on labelled rows ≤ D-1 and predicts bar D. The
> benchmark was re-run with the fixed production path. Sections 2–18 describe
> Phase 10 as it was, and section 13's numbers are the **pre-fix** run
> (artifacts moved to `walk_forward_benchmark/phase10_prefix_full/`).
> **Section 19** has the post-fix results and the pre-/post-fix comparison.

---

## 1. Purpose

Make this statement defensible:

> For a prediction made at historical time T, StockAI used only information
> available at T, generated the same production prediction it would have
> generated at that time, and we can subsequently measure its 1D/3D/5D/10D
> outcome.

The aim is trustworthy numbers, not better numbers. Where the clean benchmark
is worse than Phase 9's diagnostic figures, the worse figures are reported.

## 2. Headline integrity finding: production predicts a row it was trained on

Verifying the contract against the code found a problem Phase 9 did not
report. It is **documented here, not fixed** (fixing it changes production
predictions, which Phase 10 forbids).

`features/engineer.py:create_features()` correctly drops the final raw bar
(bar D, the bar at T) because its next-day label is unknown — this is the
Phase 9 label-fabrication fix (`8997c5a`). Every production call site then does:

```python
data, X, y, ... = prepare_data(raw)         # X ends at bar D-1
models, acc     = train_model(X, y, ...)    # trained on ALL of X, incl. bar D-1
latest          = X.iloc[-1:]               # bar D-1 — a TRAINING row
pred, conf, _   = ensemble_predict(models, latest)
```

So the production "prediction" is the ensemble's output on bar D-1, whose
training label is `Close[D] > Close[D-1]` — a move that has **already
happened** at T. RF/XGBoost largely memorise training rows (Phase 9: in-sample
accuracy ~99%), so the ML direction mostly reproduces the last realised daily
move:

| Evidence | Result |
|---|---|
| 40 cached production price series, production code unmodified | prediction == already-realised label of the predicted row in **40/40** |
| Same 40, that one row held out of training | 25/40 |
| Full benchmark, 1,350 replayed predictions (section 13) | prediction == already-realised D-1→D move in **1,313 (97.3%)** |

Consequences:

- The ML Direction / ML Confidence pillars (20% of confluence weight) mostly
  encode "did the stock close up on bar D", which the Technical/Momentum
  pillars already see. Confidence is inflated by in-sample memorisation.
- `scanner/engine.py`, `api/services.py`, `utils/recommendation_engine.py`
  and `app.py` all share this pattern (`X.iloc[-1:]` / `X.tail(1)`).
- Before `8997c5a` the row at T *was* predicted, but trained with a fabricated
  0 label — the "always bearish" bug. The fix moved the problem rather than
  removing it.
- Stored `cmp` is `Close[D-1]` (from `calculate_risk(data)` on the dropna'd
  frame), so the existing validation return includes bar D's already-known move
  (section 8).

The benchmark therefore reports two clearly separated ML views:

- **as-deployed** — exactly what production outputs (feature row D-1).
- **forward-row diagnostic** — the *same fitted models* applied to bar D's
  feature row, which no model has seen. This measures what the trained
  ensemble can do on unseen data. It is a measurement aid only, not a
  production change and not a proposal.

## 3. Evaluation contract (Task 1)

| Item | Definition |
|---|---|
| Prediction timestamp T | End of trading session D (after the close). Intraday runs cannot be replayed from daily bars and are out of scope. |
| Information set at T | Daily bars dated <= D. `replay_production_prediction()` slices the history to <= D before any production function runs; the weekly frame is resampled from those same bars. |
| Production window | Daily: bars in (D − 1 year, D] (`data.loader._fetch`, `period="1y"`). Weekly: Monday-start bars from daily bars in (D − 2 years, D] (`load_multi_timeframe_data`, `period="2y"`, `1wk`); the last week is partial, as yfinance returns mid-week. |
| Production prediction | Unmodified scanner path (`scanner/engine.py:_scan_one`): `prepare_data` → `train_model(fast=True)` → `ensemble_predict(X.iloc[-1:])` → `detect_regime` → `get_trend_signal` (weekly + daily) → `generate_signal` → `compute_pillar_scores` → `passes_quality_filters` → `calculate_risk`. A test asserts exact parity with `_scan_one` itself. |
| News | Not point-in-time reproducible (section 11). The production no-headlines path is used: `analyze_overall_sentiment([])` → 0.0. Flagged per row as `news_status = unavailable_no_point_in_time_source`. |
| Horizons | 1, 3, 5, 10 **trading sessions** after D in the symbol's own calendar. |
| Prediction price | `Close[D]`: the last price known at T. |
| Outcome price | `Close[D+h]`. NULL if that session does not exist yet: never a nearer or later price, never a shortened horizon. |
| Price series | yfinance `auto_adjust=True` (the default production uses, and the convention `get_latest_close()` uses for validation). |
| Return | `storage.recommendation_validation.calculate_return(entry, exit)` (unchanged). |
| Direction outcome | `outcome_hd = 1 if Close[D+h] > Close[D] else 0`, the same strict `>` used by the training label. |
| Signal success | `calculate_success(signal, return)` (unchanged): BUY/STRONG BUY `> 0`; SELL/STRONG SELL `< 0`; HOLD `abs <= 3.0`. |

### Price convention vs the existing system

The existing system stores `cmp = Close[D-1]` (section 2) and validates
against whatever close exists when an admin happens to trigger validation.
The benchmark's primary entry price is `Close[D]` because `Close[D-1]` → anything
includes bar D's move, which is known at T. For transparency every row also
carries `legacy_cmp_return_hd` / `legacy_cmp_signal_success_hd` (entry =
stored-style `cmp`), summarised separately and labelled as
including already-known information.

### Adjusted prices and point-in-time

Today's adjusted series is not the series production saw on day D. However,
any dividend/split after D rescales every bar <= D by the same factor. All
features are either ratios (RSI, % changes, BB position, volume ratio) or
uniformly scaled levels. StandardScaler (LR) and tree splits (RF/XGB) are
invariant to a uniform per-feature scale, so predictions are unaffected.
Returns use adjusted closes, so dividends after D appear in outcome returns
(total-return convention, the same as the existing validator).

## 4. Production ensemble (Task 3)

Verified from code (`models/trainer.py`), now a single authoritative constant
`ENSEMBLE_WEIGHTS`:

| Model | Weight | Pipeline |
|---|---|---|
| Logistic Regression | 0.20 | StandardScaler → LogisticRegression(max_iter=5000, random_state=42) |
| Random Forest | 0.30 | RandomForestClassifier(n_estimators=50, random_state=42) |
| XGBoost | 0.50 | XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42) |

Sum = 1.00 exactly; no normalisation is applied. If XGBoost is not installed
its 0.50 is applied to the RF probability (RF effectively 0.80). The blend is
a fixed-weight average of P(up); prediction = `P > 0.5`; confidence =
`max(P, 1-P) × 100`. Weights and hyperparameters are unchanged.

## 5. Walk-forward reporting fix (Task 2)

Problem (Phase 9 §7.2, confirmed): `walk_forward_validate()` scores each
model separately per fold and averages the three accuracies. That is not the
accuracy of the ensemble production uses.

Fix (additive, `models/trainer.py`):

- `ENSEMBLE_WEIGHTS`, `component_probabilities()` and `ensemble_proba()` are
  the one blend implementation. `ensemble_predict()` now calls it, and a test
  proves its output is bit-identical to the old hard-coded formula.
- New `walk_forward_validate_ensemble(X, y)`: same folds
  (`_walk_forward_splits`), same candidates, each fold fit on `[0, train_end)`
  only. It reports the blended ensemble's accuracy, the legacy
  mean-of-individuals (asserted equal to `walk_forward_validate`), and
  per-model accuracy.
- `walk_forward_validate()` / `train_model()` are **unchanged**. Their value
  feeds `MIN_ACCURACY` filtering (`scanner/filters.py`) and the `acc < 0.50`
  gate in `utils/recommendation_engine.py`. Changing it would change which
  recommendations production emits. The docstring now states what the number
  actually is.

## 6. Temporal training integrity audit (Task 4)

| Step | Finding |
|---|---|
| Target | `Up[t] = Close[t+1] > Close[t]`, final row NaN and dropped. Correct (Phase 9 tests). |
| Features | All backward-looking (rolling/ewm/shift(+n)/pct_change). Verified again by `test_issue_row_features_do_not_depend_on_the_next_bar` and the future-poisoning test. |
| Scaler | Inside the LR `Pipeline`, so it is fitted only on whatever is passed to `fit` (train fold / training rows). |
| Feature selection | None exists. |
| Imputation | None. `create_features` uses `replace(inf→NaN)` + `dropna()`, which is row-local. |
| Model fit (walk-forward CV) | `X.iloc[:train_end]` only; test block strictly after. |
| Model fit (production) | Trained on all labelled rows ≤ D-1. Labels realise at ≤ D = T, so **no future information**, but the predicted row is one of them (section 2). |
| Cached features / prices | The benchmark never reads `storage/price_cache` (1-hour rolling "current" data). It reads its own dated snapshot, sliced to ≤ D. |
| Future poisoning test | Multiplying every price after D by 3 and every volume by 0.1 leaves every replayed field identical. |

## 7. `prepare_data()` dead split (Task 5)

`prepare_data()` returns `X_train, X_test, y_train, y_test` from an 80/20
split. All four call sites (`scanner/engine.py`, `api/services.py`,
`utils/recommendation_engine.py`, `app.py`) read only `y_train`, and only as a
single-class guard; `X_train`/`X_test`/`y_test` are unused. Removing them
changes the 7-tuple contract at four call sites for no behavioural gain, so
the split is **left in place** with a comment explaining it. Model evaluation
lives in `models/trainer.py` and this benchmark.

## 8. Horizons and the existing validator (Tasks 6, 7)

- New, explicit horizons: `evaluation.walk_forward.forward_outcomes()`
  returns price/date/return/direction at exactly D+1/3/5/10 sessions, or NULL.
- **Exchange-holiday placeholders.** Yahoo emits flat, zero-volume bars on NSE
  holidays (identical dates for every symbol, e.g. 2026-01-15, 2026-05-01,
  2026-05-28, 2026-06-26, 2026-09-14). `session_closes()` excludes bars with
  `Volume == 0` and `Open == High == Low == Close`, so they never count as a
  trading day and are never issue dates. The production replay still sees
  them, because production does.
- `storage/recommendation_validation.py` is **unchanged**. Its stored
  `return_pct` stays what it is: "how did a persisted recommendation do by
  whenever validation ran" (5–39 trading days; Phase 9 §7.3). The benchmark's
  fixed-horizon outcomes are a separate concept in separate artifacts. No
  historical validation record is rewritten.

## 9. Duplicate recommendation rows (Task 8, 21)

Root cause, from the live `storage/tracker.db`:

- All 20 extra rows (5 groups) are ids 2–26, `saved_date = 2026-06-14`,
  `scan_id = NULL`. Rows 1–31 (2026-06-13 to 06-28) all have `scan_id` NULL;
  `upsert_recommendation()` (which always sets `scan_id`) was introduced on
  2026-06-29 (`69ef4bc`). Among the 2,127 rows written by upsert there are
  **zero** duplicates. The duplicates came from the legacy insert-only path,
  not from a race.
- Because those rows exist, `CREATE UNIQUE INDEX idx_rv_unique_symbol_date`
  has always failed with a logged warning, so the database never enforced
  uniqueness. Separately, upsert's SELECT-then-INSERT was not atomic.
- Identity check: `(symbol, saved_date)` is the documented key
  (`recommendation_exists`, `upsert_recommendation`, `_persist_recommendation`).
  The scanner's second persist of a symbol in the same scan (post-rerank) is
  an intentional UPDATE of the same row. Nothing creates two intentionally
  distinct rows for one symbol and day, so the key is correct.

Fix (`storage/tracker.py`):

1. When the full unique index cannot be created, a partial unique index
   `idx_rv_unique_symbol_date_scanned ON (symbol, saved_date) WHERE scan_id IS
   NOT NULL` is created instead. It enforces uniqueness for every row upsert
   writes without touching history.
2. `upsert_recommendation()` takes `BEGIN IMMEDIATE` before its existence
   check, so concurrent writers serialise.

Historical cleanup: **not performed.** 3 groups are byte-identical copies, but
RELIANCE.NS and SAREGAMA.NS groups differ in `confluence_score`, so "which row
is right" is not deterministic. `tracker.db` is local, gitignored data.
Before/after row count: 2,158 / 2,158. The existing
`dedupe_existing_recommendations()` (keep highest id) remains an explicit
operator choice. The benchmark does not read `tracker.db` at all.

Tests: `tests/test_tracker_dedup.py` (5 tests, including an 8-thread race and
a legacy-duplicate database).

## 10. Datasets (Tasks 9, 10, 13–15)

Two separate files per run, never mixed:

| File | Population | Concept |
|---|---|---|
| `raw_model_predictions.csv` | every replayed prediction | **A. Raw model benchmark** |
| `production_recommendations.csv` | rows where `passes_quality_filters` is True (what production persists) | **B. Production recommendation benchmark** |

The filtered dataset is never called "model accuracy". Columns (one row per
symbol × issue date):

- Identity/timing: `prediction_id`, `symbol`, `prediction_timestamp`,
  `model_version`, `daily_window_start/end`, `weekly_window_end`,
  `training_start_timestamp`, `training_end_timestamp`,
  `training_label_end_timestamp`, `n_training_rows`, `feature_row_timestamp`,
  `prediction_row_in_training_set`
- Model: `logistic_probability`, `rf_probability`, `xgb_probability`,
  `ensemble_probability`, `predicted_direction`, `confidence`,
  `confidence_bucket`, `production_accuracy_fast`, `forward_row_*`
- Decision: `signal`, `confluence` (0–1, as filtered/persisted),
  `weighted_score` (−1..+1), `market_regime` (per-stock `detect_regime`, from
  bars ≤ D), `timeframe_score`, `news_score`, `news_status`,
  `passes_quality_filters`, `production_cmp`
- Outcomes per h ∈ {1,3,5,10}: `outcome_date_hd`, `actual_price_hd`,
  `return_hd`, `outcome_hd`, `correct_hd`, `forward_row_correct_hd`,
  `baseline_majority_correct_hd`, `baseline_prev_direction_correct_hd`,
  `signal_success_hd`, `legacy_cmp_return_hd`, `legacy_cmp_signal_success_hd`
- `prediction_price`, `realized_issue_move`, `majority_class_training`

Confidence buckets are [50,55), [55,60) … [95,100]. They are descriptive
only; nothing is calibrated. Confluence buckets are 0.05 wide over the
observed range, and their rates are suppressed when n < 30. Regime and
confluence calculations are unchanged.

A file artifact was chosen over a DB table: the benchmark is offline,
deterministic and versionable. Nothing about it belongs in the serving DB.

## 11. News timestamp integrity (Task 16)

`news/api.py:fetch_news()` queries live Google News RSS and keeps **only
`entry.title`**. No publish timestamp is captured or persisted, and RSS
cannot be queried "as of" a past date. Persisted recommendations do not store
headlines either. Point-in-time news therefore **cannot be guaranteed** for
any historical T. The benchmark does not pretend otherwise:

- News is marked `unavailable_no_point_in_time_source` on every row.
- The production no-headlines path (score 0.0) is used. That is a real
  production state, but it means the replayed signal/confluence is
  "production with neutral news". Rows whose live signal depended on news
  can differ.
- The ML direction, confidence, regime, technical, momentum, volume and
  timeframe inputs are unaffected.
- News-related findings in Phase 9's historical data cannot be validated by
  this benchmark.

## 12. Benchmark entry point and configs (Tasks 17, 18, 20)

```
python scripts/audit/walk_forward_benchmark.py --config dev    # ~2 min
python scripts/audit/walk_forward_benchmark.py --config full   # ~22 min
```

| Config | Symbols | Issue dates | Stride |
|---|---|---|---|
| `dev` | 6 (first 2 of large/mid/small-cap CSVs) | 2026-06-01 → 2026-09-30 | every 5th session |
| `full` | 30 (first 10 of each CSV, hard-coded) | 2025-10-01 → 2026-09-30 | every 5th session |

- Stride 5 keeps 5-day outcomes non-overlapping per symbol and keeps runtime
  bounded (~0.7 s training per prediction, as in production `fast=True`).
- Prices: one yfinance download per symbol (2023-06-01 → 2026-10-01) into
  `scripts/audit/.price_snapshot/` (gitignored). They are always re-read from
  the snapshot, so a re-run against the same snapshot and code is
  byte-identical. `price_manifest.json` records the sha256, row count and
  date range of every input. `--refresh` re-downloads (Yahoo revises history,
  so results can move). Verified: two consecutive `full` runs produced
  byte-identical `raw_model_predictions.csv` (sha256 `0bc234d5…`),
  `production_recommendations.csv` (`80a28728…`) and `price_manifest.json`.
- Every row passes `assert_temporal_integrity()`: training end < T, training
  labels realised ≤ T, inputs ≤ T, feature row ≤ T, every outcome date > T. A
  violation raises `TemporalIntegrityError` and the run exits non-zero.
- Outputs: `scripts/audit/output/walk_forward_benchmark/<config>/`
  (`raw_model_predictions.csv`, `production_recommendations.csv`,
  `summary.json`, `summary.md`, `price_manifest.json`). Only `full` is
  committed.

## 13. Results — `full` config

Pre-fix (Phase 10) run. Full tables:
`scripts/audit/output/walk_forward_benchmark/phase10_prefix_full/summary.md`
(human) and `summary.json` (machine). All rates have Wilson 95% intervals.
Each rate is descriptive for this population and period only.

### 13.1 Population

| Item | Value |
|---|---|
| Symbols | 27 of 30. AEGISCHEM.NS, AKZOINDIA.NS and ALEXOTYRES.NS returned no Yahoo data. |
| Issue dates | 2025-10-01 → 2026-09-30, 50 per symbol (every 5th session) |
| Predictions | 1,350 (0 skipped, 0 temporal violations) |
| Rows with outcome | 1D 1,350 · 3D 1,323 · 5D 1,323 · 10D 1,296 (the rest are NULL: future not yet available) |
| Filter survivors (B) | 696 (51.6%) |
| Signals, all / filtered | BUY 396/165 · HOLD 637/364 · SELL 255/150 · STRONG BUY 58/16 · STRONG SELL 4/1 |

### 13.2 Temporal integrity

- Prediction row inside its own training set: **100%** of rows.
- As-deployed prediction == already-realised D-1→D move: **1,313 / 1,350 =
  97.3%** (CI 96.2–98.0).
- As-deployed and forward-row predictions agree on 59.9% of rows. Mean
  confidence is 74.9 as deployed vs 62.9 on the unseen row.

### 13.3 Ensemble walk-forward CV (Task 2), production 1-year window ending at each symbol's last issue date

| | Value |
|---|---|
| Symbols / pooled test rows | 27 / 2,430 |
| Mean actual-ensemble accuracy | 51.7% (per-symbol 40.0%–63.3%) |
| Mean legacy mean-of-individual-models (what `train_model` reports) | 51.4% |

### 13.4 A. Raw model benchmark: direction accuracy (all 1,350 predictions)

| Horizon | As deployed | Forward-row diagnostic | Baseline: majority class | Baseline: previous direction | Actual up % |
|---|---|---|---|---|---|
| 1D | 45.6% (42.9–48.2) | 50.7% (48.1–53.4) | 49.9% (47.2–52.5) | 46.8% (44.2–49.5) | 47.3 |
| 3D | 45.4% (42.8–48.1) | 49.6% (46.9–52.3) | 49.3% (46.6–52.0) | 47.2% (44.5–49.9) | 51.5 |
| 5D | 46.9% (44.2–49.6) | 51.8% (49.1–54.5) | 48.9% (46.2–51.6) | 48.6% (45.9–51.3) | 49.5 |
| 10D | 50.4% (47.7–53.1) | 51.5% (48.7–54.2) | 48.5% (45.7–51.2) | 50.3% (47.6–53.0) | 50.9 |

- The as-deployed upper bound is below 50% at 1D, 3D and 5D. Its numbers
  track the previous-direction baseline, which it agrees with 97.3% of the
  time.
- The forward-row diagnostic's interval contains 50% at every horizon.
- The majority-class baseline predicted "up" on only ~33% of rows (most
  1-year training windows were majority-down).

Confidence buckets (1D direction accuracy):

| Bucket | As deployed n / acc | Forward-row n / acc |
|---|---|---|
| 50–55 | 5 / 0.0% | 286 / 52.4% |
| 55–60 | 38 / 52.6% | 300 / 54.3% |
| 60–65 | 107 / 45.8% | 254 / 52.0% |
| 65–70 | 188 / 45.7% | 216 / 44.9% |
| 70–75 | 317 / 48.3% | 167 / 43.7% |
| 75–80 | 328 / 44.2% | 74 / 55.4% |
| 80–85 | 262 / 42.7% | 42 / 57.1% |
| 85–90 | 87 / 46.0% | 9 / 44.4% |
| 90–95 | 18 / 55.6% | 2 / 50.0% |
| 95–100 | 0 | 0 |

Neither column rises with confidence. Not calibrated, by design (measure only).

### 13.5 B. Production recommendation benchmark: signal success (696 filter survivors, entry = Close[D])

| Signal | 1D | 3D | 5D | 10D |
|---|---|---|---|---|
| BUY (n=165) | 51.5% (43.9–59.0) | 50.9% (43.3–58.4) | 52.1% (44.5–59.6) | 47.2% (39.6–54.9), n=161 |
| HOLD (n=364) | 88.2% (84.5–91.1) | 70.5% (65.5–75.0), n=352 | 58.0% (52.7–63.0), n=352 | 42.4% (37.3–47.7), n=344 |
| SELL (n=150) | 50.7% (42.7–58.6) | 45.5% (37.6–53.6), n=145 | 48.3% (40.3–56.3), n=145 | 48.6% (40.4–56.8), n=140 |
| STRONG BUY (n=16) | 31.2% (14.2–55.6) | 37.5% (18.5–61.4) | 31.2% (14.2–55.6) | 37.5% (18.5–61.4) |
| STRONG SELL (n=1) | n too small | | | |

Reference, the same success rules applied to every replayed row regardless
of signal (descriptive base rates):

| Horizon | up rule (`>0`) | down rule (`<0`) | HOLD band (`abs ≤ 3%`) |
|---|---|---|---|
| 1D | 47.3% | 52.4% | 88.4% |
| 3D | 51.5% | 48.3% | 70.5% |
| 5D | 49.5% | 50.2% | 56.3% |
| 10D | 50.9% | 48.9% | 45.1% |

HOLD success is driven mostly by the ±3% band and how wide that band is
relative to the horizon (e.g. 88% of all rows stay within ±3% after one day).

Existing-cmp convention (entry = Close[D-1]) on the same 696 rows, at 1D:
BUY 55.8%, HOLD 71.4%, SELL **78.7%** (vs 50.7% with entry = Close[D]). The
gap is the already-known bar-D move that the as-deployed ML direction
encodes. This is why existing validation records must not be read as forward
performance.

### 13.6 Confluence and regime

- Observed confluence range: 0.248–0.801 (all replayed rows). That is wider
  than Phase 9's historical maximum of ~0.69: the period is different, the
  label bug is gone, and news is neutral.
- 5D signal success by 0.05 confluence bucket (filter survivors, n ≥ 30):
  0.30–0.35 44.9%, 0.35–0.40 51.9%, 0.40–0.45 58.9%, 0.45–0.50 50.7%,
  0.50–0.55 58.9%, 0.55–0.60 61.2%, 0.60–0.65 52.1%, 0.65–0.70 50.0%. The
  intervals overlap and there is no monotonic pattern; buckets above 0.70 have n < 30.
- Filter survivors by per-stock regime, 5D success: Bearish 45.5% (n=77),
  Bullish 58.2% (134), High Volatility 47.9% (194), Sideways 58.0% (274).
  These mix signal types: Sideways contains many HOLDs, which are easier to
  satisfy.

## 14. Phase 9 diagnostic vs Phase 10 benchmark (Task 23)

| | Phase 9 `model_metrics_audit.py` | Phase 9 `horizon_audit.py` | Phase 10 benchmark |
|---|---|---|---|
| Question | Can the ensemble classify next-day direction on held-out rows? | How did persisted recommendations do at fixed horizons? | What did production actually output at T, and what happened next? |
| Population | 12 symbols, last 20% of a 2-year series, every row | 480 validated DB rows (25 most frequent symbols), filter survivors only, mostly pre-fix (contaminated) | 30 symbols × every 5th session, Oct 2025 → Sep 2026, all replayed predictions (A) and filter survivors (B) |
| Model | Fit once on first 80% | Whatever ran live (incl. label bug) | Re-fit at every T on the production 1-year window |
| Predicted row | Unseen test rows | Live `X.iloc[-1:]` | As-deployed `X.iloc[-1:]` **and** unseen bar-D row (diagnostic) |
| Entry price | n/a (direction only) | stored `cmp` (Close[D-1]) | Close[D] (plus legacy-cmp view) |
| Horizon | next row (1 day) | 1/3/5/10 from saved_date | 1/3/5/10 sessions from D, holidays excluded |
| News | not used | live at the time | neutral, flagged |

The populations, periods and predicted rows differ. **Numbers must not be
compared directly across columns.** The closest analogue to Phase 9's
55.3% is the forward-row 1D accuracy, but it uses a different fit window, a
different period and a different symbol set.

## 15. Data limitations

- 30-symbol deterministic universe, not the full scan universe.
  AEGISCHEM.NS, AKZOINDIA.NS and ALEXOTYRES.NS return no Yahoo data, so 27
  were evaluated. Delisted or renamed symbols are absent, which is a survivorship bias.
- One year of issue dates: a single market period, so results are
  regime-dependent.
- Stride-5 sampling; 10-day outcomes overlap between consecutive issue dates
  of the same symbol. Rows are not independent across symbols on the same day
  (common market moves), so the Wilson intervals are optimistic.
- The last issue dates have no 5D/10D outcome yet (NULL, counted per horizon).
- News is neutralised (section 11).
- Daily bars only: intraday scanner runs (partial bar D) are not replayed.
- Yahoo adjusted history can be revised; the snapshot hash pins the inputs used.

## 16. What was not changed (Task 24, 25)

Hyperparameters, ensemble weights, feature list, target definition,
confidence/confluence/BUY/SELL thresholds, `MIN_*` filters, Top Picks
ranking, recommendation scoring, `walk_forward_validate()` /
`train_model()` output, `storage/recommendation_validation.py`, historical DB
rows, `StockAIPro.Mobile`, authentication (password, Google/Apple SSO, OTP,
PIN, device sessions).

Code changed: `models/trainer.py` (blend extracted, ensemble walk-forward
reporting added), `storage/tracker.py` (duplicate prevention),
`utils/helpers.py` (comment only), new `evaluation/walk_forward.py`, new
`scripts/audit/walk_forward_benchmark.py`, `.gitignore`.

## 17. Open evaluation-integrity concerns (documented, not fixed)

1. ~~Production predicts its own last training row~~ (section 2). **Fixed in
   Phase 11A** (section 19).
2. ~~Stored `cmp` is `Close[D-1]`~~. **Fixed in Phase 11A**: `cmp = Close[D]`.
   Rows saved before the fix keep `Close[D-1]`.
3. News has no point-in-time record: no timestamps, no persisted headlines.
4. The existing validator's horizon is still "whenever an admin triggered it"
   (preserved by design; the benchmark measures fixed horizons separately).
5. 20 legacy duplicate rows remain in the local DB (cleanup is an explicit
   operator decision).
6. `utils/decision_engine.py` module docstring still lists stale pillar
   weights (Phase 9 finding).

## 18. Tests

- `tests/test_walk_forward_benchmark.py`: ensemble weights and bit-identical
  blend, vectorised blend, XGBoost fallback, ensemble walk-forward equals a
  manual per-fold blend, legacy value reproduced, folds train strictly before
  test, `train_model` output unchanged, exact trading-day horizons across
  weekends, NULL on missing future, holiday placeholders excluded, entry
  price at T, future-poisoning invariance, record timestamps, issue-row
  features independent of the next bar, three assertion-failure cases, news
  flagged, previous-direction baseline uses only ≤ T, **exact parity with
  `scanner/engine.py:_scan_one`**, success rules unchanged, as-deployed row is
  bar T and not in training (inverted in Phase 11A), assertion fails if the
  prediction row is in training, confidence bucket boundaries, Wilson interval, missing
  outcomes ignored.
- `tests/test_tracker_dedup.py`: idempotent upsert, 8-thread race → one row,
  unique index on fresh DB, legacy duplicates untouched and new duplicates
  blocked, upsert on a legacy key updates.

## 19. Phase 11A: post-fix benchmark and pre-/post-fix comparison

The same benchmark (`--config full`, same snapshot, same symbols, issue
dates, horizons, baselines, news treatment and summary code) was re-run
after the production fix. The only methodology change is what the
production path does: the replay calls `prepare_inference_data()` exactly as
`scanner/engine.py` now does, and `assert_temporal_integrity()` additionally
fails a row whose prediction row is in its training set.

Artifacts: `scripts/audit/output/walk_forward_benchmark/full/` (post-fix),
`phase10_prefix_full/` (pre-fix), and `full/phase10_vs_phase11a.{md,json}`
from `python scripts/audit/compare_benchmarks.py`. Two consecutive post-fix
runs produced byte-identical outputs (section 19.5).

### 19.1 Integrity

| | Phase 10 (pre-fix) | Phase 11A (post-fix) |
|---|---|---|
| Prediction row inside its own training set | 100% | **0%** |
| Prediction == already-realised D-1→D move | 97.3% (96.2–98.0) | **58.7% (56.1–61.3)** |
| Temporal-assertion violations | 0 | 0 |
| Stored entry price | Close[D-1] | Close[D] |

The memorisation behaviour is gone. The remaining 58.7% is legitimate, not
leakage. Bar D's features include `Price_Change` (the D-1→D return itself)
and momentum terms, so a model can learn continuation from information that
is known at T. Prediction and realised move were independent at their base
rates (predicted up 42.5%, realised up ≈ 47%), agreement would be ≈ 50.5%.

### 19.2 Raw model direction accuracy (1,350 predictions, 27 symbols)

| Horizon | Phase 10 model | Phase 11A model | Majority-class baseline | Previous-direction baseline |
|---|---|---|---|---|
| 1D | 45.6% (42.9–48.2) | **50.7% (48.1–53.4)** | 49.9% (47.2–52.5) | 46.8% (44.2–49.5) |
| 3D | 45.4% (42.8–48.1) | **49.6% (46.9–52.3)** | 49.3% (46.6–52.0) | 47.2% (44.5–49.9) |
| 5D | 46.9% (44.2–49.6) | **51.8% (49.1–54.5)** | 48.9% (46.2–51.6) | 48.6% (45.9–51.3) |
| 10D | 50.4% (47.7–53.1) | **51.5% (48.7–54.2)** | 48.5% (45.7–51.2) | 50.3% (47.6–53.0) |

- Baselines are identical before and after (same rows, no model involved).
- The post-fix numbers equal Phase 10's forward-row diagnostic exactly. That
  is expected (the same fitted models applied to the same bar-D row) and
  independently confirms the fix.
- Every post-fix interval contains 50%. With this population and period, the
  fixed ensemble's direction accuracy is not distinguishable from a coin flip
  or from the baselines. It is no longer systematically below 50%, but there
  is no evidence of skill either.
- Ensemble walk-forward CV is unchanged (51.7% ensemble / 51.4% legacy),
  because the training set is unchanged.

Confidence buckets (1D, post-fix): 50–55 52.4% (n=286), 55–60 54.3% (300),
60–65 52.0% (254), 65–70 44.9% (216), 70–75 43.7% (167), 75–80 55.4% (74),
80–85 57.1% (42). Confidence is lower than before (mean ≈ 63 vs 75), because
it no longer reflects in-sample memorisation. It is still not monotonic
with accuracy.

### 19.3 Production recommendations (filter survivors)

The population changed because the ML inputs to the unchanged filters
changed: survivors went from **696 to 552**, mainly because lower confidence
fails `MIN_CONFIDENCE`. All signals before filtering: BUY 371, HOLD 686,
SELL 232, STRONG BUY 56, STRONG SELL 5. After filtering: BUY 121, HOLD 327,
SELL 94, STRONG BUY 8, STRONG SELL 2.

| Signal | 1D (10 → 11A) | 3D | 5D | 10D |
|---|---|---|---|---|
| BUY | 51.5% → 51.2% (n=121) | 50.9% → 50.0% | 52.1% → 52.5% | 47.2% → 50.4% |
| HOLD | 88.2% → 88.7% (n=327) | 70.5% → 70.8% | 58.0% → 59.3% | 42.4% → 43.4% |
| SELL | 50.7% → 51.1% (n=94) | 45.5% → 42.0% | 48.3% → 53.4% | 48.6% → 46.2% |
| STRONG BUY | 31.2% → 50.0% (n=8) | 37.5% → 37.5% | 31.2% → 37.5% | 37.5% → 37.5% |
| STRONG SELL | n=1 → n=2 | too small | too small | too small |

All BUY/SELL intervals overlap 50% and the unconditional rule rates (5D: up
49.5%, down 50.2%, HOLD band 56.3%). STRONG signals have n ≤ 8. With the
stored price now Close[D], the "production stored cmp" view equals the clean
view (e.g. 1D SELL 51.1% in both; pre-fix it was inflated to 78.7%).

Confluence (5D, survivors, n ≥ 30): 0.30–0.35 50.0%, 0.35–0.40 57.9%,
0.40–0.45 55.1%, 0.45–0.50 58.4%, 0.50–0.55 60.6%, 0.55–0.60 61.5%,
0.60–0.65 46.4%, 0.65–0.70 52.9%. Observed range 0.265–0.798. No monotonic
pattern. Regime (5D, survivors): Bearish 62.1% (n=58), Bullish 56.3% (103),
High Volatility 53.2% (171), Sideways 57.7% (208), mixing signal types.

### 19.4 Reading these numbers

The fix made the reported performance *honest*, not better. Pre-fix, the ML
pillar replayed yesterday's move with ~75% "confidence", and the stored price
counted a known day into every return. Post-fix, the deployed ensemble is
genuinely out-of-sample, and on this benchmark it shows ~50% direction accuracy
across 1–10 days. Improving that is a modelling question for a later phase.
None was attempted here.

### 19.5 Reproducibility

Two consecutive post-fix `full` runs against the same price snapshot
produced byte-identical `raw_model_predictions.csv`,
`production_recommendations.csv`, `price_manifest.json`, `summary.json` and
`summary.md`: sha256 `99b074de…` (raw), `79ce5d5c…` (filtered),
`2c6daad6…` (summary.json). The price manifest (`ede607e5…`) is identical
to the Phase 10 run's, so pre- and post-fix used the same inputs.
