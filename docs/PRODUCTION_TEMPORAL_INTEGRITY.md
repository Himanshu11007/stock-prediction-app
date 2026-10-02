# Production Temporal Integrity (Phase 11A)

Status: fixed in Phase 11A. This was a correctness fix only. Model
architecture, hyperparameters, features, target, ensemble weights,
thresholds, confidence/confluence formulas, BUY/HOLD/SELL mapping and Top
Picks ranking are unchanged.

Related: [WALK_FORWARD_BENCHMARK.md](WALK_FORWARD_BENCHMARK.md) (Phase 10
benchmark, pre-/post-fix results), [RECOMMENDATION_QUALITY_AUDIT.md](RECOMMENDATION_QUALITY_AUDIT.md) (Phase 9).

---

## 1. The Phase 10 problem

The Phase 10 replay of the unmodified scanner showed that production's ML
prediction was made on a row that was **part of the model's training set**,
and that its training label was the move that had **already happened**. In
1,350 replayed predictions the ML direction equalled the realised D-1 → D
move 97.3% of the time.

## 2. Root cause (verified in code)

`features/engineer.py:create_features()` did two jobs in one `dropna()`:

1. compute the features, and
2. attach the label `Up[t] = Close[t+1] > Close[t]`.

`Up` for the latest bar D is unknown (NaN), so the final `dropna()` removed
bar D's **entire row**, features included. Every production path then
predicted from the last remaining row:

| Path | Old prediction row |
|---|---|
| `scanner/engine.py:_scan_one` (Top Picks scan → `recommendation_validation`) | `X.iloc[-1:]` |
| `api/services.py:analyze_stock` (`POST /analyze-stock`) | `X.tail(1)` |
| `utils/recommendation_engine.py:get_top_recommendations` (legacy Streamlit) | `X.iloc[-1:]` |
| `app.py` "Analyse Stock" tab (Streamlit) | `X.tail(1)` |

That row is bar D-1. It is in `X`, `train_model()` fits on all of `X`, and
its label is `Close[D] > Close[D-1]`. RF/XGBoost memorise training rows, so
the prediction mostly reproduced that already-known move.

The same coupling made every other "latest" consumer one bar stale:
`detect_regime`, `generate_signal` (technical/volume/momentum pillars),
`passes_quality_filters`, `calculate_risk` (so stored `cmp = Close[D-1]`),
and `get_trend_signal` (daily and weekly multi-timeframe trend).

## 3. Old production timeline

```
history: bars ... D-2, D-1, D            (D = latest bar, e.g. today's close)
create_features -> rows ... D-2, D-1     (D dropped: Up[D] unknown)
train_model     -> fit on rows ... D-1   (D-1 labelled with Close[D] > Close[D-1])
predict         -> row D-1               (a TRAINING row; outcome already known)
decision data   -> ends at D-1           (regime, technicals, risk one bar stale)
stored cmp      -> Close[D-1]
saved_date      -> wall-clock date of the run (≈ D)
```

## 4. Corrected production timeline

```
history: bars ... D-2, D-1, D
compute_features -> features for every bar incl. D; Up[D] = NaN
training rows    -> rows with a known label: ... D-1  (UNCHANGED set)
prediction row   -> bar D features (X_pred); never in training
decision data    -> features through D (regime, technicals, filters, risk)
trend signals    -> latest complete feature row (bar D / current week)
stored cmp       -> Close[D]
outcome          -> D+1 .. D+10 sessions, observed later
```

### Training cutoff

Training rows are the rows whose label is known at T. The last one is bar
D-1, whose label uses `Close[D]`, which is known at the close of D. The
training set (rows, features and labels) is **byte-identical to before the
fix**; `test_training_set_is_unchanged_by_the_fix` asserts this.

### Prediction cutoff

The prediction row is bar D. Its features use bars ≤ D only (all features
are backward-looking: rolling, EWM, `diff`, `pct_change`, `shift(+n)`).
Perturbing any bar after D does not change bar D's features or the
prediction (tested).

### Label availability

| Row | Label `Up` | Known at T (close of D)? | Used for |
|---|---|---|---|
| ≤ D-1 | `Close[t+1] > Close[t]` | yes | training |
| D | needs `Close[D+1]` | **no** (NaN) | prediction only |

## 5. Implementation

| File | Change |
|---|---|
| `features/engineer.py` | `compute_features()` = the existing feature + label code without the final `dropna()`. `create_features()` = `compute_features().dropna()` (identical output). `get_trend_signal()` reads the latest complete feature row instead of the latest labelled row. |
| `utils/helpers.py` | `FEATURE_COLS` (same list) and `prepare_inference_data()` returning `InferenceData(data, train_data, X, y, y_train, X_pred)`. Raises if the prediction row were ever in `X`. |
| `scanner/engine.py`, `api/services.py`, `utils/recommendation_engine.py`, `app.py` | Train on `X, y`; predict `X_pred`; decision engine, regime, filters and risk receive `data` (features through D). `app.py` keeps the labelled frame for its backtest chart. |
| `evaluation/walk_forward.py` | Replay mirrors the new production path; `assert_temporal_integrity()` now also fails if the prediction row is in the training set. |

`prepare_data()` is unchanged and is no longer used by production prediction
paths.

### When bar D has no complete feature row

If any feature of bar D is NaN, `X_pred` is `None` and the path **skips**
(scanner: symbol skipped; API: `400` with an explicit message; Streamlit:
error). It never falls back to an older row, since that would reintroduce
the bug. The known trigger is the first session after a Yahoo zero-volume
holiday placeholder bar: `Volume_Change = x / 0 = inf`. Changing that
feature definition is out of scope (see section 10).

## 6. Entry price and date convention

| Item | Before | After |
|---|---|---|
| Prediction date | `saved_date` = wall-clock date of the run | unchanged |
| Feature cutoff | bar D-1 (ML), bar D-1 (decision) | bar D |
| Stored entry price (`cmp`) | `Close[D-1]` | `Close[D]` (rounded to 2 dp by `calculate_risk`) |
| 1D / 3D / 5D / 10D outcome | — | `Close` of the 1st / 3rd / 5th / 10th trading session after D (benchmark) |

The old convention was temporally inconsistent: the recommendation was issued
at the close of D but priced at `Close[D-1]`, so every measured return
included bar D's move, which was already known. The correction was the
smallest one available: `cmp` comes from the same `data` frame, which now
ends at D. No column, schema or migration change was needed.

Unchanged and documented:

- `saved_date` is the wall-clock date, not the bar date. On a non-trading
  day it differs from D. Runs during market hours use Yahoo's partial
  current bar as D (live price), so the prediction and `cmp` reflect that
  intraday snapshot.
- The existing validator (`storage/recommendation_validation.py`) still
  validates "whenever an admin triggers it" from `saved_date`. That is
  deliberately left as-is; fixed-horizon measurement lives in the benchmark.
- Historical `recommendation_validation` rows are not rewritten. Rows saved
  before Phase 11A have `cmp = Close[D-1]` and a leaked ML direction, and
  must not be pooled with post-fix rows for model evaluation.

## 7. News treatment

- Production: news is fetched live at run time (`fetch_news`), so it cannot
  come from the future relative to the prediction. It only affects the News
  pillar of confluence, never the ML prediction (tested: ±1.0 sentiment
  leaves ML confidence/direction/accuracy identical).
- Benchmark: unchanged from Phase 10. No point-in-time news source exists, so
  news is neutral (0.0) and flagged `unavailable_no_point_in_time_source`.
  The replay never calls `fetch_news` (tested). No timestamps are fabricated.

## 8. Ensemble evaluation

Unchanged: LR 0.20 / RF 0.30 / XGBoost 0.50 (`ENSEMBLE_WEIGHTS`, sum 1.00).
Without XGBoost, its weight is applied to the RF probability. Same
hyperparameters, same `train_model()` (and its reported accuracy, which uses
the unchanged training set). The clean benchmark measures the deployed
ensemble on bar D; see [WALK_FORWARD_BENCHMARK.md](WALK_FORWARD_BENCHMARK.md) §19.

## 9. Regression tests protecting the contract

`tests/test_production_temporal_integrity.py`:

| Test | Guards |
|---|---|
| `test_prediction_row_is_the_latest_bar` | prediction row = bar D |
| `test_prediction_row_is_not_in_training` | D ∉ X, D ∉ y, max(X) < D |
| `test_only_rows_with_known_labels_are_trained_on` | every training label = recomputed `Close[t+1] > Close[t]`; no unknown label |
| `test_training_set_is_unchanged_by_the_fix` | X, y identical to `prepare_data()` |
| `test_create_features_is_compute_features_minus_incomplete_rows` | refactor is behaviour-preserving |
| `test_prediction_features_match_the_feature_definition_at_D` | X_pred = feature definition at D |
| `test_decision_data_ends_at_D` | regime/technicals/risk see bar D |
| `test_incomplete_latest_bar_yields_no_prediction_row` | no fallback to an older row |
| `test_future_prices_do_not_change_features_at_D` | D+1.. prices cannot reach D's features |
| `test_future_prices_do_not_change_the_prediction` | … nor the prediction/signal |
| `test_news_cannot_change_the_ml_prediction` | news affects only the News pillar |
| `test_benchmark_never_fetches_news` | historical replay uses no live news |
| `test_prediction_no_longer_reproduces_the_known_last_move` | **the Phase 10 bug**: same fitted models, old row reproduces the realised D-1→D move ≥ 90% (observed 20/20), bar D ≤ 75% (observed 11/20) |
| `test_scanner_predicts_bar_D_and_stores_close_D` | scanner predicts D and stores Close[D] |
| `test_api_analyze_predicts_bar_D_and_stores_close_D` | `/analyze-stock` path, same |
| `test_ensemble_weights_unchanged`, `test_xgboost_fallback_unchanged` | model definition |
| `test_production_prediction_is_deterministic` | identical output on repeat |
| `test_entry_price_and_outcome_dates` | entry = Close[D], outcomes at D+h sessions |

`tests/test_walk_forward_benchmark.py` (Phase 10) still passes, including exact
parity with `scanner/engine.py:_scan_one`. Its "prediction row inside training
set" test is inverted into `test_as_deployed_prediction_row_is_bar_T_and_not_in_training`,
plus a new assertion-failure case.

## 10. Remaining concerns and future work (not done here)

- **Post-holiday gap:** the first session after a zero-volume holiday bar
  yields no prediction (section 5). Options for a future phase: drop Yahoo
  holiday placeholder bars at load time, or guard `Volume_Change` against a
  zero denominator. Both change model inputs, so both are out of scope here.
- Historical DB rows before Phase 11A carry the leaked ML direction and the
  D-1 entry price. Keep them out of model evaluation (consider an
  `engine_version` bump in a later phase).
- `saved_date` is the wall-clock date rather than the bar date; intraday
  runs use a partial bar.
- News has no point-in-time record.
- Model quality itself (the benchmark's post-fix numbers) is a separate
  question for the next phase. No tuning was done here.
