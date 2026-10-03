# StockLens — Recommendation Quality Audit (Phase 9)

> **Naming note (2026-10-03):** the product was renamed from StockAI Pro to **StockLens**; the "StockLens Score" in this document is the same score previously called "StockAI Score". The research, data and conclusions are unchanged.

**Status: AUDIT ONLY.** Nothing in this document changed model weights,
thresholds, hyperparameters, feature sets, train/test splits, recommendation
logic, Top Picks ranking, or the mobile app. Two things were touched outside
pure documentation/scripts/tests, both explicitly permitted by the audit
scope and both described in full in section 21:

1. A regression test suite for a bug that was **already fixed** in a prior
   commit (`8997c5a`, 2026-09-27) — added because the audit found the bug
   had no test coverage at all, and the bug was severe enough (see section
   6) that a silent regression would be dangerous.
2. Three new read-only diagnostic scripts under `scripts/audit/`, which
   never write to `storage/tracker.db` and never save a model artifact.

No model was retrained for production use. No threshold was changed. No
recommendation behavior changed.

---

## 1. Executive summary

- The production ML pipeline (feature engineering → ensemble model →
  confluence scoring → quality filters → persistence → 5-trading-day
  validation) was traced end-to-end from source, not assumed.
- **The single most important finding**: a data-integrity bug in
  `features/engineer.py` fabricated a `False`/`0` ("down") label for the
  most recent trading day's `Up` target — a day whose true outcome was not
  yet knowable — for every stock, on every scan, from at least
  **2026-07-12 through 2026-09-21** (18 consecutive scan dates, 1,473
  recorded rows). This was already fixed in commit `8997c5a`
  (2026-09-27), independently of this audit, but it had **zero regression
  test coverage**, so this audit added one (section 22/23).
- Because of that bug's timing, **100% of the 1,679 validated
  recommendations currently in `storage/tracker.db` predate the fix**.
  None of the historical "track record" numbers in this document (sections
  9-15) describe the current, fixed system. They describe a system that
  was, for at least 2.5 months, structurally incapable of predicting
  "up" for any stock. This is reported in full in section 9, not hidden.
- The actual 5-trading-day validation horizon is **not actually 5 trading
  days**: because `validate_old_recommendations()` only runs when an admin
  manually triggers it (no scheduled job exists), the real elapsed time at
  validation in the current database ranges from **5 to 39 trading days**
  (mean 17.7, median 13). Every `return_pct`/`success` value in the
  database is therefore computed over an inconsistent, admin-dependent
  holding period, not a clean 5-day window.
- A fresh, out-of-sample evaluation (independent of the contaminated
  database, using the *current, fixed* code against newly downloaded
  data) found the production ensemble's out-of-sample accuracy
  (mean 55.3% across 11 sampled stocks) only modestly beats a naive
  majority-class baseline (53.2%), and is below that baseline for some
  individual stocks. In-sample accuracy for the same stocks averaged
  99.0% — a 43.7-point gap that demonstrates why an in-sample number
  would have been dangerously misleading, and underscores that the
  production model actually used for live predictions (trained on 100% of
  history, not a held-out split) likely carries meaningfully more
  overfitting risk than the validation accuracy shown to users suggests.
- The model's stated confidence is **not well-calibrated**: pooling fresh,
  uncontaminated out-of-sample predictions across 11 stocks (869
  observations) into confidence bands, observed accuracy sits in a flat
  49-60% range across every band with a usable sample (50% through 85%
  stated confidence) — the 70-75%-confidence band was actually *less*
  accurate (49.2%) than the 50-55%-confidence band (59.6%). Section 8.3.
- Several structural measurement gaps exist with no current evidence
  either way: zero sector data (no `Sector` column anywhere in the stock
  universe CSVs), zero `STRONG BUY` signals in the entire historical
  database, and a selection-bias-by-design validation dataset (only
  recommendations that already passed quality filters are ever persisted
  for later analysis).
- No claim of profitability is made anywhere in this document. The
  evidence available does not support one.

---

## 2. Current architecture (as traced from source)

```
yfinance (live fetch, 1h rolling cache, "2y"/"1y" window)
        │
        ▼
features/engineer.py :: create_features()
   Price_Change, Momentum, MA_5/10, EMA_20/50, RSI, MACD, Bollinger,
   ATR, ADX, Volume_Change/MA/Ratio, Vol_Breakout  (all backward-looking)
   + target: Up = 1 if Close[t+1] > Close[t] else 0  (NaN kept for the
     unknowable final row, then dropped)
        │
        ▼
utils/helpers.py :: prepare_data()  — picks 27 feature columns
   (incl. raw Close and Volume levels), does ITS OWN chronological 80/20
   split (X_train/X_test/y_train/y_test) — see section 7, this split's
   output is discarded by every call site except the y_train class check
        │
        ▼
models/trainer.py
   _make_candidates(): Logistic Regression, Random Forest, and
   XGBoost IF INSTALLED (optional import - see section 2.1)
   walk_forward_validate(): expanding-window, time-ordered CV -> wf_acc
     (fast=True path: single chronological 80/20 split instead, for
     background scans where speed matters)
   train_model(): validates via the above, THEN refits all 3 candidates
     on the FULL dataset for the actual deployed model
   ensemble_predict(): LR*0.20 + RF*0.30 + XGB*0.50 -> final_prob
        │
        ▼
utils/decision_engine.py :: generate_signal()
   8 weighted pillars (ML Direction 15%, ML Confidence 5%, Technical 30%,
   News 10%, Volume 5%, Regime 10%, Timeframe 15%, Momentum 10% -
   config.py is the single source of truth, asserted to sum to 1.0)
   -> weighted score in [-1,+1] -> mapped to [0,100]
   -> STRONG BUY / BUY / HOLD / SELL / STRONG SELL via config.py thresholds
   (72 / 58 / 42 / 28)
        │
        ├──> utils/risk.py (ATR-based stop/target - display only)
        ├──> utils/explainability.py (recomputes the SAME 8 pillars via
        │     decision_engine's own private functions - no duplicated
        │     formula, no drift possible)
        │
        ▼
scanner/filters.py :: passes_quality_filters()
   accuracy / confidence / confluence score / avg volume / RSI / volatility
   gates (bullish-only for several) - A RECOMMENDATION THAT FAILS THIS
   IS NEVER PERSISTED (see section 16 - selection bias)
        │
        ▼
scanner/engine.py :: get_recommendations()
   parallel scan (ThreadPoolExecutor) -> filter -> sort by score ->
   rerank top-20 using already-fetched news_score -> re-persist top-20
   with the mutated post-rerank score
        │
        ▼
storage/tracker.py :: upsert_recommendation()  (dedup key: symbol+saved_date)
        │
        ▼
storage/recommendation_validation.py :: validate_old_recommendations()
   MANUALLY TRIGGERED ONLY (admin API call / Streamlit button - no cron).
   Gate: >=5 trading days elapsed. Fetches current price via fresh
   yfinance call, computes return_pct/success, writes back.
        │
        ▼
storage/performance_analytics.py + analytics/recommendation_intelligence.py
   Read-only aggregation over recommendation_validation (signal/confidence/
   confluence/sector/regime bucketing, pillar correlation, insights)
```

### 2.1 Correcting an assumption in the task brief

The task brief stated "the current ML architecture includes Logistic
Regression and Random Forest." **That is incomplete.** `models/trainer.py`
also conditionally includes **XGBoost** (`from xgboost import XGBClassifier`,
guarded by `_XGB_AVAILABLE`), and when available it receives the **largest
single weight in the ensemble blend (50%)** — larger than LR (20%) and RF
(30%) combined is not quite right, but it is larger than either individually,
and larger than LR+RF's combined 50% is exactly tied. XGBoost **was**
installed and used in this environment (confirmed: `pip show xgboost`
succeeds, and the diagnostic scripts in this audit used it). This is
exactly the kind of "inspect, don't assume" discrepancy the audit was
asked to catch.

---

## 3. Target definition (traced, not assumed)

**Exact definition**, from `features/engineer.py`:

```python
next_close = data["Close"].shift(-1)
data["Up"] = np.where(next_close.isna(), np.nan,
                       (next_close > data["Close"]).astype(float))
```

- **Variable**: `Up`, binary (0.0 / 1.0 after `dropna()` + `.astype(int)`
  downstream in `prepare_data`).
- **Horizon**: exactly 1 trading day ahead (next row in the OHLCV series,
  not a calendar day).
- **Positive (1)**: `Close[t+1] > Close[t]` (strictly greater — a flat/zero
  move is **not** positive).
- **Negative (0)**: `Close[t+1] <= Close[t]` (flat and down are both
  labeled 0 — there is no separate "flat" class; the target is strictly
  binary).
- **Zero/flat returns**: silently folded into the negative class. A model
  trained on this target cannot distinguish "went down" from "stayed
  exactly flat" — both are label 0.
- **Final row handling**: the last row's `next_close` is `NaN` (no
  tomorrow exists in the fetched window), so `Up` is explicitly set to
  `NaN` and removed by the trailing `data.dropna()`. **This was not
  always true** — see section 6 for the pre-2026-09-27 bug where this row
  was fabricated as `0` instead.
- **Minimum data requirement**: `create_features()` needs enough rows for
  its longest `.rolling()` window (20, for `Volume_MA`/Bollinger Bands) to
  produce a non-NaN value, plus the one row lost to the target's forward
  shift. In practice this means roughly 20-25 leading rows are consumed
  before any usable training row exists; `get_trend_signal()` separately
  enforces a `len(data) >= 60` floor before calling `create_features` at
  all.
- **Rows removed**: every row inside the longest rolling window's warm-up
  period, any row where `.pct_change()`/`.diff()` produces `NaN` (just the
  first row), and the single final row (unknowable target). For a typical
  ~250-trading-day input, this removes roughly 20-21 rows total (confirmed
  empirically: a 474-row 2-year fetch yielded exactly 474 usable feature
  rows after `create_features()` in this audit's own diagnostic runs —
  i.e. ~2 years of daily data survives with a stable, modest warm-up
  cost).

---

## 4. Dataset description

Two materially different "datasets" exist and **must not be conflated**:

| | **Historical validation DB** | **Fresh audit diagnostics** |
|---|---|---|
| Source | `storage/tracker.db` → `recommendation_validation` | Live `yfinance` downloads run during this audit |
| Code version | Whatever was running in production at `saved_date` (spans the pre-fix bug) | Current `main` (post-fix) |
| Rows | 2,158 total; 1,679 validated (`is_validated=1`); 479 pending | 12-25 symbols sampled, fresh 2-year history each |
| Date range | `saved_date` 2026-06-13 → 2026-09-29 | Download executed 2026-10-02; covers ~2024-10 → 2026-10 |
| Distinct symbols | 117 (115 with ≥1 validated row) | 11-25 (two deterministic, fixed samples — see scripts) |
| Known contamination | **100% of validated rows predate the 2026-09-27 label-fabrication fix** | None — runs current code only |

No synthetic/fabricated data was used anywhere in this audit. Where data
was insufficient to answer a task, that is stated explicitly rather than
extrapolated (see section 19).

---

## 5. Feature inventory (traced from `utils/helpers.py:prepare_data` + `features/engineer.py`)

| Feature | Source | Calculation | Lookback | Available at prediction time? | Leakage risk | Notes |
|---|---|---|---|---|---|---|
| `Close` | raw | — | 0 | Yes | None | **Raw price level used as a direct model feature** — not leakage (legitimately known at T), but non-stationary across a multi-year history and across stocks of very different price scales. Flagged as a feature-quality concern, not leakage (see section 26). |
| `Volume` | raw | — | 0 | Yes | None | Same raw-level concern as `Close`. |
| `Price_Change` | derived | `Close.pct_change()` | 1 | Yes | None | |
| `MA_5`, `MA_10`, `MA_Diff` | derived | rolling mean | 5, 10 | Yes | None | |
| `EMA_20`, `EMA_50`, `EMA_Cross`, `Price_vs_EMA20` | derived | `.ewm()` | ~20, ~50 effective | Yes | None | |
| `RSI` | derived | Wilder EWM of gain/loss | 14 (com=13) | Yes | None | |
| `Momentum` | derived | `Close - Close.shift(5)` | 5 | Yes | None | |
| `Volatility` | derived | rolling std | 10 | Yes | None | |
| `Volume_Change`, `Volume_MA`, `Volume_Ratio` | derived | pct_change / rolling mean | 1, 20 | Yes | None | |
| `MACD`, `MACD_Hist`, `MACD_Cross` | derived | EWM(12)/EWM(26)/EWM(9) | ~26 effective | Yes | None | |
| `BB_Width`, `BB_Position` | derived | rolling mean/std(20) | 20 | Yes | None | |
| `ATR`, `ATR_Pct` | derived | Wilder EWM of true range | 14 (com=13) | Yes | None | |
| `ADX`, `Plus_DI`, `Minus_DI` | derived | Wilder EWM of directional movement | 14 (com=13) | Yes | None | |
| `Vol_Breakout` | derived | `Volume_Ratio>1.5 & Price_Change>0` | 20 | Yes | None | |
| `Up` (target) | derived | `Close.shift(-1) > Close` | forward 1 | **No — by design, this is the label** | N/A (label, not feature) | Correctly excluded from `feature_cols` in `prepare_data` — confirmed by direct code read, not assumption. |

**Conclusion of the per-feature trace**: every production feature is
computed strictly from data at-or-before row T (the row's own `.rolling()`/
`.ewm()`/`.shift(k)` with `k>=0` windows, never a negative/forward shift).
**No feature-level temporal leakage was found.** The only notable issue is
the inclusion of raw, non-normalized `Close`/`Volume` levels as features,
which is a model-quality concern, not a leakage concern.

---

## 6. Critical finding: the final-row label-fabrication bug (fixed, undertested)

### What was found

Querying `storage/tracker.db` directly:

```
pillar_ml_dir distribution (recommendation_validation, all rows):
  NULL : 206   (rows saved before this column existed)
  -0.7 : 1473  (i.e. EVERY recorded ML prediction was "bearish")
  +0.7 :    0  (until 2026-09-27)
```

Broken down by `saved_date`, the picture is unambiguous:

```
2026-07-12 through 2026-09-21  (18 scan dates, 1,473 rows): 100% bearish, 0% bullish
2026-09-27                      (first mixed date)         : 12 bearish / 28 bullish
2026-09-28, 2026-09-29                                     : mixed, in similar proportion
```

A per-stock, independently-trained ensemble unanimously predicting "down"
for ~90 unrelated stocks, every single day, for 2.5 months straight, is not
a plausible market outcome — it is the signature of a systematic bug.

### Root cause

`git log` shows exactly one relevant commit in this window:

```
8997c5a  Sun Sep 27 15:49:30 2026 +0530
  "features/engineer.py + utils/helpers.py: stop fabricating a False/0
   label for the last row's unknown next-day close (Up column) — keep it
   NaN so dropna() removes it instead of training on a made-up outcome"
```

Before this fix, the final row of every `create_features()` call — the
single most recent, most feature-relevant row, structurally almost
identical to the exact row later used for the live prediction
(`X.iloc[-1:]`) — carried a **fabricated label of "down"** that survived
`dropna()` (because `0` is not `NaN`). Every model, for every stock, on
every scan, was trained with one extra, false "recent-looking features →
down" example baked in, for 2.5+ months. This elegantly explains both the
unanimous bearish bias and why it disappeared precisely on 2026-09-27.

### Current state

The fix is confirmed present in the current `main` branch (read directly,
section 3 above). **No further code change was needed from this audit.**

### What this audit added

Zero tests existed for `features/engineer.py` before this audit (confirmed:
`grep` across `tests/` for `create_features`/target-construction logic
returned nothing relevant). A bug this severe, with this little test
coverage, could silently return. This audit added
`tests/test_feature_engineering.py` (8 tests, all passing) asserting:

- the input's last row is never present in `create_features()`'s output,
- the output is strictly shorter than the input,
- `Up` contains no NaN/Inf and is strictly `{0.0, 1.0}`,
- `Up[t]` exactly matches `Close[t+1] > Close[t]` recomputed independently
  from the raw input for every retained row (full alignment check, not a
  spot check),
- explicit known-up and known-down cases.

### Consequence for every number in sections 9-15 below

**All 1,679 validated recommendations in the current database were
produced before this fix.** There are zero validated post-fix
recommendations (the 158 rows saved on/after 2026-09-27 have not yet
accumulated 5+ trading days as of this audit's execution date). Every
historical performance statistic quoted below describes the broken,
always-bearish system, not the current one. This is stated at the top of
every relevant table below, not just here.

---

## 7. Train/test & validation methodology

### 7.1 ML train/test split

**Current method**: Two different, independent split mechanisms exist and
their outputs are used inconsistently:

- `utils/helpers.py:prepare_data()` computes a chronological 80/20 split
  (`X_train`/`X_test`/`y_train`/`y_test`), **but every production call
  site discards `X_train`/`X_test`/`y_test` and only reads `y_train`** (to
  check for single-class data before training). This split's actual
  train/test matrices are **dead output** in production.
- `models/trainer.py:train_model()` does its own, separate thing: either
  a full expanding-window `walk_forward_validate()` (default,
  `fast=False` — used by `app.py`, `utils/recommendation_engine.py`,
  `api/services.py`) or a single chronological 80/20 `_fast_accuracy()`
  (`fast=True` — used only by `scanner/engine.py`'s background scans).
  Both are time-ordered; **neither shuffles data** — this is the
  methodologically correct choice for financial time series and was
  confirmed present in the actual code, not assumed.
- After validation, `train_model()` **refits all three candidate models
  on 100% of the available data** (`model.fit(X, y)` — not just the
  training split) and returns *those* fitted models as the ones used for
  the live prediction (`X.iloc[-1:]`).

**Why this is worth flagging (not necessarily wrong)**: training the final
deployed model on all available history, after separately validating
methodology via a held-out split, is a defensible and common practice.
**The issue is what gets reported to the user**: the `accuracy` value
shown alongside a recommendation comes from the (smaller, more properly
held-out) walk-forward models, while the actual prediction was made by a
*different* model object trained on more data. Section 9's in-sample
evidence (99.0% mean in-sample accuracy vs. 55.3% mean out-of-sample
accuracy, a 43.7-point gap) strongly suggests the all-data-fit production
model likely carries meaningfully more overfitting risk than the reported
accuracy number implies, since that number was never measured against
*that specific* model.

**Recommended future method** (not implemented in this phase): report a
second number — the full-data model's in-sample fit quality or, better, a
rolling walk-forward accuracy computed from a *held-out-from-the-start*
test tail that is never used to fit the final model either. This is a
genuine methodology question for Phase 10, not a bug to patch now.

### 7.2 Walk-forward validation — already exists, confirmed by reading the code

`models/trainer.py:walk_forward_validate()` implements genuine expanding-
window walk-forward CV: fold *i* trains on `X[:train_end]` and tests on
`X[train_end:test_end]`, strictly forward in time, no shuffling. This
satisfies Task 5's core question — **yes, genuine out-of-sample evaluation
infrastructure exists** in this codebase already, and was not something
this audit needed to build from scratch.

**One real measurement gap found**: each fold's reported accuracy is the
**simple unweighted average of the three candidate models' individual
`.score()` values** (`sum(fold_accs) / len(fold_accs)`), not the accuracy
of the actual weighted-probability ensemble (`ensemble_predict`'s
20/30/50 blend) that produces the live prediction. These are different
numbers measuring different things. This is a genuine, demonstrable
measurement-validity gap: **the walk-forward accuracy shown to users does
not actually measure the accuracy of the ensemble mechanism used to make
the prediction** — it measures something else (the mean of three
separately-scored models). Not fixed in this phase (would require
changing what `walk_forward_validate` returns, a production-metric
change outside audit scope) — flagged as a Phase 10 experiment.

### 7.3 Recommendation validation horizon — the single biggest methodology concern

Already covered in full in the executive summary and section 6's framing;
restated precisely here because it affects every statistic below:

- `is_ready_for_validation()` only checks `elapsed >= 5` trading days — it
  is a *floor*, not a target.
- `validate_old_recommendations()` is **never scheduled** — it only runs
  when `POST /tracker/validate-old` is called (admin-only) or a Streamlit
  button is clicked.
- Measured directly from the database: actual elapsed trading days at
  validation range from **5 to 39** (mean 17.7, median 13.0), clustered
  into runs (5-10, then 12-19, then 30-39) that correspond to how
  infrequently the admin actually triggered validation.
- **Consequence**: `return_pct` and `success` are not comparable across
  rows — one row's "return" might be a true 5-day return, another's a true
  35-day return, with no column recording which. Every aggregate statistic
  computed over this table implicitly averages across wildly different
  holding periods.

---

## 8. Model metrics — fresh, out-of-sample, using CURRENT (fixed) code

Because the historical database is 100% pre-fix (section 6), this section
uses `scripts/audit/model_metrics_audit.py`, which re-runs the **exact,
unmodified** `features.engineer.create_features` and
`models.trainer._make_candidates`/`ensemble_predict` against freshly
downloaded data for a fixed, deterministic 12-symbol sample (4 each from
the large/mid/small-cap universe files' first rows). One symbol
(`AEGISCHEM.NS`) had insufficient data and was excluded; **n=11** stocks
produced a result.

**Method**: single chronological 80/20 holdout per stock (no shuffling,
no cross-stock mixing). This is intentionally the *simpler* of the two
production split strategies (matching `fast=True`'s approach) so the
in-sample/out-of-sample comparison in section 8.2 is apples-to-apples.

### 8.1 Out-of-sample metrics (mean across 11 stocks)

| Metric | Value |
|---|---|
| Accuracy | 0.553 |
| Precision | 0.586 |
| Recall | 0.194 |
| F1 | 0.245 |
| ROC-AUC | varies 0.45–0.65 per stock (see per-stock table below) — several stocks **below** 0.50 |

Per-stock detail (all 11, full confusion matrices in
`scripts/audit/output/model_metrics_audit.json`):

| Symbol | N test | Accuracy | Precision | Recall | F1 | ROC-AUC | Baseline (majority) | Baseline (prev. direction) |
|---|---|---|---|---|---|---|---|---|
| ADANIENT.NS | 95 | 0.505 | 1.000 | 0.078 | 0.146 | 0.557 | 0.463 | 0.579 |
| ADANIPORTS.NS | 95 | 0.642 | 0.813 | 0.295 | 0.433 | 0.543 | 0.537 | 0.495 |
| APOLLOHOSP.NS | 95 | 0.516 | 0.000 | 0.000 | 0.000 | 0.471 | 0.516 | 0.516 |
| ASIANPAINT.NS | 95 | 0.568 | 0.600 | 0.383 | 0.468 | 0.574 | 0.505 | 0.432 |
| ABCAPITAL.NS | 95 | 0.568 | 0.556 | 0.119 | 0.196 | 0.594 | 0.558 | 0.411 |
| ABFRL.NS | 95 | 0.453 | 0.387 | 0.632 | 0.480 | 0.448 | **0.600** | 0.558 |
| ACC.NS | 95 | 0.579 | 0.467 | 0.179 | 0.259 | 0.487 | 0.590 | 0.516 |
| AIAENG.NS | 95 | 0.579 | 0.571 | 0.098 | 0.167 | 0.462 | 0.432 | 0.505 |
| AARTIIND.NS | 95 | 0.568 | 0.571 | 0.095 | 0.163 | 0.509 | 0.558 | 0.474 |
| AAVAS.NS | 95 | 0.642 | 1.000 | 0.056 | 0.105 | 0.520 | **0.621** | 0.558 |
| AETHER.NS | 95 | 0.463 | 0.476 | 0.200 | 0.282 | 0.478 | 0.474 | 0.463 |

**Bold** = the model's own accuracy was *at or below* the naive
majority-class baseline for that stock (2 of 11 stocks). The model beat
the majority baseline in 9 of 11, by a median margin of roughly 3-5
points.

### 8.2 In-sample vs. out-of-sample (Task 5 — the overfitting demonstration)

| | Mean across 11 stocks |
|---|---|
| In-sample accuracy (fit on all data, score on the same data) | **0.990** |
| Out-of-sample accuracy (fit on first 80%, score on held-out last 20%) | **0.553** |
| Gap | **0.437** (43.7 points) |

This is unambiguous, direct evidence that these models (particularly
Random Forest and XGBoost, given tree-based models' capacity to memorize)
can fit historical data almost perfectly while generalizing only
marginally better than chance. **This demonstrates concretely why the
existing walk-forward/chronological-holdout methodology (section 7.2) is
not optional window-dressing — it is the only thing standing between the
reported accuracy and a number this misleading.**

### 8.3 Probability calibration (fresh, uncontaminated data — Task 10)

Pooling every out-of-sample (probability, actual-outcome) pair across all
11 sampled stocks (869 observations) and bucketing by the same confidence
definition the production `ensemble_predict()` reports
(`max(prob, 1-prob)*100`):

| Confidence band | N | Observed success rate |
|---|---|---|
| 50-55% | 141 | 59.6% |
| 55-60% | 188 | 55.3% |
| 60-65% | 212 | 56.6% |
| 65-70% | 192 | 55.7% |
| 70-75% | 185 | **49.2%** |
| 75-80% | 93 | 54.8% |
| 80-85% | 29 | 65.5% |
| 85-90% | 5 | 40.0% (N too small to trust) |
| 90%+ | 0 | no data |

**This data is NOT the contaminated production database** — it comes
fresh from the current (fixed) code against newly downloaded prices, so
unlike section 9.3's pre-fix table, this one is a trustworthy (if
small-sample) answer to Task 10's question. **The answer is no: stated
confidence does not correspond to observed accuracy.** A "70-75%
confidence" prediction was correct only 49.2% of the time — worse than
the "50-55% confidence" band's 59.6%. Observed success sits in a fairly
flat 49-60% range across every band with a usable sample size (50-85%),
regardless of how confident the model claimed to be. **The model is not
well-calibrated** in this sample. No calibration fix was applied — this
is reported as a measurement, consistent with the audit's no-tuning rule.

### 8.4 Baseline comparison

| | Mean |
|---|---|
| Production ensemble (out-of-sample) | 0.553 |
| Always predict majority class | 0.532 |
| "Tomorrow repeats today's direction" | 0.501 (indistinguishable from a coin flip, as expected) |

The ensemble beats the majority-class baseline by **2.1 points on
average**, is **worse than it for 2 of 11 stocks**, and the margin varies
widely by stock (from -14.7 points on ABFRL.NS to +14.7 points on
ADANIENT.NS, though ADANIENT's high accuracy comes with recall of only
7.8% — it is barely ever predicting "up" at all; see precision=1.0,
recall=0.078 in the table above). **No blanket claim that "the model beats
a naive baseline" is supportable — it depends heavily on the stock.**

---

## 9. Historical validated-recommendation metrics (⚠️ pre-fix data — see section 6)

**Every number in this section describes the always-bearish, now-fixed
system. Read section 6 before drawing any conclusion from this section.**

Computed by actually running the existing, unmodified
`analytics/recommendation_intelligence.py:generate_engine_report()`
against the real `storage/tracker.db` (1,679 validated rows).

### 9.1 Overall summary

| Metric | Value |
|---|---|
| Total validated | 1,679 |
| Successful | 777 |
| Failed | 902 |
| Success rate | 46.3% — 95% CI **[43.9%, 48.7%]** |
| Average return | **-1.32%** |
| Median return | -1.25% |
| Best / worst single return | +33.84% / -28.2% |

The average validated recommendation **lost money** over its (irregular,
5-39 day) holding period in this dataset.

### 9.2 Signal-specific performance

Confirmed by direct code read of `storage/recommendation_validation.py` —
the success rules documented in the task brief are exactly what the
code implements (no discrepancy found):

```
BUY / STRONG BUY   → success if return_pct > 0
SELL / STRONG SELL → success if return_pct < 0
HOLD               → success if abs(return_pct) <= 3.0%
```

| Signal | N | Success rate (95% CI) | Avg return | Median return |
|---|---|---|---|---|
| STRONG BUY | **0** | N/A — no data | N/A | N/A |
| BUY | 197 | 41.6% [35.0%, 48.6%] | -2.13% | -1.65% |
| HOLD | 914 | 40.2% [37.0%, 43.4%] | -1.36% | -1.62% |
| SELL | 538 | 58.2% [54.0%, 62.3%] | -1.08% | -0.81% |
| STRONG SELL | 30 | 50.0% [33.2%, 66.8%] | +0.83% | -0.73% |

**Zero STRONG BUY signals exist anywhere in the historical database** —
no performance claim about STRONG BUY can be made from this data at all.

**Interpretation caveat on "avg return"**: for SELL/STRONG SELL rows,
`return_pct` is the *raw underlying price change*, not a direction-adjusted
P&L. A successful SELL call (price fell, `return_pct<0`) still shows a
*negative* average return in this column — that is the price falling, not
a trading loss. Reading "-1.08% avg return" for SELL as "the strategy lost
1.08%" would be a misinterpretation unless one assumes the user actually
shorted the stock. This nuance exists in how the system reports these
numbers today and is noted here as a reporting-clarity issue, not a
calculation bug.

### 9.3 Confidence-band analysis

| Confidence | N | Success rate | Avg return |
|---|---|---|---|
| 50-60% | 28 | 60.7% | -1.53% |
| 60-70% | 322 | 46.9% | -1.46% |
| 70-80% | 825 | 45.3% | -1.29% |
| 80-90% | 470 | 46.2% | -1.27% |
| 90-100% | 34 | 52.9% [36.7%, 68.5%] | -1.48% |

**No monotonic relationship between stated confidence and observed
success is present in this (contaminated) data.** The lowest-N band
(50-60%, N=28) shows the *highest* success rate; 70-80% (the largest band,
N=825) shows the *lowest*. Given the entire dataset reflects an
always-bearish model, this should not be read as "confidence is
meaningless" — it may equally reflect that confidence was being computed
correctly on top of a structurally broken direction signal. This question
needs to be re-asked once genuine post-fix validated data exists (see
section 20 recommendations).

### 9.4 Confluence-score threshold analysis

| Confluence ≥ | N | Success rate | Avg return |
|---|---|---|---|
| 0.50 | 538 | 38.8% | -2.00% |
| 0.55 | 331 | 37.5% | -2.04% |
| 0.60 | 135 | 45.2% | -2.07% |
| 0.65 | 33 | 63.6% [46.6%, 77.8%] | -0.59% |
| 0.70 | **0** | no data | — |

The confluence score **never reached 0.70** anywhere in this dataset
(confirmed: `MAX(confluence_score) = 0.6899`). This is mechanically
explained by the bug in section 6: with `ML Direction` pillar score
locked at -0.7 for the entire window, and that pillar carrying real
weight in the blend, the maximum achievable confluence score was
structurally capped below what it would be with a mix of bullish and
bearish ML directions. **Any claim about "high confluence" (≥0.70)
performance is unsupported by any data in this system right now.**

### 9.5 Pillar correlation with success

| Pillar | Correlation with success | Interpretation |
|---|---|---|
| Multi-Timeframe | -0.189 | weak **negative** |
| Technical Analysis | -0.174 | weak **negative** |
| Momentum | -0.127 | weak **negative** |
| Market Regime | -0.100 | negligible |
| ML Confidence | +0.044 | negligible |
| Volume | -0.014 | negligible |
| News Sentiment | +0.003 | negligible |
| ML Direction | ~0.000 | **artifact, not a finding** — see note below |

**Important note on ML Direction's ~0 correlation**: this is not evidence
that ML direction doesn't matter. Pearson correlation against a
**constant** variable (ML Direction pillar score is exactly -0.7 for 1,473
of 1,473 non-null rows) is mathematically undefined; the code's safe
fallback returns 0.0 in that case. The ~0 correlation reported here is an
artifact of the section-6 bug, not a real measurement of ML direction's
predictive value.

**None of the eight confluence pillars show a strong positive correlation
with success in this dataset**, and three show a weak *negative* one.
Given the whole dataset is pre-fix, this cannot be read as "the confluence
weights are wrong" — but it is also not evidence that they are right. This
is a clear candidate for re-measurement once clean, post-fix data exists.

### 9.6 Market-regime performance

| Regime | N | Success rate | Avg return |
|---|---|---|---|
| Bearish | 201 | 55.2% | -0.74% |
| Unknown | 206 | 51.0% | +1.88% |
| Sideways | 651 | 47.9% | -1.81% |
| High Volatility | 387 | 41.9% | -2.23% |
| Bullish | 234 | 37.2% | -1.79% |

Regime is computed per-stock from the same already-correct feature row
(section on regime audit, section 12) — no leakage found. Observationally,
"Bullish"-regime recommendations had the *worst* average return in this
dataset, and "Bearish"-regime ones the best — plausible given every
recommendation in this window came from a model stuck predicting "down,"
meaning bearish-regime stocks matching that bearish model bias performed
best. No causal claim is made.

### 9.7 Sector analysis — no data

`get_sector()` (`utils/company_mapper.py`) returns `None` for every stock
because none of `data/largecap.csv`/`midcap.csv`/`smallcap.csv` has a
Sector/Industry column. **100% of rows are "Unknown" sector.** No
sector-level analysis is possible with the current data. This is a data
gap, not a bug.

### 9.8 Per-stock sample sizes

115 distinct stocks have at least one validated row. Distribution:

- 4 stocks: N=1
- 12 stocks: N<5 (too small to draw any conclusion about)
- 96 stocks: N≥10
- Median N per stock: 17; max N: 23

Per-stock rankings are **not reported** in this document — with the
entire dataset contaminated by the section-6 bug, a per-stock "best/worst"
table would create a false impression of stock-specific insight when the
actual driver was a system-wide defect.

---

## 10. True horizon analysis (1/3/5/10 trading days — Task 12)

The existing `return_pct` column cannot answer this (section 7.3).
`scripts/audit/horizon_audit.py` reconstructs true, consistent forward
returns from each symbol's own historical price series, using the
**unmodified** `calculate_return()`/`calculate_success()` functions, for
the 25 symbols with the most recommendation rows (480 recommendation rows,
1,920 horizon observations — still pre-fix data, see section 6 caveat).

| Horizon | N | Success rate | Avg return | Median return |
|---|---|---|---|---|
| 1 day | 480 | 76.0% | -0.09% | -0.20% |
| 3 days | 480 | 68.5% | +0.10% | -0.09% |
| 5 days | 480 | 61.0% | +0.04% | -0.14% |
| 10 days | 480 | 57.7% | -0.59% | -0.79% |

By signal (1/5/10-day shown; full 3-day table in the JSON output):

| Signal | 1d success | 5d success | 10d success | N |
|---|---|---|---|---|
| BUY | 42.9% | 54.8% | 33.3% | 42 |
| HOLD | 91.0% | 62.0% | 57.1% | 245 |
| SELL | 63.6% | 60.4% | 63.6% | 187 |
| STRONG SELL | 83.3% | 83.3% | 66.7% | 6 |

**Key observation**: success rate **changes materially with horizon** —
HOLD's apparent 91% one-day success collapses to 57% by ten days (HOLD
"succeeds" trivially over 1 day because ±3% in a single session is a wide,
easy-to-clear band; over 10 days that band is much easier to breach). This
is direct, concrete evidence that **the choice of validation horizon
materially changes the measured outcome**, independently confirming why
the admin-dependent, inconsistent horizon described in section 7.3 is a
genuine measurement problem and not a minor technicality.

---

## 11. Confluence audit (mechanism — Task 13)

Traced directly from `utils/decision_engine.py` and `config.py`:

- 8 pillars, weights from `config.py` (`W_ML_DIR=0.15, W_ML_CONF=0.05,
  W_TECH=0.30, W_NEWS=0.10, W_VOLUME=0.05, W_REGIME=0.10,
  W_TIMEFRAME=0.15, W_MOMENTUM=0.10`), asserted at import time to sum to
  1.0.
- **Found and worth noting**: `decision_engine.py`'s own module docstring
  states different weights (25/15/35/10/5/10/15/10, summing to 125%) than
  the actual `config.py` values it imports and uses. The *code* is
  correct (imports from `config.py`, asserts the sum); the *docstring
  comment* is stale/wrong. This is a documentation-accuracy bug, not a
  logic bug — flagged for a trivial doc fix in Phase 10, not changed here
  since the audit scope asked for correctness fixes only where they
  affect measurement, and this affects no computed value.
- `ML Confidence` pillar's contribution is `_ml_confidence_score(confidence)
  * ml_dir`, where `ml_dir` is ±0.7 (not ±1) — so the confidence pillar's
  effective magnitude is scaled down by the direction pillar's own 0.7
  magnitude, a compounding interaction worth knowing about but not a bug.
- Score mapping: weighted sum in [-1,+1] → `(weighted+1)*50` → [0,100] →
  bucketed via `config.py`'s `STRONG_BUY_MIN=72, BUY_MIN=58, HOLD_MIN=42,
  SELL_MIN=28` (below 28 → STRONG SELL). Matches the task brief exactly,
  verified from source.
- `utils/explainability.py:compute_pillar_scores()`/`compute_weighted_score()`
  (used to persist pillar data for later analysis) call
  `decision_engine`'s own private pillar functions directly — **confirmed
  no duplicated/drifted formula** between what produces the live signal
  and what gets persisted for analysis.

**Confluence vs. outcome** (pre-fix data, section 9.4): no reliable signal
— the top bucket (≥0.65, N=33) shows a notably higher success rate
(63.6%) than lower buckets, but with a 95% CI of [46.6%, 77.8%] on N=33,
and given the whole dataset is pre-fix, this is **suggestive at best, not
conclusive**, and the ≥0.70 bucket has zero data at all (section 9.4).

---

## 12. Market regime audit (Task 8)

Traced from `utils/regime.py:detect_regime()`:

- Despite the name "market regime," this is a **per-stock**, not a
  market-wide, classification — it reads the last row's `ADX`, `Close`,
  `EMA_20`, `EMA_50`, `MACD_Hist`, `ATR_Pct`, `RSI` (all already correctly
  computed, backward-looking features from the same `create_features()`
  output used everywhere else).
- Pure rule-based thresholding (ATR%>3%→"High Volatility"; ADX<20→
  "Sideways"; else EMA/MACD/RSI majority vote→"Bullish"/"Bearish"/mixed
  "Sideways").
- **No future data enters this calculation** — it operates on
  `data.iloc[-1]`, the same latest-known row used for every other pillar.
  **No leakage found.**
- Naming nuance worth flagging for Phase 10: renaming this to "stock
  regime" or "technical regime" would more accurately describe what it
  measures, since no market-wide index data is involved at all.

---

## 13. News sentiment audit (Task 7)

Traced from `news/api.py:fetch_news()` and `news/sentiment.py`:

- `fetch_news()` queries Google News RSS **live, for "now"** — it has
  **no timestamp field at all**: `feedparser`'s `entry.published`/
  `entry.published_parsed` fields are available on the parsed feed but
  are **never read or stored** — only `entry.title` is kept.
- For every current call site (`scanner/engine.py`, `utils/
  recommendation_engine.py`), this is **not leakage**: a live scan at
  time T fetching "current news at T" to predict T+1 is legitimate.
- **However, this function structurally cannot be used for historical
  backtesting without modification.** It has no mechanism to fetch "news
  as of a past date" — any hypothetical future backtest that called
  `fetch_news()` for a historical `saved_date` would silently receive
  *today's* news, not that date's, which would be a real and serious
  leakage path. **No code path in the current repository does this today**
  (confirmed: no backtest/historical-replay code calls `fetch_news`), so
  there is no active leakage — but the risk is structural and should be
  fixed before any future backtesting work uses this function (Phase 10
  recommendation, section 20).
- `news/sentiment.py`: FinBERT/TextBlob sentiment of a given headline
  string is deterministic and cached by headline text (process-lifetime,
  in-memory `dict`) — no temporal leakage possible here since sentiment of
  fixed text doesn't depend on when it's evaluated.
- 1-hour on-disk cache (`news/api.py:_NEWS_TTL=3600`) introduces at most
  1 hour of staleness, not a forward-looking leak.

---

## 14. Train/test temporal integrity summary (Task 4)

| Check | Finding |
|---|---|
| Training data precedes test data | **Yes**, in both split mechanisms (`prepare_data`'s 80/20 and `trainer.py`'s walk-forward/fast-accuracy) — confirmed by code read, both are strictly chronological slices with no shuffling. |
| Validation precedes final test | `walk_forward_validate` produces its own internal expanding folds; there is no separate "validation vs. test" distinction beyond that — acceptable for this use case. |
| Future observations leak backward | None found at the feature level (section 5). |
| Multiple stocks mixed incorrectly | **No** — every model is trained per-stock, on that stock's own chronological series only. No cross-sectional pooling of different stocks into one training set was found anywhere. |
| Duplicate dates/rows cross the boundary | `prepare_data`'s split output (`X_train`/`X_test`) is produced but **unused** by any production call site (section 7.1) — nothing to check here in practice. |
| Scaling fitted only on training data | **Yes, correctly** — `models/trainer.py`'s `StandardScaler` lives inside each `sklearn.Pipeline`, so `.fit()` during walk-forward/holdout only ever sees that fold's training rows; `.transform()` on the test fold uses only train-fitted statistics. Confirmed from source — no scaler is ever fit on the full dataset before splitting. |
| Feature selection uses future data | No feature-selection step exists in this pipeline at all — the feature list is a fixed, hand-written constant in `utils/helpers.py`. Nothing to leak. |

---

## 15. Validation integrity audit (Task 17)

- Recommendation generated at `saved_date` T; validated at T+N where
  **N is not fixed** (section 7.3) — this is the central finding of this
  entire audit.
- `get_latest_close()` fetches a fresh price **at validation time**, never
  from a cached/stale source — confirmed via direct `yf.download` call,
  no use of the 1-hour price cache (`data/loader.py`) for validation.
  **No evidence of validation using data unavailable at validation time.**
  The *consistency* problem (varying N) is separate from a *leakage*
  problem (using data from the future relative to validation time) — no
  leakage was found in the validation price fetch itself.
- Missing-price handling: `get_latest_close()` returns `None` on any
  failure; `validate_old_recommendations()` skips (not errors on) that
  row and leaves it pending — confirmed, no silent zero-fill.
- Failed recommendations remain in the dataset: confirmed —
  `load_validated_recommendations()`/`load_validated_df()` return all
  rows with `is_validated=1` regardless of `success` value; nothing
  filters out failures before aggregation.
- Reproducibility: re-running `validate_old_recommendations()` on an
  already-validated row is a guarded no-op (`is_validated=1` is excluded
  from `load_pending_recommendations()`'s query) — confirmed.
- **Data-integrity finding (new, found during this audit, not fixed)**:
  `storage/tracker.py:upsert_recommendation()` determines
  insert-vs-update via a separate `SELECT` then `INSERT`/`UPDATE` — not a
  single atomic `INSERT ... ON CONFLICT`. Querying the live database
  directly found **20 exact-duplicate `(symbol, saved_date)` pairs** among
  the 1,679 validated rows (1.2%), consistent with a race condition
  between concurrent scan threads/processes occasionally both seeing "no
  existing row" before either commits its insert. This causes mild
  double-counting in aggregate statistics. **Not fixed in this phase**
  (would require changing concurrent write logic in production
  persistence code, which is a larger change than this audit's "small,
  well-scoped correctness fix" allowance covers) — documented here and
  recommended for Phase 10/11 (add a `UNIQUE(symbol, saved_date)`
  constraint + `INSERT ... ON CONFLICT DO UPDATE`).

---

## 16. Top Picks audit (Task 18 — not changed)

Traced from `api/routes/top_picks.py` → `scanner/background.py` →
`scanner/engine.py:get_recommendations()`:

1. All stocks across Large/Mid/Small-cap universes are scanned in
   parallel (`ThreadPoolExecutor`).
2. Each result passes through `passes_quality_filters()` — **anything
   that fails is discarded before it is ever persisted** (see section 17
   for why this matters).
3. Surviving results are sorted by confluence `score` descending.
4. The **top 20** are re-scored using already-fetched `news_score`
   (`_rerank_top_with_news`: `score += news_score * 0.10`, clamped to
   [0,1]) and re-persisted with the new score — rows ranked 21+ keep
   their original, non-reranked score.
5. The final merged, re-sorted list is what the API/UI calls "Top Picks."

No ranking-algorithm code was changed. The selection-bias implication of
this exact mechanism is covered next.

---

## 17. Selection-bias audit (Task 16)

**Primary finding**: `scanner/engine.py:_scan_one()` calls
`_persist_recommendation()` **only if `passes_quality_filters()` returns
True** (confirmed by direct code read: the filter check is followed by an
early `return None` before persistence on failure). This means:

- `storage/tracker.db:recommendation_validation` contains **only
  recommendations that already looked good by the system's own filter
  criteria** (sufficient accuracy for bullish calls, sufficient
  confidence, sufficient confluence for bullish calls, sufficient
  liquidity, RSI range for bullish calls, no volatility spike).
- Every statistic in sections 9-11 of this document is therefore
  conditional on "recommendations the system already decided were worth
  showing," not a random or complete sample of "everything the model
  considered." **This is appropriate for answering "how good are the
  recommendations users actually see," but it is not a valid way to
  answer "how good is the raw ML model," which is why section 8's
  fresh, filter-free diagnostic exists as a separate measurement.**
- There is no evidence of survivorship bias beyond this (e.g., no
  mechanism found that removes previously-recorded *failed* validated
  recommendations after the fact — section 15 confirmed failures remain).
- The stock universe (`data/largecap.csv`/`midcap.csv`/`smallcap.csv`) is
  a fixed, current-day list — there is no evidence the system
  reconstructs a historically-accurate universe for past dates (e.g., a
  stock that has since been delisted or added to an index would not be
  retroactively excluded/included). This was not directly testable from
  the current data (no delisted-stock recommendations were found in the
  sample) but is a structural assumption worth flagging.

---

## 18. Sample-size and statistical-uncertainty notes (Task 19)

95% Wilson confidence intervals, computed directly from the live database
(not estimated):

| Statistic | N | Rate | 95% CI |
|---|---|---|---|
| Overall success rate | 1,679 | 46.3% | [43.9%, 48.7%] |
| BUY success rate | 197 | 41.6% | [35.0%, 48.6%] |
| SELL success rate | 538 | 58.2% | [54.0%, 62.3%] |
| STRONG SELL success rate | 30 | 50.0% | **[33.2%, 66.8%]** |
| Confidence 90-100% success rate | 34 | 52.9% | **[36.7%, 68.5%]** |
| Confidence 50-60% success rate | 28 | 60.7% | **[42.4%, 76.4%]** |
| Confluence ≥0.65 success rate | 33 | 63.6% | **[46.6%, 77.8%]** |

Every bolded row has a confidence interval wide enough to span both
"worse than a coin flip" and "clearly better than average" — **none of
these small-N buckets support a confident directional claim**, regardless
of which point estimate looks attractive. The task brief's own example
("78% success rate (N=9) is NOT equivalent to 78% (N=900)") is directly
illustrated here: several of this system's most attractive-looking
numbers come from N in the 28-34 range.

---

## 19. PASS / CONCERN / UNKNOWN scorecard (Task 20)

No overall score is given, per instructions. Each row is independently
assessed.

| Category | Status | Factual basis |
|---|---|---|
| **Data integrity** | CONCERN | Historical database is 100% pre-fix (section 6); 20 duplicate `(symbol, saved_date)` rows found (section 15/17) from a non-atomic upsert. |
| **Target definition** | PASS (current code) | `Up` is a clean, correctly-shifted, correctly-NaN-guarded binary target as of current `main` (section 3), confirmed by new regression tests. Historically CONCERN (section 6) until 2026-09-27. |
| **Temporal split** | PASS | Every split found (both `prepare_data`'s and `trainer.py`'s) is strictly chronological; scaler is fit train-only inside sklearn Pipelines (section 14). |
| **Feature leakage** | PASS | Full per-feature trace (section 5) found no forward-looking feature. Raw `Close`/`Volume` levels as features is a model-quality concern, not leakage. |
| **News leakage** | PASS (currently), CONCERN (structurally) | No current code path backtests with historical news (section 13); `fetch_news()` has no mechanism to fetch past-dated news at all, which would be a real risk if ever reused for backtesting. |
| **Regime leakage** | PASS | Regime is computed from the same already-correct, backward-looking feature row (section 12). |
| **Probability calibration** | CONCERN | Fresh, uncontaminated out-of-sample calibration table (section 8.3, 869 observations across 11 stocks) shows no monotonic relationship between stated confidence and observed accuracy — the 70-75% band (49.2%) is less accurate than the 50-55% band (59.6%). Sample is modest (11 stocks) but the data is clean (not the pre-fix contaminated DB), so this is a real, if preliminary, finding rather than an artifact. |
| **Validation integrity** | CONCERN | The 5-trading-day horizon is not actually enforced as a fixed horizon — real elapsed time ranges 5-39 trading days (section 7.3), making `return_pct` non-comparable across rows. |
| **Selection bias** | CONCERN (by design) | Only filter-passing recommendations are ever persisted (section 17) — appropriate for "how good are shown picks," invalid for "how good is the raw model." |
| **Reproducibility** | CONCERN | No recorded model-artifact version, no pinned data snapshot, and `fast=True` vs. `fast=False` produce materially different accuracy figures depending on which code path happened to run — see section 21. |

---

## 20. Concrete problems discovered (summary)

1. **(Fixed, now tested)** Final-row label fabrication in
   `features/engineer.py` biased every model toward "down" for 2.5+
   months (section 6). Already fixed in `8997c5a`; this audit added the
   missing regression tests (`tests/test_feature_engineering.py`).
2. **(Not fixed — measurement methodology)** The "5-trading-day"
   validation horizon is actually 5-39 trading days in practice because
   validation is never scheduled, only manually triggered (section 7.3).
3. **(Not fixed — measurement methodology)** Walk-forward CV's reported
   accuracy is the mean of three independently-scored models, not the
   accuracy of the actual weighted ensemble used for live predictions
   (section 7.2).
4. **(Not fixed — data-integrity)** `upsert_recommendation()`'s
   check-then-write is not atomic; 20 duplicate `(symbol, saved_date)`
   rows exist in the live database (section 15).
5. **(Not fixed — structural risk, currently dormant)**
   `news/api.py:fetch_news()` cannot fetch historical news and would leak
   future information if ever reused for backtesting (section 13).
6. **(Not fixed — documentation only)** `decision_engine.py`'s module
   docstring states pillar weights that don't match the actual
   `config.py` values the code uses (section 11).
7. **(Not fixed — data gap)** Zero sector data exists anywhere in the
   stock universe files; all sector analysis returns "Unknown" (section
   9.7).
8. **(Not fixed — data gap)** Zero `STRONG BUY` signals exist in the
   entire historical database — no evidence exists about this signal
   category at all (section 9.2).
9. **(Observed, not a bug)** Raw `Close`/`Volume` price levels are used
   directly as model features without normalization across time or
   across stocks (section 5) — a model-quality concern for Phase 10.
10. **(Observed, not a bug)** `utils/helpers.py:prepare_data()`'s own
    80/20 split output (`X_train`/`X_test`/`y_test`) is computed but never
    used by any production call site (section 7.1) — dead code, harmless,
    but worth removing or wiring up correctly.
11. **(Latent, unreachable today)**
    `analytics/recommendation_intelligence.py:generate_insights()` has a
    malformed format string (`f"{avg:+2.f}%"`) in a branch that is
    currently unreachable with the data this system produces (requires
    `"Signal"` to be absent from the dataframe, which never happens) — not
    fixed, since it doesn't affect any reported number today, but noted
    for cleanup.

---

## 21. Reproducibility notes (Task 24)

| | |
|---|---|
| Backend commit audited | `c5959b9` (HEAD at audit start), this audit's own commit on top |
| Database audited | `storage/tracker.db` (local dev SQLite, WAL mode) |
| Validated rows at audit time | 1,679 (`is_validated=1`), 479 pending |
| `saved_date` range | 2026-06-13 → 2026-09-29 |
| `validation_date` range | 2026-06-21 → 2026-09-20 |
| Distinct symbols in DB | 117 (115 with ≥1 validated row) |
| Audit execution date | 2026-10-02 |
| Fresh diagnostic data source | Live `yfinance` downloads executed 2026-10-02 (not reproducible bit-for-bit on a later date, since market data keeps accumulating — the *methodology* is reproducible, the *exact numbers* will drift as more trading days occur) |
| Model artifacts | None persisted/versioned anywhere in this repository — every prediction retrains from scratch on each call. There is no "model version" to pin beyond the git commit of `models/trainer.py`/`features/engineer.py`. |
| Diagnostic scripts | `scripts/audit/model_metrics_audit.py`, `scripts/audit/horizon_audit.py` — both deterministic in *symbol selection* and *split logic*, not in exact output (depends on live market data at run time) |

**No production database was modified.** All three new scripts are
read-only with respect to `storage/tracker.db` (the horizon/model-metrics
scripts never open it for writing; they only `SELECT`).

---

## 22. What was NOT changed (explicit list, per audit scope)

- `features/engineer.py`, `models/trainer.py`, `utils/decision_engine.py`,
  `utils/regime.py`, `utils/risk.py`, `utils/explainability.py` — **zero
  changes**. The label-fabrication bug (section 6) was already fixed
  before this audit began; this audit only added tests for it.
- `config.py` — **zero changes**. No threshold, weight, or model
  hyperparameter was touched.
- `scanner/engine.py`, `scanner/filters.py`, `scanner/background.py` —
  **zero changes**. Top Picks selection/ranking/filtering is exactly as
  found.
- `storage/recommendation_validation.py`,
  `storage/performance_analytics.py`,
  `analytics/recommendation_intelligence.py` — **zero changes**. The
  irregular-horizon issue and the duplicate-row issue were both
  documented, neither was patched.
- `storage/tracker.py:upsert_recommendation()` — **zero changes** despite
  the demonstrated race condition (section 15); flagged as a Phase 10/11
  fix, not made now.
- No row in `storage/tracker.db` was modified, deleted, or re-validated.
- `StockAIPro.Mobile` (the entire mobile repository) — **not touched at
  all** in this phase.
- Authentication (`auth/`, `api/routes/auth*.py`) — **not touched at
  all** in this phase.
- No new ML model, no LSTM, no additional technical indicator, no
  threshold change, no train/test split change, no target redefinition.

---

## 23. Recommended Phase 10/11 experiments (Task 26)

### 10.1 Schedule (or otherwise regularize) recommendation validation

**PROBLEM**: validation horizon is 5-39 trading days depending on when an
admin happens to trigger it (section 7.3).
**EVIDENCE**: direct measurement from the live database — distribution of
elapsed trading days at validation shown in section 7.3.
**PROPOSED EXPERIMENT**: add a scheduled job (even a simple daily cron)
that calls `validate_old_recommendations()` automatically, and/or record
`trading_days_elapsed` as its own column so historical rows can be
filtered/grouped by actual horizon rather than assumed to all be "5-day."
**EXPECTED MEASUREMENT**: `return_pct` becomes comparable across rows;
horizon-specific success rates (section 10's style of analysis) become
directly queryable from the production table instead of requiring a
separate reconstruction script.
**RISK OF OVERFITTING**: none — this is a measurement/infrastructure fix,
not a model change.

### 10.2 Report the walk-forward ensemble's actual accuracy, not the mean of its parts

**PROBLEM**: `walk_forward_validate()` reports the average of three
individually-scored models, not the accuracy of the blended
`ensemble_predict()` mechanism actually used for predictions (section
7.2).
**EVIDENCE**: direct code read of `models/trainer.py`.
**PROPOSED EXPERIMENT**: within each walk-forward fold, compute the
blended ensemble's prediction (same 20/30/50 weights) on the fold's test
rows and score *that*, instead of averaging the three models' own
`.score()` calls.
**EXPECTED MEASUREMENT**: a (likely slightly different, probably more
representative) accuracy number tied to the actual production mechanism.
**RISK OF OVERFITTING**: none if done as a pure measurement change; risk
only arises if this number is then used to re-tune the blend weights,
which would be a separate, explicit experiment requiring its own
held-out evaluation.

### 10.3 Re-measure confidence calibration and pillar correlations on clean, post-fix data

**PROBLEM**: every existing calibration/pillar-correlation number is
computed from the pre-fix, always-bearish dataset (sections 9.3, 9.5).
**EVIDENCE**: section 6's full timeline.
**PROPOSED EXPERIMENT**: simply wait for and then re-run
`analytics/recommendation_intelligence.py:generate_engine_report()` once a
meaningful number of post-2026-09-27 recommendations have been validated.
**EXPECTED MEASUREMENT**: genuinely informative confidence-band and
pillar-correlation tables for the first time.
**RISK OF OVERFITTING**: none — purely observational, same as this audit.

### 10.4 Investigate removing or transforming raw `Close`/`Volume` as features

**PROBLEM**: raw, non-normalized price/volume levels are used directly as
model inputs (section 5), which is non-stationary across a multi-year
history.
**EVIDENCE**: direct code read of `utils/helpers.py:prepare_data`.
**PROPOSED EXPERIMENT**: compare out-of-sample accuracy/ROC-AUC with vs.
without `Close`/`Volume` as raw features (keeping all derived/normalized
features unchanged), using the same chronological holdout methodology as
`scripts/audit/model_metrics_audit.py`.
**EXPECTED MEASUREMENT**: ROC-AUC/accuracy delta per stock; a reduction in
the in-sample/out-of-sample gap (section 8.2) would be a meaningful
positive signal.
**RISK OF OVERFITTING**: **moderate** — this is exactly the kind of
feature change that must be validated out-of-sample, not just tried until
accuracy improves on one sample. Must be run across many stocks and
multiple time windows before any conclusion is drawn.

### 10.5 Fix the non-atomic recommendation upsert

**PROBLEM**: `upsert_recommendation()`'s select-then-write race condition
has already produced 20 duplicate rows (section 15).
**EVIDENCE**: direct query of the live database.
**PROPOSED EXPERIMENT**: not really an "experiment" — a schema change
(`UNIQUE(symbol, saved_date)` constraint) plus `INSERT ... ON CONFLICT DO
UPDATE`, with a migration to de-duplicate existing rows first.
**EXPECTED MEASUREMENT**: zero duplicate rows going forward; a (tiny)
change in aggregate statistics once duplicates are removed.
**RISK OF OVERFITTING**: none — this is a correctness fix, not a modeling
change.

### 10.6 Decide, deliberately, what "the reported accuracy" should mean

**PROBLEM**: the accuracy shown to users comes from walk-forward-validated
models; the model actually used for the live prediction is a different
object trained on 100% of history (section 7.1), and section 8.2's
43.7-point in-sample/out-of-sample gap suggests that all-data model likely
overfits more than the shown number implies.
**EVIDENCE**: direct code read + this audit's fresh in-sample/
out-of-sample measurement.
**PROPOSED EXPERIMENT**: measure the all-data-fit model's performance on a
genuinely held-out *future* window (data not available at fit time, e.g.
the next N trading days after training) rather than only comparing
in-sample-vs-earlier-holdout.
**EXPECTED MEASUREMENT**: a true prospective accuracy figure for the
model that is actually deployed.
**RISK OF OVERFITTING**: none if done prospectively (can't overfit to data
you don't have yet); the risk is organizational (takes real calendar time
to accumulate enough future observations).

### 10.7 Investigate probability calibration (Platt scaling / isotonic regression)

**PROBLEM**: fresh, uncontaminated out-of-sample data shows the model's
stated confidence does not track observed accuracy — flat ~49-60%
observed success across the entire 50-85% confidence range (section 8.3).
**EVIDENCE**: this audit's own `scripts/audit/model_metrics_audit.py`
pooled-calibration table, 869 observations, current code, fresh data.
**PROPOSED EXPERIMENT**: fit a calibration mapping (Platt scaling or
isotonic regression, via `sklearn.calibration.CalibratedClassifierCV`) on
a held-out calibration fold per stock (or pooled across stocks, tested
both ways), then re-measure the same confidence-band table against a
*separate* held-out test fold to see whether calibrated confidence tracks
observed accuracy better than the raw ensemble blend does.
**EXPECTED MEASUREMENT**: a flatter-vs-steeper calibration curve
comparison (ideally plotted as a reliability diagram) between raw and
calibrated confidence, on data never used to fit the calibration mapping.
**RISK OF OVERFITTING**: **moderate** — calibration must be fit and
evaluated on genuinely disjoint folds, or the "fixed" calibration curve
will look good purely by construction. Must not be fit on the same fold
used to measure its own quality.

### 10.8 Harden `news/api.py:fetch_news()` before any future backtesting work touches it

**PROBLEM**: no mechanism exists to fetch date-bounded historical news
(section 13) — structurally unsafe for backtesting even though no current
code path exploits this.
**EVIDENCE**: direct code read.
**PROPOSED EXPERIMENT**: N/A — this is a prerequisite fix, not an
experiment, before any Phase 10/11 work attempts historical
sentiment-aware backtesting.
**RISK OF OVERFITTING**: none.

---

## 24. Tests run and results

```
$ python -m pytest -q
277 passed, 27 warnings (pre-existing, unrelated to this audit)
```

(Backend suite total prior to this audit's additions; see Final Report for
the exact post-audit count including the 8 new
`tests/test_feature_engineering.py` tests.)

## 25. Diagnostic scripts (this audit)

| Script | Purpose | Writes to prod DB? | Changes model/thresholds? |
|---|---|---|---|
| `scripts/audit/model_metrics_audit.py` | Fresh out-of-sample metrics, in-sample/OOS gap, baselines, confidence calibration — current code, fresh data | No | No |
| `scripts/audit/horizon_audit.py` | True 1/3/5/10-trading-day forward returns, independent of admin-triggered validation timing | No | No |
| `scripts/audit/output/recommendation_intelligence_report_snapshot.json` | Raw dump of `generate_engine_report()`'s real output against the live DB, used to source sections 9.1-9.6 verbatim | No (read-only run) | No |

All three are safe to re-run at any time; none mutate `storage/tracker.db`.
