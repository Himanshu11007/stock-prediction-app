# Target, Horizon and Decision-Engine Research (Phase 15)

Status: research complete. **No production change.** No threshold, target
or scoring value was modified.

Experiments **E15.T** (targets) and **E15.D** (decision engine), in
[ML_EXPERIMENT_REGISTRY.md](ML_EXPERIMENT_REGISTRY.md).
Scripts: `scripts/research/phase15_targets.py`, `scripts/research/phase15_decision_engine.py`.
Outputs: `scripts/research/output/phase15_targets.json`, `phase15_decision_engine.json`.

---

## Part A: target and horizon (E15.T)

### A.1 Method

The production pipeline (27 features, LR/RF/XGBoost, 20/30/50) was trained
on four separate binary targets: `Close[t+h] > Close[t]` for h = 1, 3, 5, 10
trading bars. Labels are never mixed. At a refit date R, the training rows
for target h are those whose label is known at R (t + h ≤ R; tested). Each
target is evaluated at its own horizon, on the research-harness segments
(DEV 2025-04 → 2025-09; FINAL 2025-10 → 2026-09; see
[CONFIDENCE_CALIBRATION.md §2](CONFIDENCE_CALIBRATION.md)). Each is also
compared, on the same rows, with the production 1-day-target model
evaluated at that horizon.

### A.2 Class balance

| Horizon | Training up-rate (mean) DEV / FINAL | Realised up % DEV / FINAL |
|---|---|---|
| 1D | 48.7% / 48.5% | 50.0 / 48.4 |
| 3D | 48.8% / 50.2% | 51.5 / 48.7 |
| 5D | 48.9% / 50.7% | 52.9 / 48.9 |
| 10D | 47.6% / 51.9% | 54.8 / 50.3 |

Classes are close to balanced. The base rate drifts between periods (DEV
was a rising market at 5–10 days, FINAL was not), which is why majority-class
baselines move.

### A.3 Results (each target at its own horizon)

Accuracy CIs are date-clustered. "Spread" is the mean h-day return of
up-calls minus down-calls, in percentage points, with a date-clustered CI
(optimistic for h > 1 because consecutive dates' returns overlap).

| Target | Seg | n | Acc (CI) | Bal. acc | Prec / Rec / F1 | AUC | Brier (raw / Platt) | Majority | Prev. dir. | Δacc vs 1D model (CI) | Spread (CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D | DEV | 3,375 | 50.0% (48.1–51.9) | 0.500 | 0.50 / 0.45 / 0.47 | 0.501 | 0.276 / 0.2505 | 50.0% | 51.1% | — | −0.06 (−0.20, +0.08) |
| 1D | FINAL | 6,642 | 50.3% (48.8–51.7) | 0.501 | 0.49 / 0.45 / 0.47 | 0.500 | 0.277 / 0.2499 | 50.4% | 48.9% | — | +0.05 (−0.08, +0.19) |
| 3D | DEV | 3,375 | 49.0% (47.1–51.1) | 0.491 | 0.51 / 0.46 / 0.48 | 0.492 | 0.293 / 0.2514 | 49.5% | 49.7% | −1.2 (−3.4, +1.0) | +0.05 (−0.18, +0.32) |
| 3D | FINAL | 6,588 | 50.7% (49.3–52.1) | 0.506 | 0.49 / 0.48 / 0.48 | 0.508 | 0.288 / 0.2499 | 50.9% | 49.1% | +1.2 (−0.3, +2.7) | +0.04 (−0.15, +0.23) |
| 5D | DEV | 3,375 | 51.0% (49.1–52.9) | 0.512 | 0.54 / 0.47 / 0.50 | 0.508 | 0.299 / 0.2528 | 50.0% | 48.6% | +0.6 (−1.7, +3.0) | **+0.44 (+0.08, +0.82)** |
| 5D | FINAL | 6,534 | 50.5% (49.3–51.8) | 0.504 | 0.49 / 0.48 / 0.49 | 0.505 | 0.299 / 0.2499 | 49.8% | 49.5% | +0.3 (−1.4, +2.0) | +0.03 (−0.22, +0.28) |
| 10D | DEV | 3,375 | **52.1% (50.5–53.8)** | 0.525 | 0.57 / 0.49 / 0.53 | 0.520 | 0.313 / 0.2608 | 47.7% | 48.7% | **+4.1 (+2.2, +6.0)** | **+0.71 (+0.32, +1.09)** |
| 10D | FINAL | 6,399 | 50.3% (48.8–51.9) | 0.503 | 0.51 / 0.48 / 0.49 | 0.497 | 0.320 / 0.2506 | 50.4% | 49.2% | −0.5 (−2.3, +1.4) | +0.43 (−0.03, +0.91) |

### A.4 Findings

1. **No target improves FINAL accuracy or AUC.** Every FINAL accuracy CI
   contains 50%, and AUC is 0.497–0.508.
2. **The 10-day target is a textbook DEV-only result for accuracy.** On DEV
   it shows +4.1 pp vs the 1-day model with a CI excluding 0. On FINAL that
   becomes −0.5 pp and AUC 0.497. Choosing it for its DEV accuracy would have
   been a mistake.
3. **The one lead worth recording:** the 10-day target's return spread is
   positive in both periods: +0.71 pp DEV (CI excludes 0) and +0.43 pp FINAL
   (CI −0.03 to +0.91, just including 0). Over 10 days with no costs, that
   is ~4 bp/day. Overlapping returns make the CIs optimistic. The 1-day
   production model shows no comparable spread at 10D (DEV −0.14, FINAL +0.27,
   both CIs wide). **Research candidate, insufficient evidence**: worth
   tracking prospectively on new data, not worth deploying.
4. Longer-horizon targets are **more** over-confident (raw Brier 0.29–0.32).
   Overlapping labels make consecutive training rows near-duplicates, which
   tree models fit tightly.
5. Product alignment: the product validates recommendations on multi-day
   outcomes (the existing validator, the ±3% HOLD band, ATR targets), while
   the model is trained on a 1-day target. That mismatch is real, but fixing
   it alone does not create measurable skill on this data (findings 1–3).

## Part B: decision engine (E15.D)

### B.1 Method and data

Full production replays (`evaluation/walk_forward.py`; scanner path,
decision engine, filters; news neutral), every 5th session, 27 symbols:

| Segment | Replay | Rows | Filter survivors |
|---|---|---|---|
| DEV (thresholds chosen here) | `walk_forward_benchmark/research_dev/`, 2025-01-01 → 2025-09-30 | 1,026 | 465 |
| FINAL (evaluated once) | `walk_forward_benchmark/full/` (Phase 11B), 2025-10-01 → 2026-09-30 | 1,350 | 540 |

The `research_dev` replay starts 19 months after the price snapshot begins,
so its weekly trend input has slightly less than production's 2 years of
history at the earliest dates.

**Exactness of counterfactuals:**

- The signal is a pure threshold on `confluence × 100`: STRONG BUY ≥ 72,
  BUY ≥ 58, HOLD ≥ 42, SELL ≥ 28 (`utils/decision_engine.py`), so
  alternative BUY/SELL thresholds are exact.
- Filters are ANDed gates. Stricter `MIN_CONFIDENCE` / `MIN_CONFLUENCE_SCORE` /
  `MIN_ACCURACY` values are exact. Looser ones need inputs that are not stored,
  and were not evaluated.
- **Risk/reward:** `scanner/filters.py` has no R/R gate, and
  `calculate_risk()` fixes the target at 2× the stop distance. R/R is
  therefore always 2.0 (HOLD 1.0), and an R/R filter would be vacuous.

Metric: 5-day **excess** return = a row's 5D return minus the mean 5D
return of all replayed rows on the same date (removes market moves). CIs
are date-clustered bootstrap.

Pre-registered selection: the BUY threshold maximising DEV mean 5D excess
for bullish calls (n ≥ 50), and the mirror image for bearish calls.

### B.2 Alignment of the decision engine with the model

| | DEV | FINAL |
|---|---|---|
| Spearman(confluence, ensemble P(up)) | 0.43 | 0.38 |
| BUY signals issued while the model predicts *down* | 25.5% (BUY: 75 of 257) | 32.8% (141 of 376) |
| SELL signals issued while the model predicts *up* | 8.7% (19 of 219) | 6.1% (13 of 213) |
| Mean P(up): STRONG BUY / BUY / HOLD / SELL / STRONG SELL | 0.61 / 0.55 / 0.50 / 0.38 / 0.32 | 0.61 / 0.53 / 0.47 / 0.38 / 0.29 |
| Mean ML confidence by signal | 61–68 (flat) | 61–71 (flat) |

The confluence score follows the model's direction only loosely. The ML
pillars carry 20% of the weight; technical analysis, multi-timeframe,
momentum and regime carry the rest. A quarter to a third of BUY calls
contradict the ML model, and STRONG signals always agree with it.
Confidence does not differ by signal. Because the model has no measured
skill (Phases 12–14), this alignment cannot by itself make recommendations
better or worse.

### B.3 Signal outcomes (5D, all replayed rows)

| Signal | DEV n / success / excess | FINAL n / success / excess |
|---|---|---|
| STRONG BUY | 37 / 59.5% / +0.35 | 54 / 51.9% / +0.24 |
| BUY | 257 / 50.6% / −0.25 | 372 / 54.3% / +0.28 |
| HOLD | 500 / 52.2% / −0.06 | 690 / 58.8% / −0.10 |
| SELL | 219 / 48.9% / **+0.37** | 201 / 53.7% / −0.35 |
| STRONG SELL | 13 / 61.5% / +0.19 | 6 / 50.0% / +3.52 |

- SELL calls outperformed the market on DEV (wrong direction) and
  underperformed it on FINAL (right direction). BUY calls show the reverse.
  **The direction of the edge flips between periods.**
- HOLD "success" tracks the unconditional ±3% band rate: 5D band rate 55.8%
  DEV / 56.3% FINAL, against HOLD success of 52.2% / 58.8%. The HOLD region
  adds nothing beyond the width of the band.
- STRONG signals have n = 6–54. No conclusion is possible.

### B.4 Threshold sensitivity

Bullish calls (`confluence × 100 ≥ t`), 5D mean excess return in pp (CI):

| t | DEV n / excess | FINAL n / excess |
|---|---|---|
| 50 | 526 / −0.08 (−0.31, +0.13) | 791 / −0.04 (−0.26, +0.18) |
| 54 | 414 / −0.13 | 619 / +0.02 |
| **58 (current)** | 294 / −0.17 (−0.47, +0.12) | 426 / +0.28 (−0.04, +0.59) |
| 62 | 187 / −0.19 | 257 / +0.31 |
| 66 | 112 / −0.49 | 154 / +0.28 |
| **68 (DEV pick)** | 90 / +0.06 (−0.55, +0.76) | 107 / **−0.00** (−0.66, +0.64) |
| 72 | 37 / +0.35 | 54 / +0.24 |

Bearish calls (`confluence × 100 < t`; a negative excess is the desired direction):

| t | DEV n / excess | FINAL n / excess |
|---|---|---|
| 34 | 80 / +0.30 | 60 / −0.18 |
| 38 | 138 / +0.54 | 132 / −0.15 |
| **42 (current)** | 232 / +0.36 (−0.06, +0.74) | 206 / −0.24 (−0.68, +0.19) |
| 46 | 375 / +0.13 | 361 / +0.03 |
| **48 (DEV pick)** | 434 / +0.08 (−0.16, +0.31) | 455 / +0.06 (−0.27, +0.37) |

**The DEV-selected thresholds failed on FINAL.** The DEV pick of 68 for BUY
gives 0.00 pp on FINAL, versus +0.28 pp for the current 58. The DEV pick of
48 for SELL gives +0.06 pp, the wrong direction, versus −0.24 pp for the
current 42. No threshold's CI excludes 0 on both segments. The curve is noise
around zero and changes shape between periods.

### B.5 Filter counterfactuals (stricter only; bullish survivors, 5D)

| Gate | Value | DEV n / success / excess | FINAL n / success / excess |
|---|---|---|---|
| MIN_CONFIDENCE | **55 (current)** | 121 / 50.4% / −0.01 | 120 / 56.7% / +0.09 |
| | 65 | 48 / 41.7% / −0.16 | 61 / 59.0% / +0.41 |
| | 75 | 10 / 30.0% / −0.71 | 17 / 76.5% / +1.60 |
| MIN_CONFLUENCE_SCORE | **0.55 (current)** | 121 / 50.4% / −0.01 | 120 / 56.7% / +0.09 |
| | 0.65 | 37 / 51.4% / −0.04 | 43 / 55.8% / −0.18 |
| MIN_ACCURACY | **0.44 (current)** | 121 / 50.4% / −0.01 | 120 / 56.7% / +0.09 |
| | 0.52 | 56 / 51.8% / +0.19 | 48 / 43.8% / −0.09 |
| | 0.56 | 15 / 60.0% / +0.37 | 21 / 33.3% / −1.22 (−2.31, −0.12) |

- Tightening `MIN_CONFIDENCE` moves bullish success **down on DEV and up on
  FINAL**, while `MIN_ACCURACY` does the opposite.
- At the strict ends n is 10–21, so these are opposite-signed noise.
- `MIN_ACCURACY` gates on the model's own `_fast_accuracy` estimate. Phase 14
  shows the model has no skill, so that estimate is itself noise.

### B.6 Conclusions

1. **No threshold or filter change is justified.** The pre-registered DEV
   choices did not hold on FINAL, and every effect reverses sign between the
   two periods.
2. The decision engine is **loosely aligned** with the model (Spearman
   ≈ 0.4; 25–33% of BUY calls contradict it). Given a model with no skill,
   that alignment is not the bottleneck.
3. HOLD success is almost entirely a function of the ±3% band, and STRONG
   signals are too rare to evaluate. R/R is constant by construction, so it
   cannot act as a filter.
4. What would make counterfactuals meaningful is a model with measured
   discrimination. Until then, tuning thresholds selects noise.

### B.7 Limitations

- Neutral news. The live system's News pillar (10% weight) cannot be
  replayed (no point-in-time news).
- Every 5th session only (1,026 + 1,350 rows; filtered n is 465 / 540).
- Excess returns are equal-weight, close-to-close, no costs, overlapping
  5-day windows.
- Thresholds looser than the current filters cannot be evaluated exactly.

