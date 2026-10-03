# Confidence Calibration (Phase 12)

Status: research complete. **No production change.** Recommendation: do not
deploy a calibrator as a "confidence improvement". See §7.

Experiment ID: **E12** (registry: [ML_EXPERIMENT_REGISTRY.md](ML_EXPERIMENT_REGISTRY.md)).
Script: `scripts/research/phase12_calibration.py`. Output:
`scripts/research/output/phase12_calibration.json`, `scripts/research/output/baseline.json`.

---

## 1. Question

Production reports `confidence = max(p, 1 − p) × 100`, where `p` is the
20/30/50 ensemble's P(up). Phase 10/11 found that confidence buckets did not
track success. Can `p` be calibrated, and does calibration reveal useful
probability information?

## 2. Research harness (shared by Phases 12–15)

`evaluation/research.py` is a walk-forward harness. For every symbol it
refits the **exact production pipelines** (`_make_candidates()`, run
single-threaded, which is tested to give identical output) every 5 trading
sessions. Each refit uses the production 1-year window and production
features (`compute_features`). The fitted models then predict every session
until the next refit. Every prediction is out-of-sample: the training labels
were all realised on or before the refit date, which is ≤ the prediction
date. `assert_row_integrity()` fails the run otherwise.

`tests/test_research_harness.py` proves:

- The harness reproduces the production replay's ensemble probability at a
  refit date (|Δ| < 1e-12).
- Future prices cannot change a prediction.
- Multi-horizon labels are known at the refit date.
- The calibration fit rows never overlap the evaluation segment.

| Item | Value |
|---|---|
| Universe | the 27 symbols of the clean benchmark, Phase 10 price snapshot (sha256 manifest unchanged) |
| Predictions | 15,066 (every session 2024-07-01 → 2026-09-30) |
| Difference from production | refit every 5 sessions instead of daily (a model can be up to 4 sessions old); news not used (ML only) |
| Reproducibility | two uncached runs → identical prediction fingerprint `d89eda00…` |

Chronological segments (fixed before any experiment ran):

| Segment | Dates | Rows | Use |
|---|---|---|---|
| CAL | 2024-07-01 → 2025-03-31 | 5,049 | fit calibrators |
| DEV | 2025-04-01 → 2025-09-30 | 3,375 | all comparisons and selection decisions |
| FINAL | 2025-10-01 → 2026-09-30 | 6,642 | confirmation only (same period as the Phase 10/11 benchmark) |

**Embargo:** a calibrator applied from date S is fitted only on rows whose
h-day outcome was realised before S (`fit_rows_before`). For example, at 10D
the CAL fit stops at 2025-03-12, so that its outcomes land before DEV starts
on 2025-04-01.

## 3. Uncalibrated baseline (E-BASE)

Production ensemble, 1-day target. Accuracy CIs are date-clustered bootstrap
(predictions on the same day are correlated across stocks).

| Segment | Horizon | n | Accuracy (CI) | Balanced acc. | AUC | Brier | Majority baseline | Previous-direction baseline | Actual up % |
|---|---|---|---|---|---|---|---|---|---|
| CAL | 1D | 5,049 | 48.1% (46.4–49.6) | 48.1% | 0.479 | 0.284 | 48.6% | 50.9% | 48.0 |
| DEV | 1D | 3,375 | 50.0% (48.1–51.9) | 50.0% | 0.501 | 0.276 | 50.0% | 51.1% | 50.0 |
| DEV | 5D | 3,375 | 50.4% (48.2–52.7) | 50.7% | 0.507 | 0.276 | 47.9% | 48.6% | 52.9 |
| FINAL | 1D | 6,642 | 50.3% (48.8–51.7) | 50.1% | 0.500 | 0.277 | 50.4% | 48.9% | 48.4 |
| FINAL | 3D | 6,588 | 49.6% (48.2–51.0) | 49.4% | 0.492 | 0.279 | 50.6% | 49.1% | 48.7 |
| FINAL | 5D | 6,534 | 50.2% (48.8–51.5) | 50.0% | 0.502 | 0.277 | 50.1% | 49.5% | 48.9 |
| FINAL | 10D | 6,399 | 50.8% (49.1–52.2) | 50.8% | 0.514 | 0.274 | 48.9% | 49.2% | 50.3 |

The ensemble has **no measurable discrimination**: AUC is 0.48–0.51
everywhere, and in CAL it is slightly below chance. Its Brier (~0.277) is
*worse* than the 0.25 of a constant 50% forecast, because it is
over-confident: ECE ≈ 0.13, and it predicts "up" ~45% of the time at
confidences of 60–80%.

## 4. Methods

| Method | Definition |
|---|---|
| none | raw ensemble `p` |
| platt | logistic regression on log-odds of `p` (near-unregularised) |
| isotonic | `IsotonicRegression(out_of_bounds="clip")` |
| constant_base_rate | the fit period's realised up-rate for every row: a **no-skill reference** |

Pre-registered selection: the lowest **DEV** Brier among none/platt/isotonic,
per horizon. FINAL is reported for every method, but it was not used to choose.

## 5. Results (FINAL; calibrators fitted on CAL+DEV with embargo)

| Horizon | Method | Brier | Log loss | ECE | AUC | Accuracy | Predicted up % | ΔBrier vs none (CI) |
|---|---|---|---|---|---|---|---|---|
| 1D | none | 0.2766 | 0.7549 | 0.136 | 0.500 | 50.3% | 44.8 | — |
| 1D | platt *(DEV choice)* | 0.2499 | 0.6930 | 0.003 | 0.501 | 51.3% | 13.3 | −0.027 (−0.032, −0.021) |
| 1D | isotonic | 0.2498 | 0.6928 | 0.004 | 0.502 | 51.6% | 0.2 | −0.027 (−0.032, −0.022) |
| 1D | constant | 0.2498 | 0.6927 | 0.003 | 0.500 | 51.6% | 0.0 | −0.027 |
| 3D | none | 0.2790 | 0.7604 | 0.143 | 0.492 | 49.6% | 44.7 | — |
| 3D | platt *(DEV choice)* | 0.2499 | 0.6929 | 0.003 | 0.508 | 51.2% | 23.9 | −0.029 (−0.034, −0.024) |
| 3D | constant | 0.2498 | 0.6928 | 0.002 | 0.500 | 51.3% | 0.0 | −0.029 |
| 5D | none | 0.2767 | 0.7560 | 0.137 | 0.502 | 50.2% | 44.6 | — |
| 5D | platt *(DEV choice)* | 0.2499 | 0.6929 | 0.001 | 0.498 | 51.1% | 0.4 | −0.027 (−0.032, −0.022) |
| 5D | constant | 0.2499 | 0.6929 | 0.001 | 0.500 | 51.2% | 0.0 | −0.027 |
| 10D | none | 0.2737 | 0.7495 | 0.130 | 0.514 | 50.8% | 44.9 | — |
| 10D | platt *(DEV choice)* | 0.2509 | 0.6950 | 0.026 | **0.486** | 49.2% | 20.2 | −0.023 (−0.029, −0.016) |
| 10D | isotonic | 0.2509 | 0.7003 | 0.021 | 0.501 | 49.9% | 1.6 | −0.023 (−0.029, −0.017) |
| 10D | constant | 0.2503 | 0.6938 | 0.019 | 0.500 | 49.7% | 0.0 | −0.024 |

DEV results (calibrators fitted on CAL only) show the same pattern: Platt was
chosen at every horizon by a margin of < 0.001 Brier, and every calibrated
Brier is within 0.001 of the constant forecast. Full reliability tables and
confidence buckets for every method, segment and horizon are in the JSON.

## 6. Findings

1. **Calibration reduces Brier and log loss reliably**, by 0.023–0.029
   Brier. The bootstrap CI excludes 0 at every horizon on DEV and FINAL.
   ECE falls from ~0.13 to ≤ 0.03.
2. **This gain comes entirely from removing over-confidence.** Every
   calibrated forecast lands on the no-skill constant forecast, which matches
   or beats it everywhere. The calibrated probabilities sit in a narrow band
   around the base rate (~0.48–0.50), so the "confidence" they imply is
   ~50–52% for almost every prediction.
3. **Calibration cannot add discrimination, and here there is none to
   preserve.** AUC stays ≈ 0.50. At 10D, Platt fitted on CAL+DEV *inverted*
   the ranking (AUC 0.514 → 0.486), because the fitted slope was negative.
   That is a symptom of the model–outcome relationship changing sign between
   periods (CAL AUC 0.479, FINAL 0.514). Neither sign is reliable.
4. Confidence buckets (raw): success is flat at ~50% across 55–90%
   confidence on every segment, the same as Phase 10/11.

## 7. Production recommendation

**Not a production candidate.** Deploying Platt or isotonic would make the
reported probability honest: it would show ≈ 50% for nearly every stock.
Through the unchanged decision engine that would:

- set ML Confidence to ≈ 0 for every stock and push the ML Direction pillar
  toward the base rate,
- make almost every row fail `MIN_CONFIDENCE = 55`,

which would empty the recommendation list. That is a product decision, not a
calibration improvement.

What *is* supported by evidence:

- The **raw confidence number is not a probability.** It overstates
  certainty by ~25 points on average. The UI and documentation should not
  present it as "chance of success". (A product/UX change, not made here.)
- If the product wants a calibrated number in future, the code here
  (`evaluation.research_metrics.Calibrator`, the embargoed fit split) is
  ready. It should only be wired in after a model shows real discrimination
  (Phases 13–15 found none).

## 8. Limitations

- 27 symbols, one market (NSE), ~2.25 years of predictions. Date-clustered
  CIs are used, but there is still cross-sectional correlation within dates.
- Refit every 5 sessions rather than daily.
- A single calibrator is pooled across symbols. Per-symbol calibration would
  have ~125–250 rows per fit, which is too few.
