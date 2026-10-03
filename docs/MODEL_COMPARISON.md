# Model Comparison (Phase 14)

Status: research complete. **No production change.** The pre-registered
candidate failed confirmation on FINAL.

Experiment **E14** ([ML_EXPERIMENT_REGISTRY.md](ML_EXPERIMENT_REGISTRY.md)).
Script: `scripts/research/phase14_models.py`. Output:
`scripts/research/output/phase14_models.json` (prediction fingerprint
`952009426a33…`). Harness, segments and embargo:
[CONFIDENCE_CALIBRATION.md §2](CONFIDENCE_CALIBRATION.md).

---

## 1. Models

All models were fitted **in the same run**: identical refit dates, 1-year
windows, 27 production features, 1-day target and rows (15,066
predictions, 27 symbols).

| Key | Model | Settings |
|---|---|---|
| lr | production Logistic Regression | StandardScaler + LR(max_iter=5000), seed 42 |
| rf | production Random Forest | 50 trees, seed 42 |
| xgb | production XGBoost | 50 trees, depth 4, lr 0.1, seed 42 |
| ens | **production blend** | 0.20 / 0.30 / 0.50 |
| ens_equal | equal-weight blend (research) | 1/3 each |
| hgb | HistGradientBoostingClassifier | sklearn defaults, seed 42 |
| et | ExtraTreesClassifier | sklearn defaults (100 trees), seed 42 |
| gb | GradientBoostingClassifier | sklearn defaults, seed 42 |

The candidates use library defaults on purpose. Tuning hyperparameters on
one year of DEV data for 27 stocks would select noise. Deep models (LSTM,
Transformer) were not run. With ~200 training rows per symbol-window, and
tabular models showing AUC ≈ 0.50, there is no basis for a fair or
informative deep-learning experiment on this data.

## 2. Pre-registered rule

The candidate is the model with the lowest **DEV** Platt-calibrated 1D Brier
among the non-production models whose DEV 1D AUC exceeds the production
blend's. It is then checked once on FINAL.

## 3. Results

Accuracy CIs are date-clustered bootstrap. "Cal. Brier" uses Platt
calibration under the Phase 12 protocol (CAL → DEV; CAL+DEV → FINAL with
embargo).

### DEV (2025-04 → 2025-09, n = 3,375)

| Model | Acc 1D (CI) | Bal. acc | Prec | Rec | F1 | AUC 1D | AUC 5D | Brier | Log loss | ECE | Cal. Brier 1D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ens (prod) | 50.0% (48.1–51.9) | 0.500 | 0.499 | 0.447 | 0.472 | 0.501 | 0.507 | 0.276 | 0.754 | 0.134 | 0.2505 |
| lr | 51.1% (49.4–52.7) | 0.511 | 0.511 | 0.501 | 0.506 | 0.513 | 0.522 | 0.280 | 0.783 | 0.138 | 0.2504 |
| rf | 50.0% (48.2–51.6) | 0.500 | 0.500 | 0.421 | 0.457 | 0.505 | 0.514 | 0.272 | 0.744 | 0.114 | 0.2507 |
| xgb | 49.6% (47.8–51.3) | 0.496 | 0.495 | 0.441 | 0.466 | 0.494 | 0.497 | 0.298 | 0.819 | 0.185 | 0.2504 |
| ens_equal | 50.8% (49.1–52.7) | 0.508 | 0.509 | 0.457 | 0.482 | 0.505 | 0.512 | 0.271 | 0.743 | 0.114 | 0.2505 |
| hgb | 50.5% (48.9–52.2) | 0.505 | 0.505 | 0.461 | 0.482 | 0.506 | 0.507 | 0.329 | 0.958 | 0.244 | 0.2506 |
| et | 50.9% (49.2–52.4) | 0.509 | 0.510 | 0.445 | 0.476 | 0.511 | 0.509 | 0.275 | 0.754 | 0.132 | 0.2506 |
| gb | 50.3% (48.6–52.2) | 0.503 | 0.503 | 0.472 | 0.487 | 0.502 | 0.509 | 0.318 | 0.904 | 0.221 | **0.2504** |

All four candidates were "eligible", because the production blend's DEV AUC is
0.501. **gb** had the lowest DEV calibrated Brier (0.25041 vs 0.25054 for
ens_equal, a 0.0001 difference) and was selected.

### FINAL (2025-10 → 2026-09, n ≈ 6,400–6,640)

| Model | Acc 1D / 3D / 5D / 10D | AUC 1D / 3D / 5D / 10D | Brier 1D | Cal. Brier 1D | Δacc 1D vs prod (CI) |
|---|---|---|---|---|---|
| ens (prod) | 50.3 / 49.6 / 50.2 / 50.8 | 0.500 / 0.492 / 0.502 / 0.514 | 0.277 | 0.2499 | — |
| lr | 50.9 / 49.8 / 50.6 / 51.0 | 0.505 / 0.495 / 0.502 / 0.513 | 0.283 | 0.2497 | +0.7 (−0.6, +1.9) |
| rf | 51.0 / 49.7 / 50.1 / 51.9 | 0.502 / 0.500 / 0.505 / 0.522 | 0.272 | 0.2499 | +0.7 (−0.4, +1.8) |
| xgb | 49.6 / 49.3 / 49.8 / 50.1 | 0.498 / 0.490 / 0.499 / 0.506 | 0.298 | 0.2499 | −0.7 (−1.3, −0.0) |
| ens_equal | 50.0 / 49.2 / 50.1 / 51.1 | 0.501 / 0.493 / 0.503 / 0.517 | 0.272 | 0.2499 | −0.2 (−0.8, +0.3) |
| hgb | 49.5 / 48.8 / 49.5 / 50.4 | 0.492 / 0.485 / 0.495 / 0.511 | 0.337 | 0.2497 | −0.8 (−1.9, +0.2) |
| et | 49.0 / 48.8 / 49.8 / 51.1 | 0.497 / 0.497 / 0.502 / 0.512 | 0.281 | 0.2498 | −1.3 (−2.4, −0.1) |
| **gb** (selected) | 49.4 / 49.2 / 49.6 / 50.2 | 0.491 / 0.486 / 0.493 / 0.505 | 0.325 | 0.2497 | −0.8 (−2.0, +0.2) |

**The selected candidate (gb) is below production on every FINAL horizon**
(accuracy and AUC). It is not a production candidate.

### Return diagnostics (research only: close-to-close, no costs)

"Long-only" = each day, the equal-weight mean 1D return of the names a model
calls "up". "All" = the equal-weight mean of all 27 names.

| Model | DEV long-only total / max DD | FINAL long-only total / max DD | FINAL 5D spread (up − down calls, pp) | Turnover/day |
|---|---|---|---|---|
| All 27 names | +9.8% / −7.1% | +10.3% / −14.2% | — | — |
| ens (prod) | +1.2% / −11.4% | +11.0% / −14.8% | +0.10 | 0.29 |
| lr | +7.5% / −8.6% | +18.7% / −15.1% | +0.04 | 0.22 |
| rf | +5.4% / −8.8% | +19.8% / −13.5% | +0.16 | 0.29 |
| xgb | +3.4% / −11.8% | +6.1% / −15.0% | +0.09 | 0.32 |
| ens_equal | +5.8% / −10.4% | +7.3% / −14.8% | +0.12 | 0.26 |
| hgb | +11.9% / −9.2% | +3.4% / −16.5% | +0.03 | 0.34 |
| et | +8.0% / −10.1% | +2.1% / −13.8% | +0.01 | 0.28 |
| gb | +8.3% / −9.9% | +7.0% / −15.1% | +0.06 | 0.32 |

No model's ranking holds across periods. hgb is best on DEV and worst on
FINAL; rf and lr are the reverse. Every model's FINAL up-call hit rate is
47.2–49.3%, below the 48.4% FINAL base rate or within noise of it. With
~0.29 daily turnover, any transaction cost would remove the small FINAL
differences.

## 4. Findings

1. **No model, existing or candidate, discriminates.** AUC is 0.485–0.522
   at every horizon on both segments. Accuracy CIs all contain 50%.
2. **The production models differ mainly in over-confidence:** raw Brier
   0.272 (rf) to 0.298 (xgb), and up to 0.337 (hgb). After calibration
   every model is 0.2497–0.2507, i.e. the no-skill level.
3. The DEV → FINAL rank order of models is unstable, consistent with
   selecting among noise. The pre-registered procedure worked as intended:
   it picked a model on DEV, and FINAL did not confirm it.
4. **Tradeoffs worth recording:** the boosted candidates (hgb, gb) are much
   more over-confident than the production members. LR is the most stable
   member (AUC ≥ 0.494 everywhere). RF has the least over-confident raw
   probabilities. None of these translates into measurable skill.

## 5. Production recommendation

**Do not replace or reweight the production model.** No candidate meets
the promotion rule (improvement on more than one horizon, DEV-selected and
FINAL-confirmed, with adequate CIs). The evidence points to the inputs
(the daily technical feature set and the binary target), not the
learner, as the limiting factor. See Phase 15.

## 6. Limitations

- Library-default hyperparameters. A tuned model could differ, but tuning
  would require a further held-out period that this data cannot provide.
- 27 symbols; one ~2.25-year period; NSE only; daily bars.
