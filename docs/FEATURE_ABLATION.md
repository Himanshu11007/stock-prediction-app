# Feature and Component Ablation (Phase 13)

Status: research complete. **No production change recommended.**

Experiments **E13.G.\*** (feature groups) and **E13.C.\*** (components). See
[ML_EXPERIMENT_REGISTRY.md](ML_EXPERIMENT_REGISTRY.md).
Script: `scripts/research/phase13_ablation.py`. Output: `scripts/research/output/phase13_ablation.json`.
Harness, segments and embargo: [CONFIDENCE_CALIBRATION.md §2](CONFIDENCE_CALIBRATION.md).

---

## 1. Question

Which production features and which ensemble components carry reproducible
incremental information?

## 2. Feature inventory (from code)

The production model uses the 27 columns in `utils/helpers.FEATURE_COLS`,
computed by `features/engineer.compute_features()`. Grouped by what they measure
(`evaluation.research.FEATURE_GROUPS`; a test checks the groups partition
the production list exactly):

| Group | Features |
|---|---|
| price_level | Close |
| returns_momentum | Price_Change, Momentum |
| moving_averages | MA_5, MA_10, MA_Diff, EMA_20, EMA_50, EMA_Cross, Price_vs_EMA20 |
| rsi | RSI |
| volatility_atr | Volatility, ATR, ATR_Pct |
| bollinger | BB_Width, BB_Position |
| volume | Volume, Volume_Change, Volume_MA, Volume_Ratio, Vol_Breakout |
| macd | MACD, MACD_Hist, MACD_Cross |
| trend_strength_adx | ADX, Plus_DI, Minus_DI |

Market regime (`utils/regime.py`), multi-timeframe trend and news are **not
model features**. They enter only the confluence score, so they belong to
Phase 15's decision-engine analysis.

## 3. Method

- **Baseline E-BASE:** all 27 features, production LR/RF/XGB, 20/30/50 blend,
  1-day target. Accuracy 50.0% DEV / 50.3% FINAL at 1D, AUC ≈ 0.50.
- **E13.G.\<group\>:** retrain with one group removed. Everything else is
  identical, including the rows (paired).
- **E13.C.\<variant\>:** from the baseline's stored per-model probabilities,
  with no retraining: LR, RF or XGBoost alone; an equal-weight blend; and
  leave-one-model-out blends with weights renormalised. These are research
  variants only.
- **Comparison:** a paired, date-clustered bootstrap of ΔBrier and Δaccuracy
  (variant − baseline), on DEV and FINAL, at 1/3/5/10D.
- **Pre-registered verdict:** "evidence" requires the DEV ΔBrier CI to
  exclude 0 **and** the same direction on FINAL, at ≥ 2 horizons. Otherwise
  "no reproducible evidence either way".
- Per-stock consistency: the number of the 27 symbols whose 1D accuracy
  improves vs worsens.

## 4. Feature-group results

ΔBrier is variant − baseline (positive means removing the group made
forecasts worse). Δacc is in percentage points. CIs are in the JSON.

| Removed group | DEV ΔBrier 1D / 5D / 10D | FINAL ΔBrier 1D / 5D / 10D | FINAL Δacc 1D / 5D / 10D | FINAL AUC 1D | Verdict |
|---|---|---|---|---|---|
| price_level | −0.0020 / −0.0003 / −0.0000 | +0.0016 / −0.0004 / +0.0012 | −0.2 / +0.5 / +0.4 | 0.495 | none |
| returns_momentum | −0.0029 / −0.0026 / −0.0031 | +0.0010 / −0.0004 / −0.0008 | −0.3 / +0.5 / +0.8 | 0.496 | none |
| moving_averages | −0.0018 / −0.0014 / −0.0015 | −0.0009 / −0.0025 / −0.0014 | −0.8 / +0.9 / +0.8 | 0.500 | none |
| rsi | −0.0006 / −0.0001 / −0.0004 | +0.0013 / +0.0004 / +0.0007 | −0.4 / +0.2 / +0.8 | 0.496 | none |
| volatility_atr | −0.0019 / −0.0024 / −0.0027 | −0.0016 / −0.0023 / −0.0003 | −0.1 / +0.2 / +0.8 | 0.501 | "removing improves Brier" |
| bollinger | −0.0002 / −0.0015 / −0.0037 | +0.0000 / −0.0015 / −0.0003 | −0.1 / +0.7 / +0.6 | 0.499 | none |
| volume | −0.0033 / +0.0001 / −0.0009 | +0.0010 / +0.0033 / +0.0025 | −0.6 / −0.6 / −0.3 | 0.499 | none |
| macd | −0.0021 / −0.0025 / −0.0028 | +0.0000 / +0.0011 / +0.0005 | −0.3 / −0.1 / −0.1 | 0.497 | none |
| trend_strength_adx | −0.0024 / −0.0033 / −0.0021 | −0.0002 / −0.0011 / −0.0009 | +0.1 / +0.8 / +0.8 | 0.500 | "removing improves Brier" |

Per-stock: no group improves 1D accuracy for a consistent majority of the
27 symbols in both segments. The splits range from 7/16 to 17/9 and flip
between DEV and FINAL.

**Reading:**

- Removing **any** group changes accuracy by ≤ 1.2 pp, and AUC stays within
  0.48–0.52.
- The two groups flagged by the mechanical rule (volatility_atr,
  trend_strength_adx) owe their flag to ~0.002 Brier reductions with no
  accuracy or AUC gain. Fewer inputs give a slightly less over-fitted, less
  over-confident model.
- Removing price_level, rsi, macd or volume *worsens* FINAL Brier at ≥ 2
  horizons (by ≤ 0.0033). On DEV the same removals improved Brier or left it
  unchanged, so the two periods disagree and none meets the rule.

**No feature group provides reproducible incremental discrimination.**

## 5. Component results

| Variant | Raw Brier FINAL 1D | Δ vs blend (CI) | FINAL acc 1D / 10D | FINAL AUC 1D / 10D | Verdict (raw Brier) |
|---|---|---|---|---|---|
| production 20/30/50 blend | 0.2766 | — | 50.3% / 50.8% | 0.500 / 0.514 | — |
| LR only | 0.2829 | +0.006 (+0.003, +0.010) | 50.9% / 51.0% | 0.505 / 0.513 | none |
| RF only | 0.2723 | −0.004 (−0.007, −0.002) | 51.0% / 51.9% | 0.502 / 0.522 | lower Brier than blend |
| XGBoost only | 0.2976 | +0.021 (+0.019, +0.023) | 49.6% / 50.1% | 0.498 / 0.506 | higher Brier than blend |
| equal weights | 0.2725 | −0.004 (−0.005, −0.003) | 50.0% / 51.1% | 0.501 / 0.517 | lower Brier than blend |
| no LR (RF+XGB) | 0.2824 | +0.006 | 49.9% / 50.8% | 0.499 / 0.512 | higher Brier than blend |
| no RF (LR+XGB) | 0.2837 | +0.007 | 50.1% / 50.2% | 0.499 / 0.509 | higher Brier than blend |
| no XGBoost (LR+RF) | 0.2683 | −0.008 (−0.010, −0.006) | 50.3% / 52.7% | 0.504 / 0.524 | lower Brier than blend |

**Post-hoc check (labelled as such): Platt calibration (Phase 12
protocol) equalises every component.** After calibration, FINAL 1D Brier
is 0.2497–0.2499 for LR, RF, XGBoost, the blend and the no-XGBoost blend.
5D and 10D are similar (within ±0.001). So the raw Brier ranking is a
ranking of **over-confidence**, not of information:

- **XGBoost**, at 50% of the production weight, is the most
  over-confident member (raw Brier ≈ 0.298). Most of the blend's
  over-confidence comes from it.
- No component discriminates: AUC is 0.49–0.52 on DEV and FINAL. The one
  noticeable number (no-XGBoost, FINAL 10D accuracy +1.9 pp, CI +1.0 to
  +2.9) is ≈ 0 at 1–5D and does not replicate on DEV 10D (49.5%).

## 6. Conclusions

1. **No feature group and no ensemble member provides reproducible
   incremental predictive information** on this benchmark. The question
   "where does the signal come from" has the answer: there is no detectable
   signal to attribute.
2. Removing features or the XGBoost member lowers raw Brier slightly by
   reducing over-confidence. Calibration achieves the same, and more, for
   any configuration (Phase 12).
3. **Production candidates: none.** Lowering XGBoost's weight or dropping
   volatility/ADX features is *"Research candidate — insufficient
   evidence"*. They change raw probability sharpness, not skill, and would
   alter the decision engine's ML pillars without a measured benefit to
   recommendations.

## 7. Limitations

- Leave-one-group-out cannot detect value that is redundant across groups
  (e.g. several groups encode trend). With a baseline AUC ≈ 0.50, however,
  there is no aggregate signal for redundancy to hide.
- Single-group-only models were not run: with no baseline signal they
  could only add noise to the comparison table.
- Same data limits as Phase 12: 27 symbols, ~2.25 years, NSE only.
