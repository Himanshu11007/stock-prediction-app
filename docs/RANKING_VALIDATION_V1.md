# Ranking Engine v1.0 — Out-of-Sample Validation and Freeze

> **Naming note (2026-10-03):** the product was renamed from StockAI Pro to **StockLens**; the "StockLens Score" in this document is the same score previously called "StockAI Score". The research, data and conclusions are unchanged.

**Status: Ranking Engine `ranking-v1.0` is FROZEN** (`config.RANKING_ENGINE_STATUS`).
**Verdict: Current Ranking Engine v1.0 does not demonstrate sufficient
out-of-sample evidence of incremental stock-selection skill.** The StockLens
Score is therefore an **analytical ranking, not a validated predictive
edge**. It stays in production unchanged, and prospective tracking of every
production ranking has started (section 28).

| Item | Location |
|---|---|
| Point-in-time harness (reuses the production code) | `evaluation/ranking_validation.py` |
| Validation runner (pre-registered protocol in its docstring) | `scripts/research/ranking_validation.py` |
| Machine-readable results | `scripts/research/output/ranking_validation/` (`summary.json`, `bucket_results.csv`, `top_n_results.csv`, `regime_results.csv`, `component_results.csv`, `weight_variants.csv`, `coverage_by_date.csv`, `ranking_snapshots_production.csv`) |
| Prospective tracking | `ranking/tracking.py`, tables `ranking_snapshots` / `ranking_outcomes`, admin routes `/api/v1/admin/ranking-tracking/*` |
| Tests | `tests/test_ranking_validation.py` |

Reproduce: `python scripts/research/ranking_validation.py`. The first run
downloads prices into the gitignored `scripts/research/.cache/`. The
price-snapshot sha256 is recorded in `summary.json`
(`f41ea4ea…9ce0`). Fundamentals are read from `storage/app.db`, opened
read-only.

---

## 1. Exact production methodology tested

The validation calls the unmodified production code at every ranking date:

- `ranking.technical.technical_snapshot` computes the technical, momentum,
  risk, liquidity and regime inputs from daily bars up to the ranking date.
- `fqvf.evaluate` runs the 18 fixed FQVF checks, unchanged. NOT_AVAILABLE is
  never converted to FAIL.
- `ranking.service.StockRankingService(DEFAULT_WEIGHTS, DEFAULT_RULES).rank()`
  produces the score, eligibility and rank.

**Weights:** Quality 25, Valuation 20, Financial Health 15, Technical Trend
10, Momentum 10, Risk 10, Sector Outlook 5, Market Regime 5, ML 0.

**Eligibility rules:** score coverage ≥ 0.60, FQVF coverage ≥ 0.50, 20-day
average volume ≥ 500,000, market data ≤ 4 days old.

**Rank:** dense rank over eligible stocks only. This is the production Top
Picks ordering, so "Top N" means production ranks 1..N.

The ML signal was not computed. It has weight 0, so it cannot change any
score.

## 2. Dataset

- **Prices:** Yahoo Finance daily OHLCV with dividend and split actions, from
  2021-06-01 to 2026-10-02, for the 299 universe symbols and NIFTY 50
  (`^NSEI`).
  - Valuation uses the quoted, split-adjusted close (not dividend-adjusted).
  - Technicals use the production-style adjusted frame (OHLC scaled by Adj
    Close / Close).
- **Fundamentals:** the latest stored provider snapshot per stock
  (`fundamental_snapshots`): annual statements for FY2023–FY2026. 276 stocks
  have 4 years, 5 have 5 years, 1 has 3 years, and 17 have none.
- **Classifications:** the sector and industry currently stored. No
  classification history exists.
- **Output:** 39 ranking dates × 277–282 stocks = 10,943 production-scored
  stock-date rows (`ranking_snapshots_production.csv`).

## 3. Historical period

- **Ranking dates:** the last NIFTY 50 session of each month, from 2023-07-31
  to 2026-09-30 (39 dates).
- **Forward returns:** measured up to 2026-10-01.

The start date is set by fundamental depth. The first fiscal year (FY2023,
ending March 2023) becomes usable at mid-2023.

## 4. Development (DEV) period

- **Ranking dates:** 2023-07-31 to 2024-12-31 (18 dates).
- **Outcome embargo:** a DEV observation counts only if its forward window
  ends before 2025-01-01, so no DEV outcome overlaps the final period.
- **Usable dates by horizon:** 1M 17, 3M 14, 6M 12, 12M 5.

## 5. Final test (FINAL) period

- **Ranking dates:** 2025-01-31 to 2026-09-30 (21 dates).
- **Used once,** after the DEV-based selection was fixed.
- **Dates with a complete forward window:** 1M 20, 3M 18, 6M 15, 12M 8.

## 6. Universe

The current Large Cap, Mid Cap and Small Cap universe (`stock_universe`):
301 rows, 299 distinct active symbols.

- **Priced:** 282 symbols have price history.
- **Ranked per date:** 277–282 stocks (5 listed during the period).
- **Eligible per date:** 144–190 stocks (median about 168).

Ineligible stocks are excluded from buckets and Top N, exactly as production
excludes them from Top Picks.

## 7. Benchmark

- **Primary:** NIFTY 50 price index (`^NSEI`). This is the index production
  already uses for the market regime.
- **Secondary:** the equal-weight eligible universe at the same date. It
  isolates selection skill from the universe-level biases described in
  section 9.

Stock and benchmark forward returns are both price returns (dividends
excluded). Stock dividend yields of about 1–1.5% a year are therefore
missing from both sides of the universe comparison.

## 8. Data availability

FQVF coverage depends on how many fiscal years were published by each date.
Score coverage stays at about 0.95 throughout.

| Ranking dates | Fiscal years usable | Mean FQVF coverage | Mean score coverage |
|---|---|---|---|
| 2023-07 → 2025-05 | 1–2 (FY2023, FY2024) | 0.58 | 0.95 |
| 2025-06 → 2026-05 | 3 | 0.76–0.77 | 0.95 |
| 2026-06 → 2026-09 | 4 | 0.82–0.83 | 0.95 |

Checks that are NOT_AVAILABLE at a given date:

- **Never available historically:** Historical Average PE (5–7 years) and
  Sector Outlook. No historical administrator outlook exists; production is
  also NOT_AVAILABLE here.
- **NOT_AVAILABLE until 3–4 fiscal years exist:** EPS stability, EPS
  progression and growth, PEG.

So, for all of DEV and the first five FINAL dates, the **Quality component
is computed from very few checks**.

Full per-date figures are in `coverage_by_date.csv`. No missing value was
filled with 0, a neutral value or an estimate.

## 9. Survivorship bias (quantified, not removed)

- **Index membership look-ahead:** the universe is today's index lists. Stocks
  that grew into them, which are past winners, are included from 2023. Stocks
  that dropped out are absent.
- **Unrecoverable symbols:** 17 of the 299 symbols (5.7%) have no provider
  history under their stored symbol (renamed, merged or delisted):
  AEGISCHEM, AKZOINDIA, ALEXOTYRES, CG, GUJGASLTD, ISEC, ITDCEM, LAXMIMACH,
  LTIM, MTAR, SUNDRFAST, SUVENPHAR, TATAMOTORS, ULTRAMARINE, VARUNBEV, VIP,
  ZOMATO. They cannot be reconstructed from the available sources and are
  excluded from every date.
- **Size of the effect:** the equal-weight eligible universe returned +61.1%
  cumulative over the 17 DEV months against +23.1% for NIFTY 50. Over the 20
  FINAL months it returned +9.2% against −5.3%. Most of any "Top N beats
  NIFTY" result is this universe effect (survivorship, plus the mid/small-cap
  tilt and equal weighting). It is not selection skill.
- **Comparison that matters:** **Top N versus the eligible universe** removes
  the universe-level bias. It does not remove survivorship within the
  universe.

## 10. Look-ahead controls

| Input | Rule (tested in `tests/test_ranking_validation.py`) |
|---|---|
| Prices | Only bars dated ≤ D. A missing price before D raises `LeakageError`. |
| Annual statements | Usable only when fiscal year end + 75 days ≤ D (SEBI LODR 60-day deadline + 15-day buffer). |
| PE / PB / PSR / BVPS | Recomputed at D from D's quoted close and the usable statements. The provider's current ratios are never used. |
| Dividend yield / payout | Trailing 12 months of dividend events ≤ D. No events → None, mirroring the production provider for non-payers. |
| Industry median PE | Leave-one-out median of the other stocks' point-in-time PE at D. |
| Debt/equity | From the usable statements (FQVF fallback). The provider's current value is not used. |
| Market regime | Production `detect_regime` on NIFTY 50 bars ≤ D. |
| Technicals | Production `technical_snapshot` on bars ≤ D. A later dividend rescales the whole history uniformly, so returns and volatility at D are unchanged. |
| Forward returns | Measured on the NIFTY 50 session calendar: 1M / 3M / 6M / 12M = 21 / 63 / 126 / 252 sessions. None if the window has not elapsed or the stock has no close within 5 days of the exit. |

A test rewrites all prices after D, adds future statements and adds a future
dividend. The ranking at D stays identical.

**Residual leakage that cannot be removed:**

1. Statements come from a snapshot fetched 2026-10-03, so later restatements
   are possible.
2. Classifications are current, not historical.
3. Universe membership is current (section 9).

An earlier draft of the runner valued stocks on dividend-adjusted closes,
which embed future dividends. This was found and fixed before the final run.
The results were essentially unchanged: DEV 3M IC moved from 0.006 to 0.004.

## 11. Ranking bucket results

Buckets are by rank percentile among eligible stocks at each date. The table
gives the mean per-date excess return over NIFTY 50; brackets are 95% CIs
with Newey–West standard errors (lags = overlapping months − 1).

| Segment | Horizon | Top 10% | 10–25% | 25–50% | 50–75% | Bottom 25% | Top 10% − Bottom 25% (t) |
|---|---|---|---|---|---|---|---|
| DEV | 1M | +2.1% | +2.0% | +1.0% | +1.9% | +1.8% | +0.4% (0.53) |
| DEV | 3M | +6.0% | +4.3% | +4.2% | +6.0% | +4.9% | +1.1% (0.85) |
| DEV | 6M | +12.1% | +8.9% | +10.2% | +13.4% | +11.0% | +1.0% (0.62) |
| DEV | 12M* | +35.7% | +26.7% | +20.5% | +32.4% | +28.1% | +7.6% (n/i) |
| FINAL | 1M | +1.2% | +0.9% | +0.3% | +0.8% | +1.1% | +0.2% (0.15) |
| FINAL | 3M | +4.2% | +0.8% | +0.9% | +2.3% | +2.6% | +1.6% (0.41) |
| FINAL | 6M | +6.2% | +3.4% | +2.2% | +2.3% | +3.8% | +2.4% (0.41) |
| FINAL | 12M* | +12.6% | +7.2% | +3.0% | +4.1% | +3.6% | +9.0% (n/i) |

\* 12M: only 5 (DEV) or 8 (FINAL) heavily overlapping windows, roughly one
independent observation. "n/i" means the t-statistic is not interpretable
(section 23).

**The buckets are not monotonic.** The middle buckets (50–75%) and the
Bottom 25% often match or beat the 10–25% bucket. In FINAL, Top 10% leads at
every horizon, but the Top-minus-Bottom spread is statistically
indistinguishable from zero at 1M, 3M and 6M. Full statistics are in
`bucket_results.csv`: n, mean, median, hit rate, std and CIs per bucket.

**Primary pre-registered metric: mean rank-IC** (Spearman correlation of
score with forward return, across eligible stocks, per date):

| | 1M | 3M (primary) | 6M | 12M* |
|---|---|---|---|---|
| DEV | 0.017 (t 0.64) | **0.004 (t 0.13)** | 0.000 (t 0.01) | 0.026 (n/i) |
| FINAL | 0.015 (t 0.43) | **−0.004 (t −0.06)** | 0.019 (t 0.22) | 0.103 (n/i) |

## 12–14. Top 5 / Top 10 / Top 20 results

Portfolios are equal-weight, held over each horizon, and formed from
production ranks 1..N at each date. Each cell shows the mean per-date excess
return, with the Newey–West t in parentheses.

**vs the eligible universe** (the selection-skill measure):

| Portfolio | Seg | 1M | 3M | 6M | 12M* |
|---|---|---|---|---|---|
| Top 5 | DEV | +0.6% (0.49) | −1.3% (−0.48) | +3.5% (0.80) | −1.8% |
| Top 5 | FINAL | +1.1% (1.31) | +2.6% (0.75) | +4.6% (0.96) | +6.4% |
| Top 10 | DEV | +0.0% (0.03) | +0.2% (0.16) | +1.7% (0.68) | +3.0% |
| Top 10 | FINAL | +0.7% (1.02) | +3.3% (1.36) | +5.3% (1.81) | +8.1% |
| Top 20 | DEV | +0.5% (1.10) | +0.7% (0.66) | −0.1% (−0.10) | +6.7% |
| Top 20 | FINAL | +0.5% (0.91) | +2.1% (0.98) | +2.8% (1.05) | +6.5% |

**vs NIFTY 50** (includes the universe effect from section 9):

| Portfolio | Seg | 1M | 3M | 6M | 12M* | Hit rate vs NIFTY (3M) |
|---|---|---|---|---|---|---|
| Top 5 | DEV | +2.3% | +3.7% | +14.7% | +26.1% | 57% |
| Top 5 | FINAL | +1.9% | +4.5% | +7.8% | +11.4% | 61% |
| Top 10 | DEV | +1.7% | +5.3% | +12.9% | +30.8% | 71% |
| Top 10 | FINAL | +1.5% | +5.2% | +8.5% | +13.1% | 94% |
| Top 20 | DEV | +2.2% | +5.7% | +11.1% | +34.6% | 79% |
| Top 20 | FINAL | +1.3% | +4.1% | +6.0% | +11.5% | 78% |

**Mean absolute returns, Top 10:**

- DEV: 1M +3.0%, 3M +11.3%, 6M +24.3%, 12M +57.0%.
- FINAL: 1M +1.3%, 3M +6.3%, 6M +8.7%, 12M +13.6%.

Medians, hit rates and stock-level medians are in `top_n_results.csv`.

**Average monthly turnover** (1M rebalancing): Top 5 at 45% (DEV) and 40%
(FINAL); Top 10 at 46% and 37%; Top 20 at 37% and 33%.

**Reading:** in DEV, Top N does not beat the eligible universe at any horizon
with any significance. In FINAL, all Top N portfolios beat the universe by
point estimate (Top 10: +3.3% at 3M, +5.3% at 6M), but no 1M, 3M or 6M result
reaches t ≥ 2. The FINAL sample is 18–20 months of one market (2025–2026).

## 15–18. Results by horizon

- **1M:** IC 0.017 (DEV) and 0.015 (FINAL), both indistinguishable from zero.
  Top 10 vs universe is +0.0% (DEV) and +0.7% (FINAL).
- **3M (primary):** IC 0.004 (DEV) and −0.004 (FINAL). Top 10 vs universe is
  +0.2% (DEV) and +3.3% (FINAL, t 1.36).
- **6M:** IC 0.000 (DEV) and 0.019 (FINAL). Top 10 vs universe is +1.7% (DEV)
  and +5.3% (FINAL, t 1.81).
- **12M:** only 5 (DEV) and 8 (FINAL) dates, all with overlapping windows.
  The FINAL IC of 0.10 and Top 10 vs universe of +8.1% come from roughly one
  independent year (2025-01 to 2026-08). They are reported but are **not
  evidence**.

No horizon was dropped or fabricated. Every horizon whose windows had elapsed
is reported.

## 19. Risk metrics

The portfolio rebalances monthly from 1M non-overlapping returns: 17 DEV
months and 20 FINAL months.

**Definitions:**

- Volatility = std × √12.
- Downside volatility = √mean(r²) over negative months × √12.
- Sharpe uses a 0% risk-free rate. This is a stated simplification: Indian
  T-bill yields of about 6–7% would lower every Sharpe ratio.
- Information ratio = mean(r − NIFTY) / std × √12.
- Max drawdown is computed on the compounded monthly path.

| Portfolio | Seg | Cum. return | NIFTY cum. | Ann. vol | Downside vol | Sharpe (rf 0) | IR vs NIFTY | Max DD | Months beating NIFTY |
|---|---|---|---|---|---|---|---|---|---|
| Top 5 | DEV | +72.1% | +23.1% | 28.0% | 23.2% | 1.52 | 1.29 | −14.5% | 59% |
| Top 10 | DEV | +59.2% | +23.1% | 22.6% | 22.1% | 1.58 | 1.34 | −17.0% | 82% |
| Top 20 | DEV | +74.3% | +23.1% | 19.6% | 18.4% | 2.12 | 2.24 | −10.7% | 82% |
| Eligible universe (EW) | DEV | +61.1% | +23.1% | 16.9% | 16.5% | 2.10 | 2.44 | −7.9% | 82% |
| NIFTY 50 | DEV | +23.1% | | 12.4% | 10.2% | 1.25 | | −6.6% | |
| Top 5 | FINAL | +34.1% | −5.3% | 23.0% | 20.9% | 0.88 | 1.84 | −15.9% | 65% |
| Top 10 | FINAL | +25.1% | −5.3% | 20.7% | 20.7% | 0.75 | 1.58 | −11.1% | 60% |
| Top 20 | FINAL | +22.3% | −5.3% | 18.6% | 18.1% | 0.74 | 1.68 | −9.8% | 70% |
| Eligible universe (EW) | FINAL | +9.2% | −5.3% | 20.4% | 19.8% | 0.36 | 1.14 | −15.4% | 60% |
| NIFTY 50 | FINAL | −5.3% | | 14.9% | 15.6% | −0.15 | | −15.8% | |

**In DEV,** Top N carried more volatility and deeper drawdowns than the
equal-weight eligible universe, for similar or lower returns. **In FINAL,**
Top N beat the universe with similar volatility. The two periods disagree,
which is the pattern of noise, not of a persistent edge.

## 20. Market regime analysis

The regime is the production NIFTY 50 regime at each ranking date: 21
Sideways, 9 Bullish and 9 Bearish across the 39 dates.

| Regime | Dates (ALL, 3M) | Mean 3M IC | Top 10 mean 3M excess vs NIFTY |
|---|---|---|---|
| Bullish | 9 | +0.049 | +7.4% |
| Sideways | 19 | +0.022 | +3.1% |
| Bearish | 8 | −0.081 | +3.5% |

The IC is negative after Bearish regime readings, in both DEV (one date) and
FINAL (five dates). Regime cells contain 1–11 dates, so no regime-conditional
conclusion is supported. Full results are in `regime_results.csv`.

## 21. Component analysis

`component_results.csv` reports two tests for each component, by segment and
horizon:

- **Component-alone IC:** the component score used as the ranking signal.
- **Leave-one-out:** the change in StockLens Score IC when that component's
  weight is set to 0.

| Component | IC alone, DEV 1M / 3M / 6M | IC alone, FINAL 1M / 3M / 6M | Stable? |
|---|---|---|---|
| Valuation | 0.036 / 0.059 / 0.094 | 0.071 / 0.087 / 0.092 | **Yes:** positive in both periods (t 1.2–3.4). Removing it lowers the score IC in both. |
| Quality | −0.020 / −0.060 / −0.096 | 0.015 / 0.002 / −0.000 | Negative in DEV, zero in FINAL. Removing it raises IC in both, but not significantly in DEV (t 1.32 at 3M). |
| Financial Health | 0.021 / 0.045 / 0.066 | 0.004 / −0.032 / −0.050 | No: the sign flips. |
| Technical Trend | 0.019 / 0.048 / 0.087 | −0.024 / −0.043 / −0.013 | No: the sign flips. |
| Momentum | 0.042 / 0.061 / 0.091 | −0.049 / −0.070 / −0.042 | No: the sign flips. |
| Risk (low vol/drawdown) | −0.033 / −0.035 / −0.046 | −0.011 / −0.011 / 0.018 | Weakly negative to zero. |
| Market Regime | 0.040 / 0.013 / 0.008 | 0.009 / 0.002 / 0.040 | About zero. As a cross-sectional score it mostly reflects beta to the regime. |
| Sector Outlook | NOT_AVAILABLE at every date | NOT_AVAILABLE | **Untestable:** no historical outlook exists. Its 5% weight is redistributed by production's missing-component rule. |
| ML signal | weight 0, not evaluated | | Stays 0. No new independent evidence. |

**Findings:**

- Valuation is the only component with a stable, positive out-of-sample
  association.
- Quality is weak to negative, but it is measured on 1–2 fiscal years for
  most of the sample (section 8). That is a data limitation, not a
  demonstrated design defect.
- Trend, momentum and financial health reverse sign between periods.
- In aggregate, the components largely cancel, which explains the overall IC
  of about 0.

## 22. Weight comparison (selected on DEV only)

| Variant | Weights Q/V/FH/Tr/Mo/Ri/SO/MR | DEV 3M IC | FINAL 3M IC | DEV Top 10 3M vs NIFTY | FINAL Top 10 3M vs NIFTY |
|---|---|---|---|---|---|
| **A production** | 25/20/15/10/10/10/5/5 | 0.004 | −0.004 | +5.3% | +5.2% |
| B equal | 12.5 × 8 | 0.022 | −0.028 | +3.4% | +3.2% |
| C quality-heavy | 35/15/20/5/5/10/5/5 | −0.015 | 0.000 | +3.5% | +3.8% |
| D quality-value | 25/25/15/10/5/10/5/5 | 0.006 | 0.012 | +5.9% | +4.1% |

**Pre-registered rule** (fixed before any result was seen): replace A only if
an alternative beats it on DEV 3M IC by ≥ 0.02 with a Newey–West t ≥ 2 on the
per-date IC difference.

| Alternative | DEV IC difference vs A | t | Qualifies? |
|---|---|---|---|
| B | +0.018 | 0.65 | No |
| C | −0.019 | −1.25 | No |
| D | +0.002 | 0.39 | No |

**Selected: A (production).** Note that B, the best DEV variant, was the
worst in FINAL, which illustrates why the rule exists. The FINAL figures
above were computed once and played no part in the selection.

## 23. Statistical limitations

- **Sample size:** 14 usable DEV dates and 18 FINAL dates at 3M. That is
  about 1.5 years per segment, in one market, at one interest-rate cycle.
- **Detectable effect:** with a per-date IC standard deviation of about
  0.1–0.3, the minimum IC distinguishable from zero is about 0.05–0.15. An
  IC of 0.02–0.03, which is typical of real but modest factor skill, cannot
  be confirmed or rejected with this sample.
- **Overlapping windows:** monthly ranking dates have overlapping 3M, 6M and
  12M windows. Newey–West standard errors with lags = overlapping months − 1
  are used.
  - For 12M (lags 11) with only 5–8 observations, the estimator is
    degenerate and its t-statistics (4–20) are **not interpretable**.
  - The 12M point estimates rest on roughly one independent year per
    segment.
- **Cross-sectional dependence:** stock returns on the same date are
  correlated, so stock-level counts (n = 200–800 per bucket) overstate the
  information. Inference therefore uses per-date means.
- **Multiple comparisons:** 3 Top-N sizes × 4 horizons × 2 benchmarks × 2
  segments. A single t ≈ 2 among these is expected by chance.
- **Small differences:** differences of 1–2 percentage points between
  buckets or variants are within noise and are not treated as meaningful.
- **Biases:** survivorship and index-membership look-ahead (section 9); the
  statement snapshot may contain restatements; price returns exclude
  dividends; there are no transaction costs or slippage (monthly Top 10
  turnover is about 40%).

## 24. Final out-of-sample results (production weights, used once)

| Metric | Result |
|---|---|
| **3M rank-IC (primary)** | **−0.004** (t −0.06), 18 dates |
| Rank-IC at 1M / 6M | 0.015 (t 0.43) / 0.019 (t 0.22) |
| Top 10 vs eligible universe | 1M +0.7% (t 1.02); 3M +3.3% (t 1.36); 6M +5.3% (t 1.81) |
| Top 10 vs NIFTY 50 | 1M +1.5%; 3M +5.2% (t 2.86); 6M +8.5% (t 4.13) |
| Top 10% − Bottom 25% spread | 3M +1.6% (t 0.41); 6M +2.4% (t 0.41) |
| Top 10, monthly rebalanced, 20 months | +25.1% vs NIFTY −5.3% and universe +9.2%; volatility 20.7%; max drawdown −11.1% |

## 25. Methodology selected

**Ranking Engine v1.0 is unchanged.** Weights are 25/20/15/10/10/10/5/5 with
ML at 0; the rules, the FQVF checks and the components are as in section 1.

## 26. Reason

The success criteria require that higher ranks produce reasonably better
forward outcomes, with monotonicity, persistence across periods and adequate
sample size. None of these is demonstrated:

- The primary metric (3M rank-IC) is about 0 in both DEV and FINAL.
- Buckets are not monotonic.
- Top N beats the eligible universe only in FINAL, and not significantly.
- Most outperformance against NIFTY is a universe or survivorship effect.

No alternative met the pre-registered selection rule. Two candidate single
changes were considered:

- **Removing Quality:** DEV 3M IC change +0.076, t 1.32. It is not
  statistically meaningful on DEV, and the weakness coincides with Quality
  being measured on 1–2 fiscal years, so it is not clearly a design problem.
- **Raising Valuation weight** (variant D): failed the rule (+0.002).

The one-change limit therefore did not trigger.

## 27. Whether a production change was made

**No change to the ranking, the weights, the rules or FQVF.**

The following non-scoring changes were made, and none alters any score or
rank:

1. `config.RANKING_ENGINE_STATUS = "FROZEN"`, plus a test that pins
   `DEFAULT_WEIGHTS` and `DEFAULT_RULES`.
2. Append-only prospective tracking (section 28).
3. Engine runs record a frozen snapshot of each RANKING run.

## 28. Prospective tracking design

**What is recorded.** When a RANKING engine run completes,
`ranking.tracking.record_run_snapshots` writes one `ranking_snapshots` row per
stock. Each row holds:

- `run_id`, symbol and ranking timestamp;
- StockLens Score, score coverage, eligibility and rank;
- FQVF summary (score, coverage, PASS/WARNING/FAIL/NOT_AVAILABLE counts,
  version) and every component score;
- data freshness (fundamentals fetched at, fiscal period, market data as-of,
  technicals computed at, sector outlook timestamp);
- engine and FQVF versions and the NIFTY 50 market regime;
- reference date and close, and the NIFTY 50 reference close.

Runs that used administrator-overridden weights or rules are labelled
`ranking-v1.0+custom-config`, so they are never mixed with the frozen
configuration.

**When outcomes are recorded.** `ranking.tracking.record_outcomes` is called
by `POST /api/v1/admin/ranking-tracking/outcomes`, which is audit-logged.

- It inserts a `ranking_outcomes` row (horizon, start/exit date and price,
  stock return, NIFTY 50 return, excess) only after the 1M / 3M / 6M / 12M
  horizon (21 / 63 / 126 / 252 NIFTY sessions) has fully elapsed.
- Both endpoints come from one adjusted series fetched at outcome time.
- Unelapsed horizons, and stocks without a close within 5 days of the exit,
  stay pending; nothing is estimated.

**Immutability.**

- Snapshots and outcomes are **never updated or recomputed**. The unique
  constraints (run, symbol) and (snapshot, horizon) plus insert-only code
  enforce this.
- Historical results are not modified with later information.
- A test changes the prices after an outcome is recorded and verifies that
  the stored outcome is untouched.

**Reporting.** `GET /api/v1/admin/ranking-tracking/summary` reports, per
engine version, horizon and portfolio (Top 10, Top 20, all eligible): runs,
outcomes, mean return, mean excess vs NIFTY, mean excess vs the run's
eligible universe, and hit rate vs NIFTY. Each run counts as one
observation. `GET /api/v1/admin/ranking-tracking/snapshots` lists the frozen
snapshots.

**Operation.**

1. Run `alembic upgrade head` (migration `7a1f3c9d2b10`).
2. Keep the scheduled daily RANKING run.
3. Call the outcomes endpoint periodically, for example weekly.

Evaluate prospectively with the same primary metric (3M Top-10 excess vs the
eligible universe, and rank-IC) once at least 24 monthly runs have complete
3M outcomes. Any future weight change is a new engine version, needs a new
validation, and must not be tuned on the tracked results it is judged
against.
