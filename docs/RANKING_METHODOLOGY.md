# StockLens Score and Top Investment Candidates

Implementation: `ranking/service.py` (`StockRankingService`),
`ranking/technical.py` (technical/risk metrics), `engine_runs/service.py`
(orchestration). Version: `ranking-v1.0` (`config.RANKING_ENGINE_VERSION`).
Tests: `tests/test_ranking.py`.

**Status: FROZEN** (`config.RANKING_ENGINE_STATUS`). The out-of-sample
validation in [RANKING_VALIDATION_V1.md](RANKING_VALIDATION_V1.md) found no
sufficient evidence of incremental stock-selection skill (3-month rank-IC
about 0 in both the development and the final test period). v1.0 is kept
unchanged as an **analytical ranking rather than a validated predictive
edge**, and every production run is tracked prospectively. Changing weights,
rules or components requires a new engine version and a new validation.

The StockLens Score (0–100) ranks stocks against each other on the same,
explained inputs. **It is not a probability, a price target or a return
forecast.**

## Components

Each component is scored 0–100 or is **unavailable** (never defaulted).

| Component | Default weight | Definition |
|---|---|---|
| `quality` | 25 | FQVF checks 1, 2, 5, 12, 16 (PASS 100, WARNING 50, FAIL 0; NOT_AVAILABLE excluded) |
| `valuation` | 20 | FQVF checks 6, 7, 8, 9, 10, 11, 14 |
| `financial_health` | 15 | FQVF checks 13, 15, 17 |
| `technical_trend` | 10 | (mean of daily and weekly trend score, −1..+1, from `get_trend_signal`) mapped to 0–100 |
| `momentum` | 10 | percentile of the 60-day return within the analysed universe |
| `risk` | 10 | 100 − mean percentile of (annualised volatility, |1-year max drawdown|) within the universe |
| `sector_outlook` | 5 | FQVF check 18 (administrator outlook) |
| `market_regime` | 5 | the stock's own regime score (−1..+1, `utils/regime.detect_regime`) mapped to 0–100 |
| `ml_signal` | **0** | ensemble P(up) × 100 — informational only (see below) |

Percentile components need at least 10 stocks in the pool. A single-stock
on-demand analysis is ranked against the latest full run's universe.

## Formula

```
StockLens Score = Σ (weight_i × score_i) / Σ weight_i     over available components i with weight > 0
coverage      = Σ available weights / Σ all weights
```

Missing components are excluded and the remaining weights renormalised;
`coverage` is always reported with the score. Weights are administrator-
configurable (`ranking.weights`, validated: known keys, ≥ 0, positive sum,
audited); the run stores the weights it used.

### Why the ML signal has weight 0

Walk-forward research (docs/ML_EXPERIMENT_REGISTRY.md, Phases 10–15) found
the LR/RF/XGBoost direction classifier has **no measurable out-of-sample
discrimination** (AUC ≈ 0.50 across horizons, models and feature sets). It is
computed and displayed with that disclaimer; an administrator can assign a
weight, which is recorded in the run configuration.

## Top Investment Candidates

`GET /api/v1/top-picks` returns, from the **latest completed RANKING run**,
the eligible stocks ordered by rank (descending score, ties by symbol),
limited by `top_picks.limit` (default 20).

Eligibility rules (`ranking.rules`, administrator-configurable):

| Rule | Default |
|---|---|
| Stock active and tradable in the Stock Master | required |
| Market data status OK and ≤ `max_market_data_age_days` old | 4 days |
| Fundamentals OK or PARTIAL (not UNAVAILABLE / ERROR) | required |
| FQVF coverage ≥ `min_fqvf_coverage` | 50% |
| Score coverage ≥ `min_score_coverage` | 60% |
| 20-day average volume ≥ `min_avg_volume_20d` (liquidity) | 500,000 shares |

Ineligible stocks still have a score in their own analysis, with the reasons
listed (`ineligible_reasons`).

Every candidate carries: rank, StockLens Score, coverage, FQVF score and
summary, positive factors (FQVF PASS explanations and strong components),
risks (FAIL/WARNING checks, weak components, unavailable checks, high
volatility), data freshness (fundamentals fetched, fiscal period, market data
as-of, technicals computed, sector outlook set), engine version and timestamp.

## Limitations

- Scores are relative to the analysed universe and the period's data.
- No evidence is claimed that higher scores produce higher returns; the
  ranking is a transparent screen, not a validated forecast. The
  point-in-time validation (docs/RANKING_VALIDATION_V1.md) confirms this:
  buckets are not monotonic and Top N does not significantly beat the
  eligible universe out of sample. Of the components only Valuation showed a
  stable positive association; Sector Outlook could not be tested.
- Technical/momentum/risk inputs use daily bars up to the last session.
