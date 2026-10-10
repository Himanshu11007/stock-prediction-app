# Prediction v2 baseline backtest: investigation (10 October 2026)

**Run:**
```
python scripts/research/backtest_v2_analysis.py
```
It writes `scripts/research/output/prediction_v2/backtest_analysis.json` and
took 3,364 s. Prices are cached at `storage/price_cache/backtest_v2_3y.pkl`,
which is local and gitignored.

This supersedes the first run (`backtest_v2.py` → `backtest_report.json`,
same day), which measured horizons on each stock's own bars (finding
P2-01). The 1-session holdout result for v2 is unchanged by the fix (below).

**Rules are `baseline-v0.1` exactly as written; nothing was tuned.** The
holdout of this rule version had already been read once, and is read again
here only after a verified defect fix. Any new rule version must be
developed on development and validation data only. Its holdout is forward
shadow operation.

## Configuration

| | |
|---|---|
| Data | Yahoo Finance daily bars, auto-adjusted; NIFTY 50 has 741 sessions (9 Oct 2023 – 9 Oct 2026) |
| Universe | 281 of the 299 v1 universe symbols with data (18 without: F-03) |
| Rules and features | `baseline-v0.1` (`prediction_v2/rules.py` THRESHOLDS); `v2-features-0.1` |
| Entry and exit | close of the cutoff session t → close of the **h-th NIFTY session after t**, h = 1, 3, 5. A stock without a bar on t or on that session gives no trade |
| Splits | chronological 60 / 20 / 20. **Development** Jan 2024 – Aug/Sep 2025 (about 406 sessions); **validation** Sep 2025 – Mar 2026 (about 127); **holdout** Apr 2026 – Oct 2026 (125–130). 5 + h sessions are embargoed at each boundary |
| Costs | 10, 15 and 20 bps round trip, **plus 5 bps slippage**, giving 15, 20 and 25 bps applied per trade (provisional 0.1–0.2% range plus slippage) |
| Baselines | always neutral; previous-session direction; sector direction (peer basket, ≥3 peers); random direction matched to v2's number of calls per session (seeded); v1 top 20 at the latest month-end reproduction (as UP); **always UP** (market drift) |
| Metrics | hit rate (gross direction-adjusted return > 0) with a Wilson interval; mean gross, net and NIFTY-excess return per trade; naive and **session-clustered bootstrap** 95% intervals; coverage and no-call rate; an equal-weight per-session portfolio (drawdown, concentration); breakdowns by setup, direction, sector, market regime (NIFTY vs its 50-session average; 20-session volatility vs its one-year median) and liquidity (20-session traded value) |

## Integrity checks

| Check | Result |
|---|---|
| Look-ahead, synthetic | `test_backtest_is_chronological_embargoed_and_leak_free`: changing every price after a cutoff leaves earlier decisions unchanged |
| Look-ahead, real data | 300 sampled v2 trades re-decided from bars truncated at their cutoff: **0 mismatches** |
| Regime labels | computed from NIFTY bars ≤ t only (`test_backtest_regime_breakdown_and_clustered_interval`) |
| Horizon alignment (P2-01, fixed) | under the old method, 246 of 207,753 one-session windows (0.12%), 738 of 207,191 three-session windows (0.36%) and 1,230 of 206,629 five-session windows (0.60%) were measured over the wrong end session. **Too few to explain the result**: the 1-session holdout for v2 is identical before and after the fix (1,604 calls, hit 47.3%) |
| Features | sliced at the cutoff inside `features.compute`; unit-tested formulas, missing-data and cutoff tests |

## Results: holdout, 25 bps (20 bps cost + 5 bps slippage)

Net returns are per trade. "CI" is the session-clustered 95% bootstrap interval.

| Horizon | Strategy | Calls | Hit rate | Gross | Net | Net CI | Excess vs NIFTY |
|---|---|---|---|---|---|---|---|
| **1** | **v2** | 1,604 | **47.3%** | −0.02% | **−0.27%** | **−0.45, −0.10** | +0.07% |
| 1 | random matched | 1,604 | 50.4% | −0.03% | −0.28% | −0.38, −0.16 | −0.08% |
| 1 | previous day | 36,276 | 48.1% | −0.02% | −0.27% | −0.35, −0.20 | −0.01% |
| 1 | sector | 36,400 | 49.5% | +0.03% | −0.22% | −0.34, −0.10 | +0.04% |
| 1 | v1 top 20 | 2,600 | 49.2% | +0.01% | −0.24% | −0.41, −0.09 | +0.01% |
| 1 | always UP | 36,530 | 48.5% | +0.09% | −0.16% | −0.33, −0.00 | +0.09% |
| **3** | **v2** | 1,564 | 48.1% | +0.01% | **−0.24%** | −0.57, +0.07 | +0.17% |
| 3 | random matched | 1,564 | 52.2% | +0.14% | −0.11% | −0.25, +0.04 | +0.19% |
| 3 | sector | 35,840 | 50.3% | +0.06% | −0.19% | −0.44, +0.03 | +0.07% |
| 3 | v1 top 20 | 2,560 | 46.9% | −0.01% | −0.26% | −0.54, +0.02 | +0.01% |
| 3 | always UP | 35,968 | 48.6% | +0.26% | +0.01% | −0.28, +0.29 | +0.28% |
| **5** | **v2** | 1,535 | 49.3% | +0.12% | **−0.13%** | −0.53, +0.26 | +0.34% |
| 5 | random matched | 1,535 | 49.9% | −0.10% | −0.35% | −0.59, −0.12 | −0.06% |
| 5 | sector | 35,000 | 50.1% | +0.08% | −0.17% | −0.49, +0.13 | +0.20% |
| 5 | v1 top 20 | 2,500 | 45.8% | −0.09% | −0.34% | −0.68, +0.01 | +0.03% |
| 5 | always UP | 35,125 | 48.4% | +0.37% | +0.12% | −0.23, +0.51 | +0.49% |

Always neutral makes no trades (0). v2 coverage is about 4.4% of
stock-sessions, with a no-call rate of 5–8%. All cost levels are in the JSON.

**Development and validation, v2 gross per trade:**

| Horizon | Development | Validation | Holdout |
|---|---|---|---|
| 1 | −0.02% | −0.12% | −0.02% |
| 3 | −0.07% | −0.27% | +0.01% |
| 5 | −0.21% | −0.24% | +0.12% |

v2's gross return was **never positive in development or validation** at any
horizon.

## Breakdowns (holdout, v2, 25 bps; gross / net per trade)

| | 1 session | 3 sessions | 5 sessions |
|---|---|---|---|
| UP (MOMENTUM_CONTINUATION), n ≈ 1,040 | +0.07 / −0.18 | +0.20 / −0.05 | +0.34 / +0.09 |
| DOWN (BREAKDOWN), n ≈ 500–550 | −0.20 / −0.45 | −0.37 / −0.62 | −0.34 / −0.59 |
| NIFTY above its 50-session average | +0.11 / −0.14 | −0.11 / −0.36 | −0.09 / −0.34 |
| NIFTY below its 50-session average | −0.14 / −0.39 | +0.13 / −0.12 | +0.33 / +0.08 |
| High volatility | +0.15 / −0.10 | +0.17 / −0.08 | +0.30 / +0.05 |
| Low volatility (n = 417) | −0.51 / −0.76 | −0.42 / −0.67 | −0.36 / −0.61 |
| Liquidity < ₹20 cr per day | −0.04 / −0.29 | −0.52 / −0.77 | −0.45 / −0.70 |
| Liquidity ₹20–100 cr | +0.09 / −0.16 | +0.25 / +0.00 | +0.31 / +0.06 |
| Liquidity ≥ ₹100 cr | −0.08 / −0.33 | +0.02 / −0.23 | +0.16 / −0.09 |

Per-sector results (11 sectors × 3 horizons) are in the JSON. With that
many cells, a few are positive by chance; for example Technology and Real
Estate at 5 sessions. **They are not findings, and must not be used to
select rules.**

**Per-session equal-weight portfolio (1 session, non-overlapping):**
- mean −0.30% a day;
- 38% positive sessions;
- **−33% over the 130 holdout sessions**, with a maximum drawdown of −37%;
- the top decile of trades produces 55% of positive gross returns.

The 3- and 5-session portfolios overlap, so they are indicative only.

## Is the result meaningful?

- **At 1 session, the lack of edge is statistically clear.**
  - The clustered net interval (−0.45%, −0.10%) excludes zero.
  - In gross terms the interval is about −0.20% to +0.15%. A gross edge
    large enough to cover 25 bps is ruled out at 95%.
  - The smallest edge the holdout could detect is about ±0.17% per trade
    (half-width of the clustered interval).
- **At 3 and 5 sessions the holdout is underpowered.**
  - The half-widths are ±0.32% and ±0.39%, so a cost-covering edge can't be
    excluded statistically.
  - However, the point estimates are negative or near zero.
  - The sign is inconsistent across splits: development and validation gross
    returns are negative at every horizon.
  - UP calls beat simply buying everything (always UP) only in
    development at 1 and 3 sessions (gross +0.12% vs +0.06%, and +0.22% vs
    +0.17%). They trail it in validation and in the holdout at every
    horizon (holdout: +0.07 / +0.20 / +0.34% vs +0.09 / +0.26 / +0.37%).
  - There is no evidence of an edge to promote.
- **It is not caused by a verified implementation defect.** The only defect
  found (P2-01) affected 0.1–0.6% of windows and leaves the 1-session result
  unchanged. Look-ahead checks are clean. The rules behave as specified
  (unit tests).
- **Methodology limits don't rescue the rules:**
  - DOWN calls need short selling, which Indian cash equities don't allow
    overnight, so the DOWN half isn't implementable as specified.
  - The UP half does not beat market drift outside development.
  - Survivorship bias (today's universe) flatters long-only results, so the
    true result is probably worse.
- **Not tested here:**
  - catalyst and event setups (no event data);
  - intraday confirmation (no licensed feed);
  - exits versus holding (live shadow outcomes will measure this).

## Decision

- `baseline-v0.1` has **no edge after costs**. It stays in **shadow mode**
  and is **not promoted**.
- The provisional promotion gate is not met. It requires at least 200
  holdout events per setup, a cost-adjusted interval excluding zero on the
  positive side, and 4–8 weeks of shadow operation.
- Next research steps, all on development and validation data only, with a
  new `RULE_VERSION`:
  - long-only variants judged against always-UP drift, not zero;
  - catalyst setups, once an event source exists;
  - liquidity and volatility conditioning, specified **in advance**. The
    breakdowns above suggest hypotheses; they do not confirm them.

## Limitations

- **Survivorship:** today's universe list is used for all of history.
- **Data:** Yahoo data, not exchange data; adjusted closes.
- **Sectors:** today's `companies.sector`; peers are not point-in-time.
- **Costs:** fixed per trade; no market impact; borrow is assumed for DOWN
  calls.
- **v1 baseline:** month-end research reproductions, not live runs.
- **Overlap:** windows overlap for h > 1. The clustered bootstrap resamples
  cutoff sessions, which handles same-day correlation but not overlap
  across adjacent days, so the 3- and 5-session intervals are somewhat too
  narrow.
