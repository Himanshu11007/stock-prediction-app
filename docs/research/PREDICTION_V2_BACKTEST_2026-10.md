# Prediction v2 baseline backtest (10 October 2026)

**Run:** `python scripts/research/backtest_v2.py --period 3y --horizon 1`.
**Report:** `scripts/research/output/prediction_v2/backtest_report.json`.

## Setup

| | |
|---|---|
| Data | Yahoo Finance daily bars; NIFTY 50, 741 sessions (9 Oct 2023 – 9 Oct 2026) |
| Universe | 281 of the 299 v1 universe symbols with data (18 renamed or delisted, finding F-03) |
| Rules | `baseline-v0.1` exactly as written (no tuning); features `v2-features-0.1` |
| Entry / exit | Close of the cutoff session → close one session later (TOMORROW_EOD semantics) |
| Splits | development 407 sessions (Jan 2024 – Sep 2025), validation 130 (Sep 2025 – Mar 2026), **holdout 130 (Apr – Oct 2026)**; 6-session embargo at each boundary |
| Costs | 10, 15, 20 bps per trade |
| Baselines | always neutral (no trades); previous-session direction; sector direction; random direction with the same number of calls per session (seeded); v1 top-20 at the latest month-end validation snapshot |

## Results: holdout

| Strategy | Calls | Hit rate (95% CI) | Net return per trade, 10 bps (95% CI) | Net, 20 bps |
|---|---|---|---|---|
| **v2 baseline-v0.1** | 1,604 | **47.3%** (44.9–49.8) | −0.12% (−0.25, +0.01) | −0.22% |
| random, matched | 1,604 | 50.4% (47.9–52.8) | −0.13% | −0.23% |
| previous-day direction | 36,276 | 48.1% | — | −0.22% |
| sector direction | 36,400 | 49.5% (49.0–50.0) | −0.07% | −0.17% |
| v1 top-20 (as UP) | 2,600 | 49.2% (47.2–51.1) | −0.09% | −0.19% |
| always neutral | 0 | — | 0 | 0 |

**Development and validation agree:**
- **v2:** hit 48.9% in development and 46.4% in validation; net return at
  10 bps was −0.12% and −0.22%.
- **Sector direction** was the only baseline with a positive net return in
  development (+0.02% at 10 bps). It did not hold in validation or the
  holdout.

**Coverage:**
- v2 calls about 4.4% of stock-sessions; NO_CALL is 5–8%.
- Calls by setup: MOMENTUM_CONTINUATION 5,016, BREAKDOWN 3,333.

## Interpretation

- **The baseline rules have no edge.** They are slightly worse than random
  at a 1-session horizon (below 50%; continuation after strong 5-day moves
  tended to revert), and every directional strategy loses after costs.
- **The promotion gate is not met:** 1,604 holdout events, but the
  cost-adjusted interval does not exclude zero on the positive side. v2
  stays in **shadow mode**; nothing is shown to customers.
- **The holdout has now been looked at once for `baseline-v0.1`.** Any new
  rule version must be developed on development and validation only. Its
  holdout is the forward shadow period: 4–8 weeks of live snapshots, scored
  by `prediction_outcomes`.
- **Not tested here:**
  - 3- and 5-session horizons (`--horizon 3|5`);
  - catalyst and event setups, because there is no event data;
  - intraday confirmation, because there is no licensed feed;
  - the exit engine's effect versus holding. The backtest holds for the
    horizon; `prediction_outcomes` records simulated exits for live
    snapshots.

## Limitations

- **Survivorship:** today's universe list is used for all of history.
- **Data:** Yahoo data, not exchange data, and adjusted closes.
- **Costs:** fixed per trade, with no market-impact modelling.
- **v1 baseline:** uses research month-end snapshots, not live runs.
- **Overlap:** returns are per trade and overlapping, so the intervals treat
  trades as independent and overstate certainty.
