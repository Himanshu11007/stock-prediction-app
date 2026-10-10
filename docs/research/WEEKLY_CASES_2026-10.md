# Case review: 5–9 October 2026

**Reproducible evidence:**
- `scripts/research/case_evidence_2026_10.py` →
  `scripts/research/output/prediction_v2/case_evidence_2026_10.json`. It
  covers Canara Bank forensics, intraday reaction timing, and the session
  audit with missed movers.
- `scripts/research/weekly_case_ledger.py` →
  `weekly_case_ledger.{json,md}`: per-case daily ledger.

Both scripts only read data: the local database copy, Yahoo Finance, and the
v1 month-end validation CSV. Nothing in this review was used to change, tune
or add a rule. v2 has no stock-specific logic.

## Evidence status

| Item | Status |
|---|---|
| Daily prices and volumes | Yahoo Finance daily bars, the engine's own provider: delayed, adjusted, not exchange data. VERIFIED FROM HISTORICAL SOURCE |
| Intraday timing (open gap, move by 09:45, time of the day's high or low) | Yahoo 5-minute bars in IST: unlicensed, delayed, research only. VERIFIED FROM HISTORICAL SOURCE |
| v1 ranks, components, run times | Live run `RANKING-20261003-091616-7eb2ae` in the local database copy (VERIFIED IN DATABASE), plus 39 month-end research reproductions (validation CSV) |
| Catalyst publication and ingestion times (Trent and Titan business updates, the ITC broker action, CPCL news) | **BLOCKED.** No filings or news source with publication timestamps is connected (F-05), and nothing was ingested that week, so no ingestion times exist. The intraday bars bound **when the market repriced**, not when the information was published |
| Original 5–9 Oct experiment reports | Not available to this review; their claims are REPORTED BUT NOT VERIFIED |
| Production snapshots and job history | **BLOCKED.** No read-only production access (F-13). The local database is on an older schema without `scheduled_job_runs`, so no job records exist locally |
| "v2 sim" | A **historical simulation** of `baseline-v0.1` at the previous close. **No v2 snapshot existed that week**, so these are not recovered predictions |

NSE was closed on 2 Oct (no bars for any symbol). The StockLens holiday
calendar is not configured (F-07).

## 1. Canara Bank (C-01)

**Timeline (IST)**

| When | What | Evidence |
|---|---|---|
| Thu 1 Oct, 15:30 | Data cutoff: close ₹118.36. 2 Oct was a holiday | market snapshot `as_of_date` 2026-10-01 |
| Sat 3 Oct, 14:46–14:50 | Live v1 run: **rank 1 of 157 eligible, score 77.0**, coverage 0.95, computed 14:50:35 | `engine_runs`, `stock_analysis_results` |
| Sat 3 Oct, 14:50 | Visible in Top Candidates from this time. Reference price ₹118.36 is a **fallback** (`market_snapshot_at_analysis`), because the run has **0 `ranking_snapshots` rows** (F-09) | local database |
| Mon 5 Oct, 09:15 | First tradable moment: open ₹119.39 | Yahoo daily |
| 5–9 Oct | Daily moves +0.12%, −0.57%, +1.13%, −1.51%, +1.91% | Yahoo daily |

**Components at selection (live run):**

| Component | Score | Weight | Contribution |
|---|---|---|---|
| quality | 100 | 25 | 26.3 |
| valuation | 83.3 | 20 | 17.5 |
| financial health | 100 | 15 | 15.8 |
| technical trend | **8.3** (daily and weekly bearish) | 10 | 0.9 |
| momentum | 62.7 (60-day return −2.3%) | 10 | 6.6 |
| market regime | 50 (sideways, ADX 14) | 5 | 2.6 |
| risk | 69.3 | 10 | 7.3 |
| sector outlook | not available (F-08) | 5 | — |
| ML signal | 55.3 | 0 (informational) | — |

The **30 Sep reproduction** had rank 1 of 156 and score 78.3, with
quality 100, valuation 91.7, financial health 100, trend 8.3, momentum
58.1, regime 50 and risk 69.5. These match the reported figures. It is a
research reproduction committed on 3 Oct; **no user could have seen it on
30 Sep.**

**Rank history (month-end reproductions):**
- Top-10 in **24 of 39** snapshots, not "24 of 28" as reported.
- Ranks 46–95 from Jul 2023 to May 2024.
- Then in the top 10 almost continuously from Oct 2024.
- Rank 1 in Jun 2025, Jul 2025, Dec 2025 and Sep 2026.

This is a persistent fundamental selection, not a new signal.

**Outcomes from the frozen reference (1 Oct close) and from the first tradable open (5 Oct):**

| Window | 1 session | 3 sessions | 5 sessions | MFE / MAE (5 sessions) |
|---|---|---|---|---|
| Canara from ₹118.36 | +0.12% | +0.67% | +1.04% | +2.20% / −1.56% |
| NIFTY | +0.60% | +0.81% | +0.44% | |
| Bank Nifty | +0.48% | +1.11% | **+1.48%** | |
| Canara from the 5 Oct open (₹119.39) | −0.75% | −0.20% | +0.17% | +1.32% / −2.41% |
| Bank Nifty from its 5 Oct open | −0.23% | +0.39% | +0.76% | |

**Baselines from the same live run (5 sessions from each stock's 1 Oct close):**

| Baseline | Return |
|---|---|
| Equal-weight v1 top 10 (Canara plus 9) | −2.02% |
| Top 10 by the run's own momentum component | −0.01% |
| Neutral | 0 |

**First qualifying move:** **none.** The rule, fixed in advance, needed
+3% and +2 points above Bank Nifty within 5 sessions.

**Classification** (rules fixed in advance, in the script docstring):
**"Coincidental positive outcome."**
- Canara rose +1.04% but **lagged Bank Nifty by 0.44 points**, so the move
  is explained by the sector.
- It was **not a verified short-term early-signal success, and cannot be
  one:** v1 is a long-term ranking with no frozen direction or horizon.
- The selection was driven by quality, valuation and financial health, with
  a bearish technical trend. No event input exists.
- No Canara-specific rule exists or was added.

## 2. Contrasting cases

For each case, the timing column states what the bars show, which is when
the market repriced. Publication and ingestion times remain **BLOCKED**.

| Case | v1 (3 Oct) | What happened (daily) | Intraday timing (5-minute, IST) | What could StockLens have known? |
|---|---|---|---|---|
| **Trent**: Q2 business update | rank 119 (41.8) | 5 Oct flat; **6 Oct +12.6%**; held the gain on 7–9 Oct | 6 Oct: **open gap +8.1% at 09:15; 82% of the day's move done by 09:45**; low at 09:15, high at 14:10 | The information was public **before the 6 Oct open** (fully gapped in) and absent from the 5 Oct close. A pre-open snapshot could only have used it with a timestamped filings feed (BLOCKED). The size of the gap means it was not actionable at the open; only a post-gap continuation setup was. The reported move sizes vary because they use different bases: the gap is +8.1% and the close-to-close move +12.6% |
| **ITC**: broker re-rating (reported Citi upgrade) | rank 93 (50.0) | **5 Oct +5.1%**; 8 Oct −4.0%; 9 Oct +4.3% | 5 Oct: gap only +1.4%, 35% of the move by 09:45, **high at 14:55** (built through the session); 9 Oct similar (35% by 09:45, high 13:55) | Repricing continued after the open, so a confirmed intraday setup was possible in principle. The broker note's time and target are **REPORTED, BLOCKED** (no broker-rating source) |
| **CPCL** (CHENNPETRO): commodity / factor shock and reversal | **not in v1 universe** (F-01, F-02) | 5 Oct +4.5%; **7 Oct +14.7%**; 8 Oct −5.0%; 9 Oct −2.8% | 7 Oct: gap +0.4%, **+6.2% by 09:45** (44% of the day), high 15:20. 8 Oct reversal: −3.5% by 09:45 (72%), low 15:25 | The 7 Oct move developed **during** the session, not at the open. CPCL was not ranked at all. Crude and crack observations at each cutoff: **BLOCKED** (no factor data; the `FactorObservation` table has no writer) |
| **Titan**: negative surprise despite growth | rank 54 (57.1) | weak 29 Sep – 1 Oct (−6.3%); **7 Oct −3.8%** on high volume | 7 Oct: **gap −2.4%, the whole move done by 09:45**; high 09:15, low 09:20 | Public before the 7 Oct open. Whether growth was "below expectations" needs historical consensus: **BLOCKED** |
| HCL Technologies: negative control | rank 38 | 5 Oct −3.3%; 9 Oct +3.4% (IT index about +3%) | 5 Oct: drift down all day (low 15:10); 9 Oct: 69% by 09:45 | Sector moves; one member of a negative-control set, not a tuning target |
| MRPL (refiner peer) | not in universe | 7 Oct +4.7%; 8 Oct −4.9% | 7 Oct: flat open, 88% by 09:45 | Refiner peer basket (v2 universe) |
| IOC / BPCL / HPCL (oil marketing companies) | 4 / 2 / 96 | 7 Oct −1.2 / −1.4 / −2.2%; 8 Oct −2.9 / −3.4 / −4.8% | 8 Oct: small gaps; lows around 12:30 | Exposure opposite to refiners. These are **wrong-side exposure of v1 top ranks, not false positives**, because v1 is not directional |
| Kanohar Electricals | not in master list | 17 sessions since 16 Sep; daily −7.6% to +15.8% | — | v2 gives NO_CALL (`INSUFFICIENT_HISTORY_60`); no history is fabricated |
| Moneyview | not in master list | listed 1 Oct; +19.7% on 7 Oct | — | NO_CALL (`INSUFFICIENT_HISTORY_60`) |

## 3. Session audit, missed movers and formal classification

**Formal false positives and false negatives cannot be classified** for this
week. They need frozen directional predictions made before each session, and
none existed:
- v1 ranks are not directional;
- no v2 snapshot ran;
- no job ledger exists locally.

The table below uses a **simulation** for v2.

| Session | Stocks with a bar | v2 sim at previous close (UP / DOWN / NEUTRAL / NO_CALL) | Sim calls right next session | Movers ≥ ±5% | in v1 top 20 | v1-ineligible | outside v1 | Top Candidates at 18:00 |
|---|---|---|---|---|---|---|---|---|
| 5 Oct | 285 | 1 / 8 / 260 / 16 | 3 of 9 | 10 | 0 | 5 | 1 | BEHIND (1 session) |
| 6 Oct | 285 | 1 / 12 / 256 / 16 | 5 of 13 | 14 | 1 (WELCORP) | 6 | 1 | STALE (2) |
| 7 Oct | 285 | 5 / 4 / 258 / 18 | 4 of 9 | 5 | 0 | 0 | 3 | STALE (3) |
| 8 Oct | 285 | 6 / 4 / 258 / 17 | 4 of 10 | 15 | 0 | 4 | 2 | STALE (4) |
| 9 Oct | 285 | 1 / 29 / 238 / 17 | 9 of 30 | 3 | 0 | 2 | 1 | STALE (5) |

**Week totals**

| | |
|---|---|
| Material movers | **47 stock-days across 37 stocks** (28 up, 19 down): 39 stock-days in the v1 universe and 8 outside it (CPCL, Kanohar, Moneyview) |
| Surfaced by v1 top 20 | 1 (WELCORP #14, +6.5% on 6 Oct) |
| v1-ineligible | 17 |
| v2 sim on movers | 1 directional call, in the wrong direction: KANSAINER UP, then −6.1% |
| v2 sim across all directional calls | 25 of 71 in the right direction (35%), consistent with the negative backtest. Anecdotal; not evidence of anything |

**Revision:** the earlier figure of "38 stock-days across 33 stocks" came
from an ad-hoc scan of the v1 universe only. This scan is reproducible and
also covers stocks outside the universe; in the universe it finds 39 across
34. The difference is provider data revisions.

**Data exclusions:**
- 17 v1 symbols have no Yahoo data (F-03), including TATAMOTORS, ZOMATO,
  LTIM, ISEC and GUJGASLTD.
- MRPL's moves were below 5%, so it is not counted.

**Top Candidates refresh:**
- No v1 full run happened after 3 Oct, so every session of the week served
  the 1 Oct ranking.
- That ranking was **1–5 sessions old**, and nothing in the product showed
  it.
- The new `ranking_freshness` field and the web banner (T-02) now show it:
  BEHIND on 5 Oct, then STALE.
- The reference price was a fallback (F-09) and is now labelled
  `reference_price_source`.

**Late or failed jobs:** no scheduler existed (O-01); the run was manual.
Production history is BLOCKED.

## 4. Missing parameters and catalyst gaps found by this review

| Gap | Effect this week | Status |
|---|---|---|
| No timestamped filings or news (F-05) | Trent, Titan, ITC and CPCL catalysts are unknowable in time | event schema and ingestion IMPLEMENTED; provider BLOCKED |
| No historical consensus | Titan "negative surprise" is not measurable | BLOCKED |
| No broker-rating source | ITC re-rating is not measurable | BLOCKED |
| No commodity or crack-spread observations | CPCL / OMC factor split is not measurable at the cutoff | `FactorObservation` schema only; **no writer** (DEFERRED) |
| v1 universe omissions | CPCL, MRPL, Kanohar, Moneyview | v2 universe can add them; v1 change REQUIRES_APPROVAL |
| Holiday calendar empty (F-07) | 2 Oct treated as a trading day by config | scheduler refuses to run until seeded; seeding REQUIRES_APPROVAL |
| Sector outlook unset (F-08) | 5% of v1 weight never evaluated | DEFERRED (admin data) |
| No frozen ranking snapshot rows (F-09) | reference price fallback | labelled; production BLOCKED |
| Stale Top Candidates (no daily run) | 1–5 sessions old | freshness metadata IMPLEMENTED; scheduler inactive (REQUIRES_APPROVAL) |

## 5. Hypotheses: not tested, not adopted

These are candidates only. Each must be tested across **all** comparable
events on development and validation data, never fitted to these stocks.
- Post-gap continuation after a business update (Trent).
- Intraday confirmation of a re-rating (ITC), which needs a licensed
  intraday feed and broker events.
- An event-day reference-level reversal exit (CPCL on 8 Oct opened flat,
  then fell 3.5% by 09:45).
- Refiner-versus-OMC factor exposure with past-only betas.
- A negative-surprise setup (Titan), which needs historical consensus.

The exit engine and `prediction_outcomes` will record exits for live shadow
snapshots. No retrospective claim is made that any exit "would have worked".
