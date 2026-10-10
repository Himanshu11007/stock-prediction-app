# Case review: 5–9 October 2026

**Reproducible data:** `scripts/research/weekly_case_ledger.py`, which writes
`scripts/research/output/prediction_v2/weekly_case_ledger.{json,md}`.
Additional sources:
- `scripts/research/output/ranking_validation/ranking_snapshots_production.csv`
  (v1 month-end validation snapshots);
- the local database copy of the live run `RANKING-20261003-091616-7eb2ae`.

## What is known and what is not

| Item | Status |
|---|---|
| Prices and volumes | Yahoo Finance daily bars, the engine's provider (not exchange data). VERIFIED FROM HISTORICAL SOURCE |
| v1 ranks | VERIFIED IN DATABASE (local copy of the 3 Oct live run) and validation CSV |
| Original week experiment reports | Not available to this review. REPORTED BUT NOT YET VERIFIED |
| Catalyst publication and ingestion times (ITC broker action, Trent and Titan business updates, CPCL news) | BLOCKED: no event source; nothing was ingested at the time |
| Production snapshots | BLOCKED: no read-only production access |
| "v2 sim" column | **Historical simulation** of the v2 baseline at the previous close. No v2 snapshot existed that week. 8 directional calls in 72 stock-sessions: anecdotal, not evidence, and **nothing was tuned on it** |

NSE was closed on 2 Oct (no bars). The holiday calendar in StockLens is not configured.

## Canara Bank (C-01)

| | |
|---|---|
| Validation snapshot dated 30 Sep (research reproduction, committed 3 Oct) | Rank 1 of 156; score 78.3. Components: quality 100, valuation 91.7, financial health 100, trend 8.3, momentum 58.1, stock regime 50, risk 69.5. FQVF coverage 0.812; sector outlook unset. **Matches the reported figures.** |
| Live run (3 Oct, 14:46 IST, data cutoff 1 Oct close) | Rank 1, score 77.0 |
| Top-10 frequency | 24 of **39** month-end snapshots (the report said 28); rank 1 four times |
| From the 30 Sep close ₹119.44 (1 / 3 / 5 sessions) | −0.90% / −1.36% / −1.75%. MFE +1.27%, MAE −2.45%. NIFTY −0.88 / +0.69 / −1.72; Bank Nifty −0.33 / +0.91 / −0.22 |
| From the 1 Oct close ₹118.36 (live cutoff) | +0.12% / +0.67% / +1.04%. NIFTY +0.44% and Bank Nifty +1.48% at 5 sessions |
| Largest up day | +1.91% (9 Oct) |
| Conclusion | Long-term quality and value selection driven by quality, valuation and financial health; the technical trend was bearish. **No qualifying short-term move; underperformed Bank Nifty.** Not a short-term early-signal success. No Canara-specific rule exists or was added |

## Cases

| Case | v1 (3 Oct) | What happened | Knowable in time? | Engine gap |
|---|---|---|---|---|
| ITC | rank 93 | 5 Oct +5.1% (gap +2.0%, 4.2× volume); 8 Oct −4.0%; 9 Oct +4.3% | Broker-action time unknown (BLOCKED) | No broker or event data (F-05) |
| Trent | rank 119 | 5 Oct flat; **6 Oct gap +8.5%, close +12.6%, 10× volume**; held the gain | Only after the announcement (the previous close was flat) | No filings data; v2 sim made an UP call at the 6 Oct close (Trent −0.5% next day) |
| Titan | rank 54 | Weak from 29 Sep; **7 Oct gap −2.4%, close −3.8%, 5.2× volume** | Weakness was visible in prices; the surprise needs historical consensus (BLOCKED) | v2 sim: DOWN at the 30 Sep close (next −1.6%), at the 6 Oct close (next −3.8%) and at the 8 Oct close (next +1.2%) |
| HCL Technologies (negative control) | rank 38 | 5 Oct −3.3%; 9 Oct +3.4% with the IT index +3.0% | — | v2 sim NEUTRAL throughout |
| CPCL | **not in v1 universe** (F-02) | 7 Oct +14.7% (about 13× volume; already +5.5% by 09:45 intraday); 8 Oct −5.0%; 9 Oct −2.8% | No price precursor; v2 sim NEUTRAL at the 6 Oct close and NEUTRAL at the 7 Oct close (not within 2% of its 20-session high) | Universe coverage; factor and event data. A reversal-exit hypothesis (event-day close or low) needs testing across all event days, not on CPCL alone |
| MRPL | not in v1 universe | 7 Oct +4.7%; 8 Oct −4.9% | — | peer basket |
| IOC / BPCL / HPCL | 4 / 2 / 96 | 7 Oct −1.2 / −1.4 / −2.2%; 8 Oct −2.9 / −3.4 / −4.8% (Brent +4.1%) | — | Oil-marketing exposure is the opposite of refiners'. v1 top ranks fell, but v1 is not directional, so these are not false positives. v2 sim DOWN for all three at the 8 Oct close (next day: +0.1, +0.5, −0.2%) |
| Kanohar Electricals | not in master | 17 sessions since 16 Sep; daily −7.6% to +15.8% | — | v2 gives NO_CALL (`INSUFFICIENT_HISTORY_60`) |
| Moneyview | not in master | 6 sessions since 1 Oct; −4.4% to +19.7% | — | NO_CALL; no history fabricated |

## Missed movers and formal errors

- A full v1-universe scan for 5–9 Oct found **38 stock-days of ±5% across
  33 stocks**: 21 up, 17 down. Only 1 involved a v1 top-20 stock, and 16
  involved v1-ineligible stocks. CPCL, MRPL, Kanohar and Moneyview were
  outside the universe.
- **No formal false positives or false negatives:** v1 makes no directional
  predictions and no v2 snapshots existed, so none can be computed (C-03).
  v2 now freezes directional snapshots, so this becomes measurable going
  forward.
