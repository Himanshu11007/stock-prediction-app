# Prediction Engine v2 (shadow mode)

Prediction Engine v2 makes short-horizon snapshots for the current and the
next trading session. It is **separate from Ranking Engine v1.0**:

- It has its own code (`prediction_v2/`, `events/`) and its own tables
  (`db/models/prediction.py`, migration `b78f20585037`).
- It has its own universe and its own version labels.
- It never writes a v1 table. v1 scoring, eligibility, history and API
  contracts are unchanged; `tests/test_ranking_v1_golden.py` proves it
  against the v1 freeze commit `37bd87a`.

**Status: SHADOW.** v2 output is an unvalidated experiment.
- The API serves it to administrators only (`PREDICTION_V2_PUBLIC=false`).
- No exit alert or notification is sent.
- No confidence or probability is shown, because nothing is calibrated.

## Snapshot lifecycle (Asia/Kolkata)

| Run type | When | Data cutoff | For session | Job |
|---|---|---|---|---|
| `TODAY_PREOPEN` | before 09:15 (cron 08:30, retry 08:50) | previous session's close | today | `predict_preopen` |
| `TODAY_CONFIRMED` | 09:45–15:30 (cron 09:50) | previous close + the opening move up to the cutoff instant | today | `predict_confirmed`: **SKIPPED** unless `PREDICTION_V2_INTRADAY_ENABLED` and an approved intraday feed |
| `TOMORROW_EOD` | after 16:00, once today's NIFTY bar exists (cron 19:30, retry 20:15) | today's close | next trading day (weekends and `market.holidays` skipped) | `predict_eod` |
| outcomes and exits | 18:30 | — | — | `prediction_outcomes` |
| monitor | 09:40, 20:40 | — | — | `prediction_monitor` (exit code 1 = a snapshot is missing or failed) |

**Guarantees:**
- **One snapshot per run type per trading day.** `idempotency_key` is unique;
  duplicate triggers return the existing run.
- **One run per type at a time.** At most one RUNNING run per type, across
  all processes, enforced by a partial unique index.
- **Computed, then written once.** A run is computed in memory and written
  in a single transaction. **Completed runs and predictions are never
  modified**; a recalculation is a new run.
- **Quality gate.** If fewer than 50% of the universe have usable data, the
  run is **FAILED** and publishes nothing. A failed run keeps its record but
  releases the day's key, so the next trigger retries; the job slot caps
  attempts at 3.
- **Data not ready.** If the data isn't available yet (no NIFTY bar), the
  run is SKIPPED without being recorded, so a later trigger can still run.
- **Holiday calendar.** If `market.holidays` is empty, every snapshot
  records `holiday_calendar_configured: false`. Seed the NSE holiday list in
  Admin → Application Configuration.

## Features (`prediction_v2/features.py`, `v2-features-0.1`)

| Feature | Formula | Lookback |
|---|---|---|
| `ret_1/3/5/20` | c[-1]/c[-1-n] − 1 | n |
| `gap_pct` | open[-1]/c[-2] − 1 | 2 |
| `range_pct`, `close_location` | (h−l)/c[-2]; (c−l)/(h−l) | 1 |
| `abnormal_volume` | v[-1]/median(v[-21:-1]) | 21 |
| `atr_pct` | mean(true range, 14)/c | 15 |
| `vol_20` | stdev(daily returns, 20)·√252 | 21 |
| `dist_high_20`, `dist_low_20` | c/max(h[-20:]) − 1; c/min(l[-20:]) − 1 | 20 |
| `rs_nifty_5/20` | ret_n − NIFTY ret_n | n |
| `sector_rs_5` | ret_5 − mean(sector peers' ret_5); needs ≥ 3 peers | 5 |
| `traded_value_20` | mean(c·v, 20) | 20 |

Cutoff safety: every feature uses bars **on or before the cutoff date only**.
A test perturbs later prices and checks that nothing changes. A missing input
gives `None` plus a flag, **never 0**.

Flags:
- `NO_PRICE_DATA`, `STALE_PRICE`, `INSUFFICIENT_HISTORY_20` / `_60`
- `ZERO_VOLUME_LAST`, `LOW_LIQUIDITY` (under ₹5 crore a day),
  `SUSPECT_VOLUME_SPIKE` (50× or more)
- `POSSIBLE_PRICE_BAND`, `NO_BENCHMARK`, `BENCHMARK_MISALIGNED`

Recently listed stocks, such as Moneyview with 6 sessions, get **NO_CALL**.
Their history is never filled in.

## Rules (`prediction_v2/rules.py`, `baseline-v0.1`)

All thresholds are **a-priori hypotheses**. They were not fitted to any stock
or to the October 2026 case review, and changing one creates a new rule
version.

1. Any blocking data-quality flag or missing required feature → **NO_CALL**,
   with the reason recorded.
2. `MOMENTUM_CONTINUATION` → **UP**. All of these must hold:
   - 5-session return of at least 3%;
   - at least 2 points better than NIFTY over 5 sessions;
   - close in the top 30% of the day's range;
   - volume at least 1.5× the 20-session median;
   - within 2% of the 20-session high.
3. `BREAKDOWN` → **DOWN**: the mirror image.
4. Otherwise → **NEUTRAL**. This is the expected common answer.

Levels (also hypotheses): stop 1.5 ATR, target 2 ATR, trailing 1 ATR. Each
call also records an entry condition and an invalidation condition.
`confidence` stays null until a calibration exists.

## Outcomes and performance

- `prediction_outcomes` (`outcomes-v0.1`) records 1, 3 and 5 sessions from
  the target session. Each outcome is written once all bars in its window
  are final, and is unique per (prediction, horizon).
- **Returns:** start is the frozen reference price. Returns are
  direction-adjusted. The cost assumption is 15 bps plus 5 bps slippage. MFE
  and MAE are recorded, along with the NIFTY return and the sector-basket
  return.
- **NEUTRAL and NO_CALL** predictions get NOT_APPLICABLE outcomes, which
  still record the stock's return, so it's visible what was declined.
- **Performance (`/predictions/performance`):** per setup and horizon:
  - hit rate with a 95% Wilson interval;
  - mean cost-adjusted return with a 95% interval;
  - excess return vs NIFTY;
  - MFE and MAE;
  - coverage and no-call rate.
- **Small samples:** statistics are hidden below 20 calls.
- **Promotion gate (provisional, decided by a person):**
  - at least 200 holdout events per setup;
  - a cost-adjusted interval above zero;
  - 4–8 weeks of shadow operation.

## Exit engine (`prediction_v2/exits.py`, `exit-shadow-0.1`, shadow only)

HOLD → TIGHTEN → PARTIAL_EXIT → EXIT, with direct jumps to EXIT on a stop or
an invalidation. EXIT is terminal and a repeated signal is not a transition.

Order of checks within a bar:
1. a missing bar → no change;
2. a gap through the stop → exit at the open;
3. an intraday stop breach;
4. a close beyond the reference bar's low or high (SUPPORT_BREAKDOWN);
5. invalidation or macro signals;
6. target reached → PARTIAL_EXIT, then trailing;
7. failed breakout;
8. catalyst exhaustion;
9. distribution;
10. relative-strength deterioration;
11. time stop.

Transitions are stored in `exit_transitions`, unique per (prediction,
session, state). **No notification is sent.**

## Backtest (`prediction_v2/backtest.py`, `scripts/research/backtest_v2.py`)

- Chronological development / validation / holdout splits (60/20/20), with
  an embargo of 5 + horizon sessions at each boundary.
- Entry at the cutoff close, with costs at 10, 15 and 20 bps.
- **Baselines:**
  - always neutral;
  - previous-session direction;
  - sector direction;
  - volatility-matched random (same number of calls per session, seeded);
  - v1 top-20 from the month-end validation snapshots.
- The latest report is `scripts/research/output/prediction_v2/backtest_report.json`;
  its interpretation is in `docs/research/PREDICTION_V2_BACKTEST_2026-10.md`.
- **Limitations:**
  - survivorship bias (today's universe is used for the whole period);
  - Yahoo daily data;
  - no event, consensus or intraday inputs.

## Events (`events/ingest.py`)

The model keeps raw events (`market_events`, unique per source and source id)
separate from versioned, reviewable classifications
(`event_classifications`). A correction adds a new row, and the original is
kept.

**Point in time:** `effective_available_at = max(published_at, ingested_at)`.
Backfilled history counts as known only at ingestion time, unless the
provider's times are declared point-in-time.

### Event sources: no provider is configured (BLOCKED)

| Source | Status |
|---|---|
| NSE / BSE corporate announcements (public web pages / unofficial JSON) | Not used. The terms restrict automated access and there is anti-bot protection |
| Licensed exchange feeds or vendors (filings, results, corporate actions) | Needs a contract and cost approval |
| Broker ratings, historical consensus | Paid vendors only. Expectation-based setups (re-rating, surprise) stay untestable until then |
| `FixtureProvider` | Implemented; deterministic JSON for tests and manual curation |

To add a provider, implement `fetch(since, until) -> RawEvent[]` and run
`events.ingest.ingest(...)` from a scheduled job. No classifier may invent
financial values.

## API

| Route | Notes |
|---|---|
| `GET /api/v1/predictions?session=today\|tomorrow&direction=&page=&page_size=` | One run only: today prefers today's confirmed snapshot, then pre-open. Returns `freshness` (FRESH / STALE / NONE), run metadata (cutoff, versions, counts, holiday flag) and `shadow: true` |
| `GET /api/v1/predictions/{id}` | Frozen features, outcomes, shadow exit state and events |
| `GET /api/v1/predictions/performance`, `GET /api/v1/predictions/runs` | |
| `GET /api/v1/stocks/{symbol}/events`, `GET /api/v1/watchlist/exit-signals` | Exit signals: `actionable: false` |
| `GET` / `POST /api/v1/admin/prediction-runs` | The POST runs the same calendar-aware job; audited |
| `GET /api/v1/admin/events`, `PATCH /api/v1/admin/events/classifications/{id}` | Review is audited; the original is kept |
| `GET` / `POST /api/v1/admin/v2-universe`, `POST .../deactivate` | Changes the v2 universe only; v1 is never touched |
| `GET /api/v1/admin/universe-health` | v1 symbols without market data; companies outside v1 (never ranked) |

All routes require ADMIN unless `PREDICTION_V2_PUBLIC=true`.

## Operations

1. **Enable.** Apply migrations (`alembic upgrade head`), then seed the v2
   universe once (`prediction_v2.universe.seed_from_v1`, or the admin POST),
   then choose a scheduler (`docs/OPERATIONS_AND_SECURITY.md`).
2. **Flags:**
   - `PREDICTION_V2_PUBLIC` (default false);
   - `PREDICTION_V2_INTRADAY_ENABLED` (default false; only with an approved
     intraday feed).
3. **Memory.** A run fetches 6 months of daily bars for the universe, in
   batches of 50 symbols.
