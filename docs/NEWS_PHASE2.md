# News Phase 2: live provider validation, early detection, end-to-end integration (11 Oct 2026)

**Scope:** research and shadow only.
- No scheduling, alerts, production migrations, deploys, purchases or
  credential changes.
- Ranking Engine v1.0 is only read. Prediction Engine v2 stays in shadow mode.

## 1. Baseline (stored benchmark, 69 events)

- **Reproduced offline:** `python scripts/research/news_provider_benchmark.py --resummarize`
  rebuilds every metric from the stored outcomes, with no network access.
- **Reproduced on identical events:** Google News, GDELT and the keyed
  providers were re-queried with
  `--events-from <stored report> --providers google_news,gdelt,marketaux,alphavantage,newsapi`.
  - That report predates stored definitions, so its event definitions were
    rebuilt with the rules that created them (`legacy_definition`).
  - Every new report stores them (`events[].definition`).
- **Reproducibility:**
  - **Google News:** 54 DETECTED and 12 MISSED are unchanged. The 3 other
    earlier detections are now **AMBIGUOUS** (rule below); no outcome
    changed otherwise.
  - **GDELT:** not reproducible from this network. 82 of 99 requests were
    rate-limited, so its tested set changes between runs.

### New statuses and metrics

- **AMBIGUOUS:** the only verified articles are weak. That means a market
  roundup or a headline naming several companies, or a first report more
  than 24 h after an exact official time.
  - AMBIGUOUS counts as tested.
  - **Strict** recall treats it as not detected; **lenient** recall treats
    it as detected.
- **Article precision:** unambiguous verified articles divided by the
  candidate articles returned for the subject.
- **Segments:**

  | Segment | Event kinds |
  |---|---|
  | `indian_company` | NSE and BSE filings |
  | `indian_macro` | RBI, SEBI, Government of India |
  | `international_policy` | Fed, White House, Federal Register |
  | `commodity_currency` | crude, gold, USD/INR |

- **Audit trail:** one row per provider and event, with these flags:

  | Flag | Meaning |
  |---|---|
  | `WEAK_MATCH` | the only matches are weak (AMBIGUOUS) |
  | `NO_TIMING_DATE_ONLY_EVENT` | the official time is date-only |
  | `NO_TIMING_PLACEHOLDER_TIME` | the first article has only a placeholder time |
  | `TIMEZONE_SUSPECT` | the article is earlier than the official time by more than the tolerance |
  | `OTHER_COMPANIES_TAGGED` | other Indian companies tagged as the subject |
  | `FALSE_CANDIDATES` | unverified candidate articles were returned |

  It is served by `GET /admin/news/benchmark/audit` and shown on the
  dashboard.

### Google News on identical events (strict)

| Segment | Detected / tested | Recall |
|---|---|---|
| indian_company | 19 / 28 | 0.68 |
| indian_macro | 5 / 7 | 0.71 |
| international_policy | 18 / 22 | 0.82 |
| commodity_currency | 12 / 12 | 1.00 (date-only ground truth: no timing) |
| **all** | **54 / 69** | **0.78 (95% CI 0.67–0.86)**; lenient 0.83 |

- **Timing:** median delay 42 min, p90 276 min, n=13 exact-time
  detections. All 13 arrived before the market reaction (95% CI 0.77–1.00).
- **Precision:** article precision is 0.45. More than half of the articles
  returned about a subject are not about the event.
- **Value over StockLens's own official feeds:** they lag by 39 min on
  covered kinds. The 5 actionable new detections are all NSE filings.

**Recommendation:** none.
- **Google News** now fails strict recall (0.78 < 0.8) as well as the
  licence requirement.
- **These results do not prove predictive value:** detection is not
  profit. See §6.

### Limitations of the baseline

- 69 events, so the intervals are wide.
- Commodity, currency and US-policy ground truth is date-only, so no timing
  can be measured there.
- Google News results are ranked today, and headlines can change after
  publication.
- NSE ground truth was collected before the terms review (§4); it is not
  re-collected.

## 2. Marketaux and Alpha Vantage: live validation (keys set 11 Oct 2026)

`scripts/research/news_provider_probe.py` (logic in `catalysts/probe.py`)
checks:
- connectivity and authentication;
- rate limits;
- plan restrictions;
- the raw schema;
- timestamps;
- history at 30, 365 and 730 days back;
- Indian coverage.

Keys are read from the user environment and are never stored, printed or
logged.

| Check | Marketaux (free plan) | Alpha Vantage (free key) |
|---|---|---|
| Connectivity / authentication | PASS / PASS | PASS / PASS (key accepted) |
| Plan | OK, but **3 articles per request** | **PLAN_RESTRICTED: NEWS_SENTIMENT is a premium endpoint** ("This is a premium endpoint", for both a ticker and a topic query) |
| Schema | PASS (uuid, title, url, published_at, source, entities with symbol / exchange / country / match_score) | not testable |
| Timestamps | PASS: ISO with `Z`, seconds resolution, none future-dated, no midnight placeholders | not testable |
| History | articles returned for windows 30, 365 and 730 days back | not testable |
| Indian coverage | NSE symbols in its entity tags (RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS ...); also block-deal / partly-paid tickers (`-BL`, `PP`) | not testable |

**Benchmark on the identical 69 events** (`--events-from ... --providers marketaux,alphavantage`):
- 69 Marketaux requests and 1 Alpha Vantage request. After the premium
  refusal, the runner stops asking Alpha Vantage.

| Measure | Marketaux | Google News (same events) |
|---|---|---|
| Detected / ambiguous / missed | 27 / 5 / 37 | 54 / 3 / 12 |
| Recall strict (95% CI) / lenient | **0.39 (0.28–0.51)** / 0.46 | 0.78 (0.67–0.86) / 0.83 |
| Article precision | 0.43 | 0.45 |
| Median delay (n) / p90 | **235 min (10)** / 916 min | 42 min (13) / 276 min |
| Before market reaction | 0.50 (0.24–0.76) | 1.00 (0.77–1.00) |
| Exact timestamps | 100% | 20% |
| Mapping: provider tags / StockLens | 0.89 / 1.00 (n=9) | - / 0.84 |
| By segment | Indian company 9/28, Indian macro 0/7, international policy 9/22, commodity / currency 9/12 | 19/28, 5/7, 18/22, 12/12 |

- **No unique events:** Marketaux detected **no event that Google News
  missed** (27 jointly detected, 23 detected only by Google, 0 only by
  Marketaux).
- **Free-plan cap:** at 3 articles per request, a query about a busy company
  often returns only its 3 most relevant stories, which may not be the
  event. That caps recall.
- **Date-only events:** some "detections" of the date-only Fed and commodity
  events are previews or market wraps ("Fed rate decision looms"). This is
  the known limit of date-only ground truth: no timing is claimed for them.

**Verdict:** neither provider meets the bar.
- **Marketaux** fails recall, delay, actionability and the "new events"
  criterion (4 < 5), and its licence is not reviewed.
- **Alpha Vantage** news cannot be evaluated without a paid plan. Not
  purchased.
- A paid Marketaux plan (more articles per request) is the only way to
  measure its real recall. Do this only if you approve the cost.

## 3. Event-time accuracy

For every article, `news_articles` keeps:

| Field | Meaning |
|---|---|
| `published_raw`, `published_tz` | The publication time exactly as the source wrote it, and the zone it stated. `NONE->IST` means the zone was assumed by a feed rule, e.g. RBI and NSE |
| `published_at` | Normalised UTC |
| `first_seen_at` | The provider's availability, e.g. GDELT seendate |
| `ingested_at` | StockLens's own first sight |
| `corrected_at` | A correction, if any |
| `effective_available_at` | `max(published, ingested)` |
| `provider`, `url`, `event_id` | Source, original URL and canonical event |
| `quality_flags`, `provider_entities` | Data-quality flags and the provider's own entity tags |

`event_entities.evidence` records **why** a stock is linked: a provider tag,
the title or excerpt text that matched, a hypothesis id, or an industry
peer.

The migration is `3c1f2a9d7e44`: additive, upgrade, downgrade and check
verified, **not applied in production**.

**Point in time:**
- The replay never uses an article before StockLens ingested it.
- A corrected article is unusable until its correction arrived, because the
  archive holds the corrected text.
- Timestamps are never invented: a missing source time stays missing, and
  undated feeds use first sight.

## 4. Indian-market coverage: exchange terms

- **NSE:** the Terms of Use, read on 11 Oct 2026, state: *"User is
  prohibited to conduct any systematic or automated data collection
  activities (including scraping, data mining, data extraction and data
  harvesting)"*. Content may not be *"stored ... without prior written
  permission of NSE"*. The NSE Data Policy routes commercial use through
  NSE Data licensing.
- **BSE:** the web terms on bseplus.bseindia.com read *"for your personal
  and non-commercial use"*. BSE sells a "Corporate Data" product. Its
  official RSS page could not be fetched from here (404), so the feed is
  **not verified**.
- **Consequence:**
  - Every automated NSE source refuses to run until the exchange's
    permission or licence reference is recorded. This covers the JSON
    announcements adapter and the new official RSS adapter. The variables
    are `NSE_AUTOMATED_ACCESS_PERMISSION` and
    `BSE_AUTOMATED_ACCESS_PERMISSION`.
  - The benchmark no longer rebuilds NSE or BSE ground truth without them.
- **Official NSE RSS** (`Online_announcements.xml`), verified once:
  - exact IST times, structured subjects, the latest ~35 filings, no
    history;
  - the adapter (`nse_rss`) is ready but gated.
- **Filings materiality** (`catalysts/filings.py`):

  | Class | Includes |
  |---|---|
  | HIGH | results, guidance, orders, regulatory action, defaults, M&A, pledges, management changes, fund raising, credit ratings |
  | MEDIUM | corporate actions, press releases, litigation, capacity |
  | EXCLUDED | analyst meets, prior intimations, newspaper copies, trading window, recordings, certificates, ESOPs, routine compliance |

  - EXCLUDED items are dropped unless a category is listed in
    `DEMONSTRATED_RELEVANT`, which is empty.
  - Anything unmatched is UNKNOWN and kept for review, never assumed HIGH.
- **Retained:** the RBI, SEBI, PIB, White House and Fed feeds.
  Commodity, currency and yield sources are evaluated separately: the
  date-only ground truth is listed in §1 limitations.

## 5. Event intelligence and stock filtering

Built on `catalysts/impact.py`, which already handled credibility tiers,
novelty, materiality, priced-in detection, conflicting reports, and
sentiment kept apart from direction. `catalysts/candidates.py` adds a
per-stock screen with:
- one call per stock: **UP / DOWN** only with decisive, credible,
  non-contradicted, **fresh** evidence and usable price data;
  **NEUTRAL** with its reasons; **NO_CALL** with no news or no data;
- separate scores: novelty, credibility, materiality, freshness (1 − age /
  category horizon) and surprise;
- text sentiment shown next to `expected_return_direction`, which is
  deliberately different;
- likely horizon and affected sectors;
- risks: conflicts, priced in, inferred-only, mention-only, no credible
  source, expected outcome, priced-in check not performed;
- exclusion reasons;
- evidence with source links, publication / provider / ingestion /
  availability / correction times, and the mapping evidence.

Confidence stays `null` (not calibrated).

**API** (all admin-only, shadow):
- `GET /admin/news/candidates?session=today|tomorrow` returns the screen
  next to the latest v2 snapshot call for the same session. A v2 NO_CALL
  (data quality) is never overridden. Without a snapshot, price inputs are
  marked `UNAVAILABLE`.
- `GET /admin/news/status` returns, from the database only:
  - per-provider health: OK / STALE / FAILING / NOT_CONFIGURED / REFUSED /
    NEVER_RUN;
  - the last run and last success;
  - coverage, and the stocks without news;
  - the newest article;
  - the **data basis**: REAL, STALE or UNAVAILABLE.

The **combined pipeline into v2** already existed:
- `NEWS_INGESTION` collects from every approved provider that
  `catalysts.registry.select` returns; research-only providers need
  `NEWS_ALLOW_RESEARCH_PROVIDERS=true`, and missing credentials are refused
  with a reason;
- the pipeline normalises times, groups syndicated and duplicate stories
  without merging different companies, and maps entities;
- v2 snapshots combine news with the market rules
  (`PREDICTION_V2_NEWS_ENABLED`, default true) and freeze the evidence.

## 6. Does news improve predictions? (point-in-time backtest)

Harness: `scripts/research/news_signal_backtest.py` (logic in `catalysts/signal_eval.py`, tested).

**Setup:**
- **Universe:** the 20 large caps of `INDIAN_ALIASES`, fixed in advance
  for unambiguous names.
- **Window:** 14 Oct 2025 to 9 Oct 2026, split chronologically into
  development, validation and holdout, embargoed by 5 + 1 sessions.
- **Outcome:** close(t) to close(t+1), with 30 bps round trip (20 cost +
  10 slippage); sensitivity at 10 and 50 bps.
- **Same stock-sessions for every arm.** Thresholds fixed a priori; nothing
  tuned.
- **Point in time:** each article is usable from the next full hour after
  publication. Date-only articles (94% of the Google backfill) are usable
  after the end of their day IST. The replay ingests point in time.

**Archives:**

| Arm | Status |
|---|---|
| Google News | 42,076 articles from 624 requests; 464 calls, 3,764 neutral-with-news. Not point in time: optimistic |
| StockLens archive | **UNAVAILABLE**: no news tables in the local database |
| Marketaux, Alpha Vantage | **NOT_CONFIGURED**: not tested |
| GDELT | **NOT_RUN**: rate-limited from this network and only ~3 months deep |
| Combined | identical to Google News (one archive) |

**Development (145 sessions, 2,900 stock-sessions):**

| Arm | Calls | Hit rate (95% CI) | Mean net per trade (session-bootstrap 95%) | Max drawdown |
|---|---|---|---|---|
| news_google_news | 278 | 0.50 (0.44–0.56) | −34 bps (−58 to −9) | 36% |
| v2_plus_news | 377 | 0.50 (0.45–0.55) | −26 bps (−47 to −4) | 21% |
| v2 | 122 | 0.48 (0.40–0.57) | −9 bps (−52 to +33) | 12% |
| v1_rank | 594 | 0.50 (0.46–0.54) | −37 bps (−55 to −20) | 58% |
| always_up | 2,780 | 0.49 (0.47–0.51) | −35 bps (−51 to −19) | 50% |
| random_matched | 122 | 0.50 | −31 bps | 28% |

**Holdout (43 sessions, 7 Aug to 8 Oct 2026, 860 stock-sessions):**

| Arm | Calls | Hit rate (95% CI) | Mean net (session-bootstrap 95%) | Sample |
|---|---|---|---|---|
| news_google_news | 68 | 0.37 (0.26–0.49) | −40 bps (−79 to +2) | insufficient (< 100) |
| v2_plus_news | 93 | 0.34 (0.26–0.45) | −55 bps (−93 to −18) | insufficient |
| v2 | 25 | 0.28 (0.14–0.48) | −95 bps | insufficient |
| v1_rank | 176 | 0.40 (0.33–0.47) | −55 bps (−76 to −35) | adequate |
| always_up | 860 | 0.40 (0.37–0.43) | −52 bps (−70 to −33) | adequate |

**By event category (news arm, all splits):** no category shows a
dependable edge.
- EARNINGS: 237 calls, hit 0.45, −40 bps.
- LITIGATION: 49 calls, hit 0.39, −84 bps.
- MONETARY_POLICY: 37 calls, hit 0.62, +10 bps. Too few calls, and not
  tested beyond chance.

**Verdict:** no news arm improves on v2 or on the baselines.
- Every comparison is either INSUFFICIENT_SAMPLE or
  NO_EVIDENCE_OF_IMPROVEMENT.
- In development, with an adequate sample, the news arm's direction
  accuracy equals a coin flip and its net return is negative with an
  interval excluding zero.
- This holds even though the Google arm is optimistic: Google's ranking and
  possible headline edits are later information.
- **News, as captured by these sources and rules, does not improve
  StockLens's predictions.** Nothing here supports promoting news-driven
  calls out of shadow mode.

**What would change the conclusion:**
- a genuinely point-in-time archive with exact times: StockLens's own
  scheduled ingestion, or licensed exchange filings;
- enough holdout calls (≥ 100);
- a measured edge after costs.

Rerun with `python scripts/research/news_signal_backtest.py` (cached
backfill in the system temp directory, outside the repository).

## 7. End-to-end integration and web

- **Backend:** `tests/test_news_e2e.py` runs:
  1. ingestion from four fake-HTTP providers: official RBI RSS with
     zone-less IST, Marketaux with entity tags plus a syndicated copy,
     rate-limited GDELT, and Alpha Vantage without a key;
  2. the NEWS_INGESTION job;
  3. timestamps, grouping and mapping;
  4. a v2 TOMORROW_EOD snapshot;
  5. the admin API.

  It then compares the API payloads with **contract fixtures**
  (`tests/contracts/`). The web tests render those same files.
- **Web** (StockLens.Web):
  - `/predictions/candidates/today|tomorrow` shows calls, the v2 call and
    link, evidence, source links, published / ingested / usable times,
    confidence ("Not calibrated"), scores, risks, inclusion and exclusion
    reasons, and the news-input panel (data basis, provider health,
    coverage, staleness);
  - the benchmark dashboard adds ambiguous / lenient recall, precision,
    segments, probe status, the audit trail and the backtest arms. Its test
    renders the **real** API response exported by
    `scripts/export_web_news_fixtures.py`.

## 8. Configuration

| Variable | Purpose |
|---|---|
| `MARKETAUX_API_TOKEN`, `ALPHAVANTAGE_API_KEY` | keyed providers (free: ~100 and 25 requests a day). Not set |
| `NEWS_PROVIDERS` | providers for NEWS_INGESTION (default `official_rss`) |
| `NEWS_ALLOW_RESEARCH_PROVIDERS` | must be `true` before any research-only provider is selected |
| `NSE_AUTOMATED_ACCESS_PERMISSION`, `BSE_AUTOMATED_ACCESS_PERMISSION` | written permission or licence reference; without it every exchange source refuses |
| `NEWS_BENCHMARK_REPORT` | optional report path for the dashboard |
| `PREDICTION_V2_NEWS_ENABLED` | news in v2 snapshots (default true; shadow) |

## 9. Unresolved

- **Alpha Vantage news:** needs a paid plan to be tested. Marketaux was tested on the free plan only (3 articles per request).
- **NSE / BSE:** permission or licence needed. This is the decision with the
  most measured value.
- **StockLens's own news archive does not exist yet.** The local database
  has no news tables (migrations not applied), and production ingestion is
  not scheduled. Until it runs, there is no genuinely point-in-time news
  history.
- Exact-time ground truth for commodity, currency and US-policy events.
- Clustering of Google-style syndication: about 13 canonical events per
  real event.
