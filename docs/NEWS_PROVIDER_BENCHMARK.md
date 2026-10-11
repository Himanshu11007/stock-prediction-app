# News provider benchmark (research only, 11 Oct 2026)

This benchmark answers one question: **would any provider let StockLens detect
material events before the relevant market move, beyond what StockLens's own
official sources already deliver?**

- **Scope:** research only. No provider is scheduled, alerting or used in
  predictions. Ranking Engine v1.0 is unchanged and Prediction Engine v2 stays
  in shadow mode. No production migration or deploy took place.
- **Report:** `scripts/research/output/prediction_v2/news_provider_benchmark.json`
- **Dashboard:** web `/predictions/news-benchmark`, served by
  `GET /api/v1/admin/news/benchmark` (admin only).
- **Code:** `catalysts/benchmark.py` (logic) and
  `scripts/research/news_provider_benchmark.py` (runner).

## 1. Method

### Ground truth

Ground truth means independently verified events with timestamps. It is
**built from the sources of record**, never from the providers being tested.

| Kind | Source of record | Time precision |
|---|---|---|
| nse_filing | NSE corporate announcements (`exchdisstime`, dissemination time). 20 large caps, recent window plus Jul 2025 results | exact |
| bse_filing | BSE announcements API. Material subjects only: results, board outcomes, press releases, orders. Excludes audio recordings, transcripts and newspaper copies | exact |
| rbi, sebi, fed, us_policy_whitehouse | Official RSS feeds; material titles only, deduplicated | exact (SEBI: date only) |
| india_policy | PIB RSS | **none**: the feed has no pubDate, so it is unusable as ground truth (reported as such) |
| us_policy_federal_register | Federal Register presidential documents on tariffs and sanctions | date (publication date, usually days after signing) |
| fed (rate decisions) | FRED `DFEDTARU` changes | date |
| crude, gold, currency | Brent, gold and USD/INR daily moves of at least 2 standard deviations (prior 60 days) | date |

### Outcome per provider and event

Each provider gets exactly one outcome per event. Only DETECTED and MISSED
count towards recall. **Missing history is never a pass.**

| Status | Meaning |
|---|---|
| DETECTED / MISSED | The provider was queried for the event's window |
| UNTESTABLE_NO_HISTORY | The event is older than the provider's archive, or the provider is a live-only feed |
| UNTESTABLE_NOT_CONFIGURED | No credential |
| SOURCE_OF_RECORD | The provider is the ground truth for that kind |
| OUTAGE / RATE_LIMITED | Reported separately, never as misses |

### Verification

An article counts as reporting an event only if all of the following hold:
- the subject is in the **headline** (roundups that mention it only in the
  excerpt don't count);
- the event's terms match;
- the title is not a preview;
- for exact times, the article was not published more than 30 minutes before
  the official release.

Unverified candidates are kept and reported as false matches.

### Timing

Delay is measured on exact-time events and exact-time articles only. It is
the delay of the first verified article after the official time.

- **Market reaction:** the official time if NSE is in session, otherwise the
  next 09:15 IST open. NSE holidays are inferred from days without a NIFTY
  bar.
- **Actionable:** the article arrived at or before the market reaction,
  plus 15 minutes.

### Value over existing sources

StockLens already reads RBI, SEBI, the Fed, PIB and the White House at the
official time. On those kinds a provider can only lag, so the report gives
that lag. Value is measured on the other kinds.

NSE filings are counted as *not* covered, because the NSE adapter is
research-only until its licence is decided.

### Recommendation bar

All of the following must hold. Article volume and sentiment are not
criteria.
- at least 20 tested events and recall of at least 0.8;
- median delay of 60 minutes or less, over at least 5 exact detections;
- at least 70% of detections actionable before the market reaction;
- at least 5 timed, actionable detections of events StockLens's own sources
  don't deliver;
- the licence is cleared for production use.

### Reproduce

```bash
python scripts/research/news_provider_benchmark.py
python scripts/research/news_provider_benchmark.py --resummarize
```

- The first command runs the full benchmark. It takes about 30 minutes,
  mostly from GDELT pacing.
- The second re-derives the metrics from stored outcomes, with no network
  access.

Credentials are read from the environment and never printed.

## 2. Results (run of 11 Oct 2026)

The run produced **69 ground-truth events**:

| Kind | Events |
|---|---|
| nse_filing | 28 |
| fed | 11 |
| us_policy_federal_register | 8 |
| rbi | 6 |
| crude | 4 |
| gold | 4 |
| currency | 4 |
| us_policy_whitehouse | 3 |
| sebi | 1 |

- **BSE:** reachable, but no material filing for the 20 names in the 3-day
  window.
- **PIB:** 20 undated items, so `UNAVAILABLE_NO_TIMESTAMPS`.

### All events

| Provider | Tested | Detected | Recall (95% CI) | Delay p10 / p50 / p90 (min, n) | Actionable | Exact-time share | StockLens mapping | Rate-limited | Untestable |
|---|---|---|---|---|---|---|---|---|---|
| google_news | 69 | 57 | **0.83** (0.72–0.90) | 20 / **47** / 1128 (n=15) | 0.87 (n=15) | 0.22 | 0.77 (n=22) | 0 | 0 |
| gdelt | 24 | 9 | 0.38 (0.21–0.57) | 381 / 719 / 1209 (n=4) | 0.50 | 1.00 | 1.00 (n=1) | 14 events (63 of 88 requests) | 30 no history |
| publisher_rss | 10 | 2 | 0.20 (0.06–0.51) | 526 (n=2) | 1.00 | 1.00 | - | 0 | 59 no history (live feed) |
| marketaux | 0 | - | - | not measurable | - | - | - | - | **69 not configured** |
| alphavantage | 0 | - | - | not measurable | - | - | - | - | **69 not configured** |
| newsapi | 0 | - | - | not measurable | - | - | - | - | **69 not configured** |

### Indian events only

Indian events are NSE filings, RBI and SEBI.
- **google_news:** 27/35, recall 0.77 (0.61–0.88). Median delay 45 min
  (n=12). 83% actionable.
- **gdelt:** 2/9.
- **publisher_rss:** 2/8.

### Value over existing sources

| Provider | Lag behind StockLens's official feeds, p50 | Other events: detected / tested | Timed | Actionable |
|---|---|---|---|---|
| google_news | 38.9 min | 39 / 48 | 7 | 5 (all NSE filings) |
| gdelt | 650.0 min (n=3) | 6 / 15 | 1 | 0 |
| publisher_rss | 525.6 min | 0 / 2 | 0 | 0 |

### Reading the numbers

- **Google News** finds most events, about 40–47 minutes after the official
  source.
  - On RBI, the Fed and the White House it is slower than the official feeds
    StockLens already reads.
  - All 5 of its actionable new detections are NSE filings, which it reports
    after NSE disseminates them (Trent +22, Tata Steel +19, TCS +42,
    Infosys +79, Bajaj Finance +524 min).
  - **42 of its 57 detections have no timing evidence:**
    - 24 are commodity, currency, Federal Register and rate-decision events,
      whose ground truth is date-only;
    - 18 are NSE filings, Fed and White House items whose first Google
      article has only a placeholder time (00:00, 07:00 or 08:00 UTC is
      counted as date-only). This mostly affects older articles.

    Hence the low exact-time share and n=15.
- **GDELT:** 72% of requests were rate-limited from this network. Where
  measurable, it is 6–12 hours late. Events older than 88 days are
  untestable.
- **Publisher RSS** is a live feed with no history. It is slow and sparse on
  the events in the window.
- **Marketaux, Alpha Vantage and NewsAPI are untestable**, not failed: no
  credential is configured. Their adapters are implemented and tested
  offline.

### Residual quality issues

These are visible in the report: `false_match_examples` and
`wrong_mappings`.

- **Generic "Press Release" filings** (for example Bharti Airtel or Sun Pharma)
  can still be matched to a story about the same company published a day or
  more later. The 1530- and 2156-minute delays come from this. They lower
  actionability, they don't inflate it.
- **Entity mapping:** roundups tag several other companies. StockLens mapping
  accuracy is 0.77 on Google headlines; the wrong tags are listed.
- **Duplicates:** 0 URL duplicates. Each detected event creates 13 canonical
  StockLens events on average from Google items. This is syndication and
  follow-up stories, so clustering for this source needs further tightening
  before any production use.

## 3. Measurement defects found and fixed during the benchmark

| Defect | Effect | Fix |
|---|---|---|
| RBI RSS pubDate has no zone (`Sat, 10 Oct 2026 01:25:00` is IST) and was read as UTC | Every RBI item was 5.5 h late, in ingestion **and** the benchmark. Provider delays looked like −300 min | `NAIVE_TZ_DOMAINS` / `naive_tz_for`; test `test_rbi_zone_less_pub_dates_are_ist_not_utc` |
| Pre-release articles and excerpt-only roundups counted as detections | Inflated recall; negative delays | Headline subject plus 30-min pre-release rule; test `test_pre_release_articles_and_excerpt_only_mentions_are_not_detections` |
| BSE rows unfiltered (the "event" was an earnings-call audio recording) | Non-material ground truth | Material and not-material subject filters |
| Other countries' rupees matched USD/INR moves | False detections | Exclusion of Pakistani, Sri Lankan, Nepalese and similar rupees |
| SEBI date-only pubDates; PIB 403 for a user agent containing a URL; PIB redirect to Hindi; PIB undated | SEBI and PIB items silently dropped | Fixed in the previous commit (`b9d0cb4`) |

Before these fixes Google News showed a p50 of −66 min, which looked
impossibly early. After them it is +47 min.

## 4. Configuration (none set in this run)

| Variable | Purpose |
|---|---|
| `MARKETAUX_API_TOKEN` | Marketaux (free plan: about 100 requests a day) |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage NEWS_SENTIMENT (free: 25 requests a day) |
| `NEWSAPI_KEY` | **A new key only.** The exposed key must be revoked and is never used |
| `NEWS_ALLOW_RESEARCH_PROVIDERS` | Required before any research-only provider can be selected by the pipeline. Not needed for the benchmark |
| `NEWS_BENCHMARK_REPORT` | Optional path of the report served by the admin endpoint |

## 5. Licensing

| Provider | Restriction |
|---|---|
| Google News RSS | Not licensed for commercial use or redistribution: research only |
| GDELT | Open with attribution. Unreliable from shared IPs (rate limits) |
| Publisher RSS | Generally personal or non-commercial use |
| NewsAPI | Developer plan is non-commercial and delayed; business plan paid |
| Marketaux, Alpha Vantage | Plan terms not reviewed; free tiers have tight daily limits |
| NSE announcements | Website terms for automated use not reviewed; a licensed feed is available from NSE (quote needed) |
| Official RSS, Federal Register | Official public releases; US government works are public domain |

## 6. Recommendation (measured evidence)

**No provider is recommended.**

1. **Google News** meets every measured criterion except licence. Its whole
   timing advantage, however, comes from NSE filings that it reports
   about 20–80 minutes after NSE itself. For Indian company events the
   faster source is NSE. The decision that would actually improve detection
   before the market move is **licensing NSE corporate announcements**, not a
   news aggregator.
2. **Marketaux and Alpha Vantage cannot be judged yet.**
   - Sign up for the free keys, set them locally and rerun the benchmark.
   - Their free daily limits (100 and 25) allow roughly one rerun a day.
   - Choose a paid tier only if they pass the bar above, in particular the
     "events StockLens doesn't already get" criterion.
3. **GDELT and publisher RSS** don't meet the bar: they are late, sparse and
   rate-limited.
4. **Commodity, currency and US tariff events need exact-time ground truth**
   (for example, intraday price-shock times) before any provider's timing on
   them can be judged. This is the main gap in the benchmark.
