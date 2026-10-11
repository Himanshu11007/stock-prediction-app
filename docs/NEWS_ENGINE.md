# News and catalyst engine (`catalysts/`, Prediction Engine v2, shadow)

News is the primary new input of Prediction Engine v2. Technical, fundamental
and ranking data are supporting evidence. Everything here is **shadow
mode**: stored and evaluated internally, shown to administrators only, and
never a trading signal.

## 1. Is news actually being received?

| Question | Answer (11 Oct 2026) | Evidence |
|---|---|---|
| Code exists | Yes: providers, pipeline, assessment, API, UI, tests | `catalysts/`, `tests/test_catalysts.py` (32), `test_news_api.py` (6), `test_news_replay.py` (3) |
| Provider works against the real source | **Yes, official RSS.** A read-only live run on 10 Oct fetched **29 articles → 24 canonical events**, with no errors, correct publication times and no encoding errors after a fix | live run into a throwaway database (section 7); defects it found were fixed in `9401771` |
| Ingestion runs automatically | **No, not yet.** The hourly `NEWS_INGESTION` job and workflow exist and are tested, but the scheduler is **inactive** (requires approval: `docs/SCHEDULER.md`) | `.github/workflows/stocklens-news-ingestion.yml`, `scheduling/jobs.py news_ingestion_job` |
| Production database has news | **No.** The migration `8d7b9e6c90dd` has not been applied in production | — |

So StockLens can receive news, verified against the real feeds. It does not
receive news today: nothing runs it in production until the scheduler and
the migration are approved.

## 2. Providers and coverage

| Provider | Covers | Auth / cost | Timestamps | Status |
|---|---|---|---|---|
| **RBI** press releases and notifications (RSS) | Indian monetary policy, liquidity, FX, regulation | none, free; "All rights reserved", so titles, links and short excerpts only | `pubDate` (IST offset) | **Working** (live run) |
| **SEBI** (RSS) | Indian market regulation | none, free | `pubDate` | **Working** |
| **US Federal Reserve** (RSS, all press releases) | FOMC decisions, US monetary policy | none, free | `pubDate` | **Working** |
| NewsAPI (`newsapi.org`) | international and Indian news search | needs `NEWSAPI_KEY`; the key sits in an HTTP header and is never in URLs or logs. The free plan is delayed and non-commercial; paid plans cost money | `publishedAt` | **Not configured.** The old key is in public Git history and must be **revoked**; a new key requires your decision (and cost if commercial) |
| GDELT DOC 2.0 | global news, history up to about 3 months, first-seen times | none, free; one request per 5 s | `seendate` | **Refused from this network**: every request is rate-limited, even when spaced out. Research only |
| GDELT raw 15-minute files | global events (coded actors, no headlines) and the knowledge graph | free | `DATEADDED` | reachable, but events lack company or headline information, and the knowledge graph is about 4.4 MB per 15 minutes. Not used |
| NSE / BSE corporate announcements | filings, results, orders, corporate actions | automated access is restricted by the exchanges' terms; the NSE feed timed out (anti-bot) | — | **BLOCKED**: needs a licensed feed or an exchange data agreement |
| Broker ratings, consensus estimates, licensed wires (Reuters, Bloomberg, PTI) | re-ratings, expectations, real-time international news | paid contracts | — | **BLOCKED**: needs a decision and a budget |
| Intraday prices | TODAY_CONFIRMED | licensed feed | — | **BLOCKED** (F-10) |

**Coverage today:**
- **Covered:** Indian monetary and regulatory announcements, and US Federal
  Reserve policy, from primary sources.
- **Not covered:**
  - company filings and earnings;
  - broker actions;
  - tariffs and US administration announcements (the US Trade
    Representative's feed returned 404);
  - geopolitics, commodities news, and Asian or US market news.

  These need NewsAPI or a licensed provider. Price moves of Brent, gold,
  USD/INR and US yields are available from Yahoo and used in research
  (section 6).

**Configuration** (on the API service; production change: approval):
- `NEWS_PROVIDERS=rss` (default) or `rss,newsapi`;
- `NEWSAPI_KEY` (secret store only);
- `PREDICTION_V2_NEWS_ENABLED` (default `true`; with no events it changes
  nothing).

## 3. Pipeline (`catalysts/pipeline.py`)

| Capability | Implementation | Tests |
|---|---|---|
| Untrusted text | sanitised (markup, scripts and control characters removed), length-capped, and only matched against fixed vocabularies. Never executed, never sent to a model, never used in queries or prompts | `test_sanitize_*`, `test_instruction_like_text_is_only_data` |
| Duplicates and syndication | same provider id or same canonical URL → skipped; similar titles (Jaccard ≥ 0.5) within 48 h, or the same single subject company and category within 6 h → same canonical event | `test_duplicates_syndication_and_clustering` |
| Canonical event id | `market_events` row, source `news`, `news:<fingerprint>:<first publication minute>` | same |
| Source credibility | PRIMARY / TIER1 / TIER2 / TIER3 by domain (`catalysts/sources.py`); `is_primary_source` | `test_source_tiers_*` |
| Timestamps | `published_at`, `ingested_at`, `corrected_at`, `effective_available_at = max(published, ingested)`. A later, earlier-published copy **never moves availability earlier**. Classifications and entity links carry the ingestion clock | `test_late_ingestion_is_never_backdated` |
| Classification | 17 categories by priority (`catalysts/taxonomy.py`): broker rating, business update, earnings, guidance, orders, corporate action, fundraising, management, litigation, monetary policy, trade and tariffs, geopolitics, fiscal policy, regulation, economic data, flows and currency, commodity. Plus sentiment with negation, expectedness (only when the text says so), materiality, novelty, impact horizon and expiry | `test_taxonomy`, `test_sentiment_*` |
| Entities | companies by full name, name without suffix, or capitalised ticker; sectors, countries (US, Fed and IT are case-sensitive), currencies, commodities, institutions, indices | `test_entities_*` |
| Contradiction | opposite-sentiment articles in one event → `contradicted`, direction MIXED; across events, opposing credible evidence → `CONFLICTING_REPORTS` | `test_non_english_future_dated_and_contradiction`, replay tests |
| Language | non-English articles are stored for provenance but not used | same |
| Failures | bounded retries; RATE_LIMITED, NOT_CONFIGURED, FAILED and PARTIAL recorded per run; malformed feeds and mis-declared encodings handled | `test_rss_*`, `test_provider_states_are_recorded` |
| Monitoring | `SNAPSHOT_MONITOR` alerts when no successful ingestion in 3 h; the daily quality report shows ingestion latency and provider statuses | `test_monitor_flags_stale_news_*`, `test_prediction_feedback` |

## 4. News-to-market layer

**Facts and inference are kept apart:**
- **Facts:** `news_articles` rows and factual `event_entities`.
- **Inference:** `event_entities` rows with `inferred = true`, which carry
  a hypothesis id, sign, mechanism and validation status.

**Transmission hypotheses** (`catalysts/transmission.py`, `transmission-0.1`)
are explicit and versioned:
- **Crude oil:** oil marketing companies −; upstream +; paints −;
  airlines −; tyres −; standalone refiners ambiguous (0).
- **Rupee:** a weaker rupee → IT +, OMCs −.
- **US yields:** higher yields → banks and NBFCs −, real estate −.
- **Gold:** gold lenders +; jewellers ambiguous.
- **RBI easing:** real estate, autos and NBFCs +; banks ambiguous.
- **US tariffs on India:** textiles, apparel, auto parts, generic pharma and
  seafood −.
- **Public capex:** engineering, cement and capital goods +.
- **Peer read-through** for company news.

The direction of the driver is read from explicit wording. If the text does
not say which way the driver moved, no exposure is claimed.

**Per-stock assessment** (`catalysts/impact.py`, rule `news-v0.1`) answers
the ten questions for each piece of evidence:
1. **What happened:** title, category and sources.
2. **What changed vs expectations:** stated expectedness (UNEXPECTED,
   EXPECTED or UNKNOWN). Historical consensus is BLOCKED.
3. **When it was available:** effective availability.
4. **Who is exposed:** subject, mention, hypothesis or peer.
5. **Mechanism:** from the hypothesis.
6. **Sign:** positive, negative or ambiguous.
7. **Horizon:** the category's impact sessions and expiry.
8. **Priced in:** a move of at least 1.5 ATR in the evidence's direction
   since availability → weight × 0.25.
9. **Contradiction:** contradicted or conflicting reports → no direction.
10. **Historical support:** the hypothesis' validation status (section 6).

A direction is given only for decisive, credible (PRIMARY, TIER1 or TIER2),
uncontradicted evidence. Otherwise the result is NEUTRAL with the reason,
and bad price data is always NO_CALL.

**In snapshots,** news is primary:
- a decisive assessment gives a `NEWS_CATALYST` call, with the technical
  setup recorded as agreeing or disagreeing;
- credible opposing news neutralises a technical call (`NEWS_CONFLICT`).

Every prediction freezes the evidence, event ids, `news_sentiment`,
`surprise`, `novelty`, `confidence: null` ("not calibrated") and
`data_completeness`.

**Snapshot timing:**
- `TOMORROW_EOD` (16:15 IST) uses news available by its cutoff.
- News published after it, and overnight global news, is used by the next
  `TODAY_PREOPEN` (08:30 IST); hourly ingestion runs every day, including
  weekends and holidays.
- The target is the next **trading** session (`prediction_v2.calendar`,
  holiday list).

## 5. APIs and UI

- **Shadow API, admin-only:**
  - `GET /news/events`, `/news/events/{id}`, `/news/market-summary`,
    `/news/hypotheses`;
  - `GET /admin/news/ingestion-runs`, `POST /admin/news/ingest` (audited);
  - `GET /admin/prediction-quality`;
  - `/stocks/{symbol}/events` gains `linked_news_events`;
  - `/predictions/{id}` gains `current_price`.
- **Web Labs:** the prediction detail explains the call (facts vs
  inference, sources, times, flags, separate prices). `/predictions/catalysts`
  is the daily market-intelligence summary.

## 6. Historical evidence

| Study | Method | Result | Status |
|---|---|---|---|
| Transmission validation (`scripts/research/transmission_validation.py` → `catalysts/transmission_validation.json`) | 3 years; basket excess return vs factor return; rule fixed in advance: same-day sign, \|t\| ≥ 2 in the first 2/3 and the same sign in the last 1/3 | **SUPPORTED:** crude → OMCs, crude → upstream, gold → gold lenders. **INCONCLUSIVE:** crude → paints / airlines / tyres (strong next-day effects, recorded as "predictive"), rupee → IT and OMCs, US yields | IMPLEMENTED AND TESTED |
| Factor-shock replay (`scripts/research/factor_shock_replay.py` → `output/prediction_v2/factor_shock_replay.json`) | shocks ≥ 2 sd of the prior 60 bars, known before the open; trade open → close; 15 / 25 bps; 60/20/20; random and gap-following baselines; date-clustered intervals | pooled **holdout: 606 calls, 51.8% hit, +0.02% net at 25 bps** (clustered 95% CI −0.45% to +0.57%). Every hypothesis' full-sample clustered CI includes 0. The expected move is largely **in the overnight gap** (crude → OMCs: +0.7% in the call direction at the open), so it is absorbed before it can be traded. The US-yields holdout (+0.43%) rests on 7 shock days and is experimental | IMPLEMENTED AND TESTED; **no tradable edge** |
| News replay (`catalysts/replay.py`, `scripts/research/news_replay.py`) | replays an archive in ingestion order; TOMORROW_EOD close → next close; NIFTY excess; costs; baselines; unscored, no-news and no-data cases counted separately | **No historical archive with original availability times is accessible** (GDELT refused, licensed sources absent). The archive StockLens collects itself: 29 articles (10 Oct) → `INSUFFICIENT_SAMPLE` | engine IMPLEMENTED AND TESTED; evaluation **BLOCKED** until the archive has at least 20 sessions and 50 scored calls |
| Event categories (tariffs, earnings vs expectations, broker re-ratings, RBI) | needs dated event history with expectations | — | **BLOCKED** (data) |

**Conclusion:**
- There is **no evidence yet that news improves next-session predictions.**
- The macro transmission channels are real in prices: SUPPORTED for crude
  and gold. But at the daily level they are priced in at the open.
- Company-news value cannot be measured until a point-in-time archive
  exists. That requires approved, scheduled ingestion, or a licensed
  provider with history.
- v2 stays in shadow mode and nothing is promoted.

## 7. Operations

- **Start collecting** (approval):
  1. Apply migration `8d7b9e6c90dd`.
  2. Set `NEWS_PROVIDERS`.
  3. Enable the scheduler (`docs/SCHEDULER.md`).

  The archive builds from that moment; each week of collection adds about
  5 sessions to the replay.
- **Recovery:**
  - `POST /admin/news/ingest`, or `python scripts/scheduled_jobs.py news_ingestion`.
  - Each provider resumes from its last successful window (at most 3 days
    back).
- **Live check without production** (read-only, throwaway SQLite): run
  `pipeline.ingest(session, RssProvider(), now - 14 days, now)` on an empty
  database seeded with companies. This is how the 10 Oct verification was
  done.

## 8. Limitations

- **Sentiment and categories:** lexicon and keyword rules
  (`news-rules-0.1`), not a trained model. Precision and recall on real
  news are not measured; that needs a labelled sample.
- **Entity extraction** is dictionary-based. Companies named differently
  in news (abbreviations, group names) can be missed.
- **Materiality, tiers and thresholds** are a priori hypotheses.
- **"Expected" or "unexpected"** comes only from the text, since there is
  no consensus data.
- **Excerpts only;** no full article text is stored (licensing).
