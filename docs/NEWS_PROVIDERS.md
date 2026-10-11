# News providers: audit, evaluation and recommendation (11 Oct 2026)

**Scope:** what each news source actually delivers for Indian-equity
prediction. That includes international and domestic news, US policy and
tariffs, geopolitics, economic releases, commodities, company announcements
and filings. It also covers whether a source can support leakage-free
backtests.

**Method:** every statement comes from a request made from this
environment, or is marked unverified.
- Live probes: `catalysts.registry.health`, and the probe script used for
  this audit.
- Coverage validation against real, dated events:
  `scripts/research/news_coverage_validation.py` →
  `scripts/research/output/prediction_v2/news_coverage_validation*.json`.
- Reviewer labels: `news_coverage_audit.json`.

No subscription was bought, no account was created, and no production
setting was changed.

## 1. Audit of the existing integrations

| Integration | Result | Exact cause / evidence |
|---|---|---|
| **RBI** press releases and notifications (RSS) | **Working** | 200; exact `pubDate` with offset. The feed declares UTF-8 but contains Windows-1252 bytes; decoded correctly since `9401771` |
| **SEBI** (RSS) | **Working** | 200; 30 items with exact times |
| **US Federal Reserve** (RSS, all press releases) | **Working** | 200; exact times |
| **NewsAPI** | **Integration works; the plan is unsuitable; key compromised** | The exposed key is **still active** (HTTP 200, 11 Oct) and is in the current tracked `.streamlit/secrets.toml` on public `main` (§6). Free (developer) plan: requests older than 31 days are refused ("Your plan permits you to request articles as far back as 2026-09-10"). India top-headlines returns 0 articles. For the Trent Q2 update (NSE dissemination 5 Oct 18:42:54 IST), the first relevant NewsAPI article appeared on 7 Oct at 06:43 IST, about 36 h later |
| **GDELT** DOC API | **Intermittent** | Most requests answer HTTP 429 or "Please limit requests to one every 5 seconds", even when spaced 6–20 s apart. One probe succeeded. This machine's public IP (152.59.53.62) is in a mobile carrier's shared address range. The most likely cause is GDELT's per-IP rate limit on an address shared with other users; inferred, not confirmed with GDELT. In the validation, 5 of 8 GDELT queries were rate-limited |

## 2. Sources evaluated

All of these were probed live, except where marked.

| Source | Coverage | Timestamps | History (point-in-time?) | Limits | Licence | Cost | Measured |
|---|---|---|---|---|---|---|---|
| **NSE corporate announcements** (website JSON API) | **every NSE filing**: results, business updates, orders, board outcomes, credit ratings, … | **exchange dissemination time to the second** (`exchdisstime`), plus filing time | **Yes**: queries by date range and symbol, verified back to Jan 2024. The dissemination time *is* the public availability time | none stated; anti-bot behaviour seen on BSE | **website terms for automated use / redistribution not confirmed** (the terms page was not retrievable). NSE sells licensed data | website free; licensed feed: quote needed | 1,592 filings for 1–5 Oct 2026; Trent's update at 18:42:54 IST, 5 Oct |
| BSE announcements API | BSE filings | `DT_TM` | by date | **403 on the second request** (anti-bot) | same question as NSE | — | unreliable from here |
| **White House** news and presidential-actions RSS | Trump administration announcements, executive orders, proclamations (tariffs, sanctions) | exact | no (about 30 recent items) | none | public government works | free | 200; exact times |
| **Federal Register** API | official US executive documents, tariff and sanction rules | **date only** | **Yes** (since 1994) | none | public domain | free | Tariff news appears a **median 5.5 days before** FR publication (sanctions: 2 days). It is a **late** availability source; usable in backtests only as a conservative date |
| **PIB** (Government of India) RSS | ministries, budget, policy | exact | no | none | public | free | 200 (redirect followed) |
| USTR press RSS | US trade representative | — | — | — | — | — | **404**: no RSS at the documented path |
| **Google News RSS** (search) | broad international and Indian news, named publishers | **mixed**: recent items have exact times; older items carry placeholder times (48 of 49 items from Feb 2025 stamped 08:00:00) | historical search works, but results are **capped at about 100 per query**, ranked by Google, with no first-seen time. **Not leakage-free** | undocumented | Google News terms: not for commercial redistribution | free | see §3 |
| Economic Times / Mint / Business Standard RSS | Indian market news | exact | about 2 days only | none | publisher terms (personal use) | free | 200 |
| Moneycontrol RSS | — | — | — | — | — | — | **dead**: newest item April 2024 |
| **Alpha Vantage** NEWS_SENTIMENT | claims global news with tickers and topics | `YYYYMMDDTHHMMSS` (zone not stated) | claimed `time_from` / `time_to`; **depth not verified** | free key reported as 25 requests a day (**unverified**) | terms not reviewed | free key needs **signup** | the demo key returns AAPL only; an Indian ticker returns "demo key is for demo purposes only". **Indian coverage cannot be verified without a free key** |
| FRED / ALFRED | US rates, yields, macro series; ALFRED holds vintages (point-in-time) | daily | **Yes** | none (keyless CSV) | public | free | DGS10, DFEDTARU downloaded |
| Not tested (keys or contracts needed) | Marketaux, NewsData.io, Finnhub, EODHD news, Benzinga, licensed wires (Reuters, Bloomberg, PTI), broker ratings, consensus estimates | — | — | — | — | — | marketing claims not relied on |

## 3. Coverage validation on real, dated events

**Ground truth** comes from authoritative records, never typed in by hand:
- **corporate:** NSE filings with dissemination times;
- **tariffs and sanctions:** Federal Register presidential documents;
- **Fed decisions:** FRED DFEDTARU changes;
- **crude-oil shocks:** Brent ≥ 2 sd moves;
- **US yields:** 10-year changes ≥ 12 bp.

**Google News** (the only source with usable history from here):

| Event set | Events | Found | Timestamp quality | Notes |
|---|---|---|---|---|
| Indian corporate filings (aliases, results wording, previews excluded) | 26 | **15 genuinely covered (58%)**; the automatic match said 88% | exact times for recent items only | recent window: 4 of 8 covered, delays **18.6 / 19.2 / 21.6 / 600 min** after NSE dissemination (median about 20 min). 2025 window: 11 of 18 covered, 2 with exact times. Labels: `news_coverage_audit.json` |
| US tariff actions | 10 | 6 | 64% date-only | news precedes FR publication by a median 5.5 days |
| US sanctions actions | 10 | 5 | date-only | news precedes FR publication by 2 days |
| Fed rate decisions (since Oct 2023) | 7 | 7 | date-only | generic query; the result cap was hit 6 times |
| Brent shocks | 15 | 15 | date-only | cap hit 11 times |
| US 10-year moves ≥ 12 bp | 15 | 15 | date-only | — |

**Duplicates:**
- The pipeline originally produced about 53 canonical events per real event,
  because publishers word the same story differently.
- Fixed in `01018a6`, with two rules:
  - **entity clustering:** same subject company, or the same commodity,
    institution or currency, in the same category within 24 h, forms one
    event;
  - **never merge different subject companies**, which the old title rule
    did.
- On the noisy automatic match sets, events per real event fell to about
  15–27; the automatic matches include unrelated stories, so this is an
  upper bound.

**Missed events:**
- **Google News:** misses about 40% of company filings, judged from
  headlines.
- **Official feeds:** cannot be tested historically; they only show recent
  items.
- **NewsAPI:** missed the Trent update for about 36 h.

## 4. Historical archives for leakage-free backtesting

| Source | Genuine point-in-time archive? |
|---|---|
| NSE announcements | **Yes**: dissemination times to the second, by date and symbol (licence review needed) |
| ALFRED (FRED vintages) | **Yes** for macro releases (values as first published) |
| Federal Register | Yes, but **date-only and late** (news leads by days). Conservative availability only; it under-states what was known |
| GDELT | Would be (first-seen `seendate`), but unreliable from this network |
| Google News | **No**: ranked, capped, no first-seen time, placeholder timestamps |
| NewsAPI free | No: 31 days, `publishedAt` only |
| Alpha Vantage | Not verified |
| Official RSS, publisher RSS | No: recent items only |

**Collecting from today onward is not historical evidence.** It builds an
archive for the future; it says nothing about the past.

## 5. Implementation (review branch `review/stocklens-complete`)

- **Provider-independent schema.**
  - `catalysts.providers.RawArticle` carries `time_precision`,
    provider-supplied `symbols` and the publisher's `source_domain`.
  - Pipeline, classification and predictions never depend on the provider.
  - Date-only sources set `published_at` to the end of the day, so nothing
    looks available early.
- **`catalysts/registry.py`:**
  - one spec per provider: coverage, archive, precision, quota, licence,
    cost;
  - `NEWS_PROVIDERS` selection. Research-only providers are refused unless
    `NEWS_ALLOW_RESEARCH_PROVIDERS=true`, and keyed providers without their
    credential are refused, with the reason reported;
  - daily budget check against quotas;
  - per-provider rate-limited HTTP with a per-run cap;
  - live `health()`.
- **`catalysts/adapters.py`:** Federal Register, NSE announcements, Google
  News, publisher RSS, Alpha Vantage.
- **Official RSS** adds PIB and the White House (news, presidential
  actions). RBI, SEBI and the Fed are unchanged.
- **Pipeline:**
  - provider symbols become the SUBJECT of a filing;
  - the publisher domain sets the credibility tier;
  - timestamp precision is recorded;
  - the taxonomy knows exchange-filing wording.
- **Admin API:** `GET /admin/news/providers` (no network, no secrets) and
  `POST /admin/news/providers/{name}/health` (live, audited).
- **Tests:** `tests/test_news_providers.py`, `tests/test_news_api.py`,
  `tests/test_catalysts.py`.

## 6. NewsAPI credential (owner action)

- **The exposed key is not revoked.** The owner cannot revoke it through
  the account; follow `docs/SECRET_REMEDIATION.md` §3 (account recovery,
  NewsAPI support, account deletion).
- **StockLens does not use NewsAPI.** It is disabled by default and needs
  `NEWSAPI_ENABLED=true` **and** a NEW key in `NEWSAPI_KEY`; the exposed key
  is refused by fingerprint. Nothing reads `.streamlit/secrets.toml`, which
  is untracked on the review branch.
- **History rewrite:** prepared, not executed; separate approval
  (`docs/SECRET_REMEDIATION.md` §4).

## 7. Recommendation

> **NSE terms (11 Oct 2026):** automated collection is prohibited without NSE's written permission;
> every NSE source is now gated (`docs/NEWS_PHASE2.md` §4).
>
> **Superseded by measured evidence:** see `docs/NEWS_PROVIDER_BENCHMARK.md` and `docs/NEWS_PHASE2.md`
> (69 verified events, 11 Oct 2026). No provider meets the bar. The RBI feed's
> zone-less times were found to be IST and are now read as such.

1. **Enable now, on approval of the scheduler:** `official_rss`, which is
   RBI, SEBI, Fed, PIB and the White House. Free, primary sources with
   exact times and no licence concern. It covers Indian monetary,
   regulatory and government policy, US monetary policy and US
   presidential actions (tariffs, sanctions).
2. **Indian company filings are the biggest gap, and NSE is the right
   source.**
   - It is complete, timely by definition, and has a genuine point-in-time
     archive.
   - Decide the licence first: either confirm that NSE's website terms
     allow this automated, internal use, or obtain NSE's licensed
     corporate-announcements feed (cost: quote needed).
   - Until then it stays research-only.
3. **Do not buy a general news API yet.**
   - First sign up (free) for Alpha Vantage and one alternative, for
     example Marketaux or NewsData.io.
   - Set the keys locally and rerun `news_coverage_validation.py` with
     these providers added, against the same dated events.
   - Choose only on measured coverage of Indian events, timestamp
     precision and history depth.
   - Google News is useful for coverage research only: no licence, and not
     point-in-time.
4. **GDELT:** if global first-seen history matters, run the archive pull
   from a non-shared IP, or use GDELT's BigQuery export (Google Cloud cost:
   approval needed).
5. **Measurable improvement:** the next experiment with real evidence is a
   **filings-driven replay**: NSE dissemination times, 2024–2026, the v2
   universe, through `catalysts/replay.py`, against the existing baselines.
   It needs research-only use of the NSE data, which is the licence
   question in step 2.
