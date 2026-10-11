# Prediction evidence (status 11 Oct 2026)

This document collects in one place what the measurements show about
StockLens's predictions. **Conclusion: there is no evidence of a predictive
edge after costs.** Prediction Engine v2 stays in shadow mode, and nothing
was tuned against the holdout.

## 1. Two different questions

| Question | Measure | Where | Result |
|---|---|---|---|
| **Coverage:** does a source *report* a material event, and how fast? | Recall, delay and actionability on 69 independently verified events | `docs/NEWS_PROVIDER_BENCHMARK.md`, `docs/NEWS_PHASE2.md` §1–2 | Google News: strict recall 0.78 (95% CI 0.67–0.86), median +42 min. Marketaux free plan: 0.39 (0.28–0.51), +235 min. Alpha Vantage news: premium-only, not testable |
| **Accuracy:** when StockLens calls UP or DOWN, is the next session's return in that direction, after costs? | Hit rate and mean net return, chronologically split | §2 and §3 below | **No edge.** Negative net returns in every split |

High coverage does not imply accuracy. Google News finds most events, yet
predictions built on them do not beat baselines (§3).

## 2. Prediction Engine v2 baseline rules (market / technical)

`scripts/research/output/prediction_v2/backtest_report.json`:
- rule version `baseline-v0.1`; `prediction_v2/rules.py` has had a single
  commit since it was written, so it was never tuned on results;
- universe of 281 stocks; daily bars from 9 Oct 2023 to 9 Oct 2026;
- horizon of 1 session, close to close;
- chronological splits with an embargo.

Mean net return per call, in bps, at 20 bps round-trip cost:

| Split (dates) | v2 calls | v2 hit rate | v2 net | Random (matched) net | v1 top-20 net |
|---|---|---|---|---|---|
| Development (8 Jan 2024 – 1 Sep 2025) | 5,062 | 0.489 | −21.9 | −22.4 | −13.1 |
| Validation (10 Sep 2025 – 19 Mar 2026) | 1,683 | 0.464 | −31.6 | −23.0 | −11.5 |
| Holdout (1 Apr – 8 Oct 2026) | 1,604 | 0.473 | −21.9 | −23.0 | −19.1 |

- **At 10 bps** the v2 holdout is −11.9 bps, still negative.
- **Slippage** is not modelled separately in this report. The 20 bps column
  is the realistic total.
- **The v1 column is only a benchmark.** It is a long-only basket, not a
  directional forecast.

## 3. News-driven arms (point-in-time replay, 12 months)

`scripts/research/output/prediction_v2/news_signal_backtest.json`
(`docs/NEWS_PHASE2.md` §6):
- universe of 20 large caps; 14 Oct 2025 to 9 Oct 2026;
- **30 bps round trip** (20 bps cost + 10 bps slippage); sensitivity at 10
  and 50 bps;
- every article is usable only from the next full hour after publication,
  and date-only articles only after the end of their day IST.

| Split | News arm (Google News) | v2 | v2 + news | always-up |
|---|---|---|---|---|
| Development | 278 calls, hit 0.50, **−34 bps** (CI −58 to −9) | 122, 0.48, −9 | 377, 0.50, −26 | 2,780, 0.49, −35 |
| Validation | 86 calls (insufficient), 0.40, −76 | 27 (insufficient) | 111, 0.41, −82 | 740, 0.53, −21 |
| Holdout | 68 calls (**insufficient**, < 100), 0.37, −40 | 25 (insufficient) | 93 (insufficient), 0.34, −55 | 860, 0.40, −52 |

**Not tested:**
- StockLens's own news archive: there is no point-in-time archive yet;
- Marketaux and Alpha Vantage over 12 months: the free quotas and plans
  don't allow it;
- GDELT: rate-limited, and it keeps only about 3 months.

The Google arm is optimistic by construction: headlines can be edited
later, and results are ranked today.

## 4. Safeguards that make these numbers trustworthy

| Safeguard | Test |
|---|---|
| Features use only bars up to the cutoff | `test_features_never_use_data_after_the_cutoff` |
| Snapshots are frozen when created | `test_eod_snapshot_is_created_frozen_and_complete`, `test_detail_has_frozen_features_outcomes_and_shadow_exit`, `test_snapshot_uses_news_and_freezes_the_evidence` |
| News is used only after StockLens could have had it | `test_assessment_uses_only_what_was_available_before_the_cutoff`, `test_late_ingested_article_is_not_used_before_its_arrival` |
| A corrected article is not used before its correction | `test_corrected_article_is_not_used_before_its_correction_arrived` |
| Benchmark detections must follow the official release | `test_pre_release_articles_and_excerpt_only_mentions_are_not_detections` |
| Ranking Engine v1.0 unchanged | `test_ranking_v1_golden.py`, `test_ranking_engine_v1_is_frozen` (`ranking/presenter.py` changes are additive response fields only) |
| A result is only claimed with enough calls and an interval above the comparator | `catalysts/signal_eval.verdict` (`test_signal_eval.py`) |

## 5. Decision

- **No edge:** keep Prediction Engine v2 and news-driven calls in shadow.
  No alerts, no user-facing directional predictions.
- **Don't tune:** do not tune thresholds against these results. The holdout
  is spent; any new rule needs a new, untouched holdout.
- **Keep providers research-only:** Google News, Marketaux, Alpha Vantage,
  GDELT, publisher RSS and NSE / BSE stay research-only until their
  licences are cleared. NSE / BSE automated collection also needs the
  exchange's written permission.
- **What could change the conclusion:**
  - a genuinely point-in-time news archive (approved scheduled ingestion),
    or licensed exchange filings;
  - at least 100 holdout calls per arm;
  - a measured improvement over v2 and the baselines after costs.
