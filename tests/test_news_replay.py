"""
tests/test_news_replay.py — the historical news replay never uses an article
before it was ingested, applies later contradictions from their arrival,
scores next-session outcomes against NIFTY with costs, records unscored
cases and is reproducible. Synthetic data only.
"""
import datetime as dt

import pandas as pd

from catalysts import replay
from catalysts.providers import RawArticle
from tests.test_prediction_v2 import bars, sessions, walk

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAYS = sessions(dt.date(2026, 10, 9), 90)
D1, D2, D3 = DAYS[-3], DAYS[-2], DAYS[-1]          # Wed 7, Thu 8, Fri 9 Oct 2026


def _prices():
    out = {}
    for i, sym in enumerate(("S1.NS", "S2.NS", "S3.NS")):
        closes = walk(len(DAYS), 40 + i, vol=0.004)
        df = bars(DAYS, closes, [1e6] * len(DAYS))
        out[sym] = df
    # S1 rises 3% on D2 (the session after the D1 cutoff); S2 rises 2% on D3
    out["S1.NS"].loc[pd.Timestamp(D2), "Close"] = out["S1.NS"].loc[pd.Timestamp(D1), "Close"] * 1.03
    out["S2.NS"].loc[pd.Timestamp(D3), "Close"] = out["S2.NS"].loc[pd.Timestamp(D2), "Close"] * 1.02
    return out


NIFTY = bars(DAYS, walk(len(DAYS), 1, vol=0.003, start=20000))
COMPANIES = [("S1.NS", "Sone Steelworks Limited", "Basic Materials", "Steel"),
             ("S2.NS", "Stwo Pharma Limited", "Healthcare", "Drug Manufacturers - General"),
             ("S3.NS", "Sthree Foods Limited", "Consumer Defensive", "Packaged Foods")]


def at(day, hh, mm=0):
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=IST)


def a(i, title, published, url):
    return RawArticle(f"a{i}", url, title, published.astimezone(dt.timezone.utc))


def _archive():
    good1 = a(1, "Sone Steelworks quarterly results: net profit jumps to a record, beats estimates", at(D1, 16, 5),
              "https://www.reuters.com/sone")
    good2 = a(2, "Stwo Pharma quarterly results: profit jumps to a record, beats estimates", at(D1, 16, 10),
              "https://www.reuters.com/stwo")
    return [replay.ArchivedArticle(good1, at(D1, 16, 30)),                     # ingested before the D1 cutoff
            replay.ArchivedArticle(good2, at(D2, 9, 0))]                       # published D1, ingested late (D2)


def test_late_ingested_article_is_not_used_before_its_arrival():
    cutoffs = [at(D1, 19, 30), at(D2, 19, 30)]
    res = replay.run(_archive(), cutoffs, COMPANIES, _prices(), NIFTY)
    by = {(c.cutoff, c.symbol): c for c in res.calls}
    assert (D1, "S1.NS") in by and by[(D1, "S1.NS")].direction == "UP"
    assert by[(D1, "S1.NS")].ret > 0.02                                       # S1 +3% the next session
    assert (D1, "S2.NS") not in by                                            # S2's article arrived after the cutoff
    assert (D2, "S2.NS") in by and by[(D2, "S2.NS")].ret > 0.015              # used from its arrival on
    assert res.sessions == [D1.isoformat(), D2.isoformat()]


def test_later_contradiction_applies_only_from_its_arrival():
    contra = a(3, "Sone Steelworks quarterly results: shares plunge and slump on weak margins and losses",
               at(D2, 8, 0), "https://www.livemint.com/sone")
    arch = _archive() + [replay.ArchivedArticle(contra, at(D2, 8, 30))]
    res = replay.run(arch, [at(D1, 19, 30), at(D2, 19, 30)], COMPANIES, _prices(), NIFTY)
    s1 = {c.cutoff: c.direction for c in res.calls if c.symbol == "S1.NS"}
    assert s1[D1] == "UP"                                                     # before the negative report arrived
    assert s1.get(D2) != "UP"                 # after it: the good news is priced in and a credible negative report dominates


def test_summary_costs_baselines_unscored_and_reproducibility():
    cutoffs = [at(D1, 19, 30), at(D2, 19, 30), at(D3, 19, 30)]                # D3 has no next session
    r1 = replay.run(_archive(), cutoffs, COMPANIES, _prices(), NIFTY)
    r2 = replay.run(_archive(), cutoffs, COMPANIES, _prices(), NIFTY)
    s = r1.summary(cost_bps=25)
    assert s == r2.summary(cost_bps=25)
    assert s["scored"] == 2 and s["hit_rate"] == 1.0 and s["mean_net"] < s["mean_net"] + 0.0025
    assert s["no_news"] > 0 and s["baseline_previous_day"]["calls"] > 0
    assert s["sessions"] == 2                                                  # D3 skipped: no next session
    assert abs(r1.summary(cost_bps=0)["mean_net"] - s["mean_net"] - 0.0025) < 1e-12
