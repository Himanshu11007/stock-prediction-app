"""
tests/test_prediction_feedback.py — daily prediction-quality report: correct
/ incorrect / neutral / pending counts, missed material news, false-positive
catalysts, contradicted inferred mappings, data latency and provider runs,
and recommendations that are advisory only.
"""
import datetime as dt

from sqlmodel import Session

from db.models.news import NewsArticle, NewsIngestionRun
from db.models.prediction import Prediction, PredictionOutcome
from prediction_v2 import feedback
from tests.test_prediction_api import ADMIN, _run, client, db  # noqa: F401  (fixtures)

DAY = dt.date(2026, 10, 12)
UTC = dt.timezone.utc


def ev(title, weight, inferred=False, sign=None, hyp="", cat="EARNINGS"):
    return {"title": title, "weight": weight, "inferred": inferred, "sign": sign, "hypothesis_id": hyp,
            "category": cat}


def _pred(s, pid, sym, direction, setup, news, hit=None, net=None, excess=None):
    s.add(Prediction(prediction_id=pid, run_id="RQ", symbol=sym, direction=direction, setup_type=setup,
                     horizon_sessions=1, reference_price=100.0, reference_date="2026-10-09",
                     features={"news": news}))
    s.add(PredictionOutcome(prediction_id=pid, horizon_sessions=1, evaluator_version="outcomes-v0.1",
                            outcome_status="EVALUATED", hit=hit, cost_adjusted_return=net, excess_return_nifty=excess,
                            stock_return=excess))


def _seed(db):
    with Session(db) as s:
        _run(s, "RQ", "TOMORROW_EOD", DAY)
        s.commit()
    with Session(db) as s:
        from sqlmodel import select
        for p in s.exec(select(Prediction).where(Prediction.run_id == "RQ")).all():
            s.delete(p)
        s.commit()
        _pred(s, "q1", "AAA.NS", "UP", "NEWS_CATALYST", {"evidence": [ev("AAA beats estimates", 0.7)]},
              hit=False, net=-0.012, excess=-0.01)
        _pred(s, "q2", "BBB.NS", "DOWN", "BREAKDOWN", {}, hit=True, net=0.004, excess=-0.006)
        _pred(s, "q3", "CCC.NS", "NEUTRAL", "NONE",
              {"evidence": [ev("CCC order win", 0.3)], "reasons": ["news present but not decisive: below 0.35"]},
              excess=0.05)
        _pred(s, "q4", "DDD.NS", "NEUTRAL", "NONE",
              {"evidence": [ev("Brent surges", 0.3, inferred=True, sign=-1, hyp="H-CRUDE-OMC", cat="COMMODITY")]},
              excess=0.025)
        s.add(Prediction(prediction_id="q5", run_id="RQ", symbol="EEE.NS", direction="UP", setup_type="X",
                         horizon_sessions=1, features={}))                       # not evaluated yet
        t = dt.datetime.combine(DAY, dt.time(3, 0), tzinfo=UTC)
        s.add(NewsArticle(provider="rss", provider_article_id="n1", url="u", canonical_url="u", title="t",
                          published_at=t - dt.timedelta(minutes=90), ingested_at=t, effective_available_at=t,
                          fingerprint="f"))
        s.add(NewsIngestionRun(provider="rss", status="RATE_LIMITED", started_at=t))
        s.commit()


def test_daily_report_sections(db):
    _seed(db)
    with Session(db) as s:
        r = feedback.daily_report(s, DAY)
    o = r["outcomes"]
    assert (o["correct"], o["incorrect"], o["neutral"], o["pending"], o["directional"]) == (1, 1, 2, 1, 2)
    assert o["small_sample"] is True
    assert r["false_positive_catalysts"][0]["symbol"] == "AAA.NS"
    assert [m["symbol"] for m in r["missed_news"]] == ["CCC.NS"]                 # 5% move, material news, no call
    assert r["mapping_errors"]["H-CRUDE-OMC"]["count"] == 1                     # expected down, rose 2.5%
    assert r["by_setup"]["NEWS_CATALYST"]["hit_rate"] == 0.0
    assert r["data"]["ingestion_latency_minutes"]["over_60_min"] == 1 and r["data"]["provider_runs"] == {"RATE_LIMITED": 1}
    assert "TODAY_PREOPEN" in r["data"]["missing_snapshots"]
    text = " ".join(r["recommendations"])
    assert "not evidence" in text and "provider" in text.lower() and "research review" in r["notice"]


def test_quality_report_endpoint_is_admin_only(client, db):
    _seed(db)
    assert client.get("/api/v1/admin/prediction-quality?date=2026-10-12").status_code == 401
    d = client.get("/api/v1/admin/prediction-quality?date=2026-10-12", headers=ADMIN()).json()["data"]
    assert d["target_session"] == "2026-10-12" and d["outcomes"]["correct"] == 1
    assert client.get("/api/v1/admin/prediction-quality?date=12-10-2026", headers=ADMIN()).status_code == 422
