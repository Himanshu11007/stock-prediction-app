"""
tests/test_gap_report.py — "Missed gainers": each day's top gainers that were
not in the list published before the open are explained (excluded and why, or
the components that held them back vs the list average, and news before /
after publication); false alarms; totals; admin only; read-only.
"""
import datetime as dt

from sqlmodel import Session, select

from analytics.gap_report import gap_report
from db.models.market import EngineRun, RankingSnapshot, StockAnalysisResult
from tests.test_notifications import ADMIN, H, _limit, client, db  # noqa: F401  (fixtures)
from tests.test_stock_labels import _news

UTC = dt.timezone.utc
SYMS = [f"S{i:02d}.NS" for i in range(30)]
BASE = {"quality": 70.0, "valuation": 60.0, "financial_health": 60.0, "technical_trend": 50.0, "momentum": 50.0,
        "risk": 50.0, "market_regime": 50.0, "sector_outlook": None}
COMPS = {**{s: {**BASE, "quality": 90.0, "momentum": 90.0} for s in SYMS[:5]},          # the Top 5 list
         "S10.NS": {**BASE, "momentum": 5.0}, "S11.NS": {**BASE, "valuation": 0.0}}
PRICES_A = {s: 100.0 for s in SYMS}
PRICES_B = {**PRICES_A, "S00.NS": 110.0, "S01.NS": 109.0, "S03.NS": 96.0,
            **{f"S{10 + i}.NS": 108.0 - 0.5 * i for i in range(8)}}                    # S10 +8% ... S17 +4.5%


def _rank_run(s, run_id, ref_date, finished, prices, ineligible=()):
    s.add(EngineRun(run_id=run_id, kind="RANKING", status="COMPLETED", started_at=finished - dt.timedelta(minutes=20),
                    finished_at=finished, engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0", config={}))
    s.flush()
    rank = 0
    for i, sym in enumerate(SYMS):
        ok = sym not in ineligible
        rank += ok
        s.add(RankingSnapshot(run_id=run_id, symbol=sym, ranked_at=finished, engine_version="ranking-v1.0",
                              fqvf_version="fqvf-v1.0", stockai_score=90.0 - i, eligible=ok, rank=rank if ok else None,
                              component_scores=COMPS.get(sym, BASE), reference_date=ref_date,
                              reference_price=prices[sym]))
        if not ok:
            s.add(StockAnalysisResult(run_id=run_id, symbol=sym, computed_at=finished, stockai_score=90.0 - i,
                                      eligible=False, ineligible_reasons=["20-day average volume below 100,000 (liquidity)"],
                                      engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0"))
    s.commit()


def _seed(s):
    _limit(s, 5)
    published = dt.datetime(2026, 10, 2, 11, 30, tzinfo=UTC)                              # 17:00 IST, Friday
    _rank_run(s, "A", "2026-10-02", published, PRICES_A, ineligible=("S12.NS",))
    _rank_run(s, "B", "2026-10-05", dt.datetime(2026, 10, 5, 11, 30, tzinfo=UTC), PRICES_B)
    _news(s, "S13.NS", "Stock 13 order win", 0.6, published - dt.timedelta(hours=2), ext=31)      # before the list
    _news(s, "S14.NS", "Stock 14 results", 0.6, dt.datetime(2026, 10, 5, 8, 0, tzinfo=UTC), ext=32)  # during the day


def test_missed_gainers_are_explained_and_totalled(db):
    with Session(db) as s:
        _seed(s)
        before = len(s.exec(select(RankingSnapshot)).all())
        rep = gap_report(s, now=dt.datetime(2026, 10, 6, tzinfo=UTC))
        assert len(s.exec(select(RankingSnapshot)).all()) == before                         # read-only
    (day,) = rep["days"]
    assert (day["date"], day["list_date"], day["gainers"], day["hits"]) == ("2026-10-05", "2026-10-02", 10, 2)
    m = {x["symbol"]: x for x in day["misses"]}
    assert list(m) == [f"S{10 + i}.NS" for i in range(8)]
    assert m["S10.NS"]["main_gap"] == "Momentum" and m["S10.NS"]["gaps"][0] == {
        "component": "momentum", "label": "Momentum", "score": 5.0, "list_average": 90.0, "points": 8.5}
    assert m["S11.NS"]["main_gap"] == "Valuation" and m["S11.NS"]["gaps"][0]["points"] == 12.0
    assert m["S12.NS"]["excluded_reason"] == "20-day average volume below 100,000 (liquidity)"
    assert m["S12.NS"]["rank"] is None and m["S12.NS"]["gaps"] == []
    assert m["S15.NS"]["main_gap"] == "Business quality" and m["S15.NS"]["points_below_list"] == 11.0  # 86 - 75
    assert (m["S13.NS"]["news_before_list"], m["S13.NS"]["news_after_list"]) == (1, 0)
    assert (m["S14.NS"]["news_before_list"], m["S14.NS"]["news_after_list"]) == (0, 1)
    assert day["false_alarms"] == [{"symbol": "S03.NS", "name": "Stock 03", "return": -0.04, "rank": 4}]
    sm = rep["summary"]
    assert (sm["missed"], sm["excluded"], sm["with_news_before_list"], sm["news_only_after_list"], sm["no_news"]) == \
        (8, 1, 1, 1, 6)
    assert sm["main_gap"] == {"Business quality": 5, "Momentum": 1, "Valuation": 1}
    assert sm["near_misses"] == 7 and rep["message"] is None


def test_route_is_admin_only_and_explains_an_empty_report(client):
    assert client.get("/api/v1/admin/gap-report", headers=H()).status_code == 403
    data = client.get("/api/v1/admin/gap-report", headers=ADMIN()).json()["data"]
    assert data["summary"] is None and data["message"].startswith("Not enough daily rankings yet")
