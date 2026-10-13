"""
tests/test_track_record.py — "Were the top gainers already in StockLens's
list?": point in time (only lists published before the day's open count),
gainers are stocks that rose, likely split adjustments are excluded, the
by-chance baseline is shown, failed runs are ignored, read-only.
"""
import datetime as dt

from sqlmodel import Session, select

from analytics.track_record import track_record
from db.models.market import EngineRun, RankingSnapshot
from tests.test_notifications import H, _limit, client, db  # noqa: F401  (fixtures)

UTC = dt.timezone.utc
SYMS = [f"S{i:02d}.NS" for i in range(30)]
BASE = {s: 100.0 for s in SYMS}
DAY1 = {**BASE, "S00.NS": 110.0, "S01.NS": 109.0, "S02.NS": 97.0, "S29.NS": 160.0,      # S29: split-like, excluded
        **{f"S{10 + i}.NS": 108.0 - 0.5 * i * 1 for i in range(8)}}                     # S10 +8% ... S17 +4.5%
DAY4 = {**DAY1, "S05.NS": DAY1["S05.NS"] * 1.05, "S03.NS": DAY1["S03.NS"] * 1.02}


def _rank_run(s, run_id, ref_date, finished, prices, status="COMPLETED"):
    s.add(EngineRun(run_id=run_id, kind="RANKING", status=status, started_at=finished - dt.timedelta(minutes=20),
                    finished_at=finished, engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0", config={}))
    for i, sym in enumerate(SYMS):
        s.add(RankingSnapshot(run_id=run_id, symbol=sym, ranked_at=finished - dt.timedelta(minutes=5),
                              engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0", stockai_score=90.0 - i,
                              eligible=True, rank=i + 1, reference_date=ref_date, reference_price=prices[sym]))
    s.commit()


def _evening(day):            # 17:00 IST, after the daily ranking
    return dt.datetime.fromisoformat(day).replace(hour=11, minute=30, tzinfo=UTC)


def _seed(s):
    _limit(s, 5)                                                       # Top 5 list: S00 ... S04
    _rank_run(s, "A", "2026-10-02", _evening("2026-10-02"), BASE)       # Fri
    _rank_run(s, "B", "2026-10-05", _evening("2026-10-05"), DAY1)       # Mon
    _rank_run(s, "C", "2026-10-06", dt.datetime(2026, 10, 7, 5, 0, tzinfo=UTC), DAY1)   # published 10:30 IST next day
    _rank_run(s, "D", "2026-10-07", _evening("2026-10-07"), DAY1)
    _rank_run(s, "E", "2026-10-08", _evening("2026-10-08"), DAY4)
    _rank_run(s, "F", "2026-10-09", _evening("2026-10-09"), DAY4)
    _rank_run(s, "X", "2026-10-12", _evening("2026-10-12"), BASE, status="FAILED")       # never counted


def test_next_day_record_is_point_in_time_with_a_chance_baseline(db):
    with Session(db) as s:
        _seed(s)
        before = len(s.exec(select(RankingSnapshot)).all())
        tr = track_record(s, now=dt.datetime(2026, 10, 13, 6, tzinfo=UTC))
        assert len(s.exec(select(RankingSnapshot)).all()) == before                   # read-only
    day = next(h for h in tr["horizons"] if h["key"] == "1D")
    assert [p["date"] for p in day["periods"]] == ["2026-10-09", "2026-10-08", "2026-10-06", "2026-10-05"]
    # 07 Oct is missing: the 06 Oct list was published after the 07 Oct open
    mon = day["periods"][-1]
    assert (mon["list_date"], mon["universe"], mon["list_count"], mon["gainers"], mon["hits"]) == \
        ("2026-10-02", 29, 5, 10, 2)
    assert mon["expected_by_chance"] == round(10 * 5 / 29, 2)
    assert [g["symbol"] for g in mon["top_gainers"]][:3] == ["S00.NS", "S01.NS", "S10.NS"]
    assert all(g["symbol"] != "S29.NS" for g in mon["top_gainers"])                  # split-like move excluded
    assert mon["top_gainers"][0] == {"symbol": "S00.NS", "name": "Stock 00", "return": 0.1, "in_list": True,
                                     "list_rank": 1}
    assert mon["list_avg_return"] == 0.032
    flat = day["periods"][2]                                                          # no stock rose on 06 Oct
    assert (flat["gainers"], flat["hits"], flat["top_gainers"]) == (0, 0, [])
    thu = day["periods"][1]
    assert [(g["symbol"], g["in_list"]) for g in thu["top_gainers"]] == [("S05.NS", False), ("S03.NS", True)]
    assert day["summary"] == {"periods": 4, "hits": 3, "possible": 12, "avg_hits": 0.75,
                              "avg_expected_by_chance": day["summary"]["avg_expected_by_chance"],
                              "list_avg_return": day["summary"]["list_avg_return"],
                              "universe_avg_return": day["summary"]["universe_avg_return"],
                              "periods_list_beat_universe": 2}
    week = next(h for h in tr["horizons"] if h["key"] == "1W")
    assert [(p["list_date"], p["end_date"], p["hits"]) for p in week["periods"]] == [("2026-10-02", "2026-10-09", 2)]
    assert tr["list_size"] == 5 and tr["message"] is None and "not a prediction" in tr["note"]


def test_route_explains_an_empty_record(client):
    data = client.get("/api/v1/performance/track-record", headers=H()).json()["data"]
    assert data["message"].startswith("Not enough daily rankings yet")
    assert [h["summary"] for h in data["horizons"]] == [None, None]
    assert client.get("/api/v1/performance/track-record").status_code == 401
