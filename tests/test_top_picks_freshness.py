"""
tests/test_top_picks_freshness.py — Top Candidates snapshot consistency:
additive ranking-freshness status, per-item run id and reference-price
source; every existing response field is preserved.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from db.models.market import MarketSnapshot, RankingSnapshot
from ranking.presenter import ranking_freshness
from tests.test_daily_ranking import DAY1, DAY1_SCORES, H, _publish, at, client, db  # noqa: F401  (fixtures)
from utils.market_session import IST

ROOT = Path(__file__).resolve().parents[1]

# Item fields of the published contract before this change (mobile/web depend on them).
EXISTING_ITEM_FIELDS = {"rank", "symbol", "name", "sector", "industry", "stockai_score", "stocklens_score",
                        "score_coverage", "fqvf_score", "fqvf_summary", "fqvf_counts", "positives", "risks", "labels",
                        "components", "engine_version", "computed_at", "freshness", "freshness_status",
                        "ranking_date", "reference_price", "reference_price_as_of", "current_price",
                        "current_price_as_of", "current_price_status", "current_price_source", "current_price_date",
                        "market_status"}
EXISTING_TOP_FIELDS = {"items", "total_eligible", "limit", "run", "market_regime", "disclaimer", "ranking_date",
                       "market_status"}


@pytest.mark.parametrize("now,ranking,holidays,status,behind", [
    (dt.datetime(2026, 10, 9, 18, 0, tzinfo=IST), "2026-10-09", [], "CURRENT", 0),
    (dt.datetime(2026, 10, 9, 12, 0, tzinfo=IST), "2026-10-08", [], "CURRENT", 0),   # today's bar not final yet
    (dt.datetime(2026, 10, 9, 18, 0, tzinfo=IST), "2026-10-08", [], "BEHIND", 1),
    (dt.datetime(2026, 10, 12, 9, 0, tzinfo=IST), "2026-10-09", [], "CURRENT", 0),   # Monday morning, Friday ranking
    (dt.datetime(2026, 10, 12, 18, 0, tzinfo=IST), "2026-10-07", [], "STALE", 3),
    (dt.datetime(2026, 10, 5, 18, 0, tzinfo=IST), "2026-10-01", ["2026-10-02"], "BEHIND", 1),  # holiday skipped
])
def test_ranking_freshness_counts_trading_sessions(now, ranking, holidays, status, behind):
    f = ranking_freshness(ranking, holidays, now)
    assert (f["status"], f["sessions_behind"]) == (status, behind)


def test_ranking_freshness_without_a_ranking():
    assert ranking_freshness(None, [])["status"] == "UNAVAILABLE"


def test_top_picks_keeps_every_existing_field_and_adds_freshness_metadata(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    data = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert EXISTING_TOP_FIELDS <= set(data)
    assert data["ranking_freshness"]["ranking_date"] == DAY1.isoformat()
    assert data["ranking_freshness"]["status"] in ("CURRENT", "BEHIND", "STALE")
    for item in data["items"]:
        assert EXISTING_ITEM_FIELDS <= set(item)
        assert item["run_id"] == data["run"]["run_id"] == "R1"           # one snapshot
    rel = next(i for i in data["items"] if i["symbol"] == "RELIANCE.NS")
    assert rel["reference_price_source"] == "ranking_snapshot"
    other = next(i for i in data["items"] if i["symbol"] == "S00.NS")
    assert other["reference_price"] is None and other["reference_price_source"] is None


def test_reference_price_fallback_is_labelled(client, db):
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {})                          # run without frozen snapshots
        for r in s.exec(select(RankingSnapshot)).all():
            s.delete(r)
        s.add(MarketSnapshot(symbol="RELIANCE.NS", status="OK", close=1199.0, as_of_date=DAY1.isoformat(),
                             fetched_at=at(DAY1, 16, 0).astimezone(dt.timezone.utc)))
        s.commit()
    rel = next(i for i in client.get("/api/v1/top-picks", headers=H()).json()["data"]["items"]
               if i["symbol"] == "RELIANCE.NS")
    assert rel["reference_price"] == 1199.0 and rel["reference_price_source"] == "market_snapshot_at_analysis"


def test_mobile_contract_fixture_fields_are_still_served(client, db):
    """The mobile app deserialises the captured contract; all its keys remain."""
    fixture = ROOT.parent / "StockAIPro-Mobile" / "StockAIPro.Mobile" / "StockAIPro.Mobile.Tests" / "Fixtures" / \
        "top_picks.json"
    if not fixture.exists():
        pytest.skip("mobile repository not checked out next to the backend")
    want = json.loads(fixture.read_text(encoding="utf-8"))["data"]
    with Session(db) as s:
        _publish(s, "R1", DAY1, DAY1_SCORES, {"RELIANCE.NS": 1200.0})
    got = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert set(want) <= set(got)
    assert set(want["items"][0]) <= set(got["items"][0])
