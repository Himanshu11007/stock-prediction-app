"""
tests/test_watch_warnings.py — factual watchlist warnings (presentation only):
negative news, heavy selling (close down 2 %+ on 1.5x+ the 20-day average
volume, current data only), downtrend and a large StockLens Score drop. Each
warning explains itself; missing or out-of-date data gives no warning.
"""
import datetime as dt

from sqlmodel import Session

import auth.service as auth_service
import watchlist.service as watchlist_service
from db.models.market import MarketSnapshot
from ranking import watch_warnings as ww
from tests.test_notifications import H, _two_runs, client, db  # noqa: F401  (fixtures)
from tests.test_stock_labels import _news

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
TODAY = NOW.date()


def _snap(s, symbol, day, close, volume, avg=1000.0, status="OK", fetched=None):
    s.add(MarketSnapshot(symbol=symbol, as_of_date=day, close=close, volume=volume, status=status,
                         fetched_at=fetched or dt.datetime.fromisoformat(day).replace(hour=12, tzinfo=UTC),
                         technical={"avg_volume_20d": avg} if avg is not None else {}))
    s.commit()


def test_heavy_selling_needs_a_real_fall_on_heavy_volume_with_current_data(db):
    with Session(db) as s:
        _snap(s, "S01.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S01.NS", "2026-10-12", 96.0, 2000)                    # -4 % on 2.0x volume
        w = ww.heavy_selling(s, "S01.NS", TODAY)
        assert w["key"] == "heavy_selling" and w["tone"] == "negative" and w["as_of"] == "2026-10-12"
        assert w["detail"] == "Price fell 4.0% on 12 Oct 2026 on 2.0x its 20-day average volume"

        _snap(s, "S02.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S02.NS", "2026-10-12", 99.0, 3000)                    # only -1 %
        _snap(s, "S04.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S04.NS", "2026-10-12", 95.0, 1200)                    # only 1.2x volume
        _snap(s, "S05.NS", "2026-10-01", 100.0, 900)
        _snap(s, "S05.NS", "2026-10-06", 90.0, 3000)                    # a week old: not current
        _snap(s, "S06.NS", "2026-10-02", 100.0, 900)
        _snap(s, "S06.NS", "2026-10-12", 90.0, 3000)                    # previous bar is not the previous session
        _snap(s, "S07.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S07.NS", "2026-10-12", 90.0, 3000, avg=None)          # no average volume
        _snap(s, "S08.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S08.NS", "2026-10-12", 90.0, 3000, status="ERROR")    # failed fetch is ignored
        _snap(s, "S09.NS", "2026-10-12", 90.0, 3000)                    # nothing to compare with
        for sym in ("S02.NS", "S04.NS", "S05.NS", "S06.NS", "S07.NS", "S08.NS", "S09.NS", "S10.NS"):
            assert ww.heavy_selling(s, sym, TODAY) is None, sym
        # several snapshots of the same session: compare with the previous session, not the same day
        _snap(s, "S11.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S11.NS", "2026-10-12", 97.0, 1800, fetched=dt.datetime(2026, 10, 12, 11, tzinfo=UTC))
        _snap(s, "S11.NS", "2026-10-12", 97.0, 1800, fetched=dt.datetime(2026, 10, 12, 13, tzinfo=UTC))
        assert ww.heavy_selling(s, "S11.NS", TODAY)["detail"].startswith("Price fell 3.0% on 12 Oct 2026")


def _overview(db, symbols, now=NOW):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        for sym in symbols:
            watchlist_service.add_watchlist_item(s, user, sym)
        return {i["symbol"]: i for i in watchlist_service.overview(s, user, now=now)}


def test_watchlist_items_carry_explained_warnings(db):
    with Session(db) as s:
        _two_runs(s, changes={"S03.NS": 75.0, "S05.NS": 25.0})          # S03 -12, S05 -60 and trend 25
        _snap(s, "S07.NS", "2026-10-09", 100.0, 900)
        _snap(s, "S07.NS", "2026-10-12", 95.0, 2500)
        _news(s, "S09.NS", "Stock 09 faces SEBI penalty", -0.5, NOW - dt.timedelta(hours=4), ext=21)
        _news(s, "S10.NS", "Stock 10 wins large order", 0.6, NOW - dt.timedelta(hours=4), ext=22)
    ov = _overview(db, ["S00.NS", "S03.NS", "S05.NS", "S07.NS", "S09.NS", "S10.NS"])
    keys = {sym: [w["key"] for w in i["warnings"]] for sym, i in ov.items()}
    assert keys == {"S00.NS": [], "S03.NS": ["score_drop"], "S05.NS": ["downtrend", "score_drop"],
                    "S07.NS": ["heavy_selling"], "S09.NS": ["negative_news"], "S10.NS": []}
    assert ov["S03.NS"]["warnings"][0]["detail"] == "StockLens Score fell 12.0 points since the previous ranking"
    assert ov["S05.NS"]["warnings"][0]["detail"] == "Trend score 25/100 (30 or less)"
    neg = ov["S09.NS"]["warnings"][0]
    assert neg["detail"].startswith("Stock 09 faces SEBI penalty (reuters.com, reported ") and neg["url"].startswith("https://")
    assert {"key": "news_negative", "label": "Negative news", "tone": "negative"} in ov["S09.NS"]["labels"]
    assert {"key": "news_positive", "label": "Positive news", "tone": "positive"} in ov["S10.NS"]["labels"]
    assert ov["S10.NS"]["news"][0]["title"] == "Stock 10 wins large order"
    assert all(w["tone"] == "negative" and w["detail"] for i in ov.values() for w in i["warnings"])
    # a week later the same price move is no longer current and the news is older than 72 h
    later = _overview(db, [], now=NOW + dt.timedelta(days=7))
    assert later["S07.NS"]["warnings"] == [] and later["S09.NS"]["warnings"] == []


def test_overview_route_returns_warnings_and_the_not_advice_note(client, db):
    with Session(db) as s:
        _two_runs(s, changes={"S03.NS": 75.0})
    assert client.post("/api/v1/watchlist", json={"symbol": "S03.NS"}, headers=H()).status_code == 201
    data = client.get("/api/v1/watchlist/overview", headers=H()).json()["data"]
    assert "not sell advice" in data["warnings_note"]
    item = data["items"][0]
    assert [w["key"] for w in item["warnings"]] == ["score_drop"] and item["news"] == []
    assert item["stockai_score"] == 75.0 and item["score_change"] == -12.0          # existing fields unchanged
