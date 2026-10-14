"""
tests/test_top_changes.py — "what changed since the previous ranking" on Top
Candidates: entries (newly added / moved up / now eligible), exits with the
reason, big moves inside the list; read-only; nothing to compare -> None.
"""
import datetime as dt

from sqlmodel import Session

from notifications.detector import latest_full_run
from ranking.top_changes import top_changes
from tests.test_notifications import H, _base_scores, _limit, _run, client, db  # noqa: F401  (fixtures)

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 2, 11, 0, tzinfo=UTC)


def _seed(s):
    _limit(s, 5)
    prev = {k: v for k, v in _base_scores().items() if k != "S29.NS"}     # S00 rank 1 ... S10 rank 11
    cur = {**_base_scores(), "S29.NS": 100.0, "S04.NS": 99.5, "S10.NS": 99.0}
    _run(s, "R1", T0, prev)
    _run(s, "R2", T0 + dt.timedelta(days=1), cur, ineligible=("S02.NS",))


def test_entries_exits_and_big_moves_explain_themselves(db):
    with Session(db) as s:
        _seed(s)
        ch = top_changes(s, latest_full_run(s), 5)
    assert [(e["symbol"], e["rank"], e["reason"]) for e in ch["entered"]] == [
        ("S29.NS", 1, "newly added to the stocks StockLens ranks"), ("S10.NS", 3, "up from rank 11")]
    assert [(e["symbol"], e["previous_rank"], e["reason"]) for e in ch["left"]] == [
        ("S02.NS", 3, "no longer eligible: market data stale"), ("S03.NS", 4, "now rank 6")]
    assert [(m["symbol"], m["previous_rank"], m["rank"]) for m in ch["moved"]] == [
        ("S04.NS", 5, 2), ("S00.NS", 1, 4), ("S01.NS", 2, 5)]
    assert ch["entered"][0]["name"] == "Stock 29" and ch["previous_run_id"] == "R1"
    assert ch["summary"].startswith("Since 02 Oct 2026") and ch["summary"].endswith(
        ": 2 new, 2 dropped out, 3 moved 3+ places.")


def test_no_previous_run_or_no_change(db):
    with Session(db) as s:
        _run(s, "R1", T0, _base_scores())
        assert top_changes(s, latest_full_run(s), 5) is None                 # nothing to compare with
        _run(s, "R2", T0 + dt.timedelta(days=1), _base_scores())
        ch = top_changes(s, latest_full_run(s), 5)
    assert (ch["entered"], ch["left"], ch["moved"]) == ([], [], [])
    assert ch["summary"].startswith("No changes to the Top 5 since ")


def test_top_picks_route_carries_the_changes(client, db):
    with Session(db) as s:
        _seed(s)
    data = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert [i["symbol"] for i in data["items"]] == ["S29.NS", "S04.NS", "S10.NS", "S00.NS", "S01.NS"]
    assert [e["symbol"] for e in data["changes"]["entered"]] == ["S29.NS", "S10.NS"]
    assert data["changes"]["limit"] == 5
