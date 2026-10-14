"""
tests/test_universe_admin.py — admins choose which stocks the daily ranking
covers: add one stock or a CSV list (creating Stock Master rows when
needed), check a list without saving, remove a stock while keeping its
history; every change is audited; normal users cannot do it; the shipped
NIFTY 500 file imports cleanly.
"""
import csv
from pathlib import Path

from sqlmodel import Session, select

from db.models.admin import AdminAuditLog
from db.models.stock import Company, StockUniverseMember
from engine_runs.service import resolve_universe
from tests.test_admin import _admin_headers, _normal_headers, client, engine, seed  # noqa: F401  (fixtures)

URL = "/api/v1/admin/universe"


def test_add_check_and_remove_stocks_in_the_daily_ranking(client, seed, engine):
    with Session(engine) as s:
        s.add(StockUniverseMember(symbol="TCS.NS", category="Large Cap"))
        s.commit()
    items = [{"symbol": "VEDPOWER", "name": "Vedanta Power Ltd.", "category": "Nifty 500"},     # .NS added
             {"symbol": "INFY.NS"},                                                             # existing company
             {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd."},                   # already ranked
             {"symbol": "NEWCO.NS"},                                                            # new, no name
             {"symbol": "bad symbol!", "name": "X"},
             {"symbol": "ABC.NS", "name": "Abc Ltd.", "category": "Penny"}]
    dry = client.post(URL, json={"items": items, "dry_run": True}, headers=_admin_headers()).json()
    assert dry["dry_run"] is True and dry["added"] == ["VEDPOWER.NS", "INFY.NS"]
    assert dry["created_companies"] == ["VEDPOWER.NS"] and dry["already_included"] == ["TCS.NS"]
    assert [e["row"] for e in dry["errors"]] == [4, 5, 6]
    with Session(engine) as s:                                                                 # nothing saved
        assert s.get(Company, "VEDPOWER.NS") is None
        assert len(s.exec(select(StockUniverseMember)).all()) == 1

    rep = client.post(URL, json={"items": items}, headers=_admin_headers()).json()
    assert rep["added"] == ["VEDPOWER.NS", "INFY.NS"] and rep["dry_run"] is False
    with Session(engine) as s:
        assert s.get(Company, "VEDPOWER.NS").name == "Vedanta Power Ltd."
        assert [c.symbol for c in resolve_universe(s)] == ["INFY.NS", "TCS.NS", "VEDPOWER.NS"]   # ranked next run
        log = s.exec(select(AdminAuditLog).where(AdminAuditLog.action == "add_to_universe")).one()
        assert log.extra_data["added"] == 2
    summary = client.get(URL, headers=_admin_headers()).json()
    assert summary["members"] == 3 and summary["ranked"] == 3
    assert summary["by_category"] == {"Admin added": 1, "Large Cap": 1, "Nifty 500": 1}
    again = client.post(URL, json={"items": items[:2]}, headers=_admin_headers()).json()
    assert again["added"] == [] and again["already_included"] == ["VEDPOWER.NS", "INFY.NS"]

    assert client.delete(f"{URL}/vedpower.ns", headers=_admin_headers()).json() == {"removed": "VEDPOWER.NS"}
    assert client.delete(f"{URL}/VEDPOWER.NS", headers=_admin_headers()).status_code == 404
    with Session(engine) as s:
        assert s.get(Company, "VEDPOWER.NS") is not None                                       # company kept
        assert "VEDPOWER.NS" not in [c.symbol for c in resolve_universe(s)]
        assert s.exec(select(AdminAuditLog).where(AdminAuditLog.action == "remove_from_universe")).one().entity_id \
            == "VEDPOWER.NS"


def test_only_admins_change_the_ranking_list(client, seed):
    assert client.get(URL, headers=_normal_headers()).status_code == 403
    assert client.post(URL, json={"items": [{"symbol": "TCS.NS"}]}, headers=_normal_headers()).status_code == 403
    assert client.delete(f"{URL}/TCS.NS", headers=_normal_headers()).status_code == 403
    assert client.post(URL, json={"items": [{"symbol": "TCS.NS"}]}).status_code == 401
    too_many = [{"symbol": f"S{i}.NS", "name": "x"} for i in range(1001)]
    assert client.post(URL, json={"items": too_many}, headers=_admin_headers()).status_code == 400


def test_shipped_nifty500_file_imports_cleanly(client, seed, engine):
    path = Path(__file__).resolve().parents[1] / "data" / "nifty500.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 501 and len({r["Symbol"] for r in rows}) == 501
    assert {"Symbol": "VEDPOWER.NS", "Company": "Vedanta Power Ltd.", "Category": "Nifty 500"} in rows
    items = [{"symbol": r["Symbol"], "name": r["Company"], "category": r["Category"]} for r in rows]
    rep = client.post(URL, json={"items": items, "dry_run": True}, headers=_admin_headers()).json()
    assert rep["errors"] == [] and len(rep["added"]) == 501
