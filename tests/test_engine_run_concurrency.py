"""
tests/test_engine_run_concurrency.py — only one RANKING engine run may be
RUNNING at a time, across processes (database partial unique index
uq_engine_runs_one_running_ranking + engine_runs.service.create_run).
"""
import datetime as dt
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
import engine_runs.service as runs
from db.models.market import ONE_RUNNING_RANKING_INDEX, EngineRun

ROOT = Path(__file__).resolve().parents[1]


def _file_engine(tmp_path, name="runs.db"):
    eng = create_engine(f"sqlite:///{(tmp_path / name).as_posix()}", connect_args={"timeout": 30})
    SQLModel.metadata.create_all(eng)
    return eng


def _running(eng, kind="RANKING"):
    with Session(eng) as s:
        return s.exec(select(EngineRun).where(EngineRun.status == "RUNNING", EngineRun.kind == kind)).all()


# ── the race, with real processes ────────────────────────────────────────────

def _start_run(db_url, barrier, out):
    sys.path.insert(0, str(ROOT))
    from sqlmodel import Session as S, create_engine as ce
    import engine_runs.service as r
    eng = ce(db_url, connect_args={"timeout": 30})
    with S(eng) as s:
        barrier.wait()
        try:
            out.put(("STARTED", r.create_run(s, kind="RANKING", triggered_by=None, config={}).run_id))
        except r.RunInProgressError:
            out.put(("REFUSED", None))
        except Exception as e:  # anything else is a test failure
            out.put(("ERROR", repr(e)))


@pytest.mark.parametrize("trial", range(20))
def test_two_processes_starting_together_get_exactly_one_run(tmp_path, trial):
    eng = _file_engine(tmp_path)
    url = str(eng.url)
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=_start_run, args=(url, barrier, out)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    results = sorted(out.get(timeout=10)[0] for _ in procs)
    assert results == ["REFUSED", "STARTED"], results            # never two, never an unhandled error
    assert len(_running(eng)) == 1


# ── behaviour of the guard ───────────────────────────────────────────────────

@pytest.fixture()
def eng():
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    return e


def test_index_is_declared_on_the_model(eng):
    idx = {i["name"]: i for i in inspect(eng).get_indexes("engine_runs")}
    assert idx[ONE_RUNNING_RANKING_INDEX]["unique"]


def test_database_rejects_a_second_running_ranking_row(eng):
    with Session(eng) as s:
        s.add(EngineRun(run_id="A", kind="RANKING", status="RUNNING", engine_version="v", fqvf_version="v"))
        s.commit()
        s.add(EngineRun(run_id="B", kind="RANKING", status="RUNNING", engine_version="v", fqvf_version="v"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_database_rejection_is_reported_as_run_in_progress(eng, monkeypatch):
    """The race window: the RUNNING check sees nothing, the insert collides."""
    with Session(eng) as s:
        runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        monkeypatch.setattr(runs, "_running", lambda session, kind="RANKING": None)
        with pytest.raises(runs.RunInProgressError, match="already in progress"):
            runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        # the session is usable afterwards (rolled back, not poisoned)
        assert len(s.exec(select(EngineRun)).all()) == 1


def test_finished_runs_do_not_block_a_new_run(eng):
    with Session(eng) as s:
        for i, status in enumerate(("COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED", "COMPLETED")):
            s.add(EngineRun(run_id=f"R{i}", kind="RANKING", status=status, engine_version="v", fqvf_version="v"))
        s.commit()
        assert runs.create_run(s, kind="RANKING", triggered_by=None, config={}).status == "RUNNING"


def test_abandoned_run_older_than_3_hours_is_taken_over(eng):
    with Session(eng) as s:
        s.add(EngineRun(run_id="OLD", kind="RANKING", status="RUNNING", engine_version="v", fqvf_version="v",
                        started_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)))
        s.commit()
        new = runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        old = s.exec(select(EngineRun).where(EngineRun.run_id == "OLD")).one()
        assert old.status == "FAILED" and "abandoned" in old.errors[-1]["error"]
        assert new.status == "RUNNING"
    assert len(_running(eng)) == 1


def test_single_stock_runs_are_not_restricted_by_the_guard(eng):
    with Session(eng) as s:
        runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        for _ in range(3):
            runs.create_run(s, kind="SINGLE", triggered_by=None, config={})
    assert len(_running(eng, "SINGLE")) == 3
    assert len(_running(eng, "RANKING")) == 1


def test_admin_api_still_answers_409_when_the_database_refuses(monkeypatch):
    from api.main import app
    from auth.security import create_access_token
    from db.session import get_session
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    with Session(e) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        s.add(EngineRun(run_id="BUSY", kind="RANKING", status="RUNNING", engine_version="v", fqvf_version="v"))
        s.commit()

    def _session():
        with Session(e) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    monkeypatch.setattr("api.routes.admin_masters.engine", e, raising=False)
    monkeypatch.setattr(runs, "_running", lambda session, kind="RANKING": None)   # force the DB path
    try:
        token = create_access_token(subject="admin@example.com", roles=["ADMIN"])
        r = TestClient(app).post("/api/v1/admin/engine-runs", json={}, headers={"Authorization": f"Bearer {token}"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 409
    assert "already in progress" in r.text
    assert len(_running(e)) == 1
