"""
tests/test_migration_engine_run_guard.py — migration 4e6773a9e7db (partial
unique index on RUNNING RANKING runs) on scratch SQLite databases, through
the real Alembic chain: clean upgrade / downgrade / re-upgrade, a database
with one RUNNING run, and a database with duplicate RUNNING runs (the upgrade
must stop and change nothing). Never touches a real database.
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = "4e6773a9e7db"
PREVIOUS = "66e61a7dffc5"
INDEX = "uq_engine_runs_one_running_ranking"


def alembic(db: Path, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db.as_posix()}", "APP_ENV": "development"}
    return subprocess.run([sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180)


def version(db: Path) -> str:
    with sqlite3.connect(db) as c:
        return c.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def guard_index(db: Path):
    with sqlite3.connect(db) as c:
        row = c.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (INDEX,)).fetchone()
    return row[0] if row else None


def add_run(db: Path, run_id: str, status: str = "RUNNING", kind: str = "RANKING"):
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO engine_runs (run_id, kind, status, started_at, engine_version, fqvf_version, total, "
                  "processed, succeeded, skipped, failed) VALUES (?, ?, ?, '2026-10-10 05:00:00', 'ranking-v1.0', "
                  "'fqvf-v1.0', 0, 0, 0, 0, 0)", (run_id, kind, status))


def rows(db: Path):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT run_id, kind, status FROM engine_runs ORDER BY run_id").fetchall()


@pytest.fixture()
def db_at_previous(tmp_path):
    db = tmp_path / "m.db"
    r = alembic(db, "upgrade", PREVIOUS)
    assert r.returncode == 0, r.stderr[-2000:]
    return db


def test_clean_database_upgrade_downgrade_upgrade(tmp_path):
    db = tmp_path / "clean.db"
    up = alembic(db, "upgrade", "head")
    assert up.returncode == 0, up.stderr[-2000:]
    assert version(db) == GUARD
    sql = guard_index(db)
    assert sql and "UNIQUE" in sql.upper() and "status = 'RUNNING' AND kind = 'RANKING'" in sql
    down = alembic(db, "downgrade", "-1")
    assert down.returncode == 0, down.stderr[-2000:]
    assert version(db) == PREVIOUS and guard_index(db) is None
    again = alembic(db, "upgrade", "head")
    assert again.returncode == 0 and guard_index(db)


def test_models_and_migrations_agree(tmp_path):
    db = tmp_path / "check.db"
    assert alembic(db, "upgrade", "head").returncode == 0
    check = alembic(db, "check")
    assert check.returncode == 0, check.stdout + check.stderr
    assert "No new upgrade operations detected" in check.stdout + check.stderr


def test_upgrade_with_one_running_run_succeeds_and_keeps_the_run(db_at_previous):
    add_run(db_at_previous, "R-RUNNING")
    add_run(db_at_previous, "R-DONE", status="COMPLETED")
    add_run(db_at_previous, "S-1", kind="SINGLE")
    add_run(db_at_previous, "S-2", kind="SINGLE")
    before = rows(db_at_previous)
    r = alembic(db_at_previous, "upgrade", "head")
    assert r.returncode == 0, r.stderr[-2000:]
    assert version(db_at_previous) == GUARD and guard_index(db_at_previous)
    assert rows(db_at_previous) == before                              # no data modified


def test_upgrade_with_duplicate_running_runs_stops_and_changes_nothing(db_at_previous):
    add_run(db_at_previous, "R-A")
    add_run(db_at_previous, "R-B")
    add_run(db_at_previous, "R-OK", status="COMPLETED")
    before = rows(db_at_previous)
    r = alembic(db_at_previous, "upgrade", "head")
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "DuplicateRunningRunsError" in out
    assert "2 RANKING runs are RUNNING" in out and "R-A" in out and "R-B" in out
    assert "No change was made" in out
    assert version(db_at_previous) == PREVIOUS                       # not stamped
    assert guard_index(db_at_previous) is None                        # index not created
    assert rows(db_at_previous) == before                             # run rows untouched


def test_after_manual_resolution_the_upgrade_succeeds(db_at_previous):
    add_run(db_at_previous, "R-A")
    add_run(db_at_previous, "R-B")
    assert alembic(db_at_previous, "upgrade", "head").returncode != 0
    with sqlite3.connect(db_at_previous) as c:                         # the approved, documented resolution
        c.execute("UPDATE engine_runs SET status='FAILED' WHERE run_id='R-A'")
    r = alembic(db_at_previous, "upgrade", "head")
    assert r.returncode == 0, r.stderr[-2000:]
    assert ("R-A", "RANKING", "FAILED") in rows(db_at_previous)       # kept, not deleted


def test_upgrade_is_a_no_op_when_the_index_already_exists(db_at_previous):
    """An equivalent index created elsewhere (another branch's migration)
    must not make this migration fail."""
    with sqlite3.connect(db_at_previous) as c:
        c.execute(f"CREATE UNIQUE INDEX {INDEX} ON engine_runs (kind) WHERE status = 'RUNNING' AND kind = 'RANKING'")
    add_run(db_at_previous, "R-A")
    add_run(db_at_previous, "R-B", status="COMPLETED")
    before = rows(db_at_previous)
    r = alembic(db_at_previous, "upgrade", "head")
    assert r.returncode == 0, r.stderr[-2000:]
    assert version(db_at_previous) == GUARD and rows(db_at_previous) == before
