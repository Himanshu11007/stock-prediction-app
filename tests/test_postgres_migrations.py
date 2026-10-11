"""
tests/test_postgres_migrations.py — Alembic on PostgreSQL (the production
engine). Needs TEST_DATABASE_URL pointing at a DISPOSABLE PostgreSQL
database (its public schema is dropped); skipped otherwise. Never run it
against production.

  * exactly one head; a linear chain from the production head
  * empty database -> head; `alembic check` reports no drift between the
    models and the migrations; head -> base -> head
  * upgrade from the production schema (main's head, 66e61a7dffc5) holding a
    representative row in EVERY table: no row is lost, new tables and
    columns exist, `alembic check` is clean; downgrade back to the
    production head keeps every original row
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_HEAD = "66e61a7dffc5"          # origin/main's Alembic head (11 Oct 2026)
from config import normalize_database_url  # noqa: E402

URL = normalize_database_url(os.environ.get("TEST_DATABASE_URL", ""))
needs_pg = pytest.mark.skipif(not URL.startswith("postgresql"), reason="TEST_DATABASE_URL (disposable PostgreSQL) not set")


def _script() -> ScriptDirectory:
    return ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))


def test_single_head_and_linear_chain_from_production():
    sd = _script()
    assert len(sd.get_heads()) == 1 and len(sd.get_bases()) == 1
    assert not [r for r in sd.walk_revisions() if r.is_merge_point or r.is_branch_point]
    chain = list(sd.iterate_revisions("head", PRODUCTION_HEAD))      # newest first, production head excluded
    assert chain and chain[-1].down_revision == PRODUCTION_HEAD      # production head is an ancestor of head
    assert sd.get_revision(PRODUCTION_HEAD) is not None


def _alembic(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": URL}
    return subprocess.run([sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def _reset(engine):
    with engine.begin() as c:
        c.exec_driver_sql("DROP SCHEMA public CASCADE")
        c.exec_driver_sql("CREATE SCHEMA public")


def _value(col: sa.Column, table: str, n: int):
    t = col.type
    if isinstance(t, (sa.Boolean,)):
        return False
    if isinstance(t, (sa.Integer, sa.BigInteger, sa.SmallInteger)):
        return n
    if isinstance(t, (sa.Float, sa.Numeric)):
        return 1.5
    if isinstance(t, sa.DateTime):
        return dt.datetime(2026, 10, 9, 10, 0, tzinfo=dt.timezone.utc)
    if isinstance(t, sa.Date):
        return dt.date(2026, 10, 9)
    if isinstance(t, sa.JSON) or t.__class__.__name__ in ("JSON", "JSONB"):
        return {"seed": n}
    if isinstance(t, (sa.String, sa.Text, sa.Enum)) or hasattr(t, "length"):
        s = f"{table[:12]}-{col.name}-{n}"
        return s[: t.length] if getattr(t, "length", None) else s
    return None


def _populate(engine) -> dict[str, int]:
    """One representative row in every table (parents first), with every
    column filled, as an existing production database would hold."""
    md = sa.MetaData()
    md.reflect(engine)
    inserted: dict[str, dict] = {}
    skipped = []
    for table in md.sorted_tables:
        if table.name == "alembic_version":
            continue
        row = {}
        for i, col in enumerate(table.columns):
            if col.primary_key and isinstance(col.type, sa.Integer) and col.autoincrement in (True, "auto"):
                continue
            fk = next(iter(col.foreign_keys), None)
            if fk is not None and fk.column.table.name in inserted:
                row[col.name] = inserted[fk.column.table.name].get(fk.column.name)
                continue
            row[col.name] = _value(col, table.name, i + 1)
        try:
            with engine.begin() as c:
                pk = c.execute(table.insert().values(**row).returning(*table.primary_key.columns)).first()
            inserted[table.name] = {**row, **(dict(pk._mapping) if pk else {})}
        except sa.exc.DBAPIError as e:
            skipped.append(f"{table.name}: {type(e.orig).__name__}")
    assert not skipped, f"could not seed: {skipped}"
    with engine.connect() as c:
        return {t: c.execute(sa.text(f'SELECT count(*) FROM "{t}"')).scalar() for t in inserted}


def _counts(engine, tables) -> dict[str, int]:
    with engine.connect() as c:
        return {t: c.execute(sa.text(f'SELECT count(*) FROM "{t}"')).scalar() for t in tables}


@needs_pg
def test_empty_database_to_head_has_no_drift_and_round_trips():
    eng = sa.create_engine(URL)
    _reset(eng)
    for step in (("upgrade", "head"), ("check",), ("downgrade", "base"), ("upgrade", "head"), ("check",)):
        r = _alembic(*step)
        assert r.returncode == 0, f"alembic {' '.join(step)} failed:\n{r.stdout}\n{r.stderr}"
    assert "No new upgrade operations detected" in r.stdout + r.stderr
    with eng.connect() as c:
        assert c.execute(sa.text("SELECT version_num FROM alembic_version")).scalar() == _script().get_current_head()
    eng.dispose()


@needs_pg
def test_upgrade_from_production_schema_keeps_every_row_and_downgrade_restores_it():
    eng = sa.create_engine(URL)
    _reset(eng)
    r = _alembic("upgrade", PRODUCTION_HEAD)
    assert r.returncode == 0, r.stderr
    before = _populate(eng)
    assert len(before) >= 20 and all(v == 1 for v in before.values())
    r = _alembic("upgrade", "head")
    assert r.returncode == 0, r.stderr
    assert _counts(eng, before) == before                               # nothing lost by the upgrade
    insp = sa.inspect(eng)
    new = {"prediction_runs", "predictions", "news_articles", "event_entities", "news_ingestion_runs",
           "password_reset_tokens"}
    assert new <= set(insp.get_table_names())
    cols = {c["name"] for c in insp.get_columns("news_articles")}
    assert {"published_raw", "published_tz", "first_seen_at", "quality_flags"} <= cols
    assert "token_valid_after" in {c["name"] for c in insp.get_columns("users")}
    assert _alembic("check").returncode == 0
    r = _alembic("downgrade", PRODUCTION_HEAD)
    assert r.returncode == 0, r.stderr
    assert _counts(eng, before) == before                               # rollback keeps production data
    assert not new & set(sa.inspect(eng).get_table_names())
    assert _alembic("upgrade", "head").returncode == 0
    eng.dispose()
