"""
scripts/migrate_sqlite_to_postgres.py — copy the local SQLite database into
a (new, empty) PostgreSQL database, e.g. the Render database, once.

  1. Create the schema on the target:   DATABASE_URL=<postgres url> alembic upgrade head
  2. Copy the data:                     python scripts/migrate_sqlite_to_postgres.py \
                                            --source storage/app.db --target-env TARGET_DATABASE_URL

The target URL is read from an environment variable (never typed on the
command line or committed). The script refuses to copy into a database that
already has users, reads the source through the application's own table
definitions (so types such as timestamps and booleans convert exactly as the
app reads them), copies tables in foreign-key order inside one transaction,
resets every id sequence, and verifies row counts. The source SQLite file is
opened read-only and never modified.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Integer, create_engine, inspect, select, text  # noqa: E402
from sqlmodel import SQLModel  # noqa: E402

import db.models  # noqa: E402,F401  (registers every table on SQLModel.metadata)

SKIP_TABLES = {"alembic_version"}


def normalise_url(url: str) -> str:
    """Render and others hand out postgres:// URLs; SQLAlchemy needs postgresql://."""
    return "postgresql://" + url[len("postgres://"):] if url.startswith("postgres://") else url


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="storage/app.db")
    ap.add_argument("--target-env", default="TARGET_DATABASE_URL",
                    help="name of the environment variable holding the PostgreSQL URL")
    args = ap.parse_args()
    target_url = os.environ.get(args.target_env)
    if not target_url:
        print(f"Set {args.target_env} to the PostgreSQL connection URL first.")
        return 2
    source = create_engine(f"sqlite:///file:{Path(args.source).resolve().as_posix()}?mode=ro&uri=true")
    target = create_engine(normalise_url(target_url))

    existing = set(inspect(target).get_table_names())
    missing = [t.name for t in SQLModel.metadata.sorted_tables if t.name not in existing]
    if missing:
        print(f"Target schema is incomplete (run 'alembic upgrade head' first). Missing: {missing}")
        return 2
    with target.connect() as conn:
        if conn.execute(text("SELECT COUNT(*) FROM users")).scalar():
            print("Target database already has users - refusing to copy over existing data.")
            return 2

    src_tables = set(inspect(source).get_table_names())
    counts: dict[str, int] = {}
    with source.connect() as src, target.begin() as dst:
        for table in SQLModel.metadata.sorted_tables:          # parents before children
            if table.name in SKIP_TABLES or table.name not in src_tables:
                continue
            src_cols = {c["name"] for c in inspect(source).get_columns(table.name)}
            cols = [c for c in table.columns if c.name in src_cols]
            rows = [dict(r._mapping) for r in src.execute(select(*cols))]
            for i in range(0, len(rows), 1000):
                dst.execute(table.insert(), rows[i:i + 1000])
            counts[table.name] = len(rows)
            pk = list(table.primary_key.columns)
            if len(pk) == 1 and isinstance(pk[0].type, Integer) and rows:
                dst.execute(text(f"SELECT setval(pg_get_serial_sequence('{table.name}', '{pk[0].name}'), "
                                 f"(SELECT MAX({pk[0].name}) FROM \"{table.name}\"))"))
    with target.connect() as conn:
        actual = {t: conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar() for t in counts}
    for t, n in counts.items():
        print(f"{t:32s} {n:8d}")
    mismatched = {t: (counts[t], actual[t]) for t in counts if counts[t] != actual[t]}
    if mismatched:
        print("ROW COUNT MISMATCH:", mismatched)
        return 1
    print(f"Copied {sum(counts.values())} rows in {len(counts)} tables; counts verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
