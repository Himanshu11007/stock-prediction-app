"""
scripts/research/research_db.py — locate and open the read-only research
database for the research scripts, portably.

Resolution order:
  1. the script's --db PATH argument
  2. the STOCKLENS_RESEARCH_DB environment variable
  3. <repository>/storage/app.db

The database is opened read-only. A missing file, a missing table or a
missing column raises ResearchDataError with what is missing and how to fix
it. Missing data is never silently reported as zero rows or as a successful
validation.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
ENV = "STOCKLENS_RESEARCH_DB"
DEFAULT = ROOT / "storage" / "app.db"


class ResearchDataError(RuntimeError):
    """The requested research data is missing or has an incompatible schema."""


def resolve(cli_value: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    raw = cli_value or env.get(ENV) or str(DEFAULT)
    return Path(raw).expanduser()


def connect(path: Path, required: Mapping[str, set[str]]) -> sqlite3.Connection:
    """Open `path` read-only and check that every table in `required` exists
    with at least the listed columns."""
    path = Path(path)
    if not path.is_file():
        raise ResearchDataError(f"research database not found: {path}. Pass --db PATH or set {ENV}.")
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.DatabaseError as e:
        raise ResearchDataError(f"{path} is not a readable SQLite database: {e}") from None
    problems = []
    for table, cols in required.items():
        if table not in tables:
            problems.append(f"table '{table}' is missing")
            continue
        have = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        missing = sorted(set(cols) - have)
        if missing:
            problems.append(f"table '{table}' lacks column(s) {', '.join(missing)}")
    if problems:
        con.close()
        raise ResearchDataError(f"{path}: incompatible schema - {'; '.join(problems)}. Use a database migrated to "
                                f"the current Alembic head (alembic upgrade head on a COPY, never production).")
    return con


COMPANIES = {"companies": {"symbol", "name", "sector", "industry"}}
NEWS_ARCHIVE = {"news_articles": {"provider_article_id", "url", "title", "published_at", "excerpt", "source_name",
                                  "ingested_at", "corrected_at", "provider"}}
