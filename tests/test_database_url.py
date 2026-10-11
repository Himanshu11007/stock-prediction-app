"""
tests/test_database_url.py — the PostgreSQL driver is always named
explicitly. A bare "postgresql://" means psycopg2 on SQLAlchemy 2.0 but
psycopg (v3) from SQLAlchemy 2.1, which is not installed: a fresh build
(CI, Render) would fail to connect. Render hands out "postgres://" URLs.
"""
import pytest
from sqlalchemy.engine import make_url

from config import normalize_database_url


@pytest.mark.parametrize("raw", ["postgres://u:p@db.example:5432/stocklens", "postgresql://u:p@db.example/stocklens"])
def test_render_and_bare_urls_use_psycopg2(raw):
    url = normalize_database_url(raw)
    assert url.startswith("postgresql+psycopg2://")
    assert make_url(url).get_dialect().driver == "psycopg2"        # on every SQLAlchemy version


@pytest.mark.parametrize("raw", ["postgresql+psycopg2://u:p@h/d", "postgresql+psycopg://u:p@h/d",
                                 "sqlite:///storage/app.db", "sqlite://"])
def test_explicit_drivers_and_sqlite_are_unchanged(raw):
    assert normalize_database_url(raw) == raw


def test_requirements_provide_the_pinned_driver():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for req in ("requirements.txt", "requirements-cloud.txt"):
        assert "psycopg2-binary" in (root / req).read_text(encoding="utf-8")
