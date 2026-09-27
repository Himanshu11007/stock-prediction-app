"""Database engine/session setup.

Single source of truth for the SQLAlchemy engine. Dialect-agnostic by design:
DATABASE_URL (config.py) picks SQLite for local dev or PostgreSQL in production,
and nothing in this module assumes either one.
"""
from sqlmodel import Session, create_engine

from config import DATABASE_URL

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)


def get_session():
    """FastAPI dependency: yields a Session, closes it after the request."""
    with Session(engine) as session:
        yield session
