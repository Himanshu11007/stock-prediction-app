"""
tests/conftest.py — pytest configuration.

Ensures the project root (one level above tests/) is on sys.path so test
modules can import config, utils.decision_engine, scanner.filters, etc.
the same way app.py and api/ do.

If you run `pytest` from the project root (the same place you run
`streamlit run app.py` or `uvicorn api.main:app`), this is usually
unnecessary — but it's added defensively so `pytest` also works when
invoked from other working directories or CI runners.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network_price_provider(monkeypatch):
    """Current-price refreshes (prices/service.py) must never reach the real
    provider in tests: by default the provider returns nothing, so prices
    stay NOT_AVAILABLE. Tests that need prices pass `fetch=` explicitly or
    monkeypatch prices.service._default_fetch."""
    import prices.service as price_service
    monkeypatch.setattr(price_service, "_default_fetch", lambda symbols: {})
    price_service._refresh_lock = __import__("threading").Lock()


# ── Real multi-connection databases for concurrency tests ────────────────────
# In-memory SQLite with StaticPool (what most tests use) shares ONE connection
# between threads, so it can't show a race. Concurrency tests take
# `concurrent_engine` instead: a file-backed SQLite database (always), plus
# PostgreSQL - the production database - when TEST_DATABASE_URL points at a
# disposable PostgreSQL database, e.g.
#   TEST_DATABASE_URL=postgresql://stocklens:stocklens@127.0.0.1/stocklens_test
# Every table in that database is dropped and recreated per test.

def _concurrency_backends():
    import os
    backends = ["sqlite"]
    if os.environ.get("TEST_DATABASE_URL", "").startswith("postgresql"):
        backends.append("postgresql")
    return backends


@pytest.fixture(params=_concurrency_backends())
def concurrent_engine(request, tmp_path):
    import os
    from sqlmodel import SQLModel, create_engine
    import db.models  # noqa: F401  (populate metadata)

    if request.param == "sqlite":
        engine = create_engine(f"sqlite:///{tmp_path / 'concurrency.db'}",
                               connect_args={"check_same_thread": False, "timeout": 30})
    else:
        engine = create_engine(os.environ["TEST_DATABASE_URL"], pool_size=20, max_overflow=20)
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    yield engine
    if request.param != "sqlite":
        SQLModel.metadata.drop_all(engine)
    engine.dispose()


def run_concurrently(fn, n: int = 10):
    """Runs fn(i) on n threads released at the same instant; returns the
    list of (result, exception) pairs in thread order."""
    import threading
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        try:
            results[i] = (fn(i), None)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            results[i] = (None, exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return results


@pytest.fixture()
def concurrently():
    return run_concurrently
