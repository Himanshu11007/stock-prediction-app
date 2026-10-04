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
