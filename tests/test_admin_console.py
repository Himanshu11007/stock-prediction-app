"""
tests/test_admin_console.py — the Streamlit admin console end to end.

The console's HTTP layer (admin_console.client.requests) is routed into the
real FastAPI app (api.main.app, in-memory database), so these tests exercise
the actual login form, authorization and every admin page without a server.
"""
from pathlib import Path

import pytest
import requests as real_requests
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine
from streamlit.testing.v1 import AppTest

import admin_console.client as console_client
import auth.service as auth_service
from api.main import app
from db.session import get_session
from tests.test_product_api import _seed_run

BASE = "http://testserver/api/v1"


@pytest.fixture()
def backend(monkeypatch):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
    _seed_run(eng)

    def _session():
        with Session(eng) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    tc = TestClient(app, raise_server_exceptions=False)

    class Routed:
        RequestException = real_requests.RequestException
        Response = real_requests.Response

        @staticmethod
        def request(method, url, **kw):
            kw.pop("timeout", None)
            return tc.request(method, url, **kw)

        @staticmethod
        def post(url, **kw):
            kw.pop("timeout", None)
            return tc.post(url, **kw)

    monkeypatch.setattr(console_client, "requests", Routed)
    yield
    app.dependency_overrides.clear()


def _login(email, password):
    # Absolute path: newer Streamlit resolves relative paths against the test file.
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "admin_console" / "app.py"), default_timeout=60)
    at.run()
    at.text_input[0].set_value(BASE)
    at.text_input[1].set_value(email)
    at.text_input[2].set_value(password)
    at.button[0].click().run()
    return at


def test_non_admin_cannot_sign_in(backend):
    at = _login("user@example.com", "UserPass1!")
    assert any("ADMIN role" in e.value for e in at.error)
    assert "me" not in at.session_state


def test_wrong_password_shows_error(backend):
    at = _login("admin@example.com", "wrong")
    assert at.error and "me" not in at.session_state


def test_every_admin_page_renders_for_admin(backend):
    at = _login("admin@example.com", "AdminPass1!")
    assert not at.exception and "me" in at.session_state
    radio = at.sidebar.radio[0]
    assert len(radio.options) == 25   # + Notifications, User Reports
    for page in radio.options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        assert not [e.value for e in at.error], (page, [e.value for e in at.error])


def _coverage(at):
    return next(c.value for c in at.caption if "stocks in the ranking list" in c.value)


def test_ranking_coverage_count_updates_in_the_same_page_load(backend):
    at = _login("admin@example.com", "AdminPass1!")
    at.sidebar.radio[0].set_value("Stock Master").run()
    before = int(_coverage(at).split()[0])
    next(t for t in at.text_input if t.label.startswith("New symbol")).set_value("VEDPOWER.NS")
    next(t for t in at.text_input if t.label == "Company name").set_value("Vedanta Power Ltd.")
    next(b for b in at.button if b.label == "Create").click().run()
    assert not at.exception and any("Added 1 stock(s)" in s.value for s in at.success)
    assert int(_coverage(at).split()[0]) == before + 1
    # removing (inside the coverage section) must update the count on the same page load
    next(t for t in at.text_input if t.label.startswith("Stop ranking")).set_value("VEDPOWER.NS")
    next(b for b in at.button if b.label == "Remove from the daily ranking").click().run()
    assert not at.exception and any("VEDPOWER.NS removed" in s.value for s in at.success)
    assert int(_coverage(at).split()[0]) == before                # not the count from before the change


def test_experimental_engine_list_can_be_filled_from_the_console(backend):
    at = _login("admin@example.com", "AdminPass1!")
    at.sidebar.radio[0].set_value("Stock Master").run()
    assert any("has 0 active stock(s) - empty" in c.value for c in at.caption)
    next(b for b in at.button if b.label.startswith("Copy the daily ranking list")).click().run()
    assert not at.exception
    done = next(s.value for s in at.success if s.value.startswith("Copied "))
    n = int(done.split()[1])
    assert n > 0 and any(f"has {n} active stock(s)." in c.value for c in at.caption)   # count updated in the same load
