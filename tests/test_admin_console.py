"""
tests/test_admin_console.py — the Streamlit admin console end to end.

The console's HTTP layer (admin_console.client.requests) is routed into the
real FastAPI app (api.main.app, in-memory database), so these tests exercise
the actual login form, authorization and every admin page without a server.
"""
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
    at = AppTest.from_file("admin_console/app.py", default_timeout=60)
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
