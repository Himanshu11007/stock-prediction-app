"""
tests/test_streamlit_admin_navigation.py — the Admin Console inside the main
Streamlit app (app.py -> st.navigation -> admin_console/embedded.py).

The app runs with Streamlit's AppTest; its API calls are routed into the
real FastAPI app with an in-memory database (fixture from
test_admin_console.py), so sign-in, role checks and admin pages are the real
backend behaviour.
"""
import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from tests.test_admin_console import BASE, backend  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


@pytest.fixture()
def api_env(monkeypatch):
    monkeypatch.setenv("STOCKAI_API_URL", BASE)


@pytest.fixture(autouse=True)
def _no_legacy_scan(monkeypatch):
    """The legacy dashboard page starts a background market scan when its
    cache is empty (always, in a clean checkout). That scan downloads prices,
    news and the FinBERT model in threads that outlive the test. Navigation
    tests must stay offline, so the starter is replaced and calls recorded."""
    import scanner.background as bg
    started = []
    monkeypatch.setattr(bg, "start_background_scan", lambda company_map: started.append(len(company_map)) or True)
    return started


def _open(at: AppTest, url_path: str) -> AppTest:
    """Open a page by its URL path, as a browser does (/admin, /admin-engine-runs)."""
    at._page_hash = calc_hash(url_path)    # how st.Page identifies a page
    return at.run()


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=240)


def _sign_in(at: AppTest, email: str, password: str) -> AppTest:
    _open(at, "admin")
    at.text_input[0].set_value(email)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    return at


def _texts(at: AppTest) -> str:
    return " ".join([t.value for t in at.title] + [s.value for s in at.subheader] + [m.value for m in at.markdown]
                    + [b.label for b in at.button])


def _nav(at: AppTest) -> list[str]:
    """Titles of the Administration navigation entries for this session."""
    probe = AppTest.from_function(_list_admin_pages, default_timeout=60)
    probe.session_state["me"] = at.session_state["me"] if "me" in at.session_state else None
    if probe.session_state["me"] is None:
        del probe.session_state["me"]
    probe.run()
    return [m.value for m in probe.markdown]


def _list_admin_pages():
    import streamlit as st

    from admin_console import embedded
    for p in embedded.pages():
        st.markdown(p.title)


# ── existing pages and navigation ────────────────────────────────────────────

def test_dashboard_is_the_default_page_and_shows_no_admin_controls(api_env):
    at = _app().run()
    assert not at.exception
    assert len(at.tabs) == 6                                   # the existing dashboard tabs
    assert "Daily schedule" not in _texts(at) and "Sign in" not in [b.label for b in at.button]


def test_signed_out_visitors_only_see_the_admin_sign_in_entry():
    assert _nav(_app()) == ["Admin Console"]


def test_admin_sign_in_page_uses_configured_api_and_exposes_no_secrets(api_env, backend):
    at = _open(_app(), "admin")
    assert not at.exception
    labels = [t.label for t in at.text_input]
    assert labels == ["Email", "Password"]                     # the API address is not editable here
    page = _texts(at) + " ".join(c.value for c in at.caption)
    for secret in ("DATABASE_URL", "postgres", "JWT", "API_KEY"):
        assert secret not in page


def test_admin_page_urls_do_not_exist_before_sign_in(api_env, backend):
    at = _open(_app(), "admin-engine-runs")
    assert not at.exception
    assert "Daily schedule" not in _texts(at)
    assert "me" not in at.session_state


# ── authorization (backend decides) ──────────────────────────────────────────

def test_normal_user_cannot_open_the_admin_console(api_env, backend):
    at = _sign_in(_app(), "user@example.com", "UserPass1!")
    assert any("ADMIN role" in e.value for e in at.error)
    assert "me" not in at.session_state
    assert _nav(at) == ["Admin Console"]


def test_wrong_password_is_rejected(api_env, backend):
    at = _sign_in(_app(), "admin@example.com", "wrong")
    assert at.error and "me" not in at.session_state


def test_admin_sees_the_whole_console_in_the_navigation(api_env, backend):
    from admin_console.app import PAGES
    at = _sign_in(_app(), "admin@example.com", "AdminPass1!")
    assert not at.exception and "me" in at.session_state
    assert _nav(at) == PAGES + ["Sign out"]
    assert any(t.value == "Admin Dashboard" for t in at.title)  # landed on the dashboard


# ── Engine Runs / Daily Schedule / ranking button ───────────────────────────

def test_engine_runs_page_has_daily_schedule_and_runs_the_existing_job(api_env, backend, monkeypatch):
    import scheduling.jobs as sj
    started = []
    monkeypatch.setattr(sj, "ranking_job", lambda engine: started.append(engine))
    at = _sign_in(_app(), "admin@example.com", "AdminPass1!")
    _open(at, "admin-engine-runs")
    assert not at.exception and not at.error
    assert [s.value for s in at.subheader][:2] == ["Daily schedule", "Manual run"]
    button = next(b for b in at.button if b.label == "Run the daily ranking job now (calendar-aware)")
    button.click().run()
    assert any("Daily ranking job started" in s.value for s in at.success)
    import time
    for _ in range(100):
        if started:
            break
        time.sleep(0.02)
    assert len(started) == 1                                   # the production job, via POST /admin/scheduled-jobs/ranking/run


def test_every_admin_page_renders_inside_the_main_app(api_env, backend):
    from admin_console.app import PAGES
    from admin_console.embedded import SIGN_IN_PATH, _slug
    at = _sign_in(_app(), "admin@example.com", "AdminPass1!")
    for name in PAGES:
        _open(at, SIGN_IN_PATH if name == "Admin Dashboard" else _slug(name))
        assert not at.exception, name
        assert not [e.value for e in at.error], (name, [e.value for e in at.error])
        assert at.title[0].value == name


def test_sign_out_removes_the_admin_session(api_env, backend):
    at = _sign_in(_app(), "admin@example.com", "AdminPass1!")
    _open(at, "admin-sign-out")
    assert "me" not in at.session_state and "client" not in at.session_state


# ── configuration and architecture ───────────────────────────────────────────

def test_api_url_comes_from_configuration(monkeypatch):
    from admin_console import embedded
    monkeypatch.delenv("STOCKAI_API_URL", raising=False)
    monkeypatch.setattr(embedded.st, "secrets", {})
    assert embedded.api_url() == embedded.PRODUCTION_API_URL
    monkeypatch.setenv("STOCKAI_API_URL", "https://example.test/api/v1/")
    assert embedded.api_url() == "https://example.test/api/v1"
    assert embedded.PRODUCTION_API_URL == "https://stocklens-api-otrc.onrender.com/api/v1"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_console_has_no_business_logic_or_database_access():
    for f in ("admin_console/app.py", "admin_console/embedded.py", "admin_console/client.py"):
        mods = _imports(ROOT / f)
        for forbidden in ("db", "engine_runs", "ranking", "fqvf", "scheduling", "prices", "notifications",
                          "sqlmodel", "sqlalchemy", "masters"):
            assert forbidden not in mods, (f, forbidden)
        assert "DATABASE_URL" not in (ROOT / f).read_text(encoding="utf-8")


def test_app_routes_admin_pages_before_loading_the_dashboard_engine():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    nav = src.index("st.navigation(")
    assert nav < src.index("from models.trainer import")       # admin pages never load the ML stack
    assert src.count("st.navigation(") == 1
