"""
tests/test_branding.py — StockLens branding regression tests
(docs/BRANDING.md): product metadata, icons, notification and email text,
and a repository scan that fails if the legacy brand reappears anywhere
outside the documented technical identifiers.
"""
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from notifications.settings import DEFAULT_TEMPLATES, render

ROOT = Path(__file__).resolve().parents[1]
LEGACY = re.compile(r"StockAI Pro|\bStockAI\b(?!Pro)|\bStock AI\b|Hemanshu (Stock )?Filter")
# Historical records and the brand documentation itself (which lists the old
# name to explain the retained identifiers) are exempt.
EXEMPT_PREFIX = ("alembic/versions/", "scripts/research/output/", "docs/BRANDING.md", "tests/test_branding.py")
NAMING_NOTE = "renamed from StockAI Pro to **StockLens**"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_product_name_is_centralised():
    assert config.PRODUCT_NAME == "StockLens"
    assert "guarantee" in config.PRODUCT_DESCRIPTION and "does not guarantee" in config.PRODUCT_DESCRIPTION


def test_openapi_docs_and_icons(client):
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "StockLens API"
    assert "StockLens" in spec["info"]["description"]
    assert "/api/v1/top-picks" in spec["paths"]                       # routes unchanged
    for path in ("/docs", "/redoc"):
        html = client.get(path).text
        assert "StockLens API" in html and "/static/branding/favicon-32.png" in html
    for path in ("/favicon.ico", "/static/branding/icon-192.png", "/static/branding/icon-512.png",
                 "/static/branding/apple-touch-icon.png", "/static/branding/og-image.png"):
        r = client.get(path)
        assert r.status_code == 200 and len(r.content) > 500, path
    assert client.get("/api/v1/health").json()["app"] == "StockLens API"


@pytest.fixture()
def isolated_client():
    """The app with an empty, migrated in-memory database: the branding check
    must not depend on (or read) the developer's local storage/app.db."""
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    from db.session import get_session

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)

    def _session():
        with Session(eng) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    yield TestClient(app)
    app.dependency_overrides.pop(get_session, None)


def test_app_config_and_onboarding_use_the_brand(isolated_client):
    data = isolated_client.get("/api/v1/app/config").json()["data"]
    assert data["product_name"] == "StockLens"
    assert data["disclaimer"].startswith("StockLens provides research and analysis")
    text = " ".join(p["title"] + " " + p["body"] for p in data["onboarding"]) + data["legal"]["privacy_summary"]
    assert "StockLens" in text and not LEGACY.search(text)


def test_notification_templates_are_branded():
    for key, tpl in DEFAULT_TEMPLATES.items():
        assert not LEGACY.search(tpl["title"] + tpl["body"]), key
    title, body = render(DEFAULT_TEMPLATES, "NEW_TOP_CANDIDATE", name="TCS", rank=1, score="84.0")
    assert title == "New StockLens Top Candidate" and "StockLens Score: 84.0/100" in body
    assert render(DEFAULT_TEMPLATES, "SCORE_CHANGE", name="TCS")[0] == "StockLens Score Update"


def test_otp_email_is_branded(monkeypatch):
    from auth import otp_delivery
    sent = []

    class _Smtp:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, m): sent.append(m)

    monkeypatch.setattr(otp_delivery.smtplib, "SMTP", _Smtp)
    svc = otp_delivery.SmtpOtpDeliveryService(host="h", port=25, username="u", password="p",
                                              from_address="no-reply@stocklens.app", use_tls=False)
    svc.send("user@example.com", "123456")
    assert sent[0]["Subject"] == "Your StockLens verification code"
    assert "StockLens" in sent[0].get_content()


def test_no_legacy_brand_outside_documented_identifiers():
    files = [f for f in subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.split("\0") if f]
    offenders = []
    for f in files:
        if f.startswith(EXEMPT_PREFIX) or not f.endswith((".py", ".md", ".html", ".txt", ".json", ".toml")):
            continue
        path = ROOT / f
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if LEGACY.search(line) and NAMING_NOTE not in line:
                offenders.append(f"{f}:{n}: {line.strip()[:100]}")
    assert not offenders, "legacy brand found:\n" + "\n".join(offenders[:30])
