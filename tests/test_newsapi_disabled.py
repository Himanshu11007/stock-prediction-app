"""
tests/test_newsapi_disabled.py — StockLens operates without the NewsAPI key
that was exposed in public Git history:

  * NewsAPI is disabled by default and never contacted: a key alone does not
    enable it; NEWSAPI_ENABLED=true AND a new NEWSAPI_KEY are both required
  * the exposed key is refused by fingerprint even when explicitly enabled
  * no application code reads .streamlit/secrets.toml or an API_KEY entry,
    and no tracked file contains the exposed key
  * the default news ingestion (official RSS) is unaffected, and asking for
    NewsAPI alone fails loudly instead of reporting zero articles
  * the manual revocation checker does nothing without explicit confirmation

The exposed key itself never appears here: it is matched by its SHA-256.
"""
import datetime as dt
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from catalysts import providers as prov, registry

ROOT = Path(__file__).resolve().parents[1]
T0 = dt.datetime(2026, 10, 9, 8, 0, tzinfo=dt.timezone.utc)


def _no_network(*args, **kwargs):
    raise AssertionError("NewsAPI was contacted")


def test_disabled_by_default_and_a_key_alone_never_enables_it(monkeypatch):
    for v in ("NEWSAPI_ENABLED", "NEWSAPI_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(prov.NotPermitted, match="disabled by default"):
        list(prov.NewsApiProvider(get=_no_network, sleep=lambda s: None).fetch(T0, T0))
    monkeypatch.setenv("NEWSAPI_KEY", "a-brand-new-key-0000")
    for flag in ("", "false", "1", "yes", "TRUEISH"):
        monkeypatch.setenv("NEWSAPI_ENABLED", flag)
        with pytest.raises(prov.NotPermitted):
            list(prov.NewsApiProvider(get=_no_network, sleep=lambda s: None).fetch(T0, T0))
    assert "newsapi" not in registry.select({}).enabled
    sel = registry.select({"NEWS_PROVIDERS": "official_rss,newsapi", "NEWS_ALLOW_RESEARCH_PROVIDERS": "true",
                           "NEWSAPI_KEY": "a-brand-new-key-0000"})
    assert sel.enabled == ["official_rss"] and "NEWSAPI_ENABLED=true" in sel.refused["newsapi"]
    assert registry.health("newsapi", get=_no_network, env={"NEWSAPI_KEY": "k"}, now=T0)["status"] == "DISABLED"


def test_the_exposed_key_is_refused_even_when_enabled(monkeypatch):
    stand_in = "stand-in-for-the-exposed-key-0001"          # the real key is never written into tests
    monkeypatch.setattr(prov, "EXPOSED_NEWSAPI_KEY_SHA256", hashlib.sha256(stand_in.encode()).hexdigest())
    monkeypatch.setenv("NEWSAPI_ENABLED", "true")
    for key in (stand_in, f"  {stand_in}\n"):
        with pytest.raises(prov.NotPermitted, match="exposed"):
            list(prov.NewsApiProvider(api_key=key, get=_no_network, sleep=lambda s: None).fetch(T0, T0))
    monkeypatch.setenv("NEWSAPI_KEY", stand_in)                # also when it comes from the environment
    with pytest.raises(prov.NotPermitted, match="exposed"):
        list(prov.NewsApiProvider(get=_no_network, sleep=lambda s: None).fetch(T0, T0))


def test_fingerprint_is_a_full_sha256_and_no_tracked_file_contains_the_exposed_key():
    assert re.fullmatch(r"[0-9a-f]{64}", prov.EXPOSED_NEWSAPI_KEY_SHA256)
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split("\0")
    hits = []
    for name in files:
        f = ROOT / name
        if not name or not f.is_file() or f.stat().st_size > 5_000_000:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for token in set(re.findall(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32}(?![0-9A-Za-z])", text)):
            if prov.is_exposed_newsapi_key(token):
                hits.append(name)                            # file name only, never the value
    assert hits == []


def test_no_application_code_reads_the_secrets_file_or_contacts_newsapi():
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()
    offenders = []
    for name in files:
        if name.startswith(("tests/", "scripts/security/")):
            continue
        text = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(open|read_text|read_bytes|toml\.loads?|Path)\([^)]*secrets\.toml|"
                     r"secrets\[\s*['\"]API_KEY|secrets\.get\(\s*['\"]API_KEY|"
                     r"environ[^\n]*['\"]NEWSAPI_API_KEY['\"]", text):
            offenders.append(f"{name}: reads the exposed secrets configuration")
        if "newsapi.org" in text and name != "catalysts/providers.py":
            offenders.append(f"{name}: contacts newsapi.org")
        if "check_exposed_newsapi_key" in text:
            offenders.append(f"{name}: imports the manual checker")
    assert offenders == []
    assert not (ROOT / ".streamlit" / "secrets.toml").exists() or \
        subprocess.run(["git", "ls-files", "--error-unmatch", ".streamlit/secrets.toml"], cwd=ROOT,
                       capture_output=True).returncode != 0          # a local copy may exist; it is not tracked
    assert subprocess.run(["git", "check-ignore", "-q", ".streamlit/secrets.toml"], cwd=ROOT).returncode == 0


def test_legacy_app_news_uses_google_news_rss():
    text = (ROOT / "news" / "api.py").read_text(encoding="utf-8")
    assert "news.google.com/rss/search" in text and "feedparser" in text
    assert "newsapi" not in text.lower() and "API_KEY" not in text


def test_default_ingestion_is_unaffected_and_newsapi_alone_fails_loudly(monkeypatch):
    from scheduling import jobs
    for v in ("NEWS_PROVIDERS", "NEWS_ALLOW_RESEARCH_PROVIDERS", "NEWSAPI_ENABLED", "NEWSAPI_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert [p.name for p in jobs.news_providers()] == ["official_rss"]
    monkeypatch.setenv("NEWS_PROVIDERS", "newsapi")
    monkeypatch.setenv("NEWS_ALLOW_RESEARCH_PROVIDERS", "true")
    assert jobs.news_providers() == []
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine
    import db.models  # noqa: F401
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    out = jobs.news_ingestion_job(eng, now=T0)
    assert out["status"] == "FAILED" and "no news provider configured" in out["error"]      # never "0 articles, OK"
    assert "disabled by default" in out["selection"]["refused"]["newsapi"]


def test_manual_checker_does_nothing_without_confirmation(monkeypatch, capsys):
    from scripts.security import check_exposed_newsapi_key as chk
    monkeypatch.setattr(chk.subprocess, "run", lambda *a, **k: pytest.fail("read Git history"))
    monkeypatch.setattr(chk.urllib.request, "urlopen", lambda *a, **k: pytest.fail("contacted NewsAPI"))
    assert chk.main([]) == 3
    assert "--confirm-read-exposed-key" in capsys.readouterr().out
