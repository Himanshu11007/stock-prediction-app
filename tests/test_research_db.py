"""
tests/test_research_db.py — research scripts locate their database portably
(--db, then STOCKLENS_RESEARCH_DB, then storage/app.db) and fail clearly when
it, a table or a column is missing; missing data is never reported as zero
articles or as a successful validation. No network.
"""
import sqlite3
from pathlib import Path

import pytest

from scripts.research import news_signal_backtest as nsb
from scripts.research import research_db


def _db(path: Path, news: bool = True, drop_column: str = "") -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE companies (symbol TEXT, name TEXT, sector TEXT, industry TEXT)")
    con.execute("INSERT INTO companies VALUES ('TCS.NS', 'Tata Consultancy Services', 'Technology', 'IT Services')")
    if news:
        cols = [c for c in ("provider_article_id", "url", "title", "published_at", "excerpt", "source_name",
                            "ingested_at", "corrected_at", "provider") if c != drop_column]
        con.execute(f"CREATE TABLE news_articles ({', '.join(c + ' TEXT' for c in cols)})")
    con.commit()
    con.close()
    return path


def test_resolution_order_and_no_machine_specific_default(tmp_path):
    assert research_db.resolve("a.db", {research_db.ENV: "b.db"}) == Path("a.db")
    assert research_db.resolve(None, {research_db.ENV: "b.db"}) == Path("b.db")
    assert research_db.resolve(None, {}) == research_db.ROOT / "storage" / "app.db"
    for f in Path(research_db.ROOT / "scripts" / "research").glob("*.py"):
        assert "C:\Python" not in f.read_text(encoding="utf-8"), f.name           # no hardcoded Windows path


def test_missing_file_table_and_column_are_clear_errors(tmp_path):
    with pytest.raises(research_db.ResearchDataError, match="not found.*--db PATH or set STOCKLENS_RESEARCH_DB"):
        research_db.connect(tmp_path / "nope.db", research_db.COMPANIES)
    no_news = _db(tmp_path / "a.db", news=False)
    research_db.connect(no_news, research_db.COMPANIES).close()
    with pytest.raises(research_db.ResearchDataError, match="table 'news_articles' is missing"):
        research_db.connect(no_news, research_db.NEWS_ARCHIVE)
    old = _db(tmp_path / "b.db", drop_column="corrected_at")
    with pytest.raises(research_db.ResearchDataError, match="lacks column\(s\) corrected_at"):
        research_db.connect(old, research_db.NEWS_ARCHIVE)
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database")
    with pytest.raises(research_db.ResearchDataError, match="not a readable SQLite database"):
        research_db.connect(junk, research_db.COMPANIES)


def test_archive_missing_is_an_error_or_explicitly_unavailable_never_zero(tmp_path):
    no_news = _db(tmp_path / "a.db", news=False)
    with pytest.raises(research_db.ResearchDataError):
        nsb.own_archive(no_news)
    arts, info = nsb.own_archive(no_news, allow_missing=True)
    assert arts == [] and info["status"] == "UNAVAILABLE" and info["articles"] is None and "news_articles" in info["detail"]
    arts, info = nsb.own_archive(_db(tmp_path / "b.db"))
    assert info == {"status": "EMPTY", "articles": 0, "first_ingested": None, "last_ingested": None}


def test_backtest_stops_before_any_network_work_when_data_is_missing(tmp_path, monkeypatch, capsys):
    import fundamentals.provider as fp
    monkeypatch.setattr(fp, "fetch_price_history", lambda *a, **k: pytest.fail("network used despite missing data"))
    assert nsb.main(["--db", str(tmp_path / "nope.db"), "--out", str(tmp_path / "o.json")]) == 2
    assert "research database not found" in capsys.readouterr().err
    no_news = _db(tmp_path / "a.db", news=False)
    assert nsb.main(["--db", str(no_news), "--providers", "stocklens_archive", "--out", str(tmp_path / "o.json")]) == 2
    assert "news_articles" in capsys.readouterr().err and not (tmp_path / "o.json").exists()
    monkeypatch.setenv(research_db.ENV, str(tmp_path / "nope.db"))
    assert nsb.main(["--providers", "marketaux", "--out", str(tmp_path / "o.json")]) == 2       # env var honoured
