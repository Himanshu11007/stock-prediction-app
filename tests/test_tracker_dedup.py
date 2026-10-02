"""
tests/test_tracker_dedup.py — Phase 10 regression coverage for duplicate
prevention in storage/tracker.py:upsert_recommendation().

Root cause of the 20 historical duplicate rows (see
docs/WALK_FORWARD_BENCHMARK.md): every one of them is from 2026-06-14,
has scan_id NULL and predates upsert_recommendation() (introduced
2026-06-29). Their presence made CREATE UNIQUE INDEX fail, so the database
never enforced uniqueness. These tests run against a temporary database.
"""
import sqlite3
import threading

import pytest

from storage import tracker


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "tracker.db"
    monkeypatch.setattr(tracker, "TRACKER_DB", path)
    return path


def _upsert(symbol="ABC.NS", saved_date="2026-07-01", signal="BUY", scan_id="SCAN-1"):
    return tracker.upsert_recommendation(
        symbol=symbol, stock="ABC", signal=signal, cmp=100.0,
        confluence_score=0.6, ml_confidence=70.0, news_score=0.0, accuracy=0.5,
        target=110.0, stop_loss=95.0, saved_date=saved_date, scan_id=scan_id,
    )


def _rows(db_path, symbol="ABC.NS", saved_date="2026-07-01"):
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(
            "SELECT id, signal, scan_id FROM recommendation_validation "
            "WHERE symbol = ? AND saved_date = ?", (symbol, saved_date),
        ).fetchall()
    finally:
        con.close()


def _index_names(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='recommendation_validation'")}
    finally:
        con.close()


def test_repeated_upsert_keeps_one_row_with_latest_values(db_path):
    first = _upsert(signal="BUY", scan_id="SCAN-1")
    second = _upsert(signal="SELL", scan_id="SCAN-2")
    rows = _rows(db_path)
    assert first == second
    assert rows == [(first, "SELL", "SCAN-2")]


def test_concurrent_upserts_for_one_key_create_exactly_one_row(db_path):
    _upsert(symbol="WARM.NS")  # create schema up front
    errors = []
    barrier = threading.Barrier(8)

    def worker(i):
        try:
            barrier.wait()
            _upsert(scan_id=f"SCAN-{i}")
        except Exception as e:  # pragma: no cover - surfaced by the assert
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(_rows(db_path)) == 1


def test_fresh_database_enforces_uniqueness_on_symbol_and_date(db_path):
    _upsert()
    assert "idx_rv_unique_symbol_date" in _index_names(db_path)


def test_legacy_duplicates_are_left_untouched_and_new_duplicates_are_blocked(db_path):
    # Reproduce the historical state: duplicate rows with scan_id NULL.
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE recommendation_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT, saved_date TEXT NOT NULL,
            symbol TEXT NOT NULL, stock TEXT NOT NULL, signal TEXT NOT NULL,
            cmp REAL NOT NULL, confluence_score REAL, ml_confidence REAL,
            news_score REAL, accuracy REAL, target REAL, stop_loss REAL,
            is_validated INTEGER DEFAULT 0, validation_date TEXT,
            validation_price REAL, return_pct REAL, success INTEGER
        )
    """)
    for _ in range(3):
        con.execute(
            "INSERT INTO recommendation_validation (saved_date, symbol, stock, signal, cmp) "
            "VALUES ('2026-06-14', 'OLD.NS', 'OLD', 'HOLD', 50.0)")
    con.commit()
    con.close()

    _upsert(symbol="NEW.NS")  # runs _ensure_validation_table; must not raise

    names = _index_names(db_path)
    assert "idx_rv_unique_symbol_date" not in names
    assert "idx_rv_unique_symbol_date_scanned" in names
    assert len(_rows(db_path, "OLD.NS", "2026-06-14")) == 3  # history untouched

    con = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO recommendation_validation "
                "(saved_date, symbol, stock, signal, cmp, scan_id) "
                "VALUES ('2026-07-01', 'NEW.NS', 'NEW', 'BUY', 1.0, 'SCAN-X')")
    finally:
        con.close()


def test_upsert_on_a_legacy_duplicate_key_updates_instead_of_inserting(db_path):
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE recommendation_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT, saved_date TEXT NOT NULL,
            symbol TEXT NOT NULL, stock TEXT NOT NULL, signal TEXT NOT NULL,
            cmp REAL NOT NULL, confluence_score REAL, ml_confidence REAL,
            news_score REAL, accuracy REAL, target REAL, stop_loss REAL,
            is_validated INTEGER DEFAULT 0, validation_date TEXT,
            validation_price REAL, return_pct REAL, success INTEGER
        )
    """)
    for _ in range(2):
        con.execute(
            "INSERT INTO recommendation_validation (saved_date, symbol, stock, signal, cmp) "
            "VALUES ('2026-06-14', 'OLD.NS', 'OLD', 'HOLD', 50.0)")
    con.commit()
    con.close()

    _upsert(symbol="OLD.NS", saved_date="2026-06-14")
    assert len(_rows(db_path, "OLD.NS", "2026-06-14")) == 2
