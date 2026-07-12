"""
tests/test_recommendation_intelligence.py

Unit tests for analytics/recommendation_intelligence.py.
All tests use in-memory SQLite — never touch the real tracker.db.
The intelligence engine is READ-ONLY so there is no risk of side effects.
"""
from __future__ import annotations

import sqlite3
import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_date TEXT, symbol TEXT, stock TEXT, signal TEXT,
    cmp REAL, confluence_score REAL, ml_confidence REAL,
    news_score REAL, accuracy REAL, target REAL, stop_loss REAL,
    is_validated INTEGER DEFAULT 0,
    validation_date TEXT, validation_price REAL,
    return_pct REAL, success INTEGER,
    scan_id TEXT, sector TEXT, market_regime TEXT, engine_version TEXT,
    weighted_score REAL,
    pillar_ml_dir REAL, pillar_ml_conf REAL, pillar_tech REAL,
    pillar_news REAL, pillar_volume REAL, pillar_regime REAL,
    pillar_timeframe REAL, pillar_momentum REAL
)
"""

def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Create an in-memory-style SQLite file in tmp_path and seed it."""
    db_path = tmp_path / "tracker_test.db"
    con = sqlite3.connect(str(db_path))
    con.execute(_SCHEMA)
    today = datetime.date.today().isoformat()
    for r in rows:
        r.setdefault("saved_date", today)
        r.setdefault("is_validated", 1)
        r.setdefault("validation_date", today)
        r.setdefault("engine_version", "v1.0")
        cols = ", ".join(r.keys())
        placeholders = ", ".join("?" * len(r))
        con.execute(
            f"INSERT INTO recommendation_validation ({cols}) VALUES ({placeholders})",
            list(r.values()),
        )
    con.commit()
    con.close()
    return db_path


def _patch_db(db_path: Path, monkeypatch):
    """Redirect TRACKER_DB to the test database."""
    import analytics.recommendation_intelligence as mod
    monkeypatch.setattr(mod, "TRACKER_DB", db_path)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mixed_rows():
    """20 rows with diverse signals, regimes, confidence bands, and pillar scores."""
    rows = []
    signals  = ["STRONG BUY", "BUY", "BUY", "HOLD", "SELL"]
    regimes  = ["Bullish", "Bearish", "Sideways"]
    sectors  = ["IT", "Financials", "Energy"]
    for i in range(20):
        sig = signals[i % len(signals)]
        ret = 3.0 if "BUY" in sig else (-2.0 if "SELL" in sig else 0.5)
        suc = 1 if ("BUY" in sig and ret > 0) or ("SELL" in sig and ret < 0) else 0
        rows.append({
            "symbol": f"ST{i:02d}.NS", "stock": f"Stock {i}", "signal": sig,
            "cmp": 1000.0, "confluence_score": 0.50 + (i % 5) * 0.05,
            "ml_confidence": 55.0 + (i % 5) * 10,
            "news_score": 0.1, "accuracy": 0.55,
            "return_pct": ret, "success": suc,
            "sector": sectors[i % 3], "market_regime": regimes[i % 3],
            "pillar_ml_dir": 0.7 if "BUY" in sig else -0.7,
            "pillar_ml_conf": 0.3, "pillar_tech": 0.4 + (i % 3) * 0.1,
            "pillar_news": 0.1, "pillar_volume": 0.2,
            "pillar_regime": 0.5, "pillar_timeframe": 0.3, "pillar_momentum": 0.2,
            "weighted_score": 0.4,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 1. Threshold calculations
# ══════════════════════════════════════════════════════════════════════════════

def test_threshold_calculations(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    thresh = report["threshold_analysis"]
    assert len(thresh) == 5  # 0.50, 0.55, 0.60, 0.65, 0.70
    for row in thresh:
        assert "threshold" in row
        assert "trades" in row
        assert "success_rate" in row
        assert 0.0 <= row["success_rate"] <= 100.0
    # Higher threshold → fewer or equal trades
    trades = [r["trades"] for r in thresh]
    assert trades == sorted(trades, reverse=True)


def test_threshold_returns_all_five_levels(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    levels = [r["threshold"] for r in report["threshold_analysis"]]
    assert levels == [0.50, 0.55, 0.60, 0.65, 0.70]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Confidence buckets
# ══════════════════════════════════════════════════════════════════════════════

def test_confidence_buckets(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    conf = report["confidence_analysis"]
    assert len(conf) == 5  # 50-60, 60-70, 70-80, 80-90, 90-100
    bands = [r["confidence_band"] for r in conf]
    assert bands == ["50-60", "60-70", "70-80", "80-90", "90-100"]
    for row in conf:
        assert row["success_rate"] >= 0.0


def test_confidence_buckets_sum_to_total(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    total_from_buckets = sum(r["trades"] for r in report["confidence_analysis"])
    assert total_from_buckets == report["summary"]["total"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sector grouping
# ══════════════════════════════════════════════════════════════════════════════

def test_sector_grouping(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    sectors = {r["sector"] for r in report["sector_analysis"]}
    assert "IT" in sectors
    assert "Financials" in sectors
    assert "Energy" in sectors


def test_sector_sorted_by_success_rate(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    rates = [r["success_rate"] for r in report["sector_analysis"]]
    assert rates == sorted(rates, reverse=True)


def test_missing_sector_graceful(tmp_path, monkeypatch):
    """Rows with no sector column should degrade gracefully."""
    rows = [
        {"symbol": "A.NS", "stock": "A", "signal": "BUY",
         "cmp": 100, "confluence_score": 0.6, "ml_confidence": 70,
         "news_score": 0.1, "accuracy": 0.55,
         "return_pct": 3.0, "success": 1},
    ]
    db = _make_db(tmp_path, rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    assert isinstance(report["sector_analysis"], list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Regime grouping
# ══════════════════════════════════════════════════════════════════════════════

def test_regime_grouping(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    regimes = {r["regime"] for r in report["regime_analysis"]}
    assert "Bullish" in regimes
    assert "Bearish" in regimes


def test_regime_sorted_by_success_rate(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    rates = [r["success_rate"] for r in report["regime_analysis"]]
    assert rates == sorted(rates, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Recommendation generation
# ══════════════════════════════════════════════════════════════════════════════

def test_recommendations_generated(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    recs = report["recommendations"]
    assert isinstance(recs, list)
    assert len(recs) > 0
    for r in recs:
        assert "title" in r
        assert "priority" in r
        assert "recommendation" in r
        assert "evidence" in r
        assert r["priority"] in ("High", "Medium", "Low")


def test_recommendations_never_suggest_weight_changes(tmp_path, monkeypatch, mixed_rows):
    """Intelligence recommendations must be observational, never prescriptive about model internals."""
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    forbidden = ["change weight", "retrain", "modify model", "delete"]
    for rec in report["recommendations"]:
        text = (rec.get("recommendation", "") + rec.get("evidence", "")).lower()
        for f in forbidden:
            assert f not in text, f"Recommendation contains forbidden phrase '{f}': {text}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Missing data handling
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_pillar_values_no_crash(tmp_path, monkeypatch):
    """Rows without pillar scores should not crash the engine."""
    rows = [
        {"symbol": f"X{i}.NS", "stock": f"X{i}", "signal": "BUY",
         "cmp": 100, "confluence_score": 0.65, "ml_confidence": 75,
         "news_score": 0.2, "accuracy": 0.55, "return_pct": 2.0, "success": 1}
        for i in range(5)
    ]
    db = _make_db(tmp_path, rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    for pillar in report["pillar_analysis"]:
        assert "pillar" in pillar
        assert "interpretation" in pillar


def test_corrupted_records_handled(tmp_path, monkeypatch):
    """Mix of valid and NULL/corrupted rows should degrade gracefully."""
    rows = [
        {"symbol": "GOOD.NS", "stock": "Good", "signal": "BUY",
         "cmp": 100, "confluence_score": 0.65, "ml_confidence": 72,
         "news_score": 0.1, "accuracy": 0.55, "return_pct": 3.0, "success": 1},
        {"symbol": "BAD.NS", "stock": "Bad", "signal": None,
         "cmp": None, "confluence_score": None, "ml_confidence": None,
         "news_score": None, "accuracy": None, "return_pct": None, "success": None},
    ]
    db = _make_db(tmp_path, rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    assert report["meta"]["records_analyzed"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Empty database
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_database(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [])
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    assert report["summary"]["total"] == 0
    assert report["meta"]["records_analyzed"] == 0
    assert isinstance(report["recommendations"], list)
    assert len(report["recommendations"]) > 0  # "insufficient data" rec


def test_missing_table_no_crash(tmp_path, monkeypatch):
    """No table at all should return a safe fallback."""
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(str(db_path))
    con.close()
    import analytics.recommendation_intelligence as mod
    monkeypatch.setattr(mod, "TRACKER_DB", db_path)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    assert isinstance(report, dict)
    assert "summary" in report


# ══════════════════════════════════════════════════════════════════════════════
# 8. Only BUY / Only HOLD
# ══════════════════════════════════════════════════════════════════════════════

def test_only_buy_recommendations(tmp_path, monkeypatch):
    rows = [
        {"symbol": f"B{i}.NS", "stock": f"Buy {i}", "signal": "BUY",
         "cmp": 100, "confluence_score": 0.65, "ml_confidence": 72,
         "news_score": 0.1, "accuracy": 0.55, "return_pct": 2.0, "success": 1}
        for i in range(8)
    ]
    db = _make_db(tmp_path, rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    signals = {r["signal"] for r in report["signal_analysis"] if r["trades"] > 0}
    assert signals == {"BUY"}
    assert report["summary"]["total"] == 8


def test_only_hold_recommendations(tmp_path, monkeypatch):
    rows = [
        {"symbol": f"H{i}.NS", "stock": f"Hold {i}", "signal": "HOLD",
         "cmp": 100, "confluence_score": 0.50, "ml_confidence": 58,
         "news_score": 0.0, "accuracy": 0.5, "return_pct": 0.5, "success": 1}
        for i in range(6)
    ]
    db = _make_db(tmp_path, rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    assert report["summary"]["total"] == 6
    hold = next((r for r in report["signal_analysis"] if r["signal"] == "HOLD"), None)
    assert hold is not None
    assert hold["trades"] == 6


# ══════════════════════════════════════════════════════════════════════════════
# 9. Report always contains all required keys
# ══════════════════════════════════════════════════════════════════════════════

def test_report_always_has_all_keys(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    required = {
        "summary", "threshold_analysis", "confidence_analysis",
        "pillar_analysis", "sector_analysis", "regime_analysis",
        "signal_analysis", "recommendations", "meta",
    }
    assert required.issubset(report.keys())


def test_report_keys_on_empty_db(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [])
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    required = {
        "summary", "threshold_analysis", "confidence_analysis",
        "pillar_analysis", "sector_analysis", "regime_analysis",
        "signal_analysis", "recommendations", "meta",
    }
    assert required.issubset(report.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 10. Read-only — never writes to DB
# ══════════════════════════════════════════════════════════════════════════════

def test_engine_is_read_only(tmp_path, monkeypatch, mixed_rows):
    """Row count must be identical before and after running the engine."""
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)

    con = sqlite3.connect(str(db))
    before = con.execute("SELECT COUNT(*) FROM recommendation_validation").fetchone()[0]
    con.close()

    from analytics.recommendation_intelligence import generate_engine_report
    generate_engine_report()

    con = sqlite3.connect(str(db))
    after = con.execute("SELECT COUNT(*) FROM recommendation_validation").fetchone()[0]
    con.close()

    assert before == after, "Intelligence engine must never modify the database"


# ══════════════════════════════════════════════════════════════════════════════
# 11. Meta fields present and correct types
# ══════════════════════════════════════════════════════════════════════════════

def test_meta_fields(tmp_path, monkeypatch, mixed_rows):
    db = _make_db(tmp_path, mixed_rows)
    _patch_db(db, monkeypatch)
    from analytics.recommendation_intelligence import generate_engine_report
    report = generate_engine_report()
    meta = report["meta"]
    assert isinstance(meta["execution_time_ms"], float)
    assert isinstance(meta["records_analyzed"], int)
    assert isinstance(meta["engine_version"], str)
    assert meta["records_analyzed"] == 20