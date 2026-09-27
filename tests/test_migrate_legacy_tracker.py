"""tests/test_migrate_legacy_tracker.py — scripts/migrate_legacy_tracker.py

Builds a throwaway legacy tracker.db (same schema as storage/tracker.py /
storage/watchlist.py / storage/recommendation_validation.py actually create)
and a throwaway new-schema DB, then runs the real migration functions
against them. Never touches the real storage/tracker.db or storage/app.db.
"""
import sqlite3

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
from db.models.tracker import Recommendation, RecommendationValidation, Signal, WatchlistItem
from scripts.migrate_legacy_tracker import (
    migrate_recommendations,
    migrate_signals,
    migrate_watchlist,
)


@pytest.fixture()
def legacy_conn(tmp_path):
    path = tmp_path / "legacy_tracker.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL, symbol TEXT NOT NULL, company TEXT,
            signal TEXT NOT NULL, score REAL, confidence REAL, accuracy REAL,
            close_price REAL, next_close REAL, correct INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE, stock_name TEXT NOT NULL,
            buy_price REAL NOT NULL, buy_date TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE recommendation_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_date TEXT NOT NULL, symbol TEXT NOT NULL, stock TEXT NOT NULL,
            signal TEXT NOT NULL, cmp REAL NOT NULL,
            confluence_score REAL, ml_confidence REAL, news_score REAL,
            accuracy REAL, target REAL, stop_loss REAL,
            validated INTEGER DEFAULT 0, validation_date TEXT, outcome_price REAL,
            success INTEGER, is_validated INTEGER DEFAULT 0,
            validation_price REAL, return_pct REAL, scan_id TEXT,
            pillar_ml_dir REAL, pillar_ml_conf REAL, pillar_tech REAL,
            pillar_news REAL, pillar_volume REAL, pillar_regime REAL,
            pillar_timeframe REAL, pillar_momentum REAL, weighted_score REAL,
            sector TEXT, market_regime TEXT, engine_version TEXT
        )"""
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_migrate_signals_preserves_fields_and_is_idempotent(legacy_conn, session):
    legacy_conn.execute(
        "INSERT INTO signals (date, symbol, company, signal, score, confidence, "
        "accuracy, close_price, next_close, correct) VALUES "
        "('2026-01-01','TCS.NS','Tata Consultancy','HOLD',0.0,68.0,51.64,8078.5,NULL,NULL)"
    )
    legacy_conn.commit()

    report = migrate_signals(legacy_conn, session)
    assert report.old_count == 1
    assert report.migrated_count == 1
    assert report.invalid_count == 0
    assert report.error_count == 0

    row = session.exec(select(Signal)).one()
    assert row.legacy_id == 1
    assert row.is_legacy_migration is True
    assert row.symbol == "TCS.NS"
    assert row.confidence == 68.0
    assert row.close_price == 8078.5

    # idempotent re-run
    report2 = migrate_signals(legacy_conn, session)
    assert report2.migrated_count == 0
    assert report2.duplicate_count == 1
    assert session.exec(select(Signal)).all().__len__() == 1


def test_migrate_signals_skips_invalid_rows(legacy_conn, session):
    # NOT NULL is enforced at the legacy DB level, so a genuinely missing
    # required field can't occur there in practice - but an empty string
    # (allowed by NOT NULL) is a real-world equivalent of "no symbol", and
    # the migration's own validation must still catch and skip it.
    legacy_conn.execute(
        "INSERT INTO signals (date, symbol, signal) VALUES ('2026-01-01','','HOLD')"
    )
    legacy_conn.execute(
        "INSERT INTO signals (date, symbol, signal) VALUES ('2026-01-01','TCS.NS','HOLD')"
    )
    legacy_conn.commit()

    report = migrate_signals(legacy_conn, session)
    assert report.old_count == 2
    assert report.migrated_count == 1
    assert report.invalid_count == 1
    assert "id=1" in report.field_issues[0]

    remaining = session.exec(select(Signal)).all()
    assert len(remaining) == 1
    assert remaining[0].symbol == "TCS.NS"


def test_migrate_recommendations_splits_generation_and_validation(legacy_conn, session):
    # one validated, one not-yet-validated
    legacy_conn.execute(
        "INSERT INTO recommendation_validation "
        "(saved_date, symbol, stock, signal, cmp, is_validated, validation_date, "
        " validation_price, return_pct, success, scan_id, pillar_ml_dir, sector, "
        " market_regime, engine_version) VALUES "
        "('2026-01-01','TCS.NS','Tata Consultancy','BUY',3500.0,1,'2026-01-08',3600.0,2.86,1,"
        " 'SCAN-1',0.7,NULL,'Bullish','v1.0')"
    )
    legacy_conn.execute(
        "INSERT INTO recommendation_validation "
        "(saved_date, symbol, stock, signal, cmp, is_validated) VALUES "
        "('2026-01-02','INFY.NS','Infosys','HOLD',1500.0,0)"
    )
    legacy_conn.commit()

    rec_report, val_report = migrate_recommendations(legacy_conn, session)
    assert rec_report.old_count == 2
    assert rec_report.migrated_count == 2
    assert val_report.old_count == 1  # only the is_validated=1 row
    assert val_report.migrated_count == 1

    recs = {r.symbol: r for r in session.exec(select(Recommendation)).all()}
    assert recs["TCS.NS"].pillar_ml_dir == 0.7
    assert recs["TCS.NS"].engine_version == "v1.0"
    assert recs["INFY.NS"].sector is None

    validations = session.exec(select(RecommendationValidation)).all()
    assert len(validations) == 1
    assert validations[0].recommendation_id == recs["TCS.NS"].id
    assert validations[0].return_pct == 2.86

    # not-yet-validated recommendation must NOT have a validation row
    unvalidated_has_no_validation = session.exec(
        select(RecommendationValidation).where(
            RecommendationValidation.recommendation_id == recs["INFY.NS"].id
        )
    ).first()
    assert unvalidated_has_no_validation is None

    # idempotent re-run
    rec_report2, val_report2 = migrate_recommendations(legacy_conn, session)
    assert rec_report2.migrated_count == 0
    assert rec_report2.duplicate_count == 2
    assert val_report2.migrated_count == 0
    assert val_report2.duplicate_count == 1


def test_migrate_watchlist_uses_placeholder_owner_and_preserves_created_at(legacy_conn, session):
    auth_service.ensure_roles_exist(session)
    owner = auth_service.create_user(session, "owner@example.com", "Password1!")

    legacy_conn.execute(
        "INSERT INTO watchlist (symbol, stock_name, buy_price, buy_date, quantity, created_at) "
        "VALUES ('TANLA.NS','Tanla Platforms',613.55,'2026-08-09',1,'2026-08-09 13:40:27')"
    )
    legacy_conn.commit()

    report = migrate_watchlist(legacy_conn, session, owner.id)
    assert report.old_count == 1
    assert report.migrated_count == 1

    item = session.exec(select(WatchlistItem)).one()
    assert item.user_id == owner.id
    assert item.is_legacy_migration is True
    assert item.symbol == "TANLA.NS"
    assert item.buy_price == 613.55
    # original timestamp preserved, not overwritten with migration time
    assert item.created_at.year == 2026
    assert item.created_at.month == 8
    assert item.created_at.day == 9
    assert item.created_at.hour == 13
    assert item.created_at.minute == 40

    # idempotent re-run
    report2 = migrate_watchlist(legacy_conn, session, owner.id)
    assert report2.migrated_count == 0
    assert report2.duplicate_count == 1
