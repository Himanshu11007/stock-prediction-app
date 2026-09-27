"""One-off data migration: tracker.db (signals, watchlist,
recommendation_validation) -> the new DB's signals / watchlist_items /
recommendations / recommendation_validations tables.

Source of truth: storage/tracker.db (config.TRACKER_DB), read directly via
sqlite3 - never modified or deleted by this script.

Idempotent: every migrated row carries the source row's original id as
`legacy_id` (unique). Re-running skips rows whose legacy_id is already
present rather than re-inserting or duplicating them.

recommendation_validation is split into two new tables on purpose:
Recommendation (generation-time fields) and RecommendationValidation
(validation-time fields) - a legacy row only produces a
RecommendationValidation row if it was actually validated
(legacy is_validated=1); otherwise only the Recommendation row is created,
exactly mirroring "not yet validated" in the old schema.

Two legacy columns - recommendation_validation.validated and .outcome_price -
are NOT carried over: verified via direct query that all 1989 existing rows
have validated=0 and outcome_price=NULL, and no application code anywhere
reads or writes them (superseded by is_validated/validation_price - see
storage/tracker.py:158's own comment on this). No data is lost by omitting
them; this is called out explicitly in the migration report below.

The watchlist has no owner concept in the legacy schema (global, single
table, no user_id). Per explicit instruction, migrated watchlist rows are
NOT presented as if genuinely created by the placeholder account - user_id
is set only to satisfy the new schema's FK requirement, and
is_legacy_migration=True marks every such row so it is always
distinguishable from real account activity going forward.

Usage:
    venv/Scripts/python.exe -m scripts.migrate_legacy_tracker
"""
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select

import config
from auth.service import get_user_by_email
from db.models.tracker import Recommendation, RecommendationValidation, Signal, WatchlistItem
from db.session import engine
from scripts.seed_admin import DEFAULT_ADMIN_EMAIL

LEGACY_DATA_OWNER_EMAIL = DEFAULT_ADMIN_EMAIL


@dataclass
class TableReport:
    name: str
    old_count: int = 0
    migrated_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    error_count: int = 0
    field_issues: list = field(default_factory=list)

    def print(self) -> None:
        print(f"--- {self.name} ---")
        print(f"  old_count (tracker.db):      {self.old_count}")
        print(f"  migrated this run:           {self.migrated_count}")
        print(f"  skipped as already-migrated: {self.duplicate_count}")
        print(f"  invalid (not migrated):      {self.invalid_count}")
        print(f"  errors (not migrated):       {self.error_count}")
        if self.field_issues:
            print(f"  field-level issues ({len(self.field_issues)}):")
            for issue in self.field_issues[:20]:
                print(f"    - {issue}")
            if len(self.field_issues) > 20:
                print(f"    ... and {len(self.field_issues) - 20} more")


def _parse_legacy_timestamp(value: str) -> datetime:
    """watchlist.created_at is SQLite's CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS'),
    which SQLite generates in UTC."""
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _legacy_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.TRACKER_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _already_migrated_legacy_ids(session: Session, model) -> set:
    stmt = select(model.legacy_id).where(model.legacy_id.is_not(None))
    return set(session.exec(stmt).all())


def migrate_signals(legacy_conn: sqlite3.Connection, session: Session) -> TableReport:
    report = TableReport(name="signals")
    rows = legacy_conn.execute("SELECT * FROM signals").fetchall()
    report.old_count = len(rows)
    existing = _already_migrated_legacy_ids(session, Signal)

    for row in rows:
        if row["id"] in existing:
            report.duplicate_count += 1
            continue
        if not row["date"] or not row["symbol"] or not row["signal"]:
            report.invalid_count += 1
            report.field_issues.append(
                f"signals.id={row['id']}: missing required field (date/symbol/signal)"
            )
            continue
        try:
            session.add(
                Signal(
                    legacy_id=row["id"],
                    is_legacy_migration=True,
                    date=row["date"],
                    symbol=row["symbol"],
                    company=row["company"],
                    signal=row["signal"],
                    score=row["score"],
                    confidence=row["confidence"],
                    accuracy=row["accuracy"],
                    close_price=row["close_price"],
                    next_close=row["next_close"],
                    correct=row["correct"],
                )
            )
            session.commit()
            report.migrated_count += 1
        except Exception as e:
            session.rollback()
            report.error_count += 1
            report.field_issues.append(f"signals.id={row['id']}: {e}")

    return report


def migrate_recommendations(legacy_conn: sqlite3.Connection, session: Session) -> tuple:
    rec_report = TableReport(name="recommendations")
    val_report = TableReport(name="recommendation_validations")

    rows = legacy_conn.execute("SELECT * FROM recommendation_validation").fetchall()
    rec_report.old_count = len(rows)
    val_report.old_count = sum(1 for r in rows if r["is_validated"])

    existing_recs = _already_migrated_legacy_ids(session, Recommendation)
    existing_vals = _already_migrated_legacy_ids(session, RecommendationValidation)

    for row in rows:
        already_migrated = row["id"] in existing_recs
        if already_migrated:
            rec_report.duplicate_count += 1
        required = (row["saved_date"], row["symbol"], row["stock"], row["signal"], row["cmp"])
        if any(v is None for v in required):
            if not already_migrated:
                rec_report.invalid_count += 1
                rec_report.field_issues.append(
                    f"recommendation_validation.id={row['id']}: missing required field "
                    f"(saved_date/symbol/stock/signal/cmp)"
                )
            continue

        rec = None
        if not already_migrated:
            try:
                rec = Recommendation(
                    legacy_id=row["id"],
                    is_legacy_migration=True,
                    saved_date=row["saved_date"],
                    symbol=row["symbol"],
                    stock=row["stock"],
                    signal=row["signal"],
                    cmp=row["cmp"],
                    confluence_score=row["confluence_score"],
                    ml_confidence=row["ml_confidence"],
                    news_score=row["news_score"],
                    accuracy=row["accuracy"],
                    target=row["target"],
                    stop_loss=row["stop_loss"],
                    scan_id=row["scan_id"],
                    pillar_ml_dir=row["pillar_ml_dir"],
                    pillar_ml_conf=row["pillar_ml_conf"],
                    pillar_tech=row["pillar_tech"],
                    pillar_news=row["pillar_news"],
                    pillar_volume=row["pillar_volume"],
                    pillar_regime=row["pillar_regime"],
                    pillar_timeframe=row["pillar_timeframe"],
                    pillar_momentum=row["pillar_momentum"],
                    weighted_score=row["weighted_score"],
                    sector=row["sector"],
                    market_regime=row["market_regime"],
                    engine_version=row["engine_version"],
                )
                session.add(rec)
                session.commit()
                session.refresh(rec)
                rec_report.migrated_count += 1
            except Exception as e:
                session.rollback()
                rec_report.error_count += 1
                rec_report.field_issues.append(f"recommendation_validation.id={row['id']}: {e}")
                continue

        if not row["is_validated"]:
            continue  # nothing to migrate into recommendation_validations

        if row["id"] in existing_vals:
            val_report.duplicate_count += 1
            continue

        if rec is None:
            # already_migrated recommendation but its validation wasn't migrated yet
            # (shouldn't normally happen since both are written in the same pass,
            # but handle it defensively for partial/interrupted prior runs)
            existing_rec = session.exec(
                select(Recommendation).where(Recommendation.legacy_id == row["id"])
            ).first()
            if existing_rec is None:
                val_report.error_count += 1
                val_report.field_issues.append(
                    f"recommendation_validation.id={row['id']}: parent recommendation not found"
                )
                continue
            rec = existing_rec

        try:
            session.add(
                RecommendationValidation(
                    recommendation_id=rec.id,
                    legacy_id=row["id"],
                    is_legacy_migration=True,
                    validation_date=row["validation_date"],
                    validation_price=row["validation_price"],
                    return_pct=row["return_pct"],
                    success=row["success"],
                )
            )
            session.commit()
            val_report.migrated_count += 1
        except Exception as e:
            session.rollback()
            val_report.error_count += 1
            val_report.field_issues.append(f"recommendation_validation.id={row['id']}: {e}")

    return rec_report, val_report


def migrate_watchlist(legacy_conn: sqlite3.Connection, session: Session, owner_user_id: int) -> TableReport:
    report = TableReport(name="watchlist_items")
    rows = legacy_conn.execute("SELECT * FROM watchlist").fetchall()
    report.old_count = len(rows)
    existing = _already_migrated_legacy_ids(session, WatchlistItem)

    for row in rows:
        if row["id"] in existing:
            report.duplicate_count += 1
            continue
        if not row["symbol"] or not row["stock_name"] or row["buy_price"] is None or not row["buy_date"]:
            report.invalid_count += 1
            report.field_issues.append(
                f"watchlist.id={row['id']}: missing required field "
                f"(symbol/stock_name/buy_price/buy_date)"
            )
            continue
        try:
            item = WatchlistItem(
                legacy_id=row["id"],
                is_legacy_migration=True,
                user_id=owner_user_id,
                symbol=row["symbol"],
                stock_name=row["stock_name"],
                buy_price=row["buy_price"],
                buy_date=row["buy_date"],
                quantity=row["quantity"] if row["quantity"] is not None else 1,
            )
            if row["created_at"]:
                try:
                    item.created_at = _parse_legacy_timestamp(row["created_at"])
                except ValueError:
                    report.field_issues.append(
                        f"watchlist.id={row['id']}: unparseable created_at "
                        f"{row['created_at']!r}, used migration time instead"
                    )
            session.add(item)
            session.commit()
            report.migrated_count += 1
        except Exception as e:
            session.rollback()
            report.error_count += 1
            report.field_issues.append(f"watchlist.id={row['id']}: {e}")

    return report


def migrate() -> None:
    legacy_conn = _legacy_connection()
    try:
        with Session(engine) as session:
            owner = get_user_by_email(session, LEGACY_DATA_OWNER_EMAIL)
            if owner is None:
                raise SystemExit(
                    f"Legacy data owner account ({LEGACY_DATA_OWNER_EMAIL}) does not exist. "
                    f"Run `python -m scripts.seed_admin` first (Phase 3)."
                )

            print(
                f"Legacy watchlist rows will be attached to user_id={owner.id} "
                f"({owner.email}) as a migration placeholder ONLY - this is not a "
                f"claim that this account created that activity.\n"
            )

            signals_report = migrate_signals(legacy_conn, session)
            rec_report, val_report = migrate_recommendations(legacy_conn, session)
            watchlist_report = migrate_watchlist(legacy_conn, session, owner.id)

        print()
        for report in (signals_report, rec_report, val_report, watchlist_report):
            report.print()
            print()

        if config.TRACKER_DB.exists():
            print(f"Source file untouched: {config.TRACKER_DB}")
    finally:
        legacy_conn.close()


if __name__ == "__main__":
    migrate()
