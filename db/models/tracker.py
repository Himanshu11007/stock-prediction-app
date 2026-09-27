"""Signals, recommendations, recommendation validation, and watchlist items.

These mirror three legacy tracker.db tables (signals, recommendation_validation,
watchlist - see storage/tracker.py, storage/watchlist.py,
storage/recommendation_validation.py) with the same fields, but split
recommendation_validation's generation-time and validation-time data into two
related tables per the architecture decision: a "recommendation" (what the
system generated) is a distinct concept from its "validation" (what actually
happened, filled in later) - not one flat record.

Every row migrated from tracker.db carries legacy_id (the source row's
original id, unique) and is_legacy_migration=True, so migrated history is
always distinguishable from data the running application creates going
forward. See scripts/migrate_legacy_tracker.py for the migration itself.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Signal(SQLModel, table=True):
    """Mirrors legacy tracker.db `signals` (My Tracker tab / analyze-stock).
    Kept as its own table/concept - not merged into recommendations, per the
    architecture decision that signals and recommendations are distinct."""

    __tablename__ = "signals"

    id: Optional[int] = Field(default=None, primary_key=True)
    legacy_id: Optional[int] = Field(default=None, unique=True, index=True)
    is_legacy_migration: bool = Field(default=False)

    date: str = Field(index=True)
    symbol: str = Field(index=True)
    company: Optional[str] = None
    signal: str
    score: Optional[float] = None
    confidence: Optional[float] = None
    accuracy: Optional[float] = None
    close_price: Optional[float] = None
    next_close: Optional[float] = None
    correct: Optional[int] = None

    created_at: datetime = Field(default_factory=utcnow)


class Recommendation(SQLModel, table=True):
    """Generation-time data only - what the system produced and why (score,
    pillars, regime, sector, scan/engine traceability). See
    RecommendationValidation for the separate, later-filled-in outcome."""

    __tablename__ = "recommendations"

    id: Optional[int] = Field(default=None, primary_key=True)
    legacy_id: Optional[int] = Field(default=None, unique=True, index=True)
    is_legacy_migration: bool = Field(default=False)

    saved_date: str = Field(index=True)
    symbol: str = Field(index=True)
    stock: str
    signal: str
    cmp: float
    confluence_score: Optional[float] = None
    ml_confidence: Optional[float] = None
    news_score: Optional[float] = None
    accuracy: Optional[float] = None
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    scan_id: Optional[str] = Field(default=None, index=True)

    pillar_ml_dir: Optional[float] = None
    pillar_ml_conf: Optional[float] = None
    pillar_tech: Optional[float] = None
    pillar_news: Optional[float] = None
    pillar_volume: Optional[float] = None
    pillar_regime: Optional[float] = None
    pillar_timeframe: Optional[float] = None
    pillar_momentum: Optional[float] = None
    weighted_score: Optional[float] = None

    sector: Optional[str] = None
    market_regime: Optional[str] = None
    engine_version: Optional[str] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=utcnow)


class RecommendationValidation(SQLModel, table=True):
    """The subsequent, separate validation/performance result for a
    Recommendation - only exists once that recommendation has actually been
    validated (legacy is_validated=1). One-to-one with Recommendation."""

    __tablename__ = "recommendation_validations"

    id: Optional[int] = Field(default=None, primary_key=True)
    recommendation_id: int = Field(foreign_key="recommendations.id", unique=True, index=True)
    legacy_id: Optional[int] = Field(default=None, unique=True, index=True)
    is_legacy_migration: bool = Field(default=False)

    validation_date: Optional[str] = None
    validation_price: Optional[float] = None
    return_pct: Optional[float] = None
    success: Optional[int] = None

    created_at: datetime = Field(default_factory=utcnow)


class WatchlistItem(SQLModel, table=True):
    """Mirrors legacy tracker.db `watchlist`, now with an explicit owner
    (user_id). Migrated legacy rows point user_id at a documented
    placeholder account (see scripts/migrate_legacy_tracker.py) - that FK
    satisfies the schema requirement only, it is NOT a claim that the
    placeholder account genuinely created that watchlist entry."""

    __tablename__ = "watchlist_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    legacy_id: Optional[int] = Field(default=None, unique=True, index=True)
    is_legacy_migration: bool = Field(default=False)

    user_id: int = Field(foreign_key="users.id", index=True)
    symbol: str = Field(index=True)
    stock_name: str
    buy_price: float
    buy_date: str
    quantity: float = Field(default=1)

    created_at: datetime = Field(default_factory=utcnow)
