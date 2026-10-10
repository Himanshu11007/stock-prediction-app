"""
db/models/prediction.py — Prediction Engine v2 persistence (separate from
Ranking Engine v1; no v1 table is read-modified-written by v2).

Snapshot discipline: a prediction run and its predictions are written once
(status RUNNING -> COMPLETED/FAILED) and never recalculated in place. A new
calculation is a new run. Outcomes and exit transitions are append-only and
unique per (prediction, horizon) / (prediction, session, state), so retried
jobs cannot duplicate them.

Dates are ISO strings (YYYY-MM-DD) like the rest of the schema; times are
timezone-aware UTC datetimes.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, text
from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


RUN_TYPES = ("TODAY_PREOPEN", "TODAY_CONFIRMED", "TOMORROW_EOD")
RUN_STATUSES = ("RUNNING", "COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED")
DIRECTIONS = ("UP", "DOWN", "NEUTRAL", "NO_CALL")
OUTCOME_STATUSES = ("EVALUATED", "NOT_APPLICABLE", "INSUFFICIENT_DATA")
EXIT_STATES = ("HOLD", "TIGHTEN", "PARTIAL_EXIT", "EXIT")
REVIEW_STATUSES = ("UNREVIEWED", "CONFIRMED", "CORRECTED", "REJECTED")

ONE_RUNNING_PREDICTION_INDEX = "uq_prediction_runs_one_running_per_type"


class V2UniverseMember(SQLModel, table=True):
    """Prediction v2 universe. Independent of v1's stock_universe: adding or
    removing a symbol here never changes the v1 ranking universe."""
    __tablename__ = "v2_universe_members"
    __table_args__ = (UniqueConstraint("symbol", name="uq_v2_universe_symbol"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    source: str = Field(default="SEED")            # SEED (from v1 CSV lists) | ADMIN | IMPORT
    active: bool = Field(default=True, index=True)
    added_by: Optional[int] = Field(default=None, foreign_key="users.id")
    added_at: datetime = Field(default_factory=utcnow)
    note: Optional[str] = Field(default=None)


class PredictionRun(SQLModel, table=True):
    __tablename__ = "prediction_runs"
    __table_args__ = (
        Index(ONE_RUNNING_PREDICTION_INDEX, "run_type", unique=True,
              sqlite_where=text("status = 'RUNNING'"), postgresql_where=text("status = 'RUNNING'")),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    idempotency_key: str = Field(unique=True)       # "<run_type>:<trading_date>"
    engine_version: str
    rule_version: str
    feature_set_version: str
    run_type: str = Field(index=True)               # RUN_TYPES
    trading_date: str = Field(index=True)           # session the run belongs to (IST date)
    target_session_date: str = Field(index=True)    # session the predictions are for
    data_cutoff_at: datetime                        # nothing after this instant is used
    status: str = Field(default="RUNNING", index=True)
    config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    config_hash: str
    started_at: datetime = Field(default_factory=utcnow, index=True)
    completed_at: Optional[datetime] = Field(default=None)
    universe_count: int = Field(default=0)
    eligible_count: int = Field(default=0)
    prediction_count: int = Field(default=0)
    counts: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    failure_reason: Optional[str] = Field(default=None)
    errors: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))


class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("run_id", "symbol", name="uq_predictions_run_symbol"),
                      Index("ix_predictions_symbol_created", "symbol", "created_at"))

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(unique=True, index=True)
    run_id: str = Field(foreign_key="prediction_runs.run_id", index=True)
    symbol: str = Field(index=True)
    direction: str                                  # DIRECTIONS
    setup_type: str                                 # rule that produced it, or NONE / reason for NO_CALL
    horizon_sessions: int = Field(default=1)
    confidence: Optional[float] = Field(default=None)            # null until calibrated
    calibration_version: Optional[str] = Field(default=None)
    reference_price: Optional[float] = Field(default=None)       # close at the data cutoff
    reference_date: Optional[str] = Field(default=None)
    entry_condition: Optional[str] = Field(default=None)
    stop_loss: Optional[float] = Field(default=None)
    target: Optional[float] = Field(default=None)
    trailing_stop_rule: Optional[str] = Field(default=None)
    invalidation_condition: Optional[str] = Field(default=None)
    features: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))   # frozen inputs
    reasons: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    quality_flags: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    event_ids: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class PredictionOutcome(SQLModel, table=True):
    __tablename__ = "prediction_outcomes"
    __table_args__ = (UniqueConstraint("prediction_id", "horizon_sessions", name="uq_outcome_prediction_horizon"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(foreign_key="predictions.prediction_id", index=True)
    horizon_sessions: int
    evaluator_version: str
    outcome_status: str                              # OUTCOME_STATUSES
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    start_price: Optional[float] = Field(default=None)
    end_price: Optional[float] = Field(default=None)
    stock_return: Optional[float] = Field(default=None)
    nifty_return: Optional[float] = Field(default=None)
    sector_return: Optional[float] = Field(default=None)
    excess_return_nifty: Optional[float] = Field(default=None)
    mfe: Optional[float] = Field(default=None)       # max favourable excursion (direction-adjusted)
    mae: Optional[float] = Field(default=None)       # max adverse excursion (direction-adjusted)
    directional_return: Optional[float] = Field(default=None)
    cost_bps: Optional[float] = Field(default=None)
    slippage_bps: Optional[float] = Field(default=None)
    cost_adjusted_return: Optional[float] = Field(default=None)
    hit: Optional[bool] = Field(default=None)
    simulated_exit_reason: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)
    evaluated_at: datetime = Field(default_factory=utcnow)


class FeatureSnapshot(SQLModel, table=True):
    """Daily short-horizon features per symbol as of a session's close."""
    __tablename__ = "v2_daily_features"
    __table_args__ = (UniqueConstraint("symbol", "session_date", "feature_set_version", name="uq_v2_features_key"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    session_date: str = Field(index=True)
    feature_set_version: str
    values: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    flags: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    data_cutoff_at: datetime
    computed_at: datetime = Field(default_factory=utcnow)


class FactorObservation(SQLModel, table=True):
    """Index, sector-basket and macro/commodity observations (e.g. ^NSEI,
    SECTOR:Energy, BZ=F) with the time they became available."""
    __tablename__ = "factor_observations"
    __table_args__ = (UniqueConstraint("key", "session_date", name="uq_factor_key_date"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(index=True)
    session_date: str = Field(index=True)
    close: Optional[float] = Field(default=None)
    return_1d: Optional[float] = Field(default=None)
    source: str
    available_at: datetime


class MarketEvent(SQLModel, table=True):
    """Raw, source-attributed event (filing, results, broker action, ...).
    Derived labels live in EventClassification, never here."""
    __tablename__ = "market_events"
    __table_args__ = (UniqueConstraint("source", "source_event_id", name="uq_market_events_source_id"),
                      Index("ix_market_events_symbol_available", "symbol", "effective_available_at"))

    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    source_event_id: str
    symbol: Optional[str] = Field(default=None, index=True)
    event_type: str = Field(index=True)              # taxonomy, e.g. BUSINESS_UPDATE, RE_RATING
    title: str
    published_at: datetime
    ingested_at: datetime = Field(default_factory=utcnow)
    effective_available_at: datetime                 # max(published_at, ingested_at)
    source_reference: Optional[str] = Field(default=None)
    raw: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class EventClassification(SQLModel, table=True):
    """Versioned, reviewable classification of an event. Corrections add a
    new row; originals are kept."""
    __tablename__ = "event_classifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="market_events.id", index=True)
    classifier_version: str
    direction: Optional[str] = Field(default=None)   # POSITIVE | NEGATIVE | MIXED | UNKNOWN
    severity: Optional[str] = Field(default=None)
    credibility: Optional[str] = Field(default=None)
    expected_value: Optional[float] = Field(default=None)
    actual_value: Optional[float] = Field(default=None)
    surprise_score: Optional[float] = Field(default=None)
    review_status: str = Field(default="UNREVIEWED")
    reviewed_by: Optional[int] = Field(default=None, foreign_key="users.id")
    reviewed_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None)
    supersedes_id: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    # News intelligence (catalysts/, additive, all nullable): derived labels only.
    category: Optional[str] = Field(default=None)        # catalysts.taxonomy.CATEGORIES
    sentiment: Optional[float] = Field(default=None)     # -1 .. +1, lexicon score of the text
    materiality: Optional[float] = Field(default=None)   # 0 .. 1
    novelty: Optional[str] = Field(default=None)         # NEW | FOLLOW_UP | REPEAT
    expectedness: Optional[str] = Field(default=None)    # UNEXPECTED | EXPECTED | UNKNOWN (from the text only)
    contradicted: Optional[bool] = Field(default=None)   # sources disagree on direction
    impact_sessions: Optional[int] = Field(default=None) # how long the catalyst is considered live
    expires_at: Optional[datetime] = Field(default=None)


class ExitState(SQLModel, table=True):
    """Current shadow exit state of a directional prediction."""
    __tablename__ = "exit_states"

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(foreign_key="predictions.prediction_id", unique=True)
    state: str = Field(default="HOLD")               # EXIT_STATES
    reason: Optional[str] = Field(default=None)
    rule_version: str
    stop_level: Optional[float] = Field(default=None)
    target_level: Optional[float] = Field(default=None)
    peak_price: Optional[float] = Field(default=None)
    last_session_date: Optional[str] = Field(default=None)
    updated_at: datetime = Field(default_factory=utcnow)


class ExitTransition(SQLModel, table=True):
    __tablename__ = "exit_transitions"
    __table_args__ = (UniqueConstraint("prediction_id", "session_date", "to_state", name="uq_exit_transition"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(foreign_key="predictions.prediction_id", index=True)
    session_date: str
    from_state: str
    to_state: str
    reason: str
    price: Optional[float] = Field(default=None)
    rule_version: str
    created_at: datetime = Field(default_factory=utcnow)
