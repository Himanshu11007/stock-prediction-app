"""Market, fundamental and analysis-engine master data.

Everything the product analysis engine reads or produces lives here, so the
backend is the single source of truth for what the mobile app displays:

  Sector / Industry           classification masters (sector outlook is an
                              admin-controlled input to FQVF check 18)
  FundamentalSnapshot         one provider fetch of fundamentals (history kept)
  MarketSnapshot              latest price bar + technical/risk metrics per fetch
  MarketRegimeSnapshot        market-wide regime (NIFTY 50) per computation
  EngineRun                   one controlled analysis run (counts, errors)
  StockAnalysisResult         FQVF + StockAI Score for one stock in one run
  AppConfig                   key/value application configuration

No row ever holds a fabricated value: missing data is stored as NULL with a
status and reason. Snapshots are append-only; "latest" is the newest row.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Classification masters ───────────────────────────────────────────────────

SECTOR_OUTLOOKS = ("POSITIVE", "NEUTRAL", "NEGATIVE")


class Sector(SQLModel, table=True):
    __tablename__ = "sectors"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    active: bool = Field(default=True)
    # Admin-controlled outlook (FQVF check 18). NULL = not assessed.
    outlook: Optional[str] = Field(default=None)
    outlook_notes: Optional[str] = Field(default=None)
    outlook_updated_at: Optional[datetime] = Field(default=None)
    outlook_updated_by: Optional[int] = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Industry(SQLModel, table=True):
    __tablename__ = "industries"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    sector_id: Optional[int] = Field(default=None, foreign_key="sectors.id", index=True)
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Provider snapshots ───────────────────────────────────────────────────────

DATA_STATUSES = ("OK", "PARTIAL", "STALE", "UNAVAILABLE", "ERROR")


class FundamentalSnapshot(SQLModel, table=True):
    """One fetch of a symbol's fundamentals. `data` holds the normalised
    fields and annual series (see fundamentals/provider.py); each value is
    either provider-reported or NULL."""

    __tablename__ = "fundamental_snapshots"
    __table_args__ = (Index("ix_fundamental_snapshots_symbol_fetched", "symbol", "fetched_at"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    fetched_at: datetime = Field(default_factory=utcnow)
    source: str = Field(default="yfinance")
    status: str = Field(default="OK")
    error: Optional[str] = Field(default=None)
    fiscal_period_end: Optional[str] = Field(default=None)
    data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class MarketSnapshot(SQLModel, table=True):
    """Latest daily bar plus technical/risk metrics computed from the
    1-year price history available at fetch time."""

    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_market_snapshots_symbol_fetched", "symbol", "fetched_at"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    fetched_at: datetime = Field(default_factory=utcnow)
    source: str = Field(default="yfinance")
    status: str = Field(default="OK")
    error: Optional[str] = Field(default=None)
    as_of_date: Optional[str] = Field(default=None)
    close: Optional[float] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    bars: Optional[int] = Field(default=None)
    technical: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    issues: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))


class MarketRegimeSnapshot(SQLModel, table=True):
    __tablename__ = "market_regime_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    index_symbol: str = Field(default="^NSEI")
    computed_at: datetime = Field(default_factory=utcnow, index=True)
    as_of_date: Optional[str] = Field(default=None)
    status: str = Field(default="OK")
    regime: Optional[str] = Field(default=None)
    regime_score: Optional[float] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    run_id: Optional[str] = Field(default=None, index=True)


# ── Engine runs and results ──────────────────────────────────────────────────

RUN_STATUSES = ("RUNNING", "COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED")


class EngineRun(SQLModel, table=True):
    __tablename__ = "engine_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    kind: str = Field(default="RANKING")
    status: str = Field(default="RUNNING", index=True)
    started_at: datetime = Field(default_factory=utcnow, index=True)
    finished_at: Optional[datetime] = Field(default=None)
    triggered_by: Optional[int] = Field(default=None, foreign_key="users.id")
    engine_version: str
    fqvf_version: str
    config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    total: int = Field(default=0)
    processed: int = Field(default=0)
    succeeded: int = Field(default=0)
    skipped: int = Field(default=0)
    failed: int = Field(default=0)
    errors: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))


class StockAnalysisResult(SQLModel, table=True):
    __tablename__ = "stock_analysis_results"
    __table_args__ = (UniqueConstraint("run_id", "symbol", name="uq_analysis_run_symbol"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="engine_runs.run_id", index=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    computed_at: datetime = Field(default_factory=utcnow)
    stockai_score: Optional[float] = Field(default=None, index=True)
    score_coverage: Optional[float] = Field(default=None)
    eligible: bool = Field(default=False, index=True)
    ineligible_reasons: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    rank: Optional[int] = Field(default=None)
    components: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    fqvf: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    positives: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    risks: Optional[list[Any]] = Field(default=None, sa_column=Column(JSON))
    freshness: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    ml_signal: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    engine_version: str
    fqvf_version: str


# ── Prospective ranking tracking (docs/RANKING_VALIDATION_V1.md) ─────────────
# Append-only. A RankingSnapshot is written once per stock when a RANKING run
# completes and is never modified; a RankingOutcome is inserted only after its
# horizon has fully elapsed and is never recomputed or overwritten.

class RankingSnapshot(SQLModel, table=True):
    __tablename__ = "ranking_snapshots"
    __table_args__ = (UniqueConstraint("run_id", "symbol", name="uq_ranking_snapshot_run_symbol"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="engine_runs.run_id", index=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    ranked_at: datetime = Field(default_factory=utcnow, index=True)
    engine_version: str
    fqvf_version: str
    stockai_score: Optional[float] = Field(default=None)
    score_coverage: Optional[float] = Field(default=None)
    eligible: bool = Field(default=False)
    rank: Optional[int] = Field(default=None)
    fqvf_summary: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    component_scores: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    freshness: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    market_regime: Optional[str] = Field(default=None)
    reference_date: Optional[str] = Field(default=None)
    reference_price: Optional[float] = Field(default=None)
    benchmark_symbol: str = Field(default="^NSEI")
    benchmark_reference_price: Optional[float] = Field(default=None)


class RankingOutcome(SQLModel, table=True):
    __tablename__ = "ranking_outcomes"
    __table_args__ = (UniqueConstraint("snapshot_id", "horizon", name="uq_ranking_outcome_snapshot_horizon"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    snapshot_id: int = Field(foreign_key="ranking_snapshots.id", index=True)
    horizon: str                       # 1M / 3M / 6M / 12M (21 / 63 / 126 / 252 sessions)
    start_date: str
    start_price: float
    outcome_date: str
    outcome_price: float
    stock_return: float
    benchmark_return: Optional[float] = Field(default=None)
    excess_return: Optional[float] = Field(default=None)
    computed_at: datetime = Field(default_factory=utcnow)


class AppConfig(SQLModel, table=True):
    __tablename__ = "app_config"

    key: str = Field(primary_key=True)
    value: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    description: Optional[str] = Field(default=None)
    updated_at: datetime = Field(default_factory=utcnow)
    updated_by: Optional[int] = Field(default=None, foreign_key="users.id")
