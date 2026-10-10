"""
db/models/news.py — news and catalyst intelligence for Prediction Engine v2
(additive; Ranking v1 never reads these tables).

Layers (raw evidence is never mixed with inference):
  news_articles        one row per provider article: source attribution,
                       timestamps, a short excerpt only (licensing), a content
                       fingerprint. Each article belongs to one canonical
                       event (market_events row, source "news").
  market_events        (existing) the canonical event = a cluster of articles
                       reporting the same thing; first publication and the
                       time StockLens could first use it.
  event_entities       what an event is about. FACTUAL rows come from the text
                       (company, sector, country, currency, commodity,
                       institution). INFERRED rows come from a versioned
                       transmission hypothesis and carry its id, sign and
                       mechanism - they are explanations to test, not facts.
  event_classifications (existing, extended) versioned derived labels:
                       category, sentiment, materiality, novelty,
                       expectedness, contradiction, expiry.
  news_ingestion_runs  provider runs: counts, status, errors, rate limiting.

Point in time: an article / event is usable at T only when
effective_available_at = max(published_at, ingested_at) <= T. Late ingestion
is never backdated.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


ENTITY_TYPES = ("COMPANY", "SECTOR", "INDUSTRY", "COUNTRY", "CURRENCY", "COMMODITY", "INSTITUTION", "INDEX")
SOURCE_TIERS = ("PRIMARY", "TIER1", "TIER2", "TIER3", "UNKNOWN")
INGESTION_STATUSES = ("RUNNING", "COMPLETED", "PARTIAL", "FAILED", "RATE_LIMITED", "NOT_CONFIGURED")


class NewsArticle(SQLModel, table=True):
    __tablename__ = "news_articles"
    __table_args__ = (UniqueConstraint("provider", "provider_article_id", name="uq_news_articles_provider_id"),
                      Index("ix_news_articles_event", "event_id"),
                      Index("ix_news_articles_available", "effective_available_at"))

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    provider_article_id: str
    url: str
    canonical_url: str = Field(index=True)
    title: str
    excerpt: Optional[str] = Field(default=None)      # <= 500 chars of provider text; never full articles
    language: str = Field(default="en")
    source_name: Optional[str] = Field(default=None)
    source_domain: Optional[str] = Field(default=None, index=True)
    source_tier: str = Field(default="UNKNOWN")       # SOURCE_TIERS
    is_primary_source: bool = Field(default=False)    # regulator / central bank / exchange / company itself
    published_at: datetime
    ingested_at: datetime = Field(default_factory=utcnow)
    corrected_at: Optional[datetime] = Field(default=None)
    effective_available_at: datetime                   # max(published_at, ingested_at)
    fingerprint: str = Field(index=True)               # normalised-title token signature
    event_id: Optional[int] = Field(default=None, foreign_key="market_events.id")
    duplicate_of_id: Optional[int] = Field(default=None)   # same story syndicated (another article id)
    raw: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class EventEntity(SQLModel, table=True):
    __tablename__ = "event_entities"
    __table_args__ = (UniqueConstraint("event_id", "entity_type", "entity_key", "relation", "hypothesis_id",
                                       name="uq_event_entities"),
                      Index("ix_event_entities_symbol", "symbol"))

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="market_events.id", index=True)
    entity_type: str                                  # ENTITY_TYPES
    entity_key: str                                   # e.g. CANBK.NS, Financial Services, CRUDE_OIL, USD, RBI
    symbol: Optional[str] = Field(default=None)       # set for companies (direct or inferred)
    relation: str = Field(default="MENTIONED")        # MENTIONED | SUBJECT | EXPOSED (inferred)
    inferred: bool = Field(default=False)
    method: str = Field(default="DICTIONARY")         # DICTIONARY | HYPOTHESIS | REVIEW
    hypothesis_id: str = Field(default="")            # transmission hypothesis id for inferred rows
    sign: Optional[int] = Field(default=None)         # inferred direction for this entity: +1 / -1 / 0 (ambiguous)
    mechanism: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class NewsIngestionRun(SQLModel, table=True):
    __tablename__ = "news_ingestion_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    status: str = Field(default="RUNNING")            # INGESTION_STATUSES
    window_since: Optional[datetime] = Field(default=None)
    window_until: Optional[datetime] = Field(default=None)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = Field(default=None)
    fetched: int = Field(default=0)
    inserted: int = Field(default=0)
    duplicates: int = Field(default=0)
    new_events: int = Field(default=0)
    rejected: int = Field(default=0)
    error: Optional[str] = Field(default=None)
