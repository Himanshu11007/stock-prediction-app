"""
events/ingest.py — provider-agnostic ingestion of timestamped events.

No production event provider is configured: no licensed corporate-filings,
broker-rating or consensus feed is available, and the exchanges' public
pages are not used because their terms restrict automated access. This
module provides the contract, idempotent persistence and point-in-time
queries; a provider is plugged in once a source is approved
(docs/PREDICTION_V2.md, "Event sources").

Point-in-time rule: an event is usable at time T only if
  effective_available_at = max(published_at, ingested_at) <= T
so backfilled history ingested today is NOT available to past snapshots,
unless the provider supplies a trustworthy original ingestion/publication
time and the operator marks the backfill as point-in-time (backfill=True).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db.models.prediction import EventClassification, MarketEvent

EVENT_TYPES = ("EARNINGS_SURPRISE", "BUSINESS_UPDATE", "RE_RATING", "DOWNGRADE", "GUIDANCE_CHANGE",
               "CORPORATE_ACTION", "REGULATORY_EVENT", "SECTOR_SHOCK", "MACRO_SHOCK", "COMMODITY_SHOCK", "OTHER")


@dataclass(frozen=True)
class RawEvent:
    source_event_id: str
    symbol: Optional[str]
    event_type: str
    title: str
    published_at: dt.datetime
    source_reference: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


class EventProvider(Protocol):
    name: str

    def fetch(self, since: dt.datetime, until: dt.datetime) -> Iterable[RawEvent]: ...


class NotConfiguredProvider:
    """Default: makes the absence of a provider explicit."""
    name = "none"

    def fetch(self, since, until):
        raise ProviderNotConfigured("No event provider is configured (see docs/PREDICTION_V2.md, 'Event sources').")


class ProviderNotConfigured(RuntimeError):
    pass


class FixtureProvider:
    """Deterministic provider reading a JSON list (tests, manual curation)."""

    def __init__(self, path: Path, name: str = "fixture"):
        self.path, self.name = Path(path), name

    def fetch(self, since, until):
        for e in json.loads(self.path.read_text(encoding="utf-8")):
            published = dt.datetime.fromisoformat(e["published_at"])
            if since <= published <= until:
                yield RawEvent(source_event_id=str(e["id"]), symbol=e.get("symbol"), event_type=e["event_type"],
                               title=e["title"], published_at=published, source_reference=e.get("source_reference"),
                               raw=e)


def _utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def ingest(session: Session, provider: EventProvider, since: dt.datetime, until: dt.datetime,
           now: Optional[dt.datetime] = None, backfill: bool = False) -> dict[str, int]:
    """Store new events; duplicates (same source + source_event_id) are
    skipped, so re-running is safe. `backfill` declares that the provider's
    publication time is also when the event was knowable (point-in-time
    history); otherwise availability is the ingestion time."""
    now = _utc(now or dt.datetime.now(dt.timezone.utc))
    out = {"fetched": 0, "inserted": 0, "duplicates": 0, "rejected": 0}
    for e in provider.fetch(since, until):
        out["fetched"] += 1
        if e.event_type not in EVENT_TYPES or not e.title or e.published_at is None:
            out["rejected"] += 1
            continue
        if session.exec(select(MarketEvent.id).where(MarketEvent.source == provider.name,
                                                     MarketEvent.source_event_id == e.source_event_id)).first():
            out["duplicates"] += 1
            continue
        published = _utc(e.published_at)
        ingested = published if backfill else now
        row = MarketEvent(source=provider.name, source_event_id=e.source_event_id,
                          symbol=e.symbol.upper() if e.symbol else None, event_type=e.event_type, title=e.title,
                          published_at=published, ingested_at=ingested,
                          effective_available_at=max(published, ingested), source_reference=e.source_reference,
                          raw=e.raw)
        session.add(row)
        try:
            session.commit()
            out["inserted"] += 1
        except IntegrityError:          # concurrent ingest of the same event
            session.rollback()
            out["duplicates"] += 1
    return out


def available_events(session: Session, symbol: str, as_of: dt.datetime,
                     lookback: dt.timedelta = dt.timedelta(days=5)) -> list[MarketEvent]:
    """Events for `symbol` that were knowable at `as_of` (point-in-time)."""
    as_of = _utc(as_of)
    rows = session.exec(select(MarketEvent).where(MarketEvent.symbol == symbol.upper())
                        .order_by(MarketEvent.effective_available_at)).all()
    return [r for r in rows if as_of - lookback <= _utc(r.effective_available_at) <= as_of]


def classify(session: Session, event: MarketEvent, *, classifier_version: str, direction: Optional[str] = None,
             severity: Optional[str] = None, credibility: Optional[str] = None, expected_value: Optional[float] = None,
             actual_value: Optional[float] = None, notes: Optional[str] = None) -> EventClassification:
    """Record a classification. Numbers are only stored when given by the
    source or a reviewer; a classifier must never invent them."""
    surprise = None
    if expected_value is not None and actual_value is not None and expected_value != 0:
        surprise = (actual_value - expected_value) / abs(expected_value)
    row = EventClassification(event_id=event.id, classifier_version=classifier_version, direction=direction,
                              severity=severity, credibility=credibility, expected_value=expected_value,
                              actual_value=actual_value, surprise_score=surprise, notes=notes)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def review(session: Session, classification: EventClassification, *, reviewer_id: int, status: str,
           corrected: Optional[dict[str, Any]] = None, notes: Optional[str] = None) -> EventClassification:
    """Review a classification. CORRECTED creates a new row that supersedes
    the original (the original is kept for audit)."""
    if status not in ("CONFIRMED", "CORRECTED", "REJECTED"):
        raise ValueError("status must be CONFIRMED, CORRECTED or REJECTED")
    now = dt.datetime.now(dt.timezone.utc)
    if status == "CORRECTED":
        fields = {k: getattr(classification, k) for k in ("direction", "severity", "credibility", "expected_value",
                                                          "actual_value")}
        fields.update(corrected or {})
        new = EventClassification(event_id=classification.event_id, classifier_version="manual-review",
                                  review_status="CORRECTED", reviewed_by=reviewer_id, reviewed_at=now,
                                  notes=notes, supersedes_id=classification.id, **fields)
        if new.expected_value not in (None, 0) and new.actual_value is not None:
            new.surprise_score = (new.actual_value - new.expected_value) / abs(new.expected_value)
        session.add(new)
        session.commit()
        session.refresh(new)
        return new
    classification.review_status, classification.reviewed_by, classification.reviewed_at = status, reviewer_id, now
    classification.notes = notes or classification.notes
    session.add(classification)
    session.commit()
    return classification
