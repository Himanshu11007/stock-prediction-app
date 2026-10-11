"""
catalysts/pipeline.py — ingest provider articles into canonical news events.

For each article (oldest first):
  1. sanitise title / excerpt; detect language; canonical URL; source tier
  2. duplicates: same provider id, or the same canonical URL from any provider
     -> skipped (counted); never stored twice
  3. clustering: an English article joins an existing news event when its
     title is similar (Jaccard >= CLUSTER_SIMILARITY) and it was published
     within CLUSTER_WINDOW of the event; otherwise it starts a new event.
     A later article never moves an event's effective availability earlier.
  4. classification (versioned, catalysts.taxonomy): a new row whenever the
     event's assessment changes (the previous row is superseded, kept);
     contradiction = articles in the cluster disagree in sentiment
  5. factual entities from the text; inferred exposures from the
     transmission hypotheses (relation EXPOSED, inferred = True) and peer
     read-through for company events (H-PEER-READTHROUGH, unvalidated)
  6. a news_ingestion_runs row records counts, status and errors

Non-English articles are stored for provenance but not clustered, classified
or used (language "unknown").
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

import masters.service as masters
from catalysts import CLASSIFIER_VERSION, entities as ent, sources, taxonomy, text as tx, transmission
from catalysts.providers import NotConfigured, ProviderError, RateLimited, RawArticle
from db.models.news import EventEntity, NewsArticle, NewsIngestionRun
from db.models.prediction import EventClassification, MarketEvent
from db.models.stock import Company
from prediction_v2 import calendar
from utils.logger import get_logger
from utils.market_session import now_ist

logger = get_logger(__name__)

CLUSTER_SIMILARITY = 0.5
CLUSTER_WINDOW = dt.timedelta(hours=48)
PEER_SIGN = 0.3                        # weight of a peer read-through relative to the subject
NEWS_SOURCE = "news"


def _utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _universe(session: Session) -> list[Company]:
    from prediction_v2.universe import members
    comps = members(session)
    return comps or list(session.exec(select(Company)).all())


def _direction(sentiment: float) -> str:
    return "POSITIVE" if sentiment >= 0.2 else "NEGATIVE" if sentiment <= -0.2 else "UNKNOWN"


LATE_INGESTION = dt.timedelta(hours=6)


def quality_flags(a: RawArticle, published: dt.datetime, ingested: dt.datetime, lang: str, excerpt: str,
                  domain: str, universe: set[str]) -> list[str]:
    """Data-quality facts about one article (stored with it; never used to
    invent values)."""
    flags = []
    if a.time_precision == "date":
        flags.append("DATE_ONLY_TIMESTAMP")
    if (a.raw or {}).get("published_is_first_seen"):
        flags.append("PUBLISHED_TIME_IS_FIRST_SEEN")
    link = tx.domain(a.url)
    if domain and link and not (link == domain or link.endswith("." + domain) or domain.endswith("." + link)):
        flags.append("AGGREGATOR_LINK")
    if ingested - published > LATE_INGESTION:
        flags.append("LATE_INGESTION")
    if lang != "en":
        flags.append("NON_ENGLISH")
    if not excerpt:
        flags.append("NO_EXCERPT")
    if not (a.source_name or domain):
        flags.append("MISSING_SOURCE")
    if a.provider_entities and not any(s in universe for s in a.symbols):
        flags.append("PROVIDER_ENTITIES_UNMAPPED")
    return flags


def _expires(effective: dt.datetime, sessions: int, holidays: list[str]) -> dt.datetime:
    day = calendar.add_trading_days(now_ist(effective).date(), sessions, holidays)
    return dt.datetime.combine(day, dt.time(10, 0), tzinfo=dt.timezone.utc)      # 15:30 IST on that session


def assess(session: Session, event: MarketEvent, holidays: list[str],
           now: Optional[dt.datetime] = None) -> EventClassification:
    """(Re)classify an event from all its articles; supersede the previous
    classification only when something changed."""
    arts = session.exec(select(NewsArticle).where(NewsArticle.event_id == event.id)
                        .order_by(NewsArticle.effective_available_at)).all()
    first = arts[0]
    c = taxonomy.classify(first.title, first.excerpt or "")
    sentiments = [taxonomy.classify(a.title, a.excerpt or "").sentiment for a in arts]
    best_tier = min((a.source_tier for a in arts), key=lambda t: list(sources.CREDIBILITY).index(t))
    credibility = sources.CREDIBILITY[best_tier] * min(1.0, 0.85 + 0.05 * len({a.source_domain for a in arts}))
    contradicted = any(s >= 0.3 for s in sentiments) and any(s <= -0.3 for s in sentiments)
    mean_sent = sum(sentiments) / len(sentiments)
    mat = taxonomy.materiality(c, credibility)
    fields = dict(category=c.category, sentiment=round(mean_sent, 3), materiality=mat,
                  novelty="NEW" if len(arts) == 1 else "FOLLOW_UP", expectedness=c.expectedness,
                  contradicted=contradicted, impact_sessions=c.impact_sessions,
                  expires_at=_expires(_utc(event.effective_available_at), c.impact_sessions, holidays),
                  direction="MIXED" if contradicted else _direction(mean_sent),
                  severity="HIGH" if mat >= 0.6 else "MEDIUM" if mat >= 0.35 else "LOW", credibility=best_tier)
    prev = session.exec(select(EventClassification).where(EventClassification.event_id == event.id,
                                                          EventClassification.classifier_version == CLASSIFIER_VERSION)
                        .order_by(EventClassification.id.desc())).first()
    if prev is not None and all(getattr(prev, k) == v for k, v in fields.items() if k != "expires_at"):
        return prev
    row = EventClassification(event_id=event.id, classifier_version=CLASSIFIER_VERSION,
                              supersedes_id=prev.id if prev else None,
                              created_at=_utc(now or dt.datetime.now(dt.timezone.utc)),
                              notes=f"{len(arts)} article(s); matched: {', '.join(c.matched)[:200]}", **fields)
    session.add(row)
    session.flush()
    return row


def _link_entities(session: Session, event: MarketEvent, art: NewsArticle, category: str, scope: str,
                   index: ent.CompanyIndex, universe: list[Company], now: dt.datetime,
                   provider_symbols: tuple[str, ...] = ()) -> None:
    body = f"{art.title}. {art.excerpt or ''}"
    found = ent.extract(body, index)
    title_companies = {e.symbol for e in index.find(art.title)}
    known = {c.symbol for c in universe}
    rows: list[EventEntity] = []
    for sym in provider_symbols:                    # e.g. the company that filed with the exchange: a fact
        if sym in known:
            rows.append(EventEntity(event_id=event.id, entity_type="COMPANY", entity_key=sym, symbol=sym,
                                    relation="SUBJECT", method="PROVIDER", confidence=1.0,
                                    evidence={"source": "provider_tag", "matched": sym, "article_id": art.id}))
            title_companies.add(sym)
            scope = "COMPANY"
    for e in found:
        relation = "SUBJECT" if e.symbol and e.symbol in title_companies and scope == "COMPANY" else "MENTIONED"
        in_title = bool(e.matched) and e.matched.lower() in art.title.lower()
        rows.append(EventEntity(event_id=event.id, entity_type=e.entity_type, entity_key=e.entity_key, symbol=e.symbol,
                                relation=relation, method="DICTIONARY", confidence=1.0,
                                evidence={"source": "title" if in_title else "excerpt", "matched": e.matched[:80],
                                          "article_id": art.id}))
    for x in transmission.exposures(category, body, [(c.symbol, c.sector, c.industry) for c in universe]):
        rows.append(EventEntity(event_id=event.id, entity_type="COMPANY", entity_key=x.symbol, symbol=x.symbol,
                                relation="EXPOSED", inferred=True, method="HYPOTHESIS", hypothesis_id=x.hypothesis_id,
                                sign=x.sign, mechanism=f"{x.mechanism} [{x.status}]", confidence=x.weight,
                                evidence={"source": "hypothesis", "hypothesis_id": x.hypothesis_id,
                                          "status": x.status, "article_id": art.id}))
    subjects = {r.symbol for r in rows if r.relation == "SUBJECT"}
    if scope == "COMPANY" and subjects:
        industry = {c.symbol: c.industry for c in universe}
        for s in subjects:
            for c in universe:
                if c.symbol not in subjects and industry.get(s) and c.industry == industry.get(s):
                    rows.append(EventEntity(event_id=event.id, entity_type="COMPANY", entity_key=c.symbol,
                                            symbol=c.symbol, relation="EXPOSED", inferred=True, method="HYPOTHESIS",
                                            hypothesis_id="H-PEER-READTHROUGH", sign=None,
                                            mechanism=f"same industry as {s} ({c.industry}); read-through "
                                                      f"[UNVALIDATED]", confidence=PEER_SIGN * 0.5,
                                            evidence={"source": "industry", "peer_of": s, "industry": c.industry,
                                                      "article_id": art.id}))
    for r in rows:
        exists = session.exec(select(EventEntity.id).where(
            EventEntity.event_id == r.event_id, EventEntity.entity_type == r.entity_type,
            EventEntity.entity_key == r.entity_key, EventEntity.relation == r.relation,
            EventEntity.hypothesis_id == r.hypothesis_id)).first()
        if not exists:
            r.created_at = now                          # ingestion clock, so replays stay point-in-time
            session.add(r)
    if subjects and event.symbol is None and len(subjects) == 1:
        event.symbol = next(iter(subjects))
        session.add(event)


ENTITY_WINDOW = dt.timedelta(hours=24)
# Which factual entities identify "the same story" for each category. Different
# publishers word the same event differently, so titles alone under-merge
# (measured: about 53 events per real event before this rule).
CLUSTER_KEYS: dict[str, tuple[str, ...]] = {
    "COMMODITY": ("COMMODITY",), "MONETARY_POLICY": ("INSTITUTION",), "FLOWS_CURRENCY": ("CURRENCY",),
    "ECONOMIC_DATA": ("COUNTRY", "INSTITUTION"), "TRADE_TARIFF": ("COUNTRY",), "GEOPOLITICAL": ("COUNTRY",),
    "FISCAL_POLICY": ("COUNTRY", "INSTITUTION"), "REGULATORY": ("INSTITUTION",),
}
BROAD_KEYS = {"COUNTRY:IN", "COUNTRY:US"}       # too common on their own to identify a story


def cluster_keys(category: str, scope: str, entities: list, subjects: set[str]) -> list[str]:
    if scope == "COMPANY" and subjects:
        return sorted(f"COMPANY:{s}" for s in subjects)
    types = CLUSTER_KEYS.get(category, ())
    keys = {f"{e.entity_type}:{e.entity_key}" for e in entities if e.entity_type in types}
    specific = keys - BROAD_KEYS
    return sorted(specific or keys)


def _cluster(session: Session, art: NewsArticle, category: str, subjects: set[str],
             keys: Optional[list[str]] = None) -> Optional[MarketEvent]:
    """Same story when, in the same category: the titles are similar (Jaccard
    >= CLUSTER_SIMILARITY) within 48 h, or the identifying entities overlap
    (same subject company; same commodity / institution / currency / country)
    within 24 h. Contradictory reports therefore join one event, where the
    contradiction is detected."""
    lo, hi = art.published_at - CLUSTER_WINDOW, art.published_at + CLUSTER_WINDOW
    cands = session.exec(select(MarketEvent).where(MarketEvent.source == NEWS_SOURCE, MarketEvent.published_at >= lo,
                                                   MarketEvent.published_at <= hi)).all()
    mine = set(keys or []) | {f"COMPANY:{s}" for s in subjects}
    best, score = None, 0.0
    for ev in cands:
        same_cat = (ev.raw or {}).get("category") in (category, None)
        if not same_cat:
            continue
        s = tx.similarity(ev.title, art.title)
        theirs = set((ev.raw or {}).get("keys") or []) | ({f"COMPANY:{ev.symbol}"} if ev.symbol else set())
        mine_co, theirs_co = {k for k in mine if k.startswith("COMPANY:")}, {k for k in theirs if k.startswith("COMPANY:")}
        if mine_co and theirs_co and not (mine_co & theirs_co):
            continue                              # different subject companies are never one story
        if mine and theirs and (mine & theirs) and abs(_utc(ev.published_at) - art.published_at) <= ENTITY_WINDOW:
            s = max(s, CLUSTER_SIMILARITY)
        if s > score:
            best, score = ev, s
    return best if score >= CLUSTER_SIMILARITY else None


def ingest(session: Session, provider, since: dt.datetime, until: dt.datetime,
           now: Optional[dt.datetime] = None) -> NewsIngestionRun:
    now = _utc(now or dt.datetime.now(dt.timezone.utc))
    run = NewsIngestionRun(provider=provider.name, window_since=since, window_until=until, started_at=now)
    session.add(run)
    session.commit()
    holidays = masters.get_config(session, "market.holidays") or []
    universe = _universe(session)
    index = ent.CompanyIndex([(c.symbol, c.name) for c in universe])
    try:
        articles: list[RawArticle] = sorted(provider.fetch(_utc(since), _utc(until)), key=lambda a: a.published_at)
    except NotConfigured as e:
        return _finish(session, run, "NOT_CONFIGURED", str(e))
    except RateLimited as e:
        return _finish(session, run, "RATE_LIMITED", str(e))
    except ProviderError as e:
        return _finish(session, run, "FAILED", str(e))
    touched: set[int] = set()
    for a in articles:
        run.fetched += 1
        title = tx.sanitize(a.title, tx.MAX_TITLE)
        if not title or not a.url or a.published_at is None:
            run.rejected += 1
            continue
        canon = tx.canonical_url(a.url)
        if session.exec(select(NewsArticle.id).where(NewsArticle.provider == provider.name,
                                                     NewsArticle.provider_article_id == a.provider_article_id)).first() \
                or session.exec(select(NewsArticle.id).where(NewsArticle.canonical_url == canon)).first():
            run.duplicates += 1
            continue
        published = _utc(a.published_at)
        if published > now + dt.timedelta(minutes=10):
            run.rejected += 1                               # future-dated: provider clock or data error
            continue
        excerpt = tx.sanitize(a.excerpt)
        dom = a.source_domain or tx.domain(a.url)      # the publisher, not an aggregator's redirect link
        tier = sources.tier(dom)
        lang = tx.language(f"{title}. {excerpt}")
        art = NewsArticle(provider=provider.name, provider_article_id=a.provider_article_id[:300], url=a.url[:1000],
                          canonical_url=canon[:1000], title=title, excerpt=excerpt or None, language=lang,
                          source_name=(a.source_name or "")[:200] or None, source_domain=dom or None,
                          source_tier=tier, is_primary_source=tier == "PRIMARY", published_at=published,
                          ingested_at=now, corrected_at=_utc(a.corrected_at) if a.corrected_at else None,
                          effective_available_at=max(published, now), fingerprint=tx.fingerprint(title),
                          raw={"time_precision": a.time_precision, "provider_symbols": list(a.symbols),
                               **{k: v for k, v in (a.raw or {}).items() if isinstance(v, (str, int, float, bool))}},
                          first_seen_at=_utc(a.first_seen_at) if a.first_seen_at else None,
                          provider_entities=[dict(e) for e in a.provider_entities][:20] or None,
                          published_raw=(a.published_raw or None) and a.published_raw[:80],
                          published_tz=(a.published_tz or None) and a.published_tz[:40],
                          quality_flags=quality_flags(a, published, now, lang, excerpt, dom,
                                                      {u.symbol for u in universe}))
        try:
            with session.begin_nested():                    # savepoint: a concurrent duplicate only undoes this one
                session.add(art)
        except IntegrityError:
            run.duplicates += 1
            continue
        run.inserted += 1
        if lang != "en":
            continue
        c = taxonomy.classify(title, excerpt)
        art.category = c.category
        subjects = {e.symbol for e in index.find(title)} if c.scope == "COMPANY" else set()
        subjects |= {s for s in a.symbols if s in {u.symbol for u in universe}}
        keys = cluster_keys(c.category, c.scope, ent.extract(f"{title}. {excerpt}", index), subjects)
        event = _cluster(session, art, c.category, subjects, keys)
        if event is None:
            event = MarketEvent(source=NEWS_SOURCE, source_event_id=f"news:{art.fingerprint}:{published:%Y%m%d%H%M}",
                                symbol=None, event_type=c.event_type, title=title, published_at=published,
                                ingested_at=now, effective_available_at=max(published, now),
                                source_reference=a.url[:1000], raw={"category": c.category, "scope": c.scope,
                                                                    "keys": keys})
            session.add(event)
            session.flush()
            run.new_events += 1
        elif published < _utc(event.published_at):
            event.published_at = published                  # earliest report; availability is never moved earlier
            session.add(event)
        art.event_id = event.id
        session.add(art)
        _link_entities(session, event, art, c.category, c.scope, index, universe, now, tuple(a.symbols))
        touched.add(event.id)
    for eid in touched:
        assess(session, session.get(MarketEvent, eid), holidays, now)
    errors = getattr(provider, "errors", [])
    session.commit()
    return _finish(session, run, "PARTIAL" if errors else "COMPLETED", "; ".join(errors)[:500] or None)


def _finish(session: Session, run: NewsIngestionRun, status: str, error: Optional[str]) -> NewsIngestionRun:
    run.status, run.error, run.finished_at = status, error, dt.datetime.now(dt.timezone.utc)
    session.add(run)
    session.commit()
    session.refresh(run)
    if status not in ("COMPLETED",):
        logger.warning("NEWS_INGESTION | %s | %s | %s", run.provider, status, (error or "")[:200])
    return run
