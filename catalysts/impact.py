"""
catalysts/impact.py — point-in-time news assessment of one stock, and its
combination with the technical baseline (rule news-v0.1, SHADOW).

Evidence for `symbol` at cutoff T comes only from events with
  effective_available_at <= T  (ingested and usable by then; never backdated)
  expires_at > T               (catalyst still live)
  classification created <= T  (no later re-labelling leaks in)
  entity link created <= T

Each piece of evidence contributes sign x weight:
  SUBJECT     the company is the subject: sign from the text's sentiment
              (|s| >= 0.2), weight = materiality x novelty factor
  MENTIONED   named but not the subject: weight x 0.3
  EXPOSED     inferred through a transmission hypothesis: sign from the
              hypothesis x driver direction, weight = materiality x status
              weight (SUPPORTED 1.0, UNVALIDATED 0.5, INCONCLUSIVE 0.3)
  PEER        industry read-through of a peer's company event: x0.15
  contradicted events contribute 0; ambiguous (sign 0) contribute 0
  PRICED_IN   if the stock has already moved >= 1.5 ATR in the evidence's
              direction (excess over NIFTY) since availability: x0.25

Decision (never forced): a direction only when |score| >= MIN_SCORE and at
least one non-contradicted contribution comes from a credible source
(PRIMARY / TIER1 / TIER2) with weight >= MIN_SINGLE. Otherwise NEUTRAL
with the reason. Missing price data stays NO_CALL (news cannot repair it).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sqlmodel import Session, select

from catalysts import CLASSIFIER_VERSION, NEWS_RULE_VERSION
from catalysts.transmission import TRANSMISSION_VERSION
from db.models.news import EventEntity, NewsArticle
from db.models.prediction import EventClassification, MarketEvent

MIN_SCORE = 0.35
MIN_SINGLE = 0.25
PRICED_IN_ATR = 1.5
CREDIBLE = ("PRIMARY", "TIER1", "TIER2")
NOVELTY = {"NEW": 1.0, "FOLLOW_UP": 0.7, "REPEAT": 0.3}


def _utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


@dataclass
class Evidence:
    event_id: int
    title: str
    category: Optional[str]
    relation: str
    inferred: bool
    sign: int
    weight: float
    available_at: str
    source_tier: str
    sources: list[str]
    hypothesis_id: str = ""
    mechanism: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    sentiment: Optional[float] = None          # text sentiment of the event (-1..1); not a price forecast
    novelty: Optional[str] = None              # NEW | FOLLOW_UP | REPEAT
    expectedness: Optional[str] = None         # UNEXPECTED | EXPECTED | UNKNOWN (stated by the source text)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Assessment:
    symbol: str
    as_of: str
    score: float = 0.0
    direction: Optional[str] = None            # UP / DOWN when justified, else None
    status: str = "NO_NEWS"                     # NO_NEWS | NEUTRAL | DIRECTIONAL
    reasons: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        ev = [e for e in self.evidence if e.weight > 0]
        w = sum(e.weight for e in ev)
        return {"symbol": self.symbol, "as_of": self.as_of, "score": round(self.score, 3), "direction": self.direction,
                "status": self.status, "reasons": self.reasons, "evidence": [e.as_dict() for e in self.evidence],
                # kept separate on purpose: sentiment of the text != surprise != predicted price direction
                "news_sentiment": round(sum((e.sentiment or 0) * e.weight for e in ev) / w, 3) if w else None,
                "surprise": "UNEXPECTED" if any(e.expectedness == "UNEXPECTED" for e in ev) else
                            "EXPECTED" if ev and all(e.expectedness == "EXPECTED" for e in ev) else
                            ("UNKNOWN" if ev else None),
                "novelty": "NEW" if any(e.novelty == "NEW" for e in ev) else (ev[0].novelty if ev else None),
                "confidence": None, "confidence_note": "not calibrated: the score is evidence strength, not a probability",
                "rule_version": NEWS_RULE_VERSION, "classifier_version": CLASSIFIER_VERSION,
                "transmission_version": TRANSMISSION_VERSION}


def _classification(session: Session, event_id: int, as_of: dt.datetime) -> Optional[EventClassification]:
    rows = session.exec(select(EventClassification).where(EventClassification.event_id == event_id,
                                                          EventClassification.classifier_version == CLASSIFIER_VERSION)
                        .order_by(EventClassification.id.desc())).all()
    for r in rows:
        if _utc(r.created_at) <= as_of:
            return r
    return None


def assess_stock(session: Session, symbol: str, as_of: dt.datetime,
                 price_move: Optional[Callable[[dt.datetime], Optional[tuple[float, float]]]] = None) -> Assessment:
    """`price_move(since)` -> (excess return over NIFTY since `since`, atr_pct)
    measured up to the cutoff, or None when unknown."""
    as_of = _utc(as_of)
    out = Assessment(symbol=symbol, as_of=as_of.isoformat())
    links = session.exec(select(EventEntity).where(EventEntity.symbol == symbol)).all()
    by_event: dict[int, list[EventEntity]] = {}
    for link in links:
        if _utc(link.created_at) <= as_of:
            by_event.setdefault(link.event_id, []).append(link)
    for event_id, ents in by_event.items():
        ev = session.get(MarketEvent, event_id)
        if ev is None or _utc(ev.effective_available_at) > as_of:
            continue
        cls = _classification(session, event_id, as_of)
        if cls is None or (cls.expires_at is not None and _utc(cls.expires_at) <= as_of):
            continue
        arts = session.exec(select(NewsArticle).where(NewsArticle.event_id == event_id,
                                                      NewsArticle.effective_available_at <= as_of)).all()
        if not arts:
            continue
        mat = cls.materiality or 0.0
        nov = NOVELTY.get(cls.novelty or "NEW", 1.0)
        # strongest relation first: SUBJECT > EXPOSED (hypothesis) > MENTIONED
        order = {"SUBJECT": 0, "EXPOSED": 1, "MENTIONED": 2}
        link = sorted(ents, key=lambda e: (order.get(e.relation, 3), -(e.confidence or 0)))[0]
        flags: list[str] = []
        sent = cls.sentiment or 0.0
        if link.relation == "SUBJECT":
            sign, weight = (1 if sent >= 0.2 else -1 if sent <= -0.2 else 0), mat * nov
        elif link.relation == "EXPOSED" and link.hypothesis_id == "H-PEER-READTHROUGH":
            sign, weight = (1 if sent >= 0.2 else -1 if sent <= -0.2 else 0), mat * nov * 0.15
            flags.append("PEER_READTHROUGH")
        elif link.relation == "EXPOSED":
            sign, weight = int(link.sign or 0), mat * nov * (link.confidence or 0.5)
            flags.append("INFERRED")
        else:
            sign, weight = (1 if sent >= 0.2 else -1 if sent <= -0.2 else 0), mat * nov * 0.3
            flags.append("MENTION_ONLY")
        if cls.contradicted:
            flags.append("CONTRADICTED")
            weight = 0.0
        if sign == 0:
            flags.append("AMBIGUOUS_OR_NO_DIRECTION")
        if price_move is not None and sign != 0 and weight > 0:
            pm = price_move(_utc(ev.effective_available_at))
            if pm is not None:
                excess, atr = pm
                if atr and sign * excess >= PRICED_IN_ATR * atr:
                    flags.append("PRICED_IN")
                    weight *= 0.25
        tier = min((a.source_tier for a in arts), key=lambda t: ["PRIMARY", "TIER1", "TIER2", "TIER3", "UNKNOWN"].index(t))
        out.evidence.append(Evidence(event_id=event_id, title=ev.title, category=cls.category, relation=link.relation,
                                     inferred=link.inferred, sign=sign, weight=round(weight, 3),
                                     available_at=_utc(ev.effective_available_at).isoformat(), source_tier=tier,
                                     sources=sorted({a.source_domain or a.provider for a in arts}),
                                     hypothesis_id=link.hypothesis_id, mechanism=link.mechanism, flags=flags,
                                     sentiment=cls.sentiment, novelty=cls.novelty, expectedness=cls.expectedness))
    if not out.evidence:
        out.reasons = ["no material news captured for this stock before the cutoff"]
        return out
    out.score = sum(e.sign * e.weight for e in out.evidence)
    credible = [e for e in out.evidence if e.sign != 0 and e.weight >= MIN_SINGLE and e.source_tier in CREDIBLE
                and "CONTRADICTED" not in e.flags and (e.sign > 0) == (out.score > 0)]
    strong = [e for e in out.evidence if e.sign != 0 and e.weight >= MIN_SINGLE and e.source_tier in CREDIBLE]
    conflicting = any(e.sign > 0 for e in strong) and any(e.sign < 0 for e in strong)
    if conflicting:
        for e in strong:
            e.flags.append("CONFLICTING_REPORTS")
        credible = []
    out.status = "NEUTRAL"
    if abs(out.score) >= MIN_SCORE and credible:
        out.direction, out.status = ("UP" if out.score > 0 else "DOWN"), "DIRECTIONAL"
        out.reasons = [f"news evidence {out.score:+.2f} from {len(out.evidence)} event(s); strongest: "
                       f"{max(credible, key=lambda e: e.weight).title[:120]}"]
    else:
        why = []
        if any("CONTRADICTED" in e.flags or "CONFLICTING_REPORTS" in e.flags for e in out.evidence):
            why.append("sources contradict each other")
        if any("PRICED_IN" in e.flags for e in out.evidence):
            why.append("the move already happened")
        if not credible:
            why.append("no credible, directional evidence")
        if abs(out.score) < MIN_SCORE:
            why.append(f"net evidence {out.score:+.2f} below {MIN_SCORE}")
        out.reasons = ["news present but not decisive: " + "; ".join(why)]
    return out


def combine(base: dict[str, Any], news: Assessment, features: dict[str, Any],
            call: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """News-first combination with the technical baseline decision `base`
    (prediction_v2.rules.decide output). `call(direction, setup, features,
    thresholds, reasons)` builds levels for a directional call."""
    if base["direction"] == "NO_CALL":
        return base                                             # data quality first
    if news.status == "DIRECTIONAL":
        tech = base["direction"]
        context = ("technical setup agrees" if tech == news.direction else
                   "no technical setup" if tech == "NEUTRAL" else f"technical setup disagrees ({tech})")
        out = call(news.direction, "NEWS_CATALYST", features, None, news.reasons + [context])
        out["quality_flags"] = sorted(set(out.get("quality_flags", []) + ["NEWS_DRIVEN"]))
        out["invalidation_condition"] = (f"{out['invalidation_condition']}; or a credible report contradicts or "
                                         f"reverses the catalyst")
        return out
    if news.status == "NEUTRAL" and base["direction"] in ("UP", "DOWN"):
        opposing = [e for e in news.evidence if e.sign != 0 and e.weight >= MIN_SINGLE
                    and (e.sign > 0) != (base["direction"] == "UP")]
        if opposing:
            return {**base, "direction": "NEUTRAL", "setup_type": "NEWS_CONFLICT", "stop_loss": None, "target": None,
                    "entry_condition": None, "trailing_stop_rule": None, "invalidation_condition": None,
                    "reasons": base["reasons"] + [f"opposing news: {opposing[0].title[:120]}"]}
    return base
