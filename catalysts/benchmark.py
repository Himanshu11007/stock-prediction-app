"""
catalysts/benchmark.py — news-provider benchmark logic (research only).

For every ground-truth event (an independently verified, timestamped market
event: an NSE filing, an RBI / SEBI / PIB / White House / Fed release, a
Federal Register document, a Fed rate change, a crude / gold / rupee shock)
and every provider, the outcome is exactly one of:

  DETECTED                    a VERIFIED article exists in the event window
  MISSED                      the provider was queried successfully and has none
  UNTESTABLE_NO_HISTORY       the event is older than the provider's history
  UNTESTABLE_NOT_CONFIGURED   no credential: the provider was not queried
  SOURCE_OF_RECORD            the provider is the event's own official source
  OUTAGE / RATE_LIMITED       the query failed: not a miss, reported separately

Only DETECTED and MISSED count towards recall. Missing history is never a
pass.

Matching, per article title (and excerpt):
  candidate   the event's search pattern matches (e.g. the company alias)
  verified    every `must` pattern matches and the text is not a preview
  false match a candidate that is not verified (examples are reported)

Timing (exact-time events only; date-only events get no latency):
  delay              first verified exact-time article - official time
  market reaction    the official time if NSE is in session (09:15-15:30 IST,
                     trading day), else the next session's open (09:15 IST)
  actionable         first verified exact article <= reaction time + 15 min
Mapping (events with an expected Indian symbol):
  provider tags      the provider's own entity tags include the symbol
  stocklens          StockLens' entity extraction finds the symbol
  wrong              other Indian companies tagged as the subject
Duplicates: canonical events the StockLens pipeline creates from the verified
articles (1 = perfect grouping) and repeated URLs.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Iterable, Optional

from catalysts import entities as ent
from catalysts.providers import RawArticle
from prediction_v2.performance import wilson

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OPEN, CLOSE = dt.time(9, 15), dt.time(15, 30)
GRACE = dt.timedelta(minutes=15)
PREVIEW = re.compile(r"\b(preview|expect(ed|s)?|likely to|ahead of|estimates for|what to watch|may report|to announce)\b", re.I)
DETECTED, MISSED = "DETECTED", "MISSED"
NO_HISTORY, NOT_CONFIGURED, OF_RECORD = "UNTESTABLE_NO_HISTORY", "UNTESTABLE_NOT_CONFIGURED", "SOURCE_OF_RECORD"
OUTAGE, RATE_LIMITED = "OUTAGE", "RATE_LIMITED"
TESTED = (DETECTED, MISSED)


@dataclass
class Event:
    id: str
    kind: str                         # nse_filing, bse_filing, rbi, sebi, india_policy, us_trade_policy, fed, crude, gold, currency
    title: str
    official_time: dt.datetime        # aware
    precision: str                    # exact | date
    source: str                       # where the ground truth comes from
    evidence_url: Optional[str]
    query: str                        # provider search string
    candidate: str                    # regex: is the article about this subject at all?
    must: list[str]                   # regexes: does it report THIS event?
    expected_symbols: tuple[str, ...] = ()
    window_before: dt.timedelta = dt.timedelta(hours=6)
    window_after: dt.timedelta = dt.timedelta(days=2)
    indian_material: bool = True       # material for Indian equities by construction of the event set

    @property
    def window(self) -> tuple[dt.datetime, dt.datetime]:
        if self.precision == "date":
            day = self.official_time.date()
            return (dt.datetime.combine(day - dt.timedelta(days=1), dt.time(0), tzinfo=dt.timezone.utc),
                    dt.datetime.combine(day + dt.timedelta(days=1), dt.time(23, 59), tzinfo=dt.timezone.utc))
        return self.official_time - self.window_before, self.official_time + self.window_after


@dataclass
class Capability:
    name: str
    configured: bool
    max_age: Optional[dt.timedelta] = None       # None = no stated history limit
    live_since: Optional[dt.datetime] = None     # live-only feeds: oldest item available now
    of_record_kinds: tuple[str, ...] = ()        # kinds whose ground truth IS this provider


def market_reaction_time(t: dt.datetime, holidays: Iterable[str] = ()) -> dt.datetime:
    hol = set(holidays)
    local = t.astimezone(IST)

    def trading(d: dt.date) -> bool:
        return d.weekday() < 5 and d.isoformat() not in hol
    if trading(local.date()) and OPEN <= local.time() < CLOSE:
        return t
    d = local.date() if local.time() < OPEN and trading(local.date()) else local.date() + dt.timedelta(days=1)
    while not trading(d):
        d += dt.timedelta(days=1)
    return dt.datetime.combine(d, OPEN, tzinfo=IST).astimezone(dt.timezone.utc)


def testability(cap: Capability, ev: Event, now: dt.datetime) -> Optional[str]:
    if ev.kind in cap.of_record_kinds:
        return OF_RECORD
    if not cap.configured:
        return NOT_CONFIGURED
    if cap.max_age is not None and ev.window[0] < now - cap.max_age:
        return NO_HISTORY
    if cap.live_since is not None and ev.window[1] < cap.live_since:
        return NO_HISTORY
    return None


@dataclass
class Outcome:
    provider: str
    event_id: str
    kind: str
    status: str
    detail: Optional[str] = None
    candidates: int = 0
    verified: int = 0
    exact_share: Optional[float] = None
    delay_minutes: Optional[float] = None
    actionable: Optional[bool] = None
    first_title: Optional[str] = None
    false_matches: list[str] = field(default_factory=list)
    mapping_provider: Optional[bool] = None
    mapping_stocklens: Optional[bool] = None
    wrong_symbols: list[str] = field(default_factory=list)
    canonical_events: Optional[int] = None
    url_duplicates: int = 0


def _text(a: RawArticle) -> str:
    return f"{a.title}. {a.excerpt or ''}"


def evaluate(provider: str, ev: Event, items: list[RawArticle], index: Optional[ent.CompanyIndex] = None,
             holidays: Iterable[str] = (), canonical: Optional[Callable[[list[RawArticle]], int]] = None) -> Outcome:
    lo, hi = ev.window
    inside = [a for a in items if lo <= a.published_at <= hi]
    cand_re, must_re = re.compile(ev.candidate, re.I), [re.compile(m, re.I) for m in ev.must]
    candidates = [a for a in inside if cand_re.search(_text(a))]
    verified = [a for a in candidates if all(m.search(_text(a)) for m in must_re) and not PREVIEW.search(a.title)]
    out = Outcome(provider, ev.id, ev.kind, DETECTED if verified else MISSED, candidates=len(candidates),
                  verified=len(verified))
    out.false_matches = [a.title[:140] for a in candidates if a not in verified][:3]
    urls = [a.url for a in verified if a.url]
    out.url_duplicates = len(urls) - len(set(urls))
    if not verified:
        return out
    exact = [a for a in verified if a.time_precision == "exact"]
    out.exact_share = round(len(exact) / len(verified), 3)
    first = min(verified, key=lambda a: a.published_at)
    out.first_title = first.title[:140]
    if ev.precision == "exact" and exact:
        f = min(exact, key=lambda a: a.published_at)
        out.delay_minutes = round((f.published_at - ev.official_time).total_seconds() / 60, 1)
        out.actionable = f.published_at <= market_reaction_time(ev.official_time, holidays) + GRACE
    if ev.expected_symbols:
        exp = set(ev.expected_symbols)
        tagged = {s for a in verified for s in a.symbols}
        out.mapping_provider = bool(exp & tagged) if any(a.provider_entities for a in verified) else None
        if index is not None:
            found = {e.symbol for a in verified for e in index.find(a.title)}
            out.mapping_stocklens = bool(exp & found)
            out.wrong_symbols = sorted((found | tagged) - exp)[:5]
    if canonical is not None:
        out.canonical_events = canonical(verified)
    return out


def percentile(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    by_provider: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_provider.setdefault(o.provider, []).append(o)
    out = {}
    for prov, rows in sorted(by_provider.items()):
        tested = [o for o in rows if o.status in TESTED]
        det = [o for o in tested if o.status == DETECTED]
        delays = [o.delay_minutes for o in det if o.delay_minutes is not None]
        act = [o.actionable for o in det if o.actionable is not None]
        mp = [o.mapping_provider for o in det if o.mapping_provider is not None]
        ms = [o.mapping_stocklens for o in det if o.mapping_stocklens is not None]
        canon = [o.canonical_events for o in det if o.canonical_events]
        statuses: dict[str, int] = {}
        for o in rows:
            statuses[o.status] = statuses.get(o.status, 0) + 1
        kinds: dict[str, dict[str, int]] = {}
        for o in rows:
            k = kinds.setdefault(o.kind, {})
            k[o.status] = k.get(o.status, 0) + 1
        out[prov] = {
            "events": len(rows), "statuses": statuses, "tested": len(tested), "detected": len(det),
            "missed": len(tested) - len(det),
            "recall": round(len(det) / len(tested), 3) if tested else None,
            "recall_ci95": wilson(len(det), len(tested)) if tested else None,
            "latency_minutes": {"n": len(delays), "p10": percentile(delays, 0.1), "p50": percentile(delays, 0.5),
                                "p90": percentile(delays, 0.9), "max": max(delays) if delays else None},
            "actionable_before_market_reaction": {"n": len(act), "share": round(sum(act) / len(act), 3) if act else None},
            "exact_timestamp_share": round(sum(o.exact_share or 0 for o in det) / len(det), 3) if det else None,
            "mapping_accuracy_provider_tags": {"n": len(mp), "share": round(sum(mp) / len(mp), 3) if mp else None},
            "mapping_accuracy_stocklens": {"n": len(ms), "share": round(sum(ms) / len(ms), 3) if ms else None},
            "canonical_events_per_detected_event": round(sum(canon) / len(canon), 2) if canon else None,
            "url_duplicates": sum(o.url_duplicates for o in rows),
            "outages": statuses.get(OUTAGE, 0), "rate_limited": statuses.get(RATE_LIMITED, 0),
            "by_kind": kinds,
        }
    return out


def recommend(summary: dict[str, Any], licence_ok: dict[str, bool], min_tested: int = 20, min_recall: float = 0.8,
              max_p50_minutes: float = 60.0, min_actionable: float = 0.7) -> dict[str, Any]:
    """A provider is recommended only if, on tested events, it detects most of
    them, early enough to act before the market reacts, with exact times, and
    its licence permits use. Volume and sentiment are not criteria."""
    verdicts = {}
    for prov, s in summary.items():
        reasons = []
        if s["tested"] < min_tested:
            reasons.append(f"only {s['tested']} tested events (< {min_tested})")
        if s["recall"] is None or s["recall"] < min_recall:
            reasons.append(f"recall {s['recall']} < {min_recall}")
        lat = s["latency_minutes"]
        if lat["n"] < 5 or (lat["p50"] is not None and lat["p50"] > max_p50_minutes):
            reasons.append(f"median delay {lat['p50']} min over {lat['n']} exact detections (need <= {max_p50_minutes}, n >= 5)")
        act = s["actionable_before_market_reaction"]
        if act["share"] is None or act["share"] < min_actionable:
            reasons.append(f"actionable before the market reaction {act['share']} (< {min_actionable})")
        if not licence_ok.get(prov, False):
            reasons.append("licence not cleared for production use")
        verdicts[prov] = {"meets_bar": not reasons, "reasons": reasons}
    chosen = [p for p, v in verdicts.items() if v["meets_bar"]]
    return {"criteria": {"min_tested": min_tested, "min_recall": min_recall, "max_p50_minutes": max_p50_minutes,
                         "min_actionable_share": min_actionable, "licence_required": True},
            "recommended": chosen, "verdicts": verdicts}
