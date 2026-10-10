"""
catalysts/taxonomy.py — rule-based classification of sanitised article text
(classifier news-rules-0.1). Fixed vocabularies only; nothing is inferred
that the text does not say, and no number is ever invented.

category      first matching category in priority order (most specific
              first); maps to the existing market_events.event_type taxonomy
scope         COMPANY / SECTOR / MACRO (where the effect starts)
impact        sessions the catalyst is considered live (expiry)
sentiment     lexicon score in [-1, 1] with simple negation handling
expectedness  UNEXPECTED / EXPECTED / UNKNOWN, only from explicit wording
materiality   0..1 from category weight x source credibility x magnitude words
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (category, event_type, scope, impact_sessions, base_weight, pattern)
CATEGORIES: list[tuple[str, str, str, int, float, str]] = [
    ("BROKER_RATING", "RE_RATING", "COMPANY", 2, 0.5,
     r"\b(upgrades?|downgrades?|upgraded|downgraded|target price|price target|initiates? coverage|overweight|"
     r"underweight|outperform|underperform|buy rating|sell rating)\b"),
    ("BUSINESS_UPDATE", "BUSINESS_UPDATE", "COMPANY", 3, 0.7,
     r"\b(business update|sales update|monthly sales|deliveries|volume growth|store additions|operational update)\b"),
    ("EARNINGS", "BUSINESS_UPDATE", "COMPANY", 3, 0.8,
     r"\b(q[1-4] (results|earnings|profit)|quarterly (results|earnings)|net profit|profit (rises|falls|jumps|drops|"
     r"surges|slumps)|ebitda|revenue (rises|falls|grows|declines)|beats? (estimates|expectations)|"
     r"miss(es)? (estimates|expectations)|earnings)\b"),
    ("GUIDANCE", "GUIDANCE_CHANGE", "COMPANY", 3, 0.7,
     r"\b(guidance|raises? (its )?(outlook|forecast)|cuts? (its )?(outlook|forecast)|lowers? (its )?forecast)\b"),
    ("ORDER_CONTRACT", "BUSINESS_UPDATE", "COMPANY", 3, 0.6,
     r"\b(bags? (an? )?order|order win|wins? (an? )?(order|contract)|order inflow|contract worth|letter of award|"
     r"contract (cancelled|terminated)|order (cancelled|cancellation))\b"),
    ("CORPORATE_ACTION", "CORPORATE_ACTION", "COMPANY", 5, 0.6,
     r"\b(acquisition|acquires?|merger|demerger|stake sale|buyback|bonus issue|stock split|open offer|delisting)\b"),
    ("FUNDRAISING", "CORPORATE_ACTION", "COMPANY", 2, 0.4,
     r"\b(qip|rights issue|preferential issue|raises? rs|fund ?rais(e|ing)|ncd issue|ipo)\b"),
    ("MANAGEMENT", "CORPORATE_ACTION", "COMPANY", 2, 0.4,
     r"\b(ceo|cfo|managing director|chairman) (resigns|quits|steps down|appointed)|\bappoints? (new )?(ceo|cfo|md)\b"),
    ("LITIGATION", "REGULATORY_EVENT", "COMPANY", 3, 0.6,
     r"\b(lawsuit|probe|raid|fraud|show[- ]cause|nclt|insolvency|tribunal|court (order|ruling)|penalty|fined)\b"),
    ("MONETARY_POLICY", "MACRO_SHOCK", "MACRO", 2, 0.9,
     r"\b(repo rate|rate cut|rate hike|cuts? (interest )?rates?|hikes? (interest )?rates?|monetary policy|fomc|"
     r"policy rate|basis points|crr|liquidity window|special window)\b"),
    ("TRADE_TARIFF", "MACRO_SHOCK", "MACRO", 3, 0.8,
     r"\b(tariffs?|import duty|export duty|export ban|trade deal|trade war|anti[- ]dumping|sanctions?|"
     r"countervailing)\b"),
    ("GEOPOLITICAL", "MACRO_SHOCK", "MACRO", 2, 0.7,
     r"\b(war|missile|air ?strikes?|military|ceasefire|border clash|invasion|terror attack|strait of hormuz|"
     r"red sea|shipping disruption)\b"),
    ("FISCAL_POLICY", "MACRO_SHOCK", "MACRO", 3, 0.7,
     r"\b(union budget|budget|capex|capital expenditure|infrastructure spending|gst|tax cut|fiscal deficit|"
     r"pli scheme|disinvestment)\b"),
    ("REGULATORY", "REGULATORY_EVENT", "SECTOR", 3, 0.6,
     r"\b(sebi|irdai|trai|circular|regulator|guidelines|framework|norms|licen[cs]e (cancelled|suspended))\b"),
    ("ECONOMIC_DATA", "MACRO_SHOCK", "MACRO", 1, 0.6,
     r"\b(cpi|wpi|inflation|gdp|iip|pmi|payrolls|jobless|unemployment|trade deficit|current account)\b"),
    ("FLOWS_CURRENCY", "MACRO_SHOCK", "MACRO", 1, 0.5,
     r"\b(fii|fpi|foreign (portfolio )?investors|rupee|dollar index|treasury yields?|bond yields?|forex reserves)\b"),
    ("COMMODITY", "COMMODITY_SHOCK", "MACRO", 2, 0.6,
     r"\b(crude|brent|wti|oil prices?|opec|natural gas|gold prices?|steel prices?|copper|aluminium|aluminum|coal "
     r"prices?|iron ore)\b"),
]
OTHER = ("OTHER", "OTHER", "COMPANY", 1, 0.2)

POSITIVE = frozenset("""beat beats rise rises rose surge surges surged jump jumps jumped gain gains gained record
growth grows grew upgrade upgrades upgraded raise raises raised strong stronger robust boost boosts boosted
expands expansion approval approved wins win won bags rally rallies recovers recovery improve improves improved
outperform overweight buy cut cuts eases easing relief ceasefire deal agreement exceeds tops higher profit""".split())
NEGATIVE = frozenset("""miss misses missed fall falls fell drop drops dropped plunge plunges plunged slump slumps
slumped decline declines declined weak weaker downgrade downgrades downgraded lower lowers lowered loss losses
probe raid fraud penalty fined ban bans banned cancel cancelled cancellation default delay delayed halt halted
war attack strike sanctions tariff tariffs hike hikes hiked underperform underweight sell concern concerns warns
warning slowdown slows slowed contraction resigns quits shortfall""".split())
NEGATORS = frozenset("not no never without fails fail failed denies denied unlikely".split())
UNEXPECTED = re.compile(r"\b(unexpected(ly)?|surprise[sd]?|surprising(ly)?|shock|unscheduled|out of the blue|"
                        r"beats? (estimates|expectations)|miss(es|ed)? (estimates|expectations))\b")
EXPECTED = re.compile(r"\b(as expected|in line with (estimates|expectations)|on expected lines|widely expected|"
                      r"priced in|anticipated)\b")
MAGNITUDE = re.compile(r"\b(record|biggest|largest|sharp(ly)?|plunge|surge|soar|crash|historic|massive|"
                       r"\d{2,}\s?(%|per ?cent|bps|basis points))\b")


@dataclass(frozen=True)
class Classification:
    category: str
    event_type: str
    scope: str
    impact_sessions: int
    base_weight: float
    matched: tuple[str, ...]
    sentiment: float
    expectedness: str
    magnitude: bool


def _sentiment(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    score, n = 0, 0
    for i, w in enumerate(words):
        s = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
        if not s:
            continue
        if any(x in NEGATORS for x in words[max(0, i - 3):i]):
            s = -s
        score += s
        n += 1
    return 0.0 if n == 0 else max(-1.0, min(1.0, score / (n + 1)))


def classify(title: str, excerpt: str = "") -> Classification:
    text = f"{title}. {excerpt}".lower()
    for cat, etype, scope, impact, weight, pattern in CATEGORIES:
        found = tuple(sorted({m.group(0) for m in re.finditer(pattern, text)}))
        if found:
            break
    else:
        cat, etype, scope, impact, weight = OTHER
        found = ()
    sent = _sentiment(title + ". " + excerpt)
    if cat == "BROKER_RATING":
        etype = "DOWNGRADE" if re.search(r"\b(downgrade[sd]?|underweight|underperform|sell rating)\b", text) else "RE_RATING"
    if cat == "EARNINGS" and UNEXPECTED.search(text):
        etype = "EARNINGS_SURPRISE"
    exp = "UNEXPECTED" if UNEXPECTED.search(text) else "EXPECTED" if EXPECTED.search(text) else "UNKNOWN"
    return Classification(cat, etype, scope, impact, weight, found, round(sent, 3), exp, bool(MAGNITUDE.search(text)))


def materiality(c: Classification, credibility: float) -> float:
    """0..1. Category weight x source credibility, +25% for explicit magnitude
    wording, x0.6 when the text says it was expected (likely priced)."""
    m = c.base_weight * credibility * (1.25 if c.magnitude else 1.0)
    if c.expectedness == "EXPECTED":
        m *= 0.6
    return round(min(1.0, m), 3)
