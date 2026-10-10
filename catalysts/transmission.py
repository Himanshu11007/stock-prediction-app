"""
catalysts/transmission.py — global / domestic event -> Indian asset exposure.

Each Hypothesis is an explicit, versioned, testable statement: "when <driver>
moves in direction d, <targets> tend to move with sign s x d, because
<mechanism>". They are HYPOTHESES, not rules:

  status   UNVALIDATED   not tested (event history not available yet)
           SUPPORTED     price-based test: sign matches with |t| >= 2
           INCONCLUSIVE  tested, not significant
           CONTRARY      tested, significant with the opposite sign (never used)
  weight   SUPPORTED 1.0, UNVALIDATED 0.5, INCONCLUSIVE 0.3, CONTRARY 0

Price-testable hypotheses name a factor series (`factor`); their status comes
from scripts/research/transmission_validation.py, stored in
catalysts/transmission_validation.json (versioned with the code). Sign 0
means the effect is genuinely ambiguous (e.g. standalone refiners depend on
product cracks, not on the crude price) - such exposure is shown, never
traded.

Driver direction is read from the text with fixed vocabularies
(`driver_direction`), e.g. "Brent surges" -> +1 for CRUDE_OIL, "RBI cuts
repo rate" -> +1 for EASING. If the direction cannot be read, the exposure is
recorded with direction 0 and contributes nothing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

TRANSMISSION_VERSION = "transmission-0.1"
VALIDATION_FILE = Path(__file__).with_name("transmission_validation.json")
STATUS_WEIGHT = {"SUPPORTED": 1.0, "UNVALIDATED": 0.5, "INCONCLUSIVE": 0.3, "CONTRARY": 0.0}

UP = r"(surge[sd]?|soar(s|ed)?|jump(s|ed)?|ris(e|es|ing)|rose|spike[sd]?|climb(s|ed)?|rall(y|ies|ied)|higher|" \
     r"gain(s|ed)?|up \d|hits? (a )?(\d+-\w+ )?high|strengthen(s|ed)?|firm(s|ed)?)"
DOWN = r"(fall(s|ing)?|fell|drop(s|ped)?|plunge[sd]?|slump(s|ed)?|slide[sd]?|tumble[sd]?|lower|declin(e|es|ed)|" \
       r"ease[sd]?|weaken(s|ed)?|hits? (a )?(\d+-\w+ )?low|crash(es|ed)?|down \d)"


@dataclass(frozen=True)
class Target:
    sign: int                                    # +1 moves with the driver, -1 against, 0 ambiguous
    symbols: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    mechanism: str = ""


@dataclass(frozen=True)
class Hypothesis:
    id: str
    driver: str                                  # CRUDE_OIL, INR_WEAKNESS, US_YIELDS, EASING, TARIFFS_ON_INDIA, ...
    categories: tuple[str, ...]                  # taxonomy categories that can carry this driver
    targets: tuple[Target, ...]
    horizon_sessions: int = 2
    factor: Optional[str] = None                 # price series for validation (Yahoo symbol)
    factor_sign: int = 1                         # factor return sign that corresponds to driver +1
    notes: str = ""


OMCS = ("IOC.NS", "BPCL.NS", "HINDPETRO.NS")
HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("H-CRUDE-OMC", "CRUDE_OIL", ("COMMODITY", "GEOPOLITICAL"), (
        Target(-1, symbols=OMCS, mechanism="retail fuel prices are administered, so higher crude compresses "
                                           "marketing margins and raises working capital"),), factor="BZ=F"),
    Hypothesis("H-CRUDE-UPSTREAM", "CRUDE_OIL", ("COMMODITY", "GEOPOLITICAL"), (
        Target(+1, symbols=("ONGC.NS", "OIL.NS"), mechanism="higher crude raises upstream realisations "
                                                            "(subject to windfall taxes)"),), factor="BZ=F"),
    Hypothesis("H-CRUDE-PAINTS", "CRUDE_OIL", ("COMMODITY", "GEOPOLITICAL"), (
        Target(-1, symbols=("ASIANPAINT.NS", "BERGEPAINT.NS", "KANSAINER.NS", "INDIGOPNTS.NS"),
               mechanism="crude derivatives are a large share of raw-material cost"),), factor="BZ=F"),
    Hypothesis("H-CRUDE-AIRLINES", "CRUDE_OIL", ("COMMODITY", "GEOPOLITICAL"), (
        Target(-1, symbols=("INDIGO.NS", "SPICEJET.NS"), mechanism="jet fuel is the largest operating cost"),),
        factor="BZ=F"),
    Hypothesis("H-CRUDE-TYRES", "CRUDE_OIL", ("COMMODITY",), (
        Target(-1, symbols=("MRF.NS", "APOLLOTYRE.NS", "CEATLTD.NS", "JKTYRE.NS"),
               mechanism="synthetic rubber and carbon black are crude-linked inputs"),), factor="BZ=F"),
    Hypothesis("H-CRUDE-REFINERS", "CRUDE_OIL", ("COMMODITY", "GEOPOLITICAL"), (
        Target(0, symbols=("CHENNPETRO.NS", "MRPL.NS"),
               mechanism="standalone refiners depend on product cracks, not on the crude level: ambiguous"),),
        factor="BZ=F"),
    Hypothesis("H-INR-IT", "INR_WEAKNESS", ("FLOWS_CURRENCY", "MONETARY_POLICY"), (
        Target(+1, industries=("Information Technology Services",),
               mechanism="dollar revenues with rupee costs: a weaker rupee lifts margins"),),
        factor="INR=X"),
    Hypothesis("H-INR-OMC", "INR_WEAKNESS", ("FLOWS_CURRENCY",), (
        Target(-1, symbols=OMCS, mechanism="crude imports are dollar-denominated"),), factor="INR=X"),
    Hypothesis("H-US-YIELDS", "US_YIELDS", ("FLOWS_CURRENCY", "MONETARY_POLICY", "ECONOMIC_DATA"), (
        Target(-1, industries=("Banks - Regional", "Credit Services"), sectors=("Real Estate",),
               mechanism="higher US yields draw foreign portfolio money out of rate-sensitive Indian equities"),),
        factor="^TNX"),
    Hypothesis("H-GOLD-LENDERS", "GOLD", ("COMMODITY",), (
        Target(+1, symbols=("MUTHOOTFIN.NS", "MANAPPURAM.NS"), mechanism="higher gold raises collateral value "
                                                                         "and loan-to-value headroom"),
        Target(0, symbols=("TITAN.NS", "KALYANKJIL.NS"),
               mechanism="jewellers gain on inventory but lose on demand: ambiguous")), factor="GC=F"),
    Hypothesis("H-RBI-EASING", "EASING", ("MONETARY_POLICY",), (
        Target(+1, sectors=("Real Estate",), industries=("Auto Manufacturers", "Credit Services", "Mortgage Finance"),
               mechanism="lower policy rates cut borrowing costs for rate-sensitive demand and lenders' funding"),
        Target(0, industries=("Banks - Regional",),
               mechanism="banks: credit growth vs net interest margin compression - ambiguous")), horizon_sessions=3),
    Hypothesis("H-US-TARIFF-EXPORTERS", "TARIFFS_ON_INDIA", ("TRADE_TARIFF",), (
        Target(-1, industries=("Textile Manufacturing", "Apparel Manufacturing", "Auto Parts",
                               "Drug Manufacturers - Specialty & Generic"), symbols=("AVANTIFEED.NS",),
               mechanism="higher US import tariffs reduce price competitiveness of Indian exports to the US"),),
        horizon_sessions=3),
    Hypothesis("H-INFRA-CAPEX", "PUBLIC_CAPEX", ("FISCAL_POLICY",), (
        Target(+1, industries=("Engineering & Construction", "Building Materials", "Specialty Industrial Machinery",
                               "Farm & Heavy Construction Machinery", "Electrical Equipment & Parts"),
               mechanism="higher government infrastructure spending raises order inflows"),), horizon_sessions=3),
)

DRIVER_PATTERNS: dict[str, tuple[str, str]] = {
    # driver: (subject pattern, special direction rule)
    "CRUDE_OIL": (r"(crude|brent|wti|oil prices?)", "move"),
    "GOLD": (r"(gold)", "move"),
    "US_YIELDS": (r"((us|u\.s\.|treasury) (bond )?yields?|treasur(y|ies))", "move"),
    "INR_WEAKNESS": (r"(rupee|inr)", "inverse_move"),          # rupee falls -> weakness +1
    "EASING": (r"(repo rate|policy rate|interest rates?|rates?)", "policy"),
    "TARIFFS_ON_INDIA": (r"(tariffs?|import dut(y|ies)|duties)", "tariff"),
    "PUBLIC_CAPEX": (r"(capex|capital expenditure|infrastructure (spending|outlay|investment))", "spending"),
}


def driver_direction(driver: str, text: str) -> int:
    """+1 / -1 for the driver from explicit wording near its subject, 0 when
    the text does not say."""
    subj, rule = DRIVER_PATTERNS[driver]
    t = text.lower()
    if not re.search(subj, t):
        return 0
    if rule in ("move", "inverse_move"):
        window = r"{s}\W+(\w+\W+){{0,4}}?{v}|{v}\W+(\w+\W+){{0,3}}?{s}"
        up = re.search(window.format(s=subj, v=UP), t) is not None
        down = re.search(window.format(s=subj, v=DOWN), t) is not None
        d = (1 if up else 0) - (1 if down else 0)
        return -d if rule == "inverse_move" else d
    if rule == "policy":
        if re.search(r"\b(cuts?|reduc(e|es|ed)|lower(s|ed)?|eas(e|es|ed|ing))\b.{0,40}\b(repo|policy|interest)?\s?rates?",
                     t) or re.search(r"rate cut", t):
            return 1
        if re.search(r"\b(hikes?|rais(e|es|ed)|increas(e|es|ed))\b.{0,40}\b(repo|policy|interest)?\s?rates?", t) \
                or re.search(r"rate hike", t):
            return -1
        return 0
    if rule == "tariff":
        if not re.search(r"\b(india|indian)\b", t):
            return 0
        if re.search(r"\b(impos(e|es|ed)|rais(e|es|ed)|hik(e|es|ed)|doubl(e|es|ed)|slap(s|ped)?|additional|new)\b"
                     r".{0,40}\b(tariffs?|dut(y|ies))", t):
            return 1
        if re.search(r"\b(trade deal|cut(s)?|reduc(e|es|ed)|exempt(s|ion|ed)?|roll(s|ed)? back|lift(s|ed)?)\b"
                     r".{0,40}\b(tariffs?|dut(y|ies))", t) or "trade deal" in t:
            return -1
        return 0
    if rule == "spending":
        if re.search(r"\b(rais(e|es|ed)|increas(e|es|ed)|boost(s|ed)?|hike[sd]?|record|higher)\b", t):
            return 1
        if re.search(r"\b(cut(s)?|reduc(e|es|ed)|lower|slash(es|ed)?)\b", t):
            return -1
    return 0


def validation_status() -> dict[str, dict]:
    if VALIDATION_FILE.exists():
        return json.loads(VALIDATION_FILE.read_text(encoding="utf-8")).get("hypotheses", {})
    return {}


@dataclass(frozen=True)
class Exposure:
    hypothesis_id: str
    symbol: str
    sign: int                    # target sign x driver direction (0 = ambiguous or direction unknown)
    driver: str
    driver_direction: int
    mechanism: str
    status: str
    weight: float


def exposures(category: str, text: str, companies: Iterable[tuple[str, Optional[str], Optional[str]]],
              status: Optional[dict[str, dict]] = None) -> list[Exposure]:
    """Inferred exposures of the given universe (symbol, sector, industry) to
    one classified event. Only hypotheses whose categories include the
    event's category are considered."""
    status = validation_status() if status is None else status
    comps = list(companies)
    out: list[Exposure] = []
    for h in HYPOTHESES:
        if category not in h.categories:
            continue
        d = driver_direction(h.driver, text)
        st = status.get(h.id, {}).get("status", "UNVALIDATED")
        w = STATUS_WEIGHT.get(st, 0.5)
        for tg in h.targets:
            for sym, sector, industry in comps:
                if sym in tg.symbols or (industry and industry in tg.industries) or (sector and sector in tg.sectors):
                    out.append(Exposure(h.id, sym, tg.sign * d, h.driver, d, tg.mechanism, st, w))
    return out
