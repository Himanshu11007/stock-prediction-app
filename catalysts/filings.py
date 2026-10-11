"""
catalysts/filings.py — materiality of exchange filings (NSE / BSE corporate
announcements), from the filing's own subject line.

Price-sensitive disclosures are kept; routine notices are excluded unless
they are listed in DEMONSTRATED_RELEVANT (empty: no routine category has
shown relevance in StockLens's measurements yet).

  materiality  HIGH     results, guidance, large orders, regulatory or
                        enforcement action, defaults, M&A, pledges,
                        management changes at the top, fund raising,
                        credit-rating changes
               MEDIUM   dividends / buybacks / bonus / splits, press
                        releases and business updates, litigation, capacity
               EXCLUDED routine notices (analyst meets, newspaper copies,
                        trading-window closures, audio / transcripts,
                        share-certificate losses, ESOP allotments,
                        compliance certificates, prior intimations ...)
               UNKNOWN  nothing matched (kept for review, never assumed HIGH)

Order matters: exclusions are checked first, so "Intimation of board meeting
to consider results" is a prior notice, not a result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

FILINGS_VERSION = "filings-v0.1"

EXCLUDED: tuple[tuple[str, str], ...] = (
    ("ANALYST_MEET", r"analysts?\s*(or|/|&)?\s*institutional investors? meet|analyst meet|investor meet|con(ference)? call (schedule|intimation)|schedule of analyst"),
    ("PRIOR_INTIMATION", r"prior intimation|board meeting (intimation|to be held)|intimation of board meeting|to consider and approve"),
    ("NEWSPAPER", r"newspaper (publication|advertisement)|copy of newspaper|publication of"),
    ("TRADING_WINDOW", r"trading window"),
    ("RECORDING", r"audio|video recording|transcript"),
    ("SHARE_CERTIFICATE", r"loss of share certificate|duplicate share certificate|issue of duplicate"),
    ("ESOP", r"\besop\b|\besos\b|employee stock option|allotment of (equity )?shares under"),
    ("COMPLIANCE", r"compliance certificate|regulation 74\(5\)|regulation 7\(3\)|reg\.? 40\(9\)|secretarial compliance|certificate under regulation"),
    ("ROUTINE", r"change in (registrar|rta)|closure of register|book closure|record date(?! .*dividend)|updates? on (agm|egm)|notice of (agm|egm)|annual report|e-?voting|scrutini[sz]er|postal ballot"),
)

HIGH: tuple[tuple[str, str], ...] = (
    ("RESULTS", r"financial results?|un-?audited|audited results|outcome of board meeting.*results|quarterly results|results? for the (quarter|period)"),
    ("GUIDANCE", r"guidance|outlook|business update|update for q[1-4]|q[1-4] ?(fy\d{2})? ?(business )?update"),
    ("ORDER", r"(bagging|receipt|award|receiving|securing|win(ning)?) of (an? )?(new |large |significant )?(orders?|contracts?|work orders?)|order (win|inflow|book)|letter of (award|intent)"),
    ("REGULATORY_ACTION", r"show cause|penalt|sebi order|order (passed|received) (by|from)|enforcement|search and seizure|income tax (search|demand)|gst (demand|notice)|regulatory action|warning letter|import alert|form 483|usfda"),
    ("DEFAULT", r"default|delay in (payment|servicing)|non-payment|insolvency|nclt|corporate insolvency|resolution plan"),
    ("M_AND_A", r"acquisition|amalgamation|merger|demerger|scheme of arrangement|divest|sale of (stake|business|undertaking)|slump sale|takeover|open offer|joint venture"),
    ("PLEDGE", r"pledge|encumbrance|invocation"),
    ("MANAGEMENT_CHANGE", r"resignation|appointment|cessation|change in (management|directors?|kmp|key managerial)|(md|ceo|cfo|managing director|chief executive|chairman)\b.*(resign|appoint|ceas)"),
    ("FUND_RAISING", r"fund ?raising|qip|qualified institutions placement|preferential (issue|allotment)|rights issue|issue of (ncds?|debentures|bonds)|warrants"),
    ("CREDIT_RATING", r"credit rating"),
)

MEDIUM: tuple[tuple[str, str], ...] = (
    ("CORPORATE_ACTION", r"dividend|buy-?back|bonus|stock split|sub-?division"),
    ("PRESS_RELEASE", r"press release|media release"),
    ("LITIGATION", r"litigation|dispute|arbitration|court"),
    ("CAPACITY", r"capacity|commission(ing|ed)|plant|expansion|capex"),
)

# Routine categories with demonstrated relevance (by measurement) that should
# be kept despite EXCLUDED. Empty until such evidence exists.
DEMONSTRATED_RELEVANT: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FilingClass:
    category: str
    materiality: str          # HIGH | MEDIUM | EXCLUDED | UNKNOWN
    matched: str = ""

    @property
    def keep(self) -> bool:
        return self.materiality in ("HIGH", "MEDIUM", "UNKNOWN")


def classify(subject: Optional[str], detail: Optional[str] = None) -> FilingClass:
    text = f"{subject or ''} | {detail or ''}"
    low = text.lower()
    for cat, pat in EXCLUDED:
        m = re.search(pat, low)
        if m and cat not in DEMONSTRATED_RELEVANT:
            return FilingClass(cat, "EXCLUDED", m.group(0))
    for level, table in (("HIGH", HIGH), ("MEDIUM", MEDIUM)):
        for cat, pat in table:
            m = re.search(pat, low)
            if m:
                return FilingClass(cat, level, m.group(0))
    return FilingClass("OTHER", "UNKNOWN")
