"""
catalysts/entities.py — factual entity extraction by dictionary match.

Only what the text names is extracted (FACTUAL). Companies come from the
companies table: the full name, the name without legal suffixes, and the
ticker when it appears in capitals as a separate word (at least 3 letters),
so ordinary words are not mistaken for companies. Everything else (sector,
country, currency, commodity, institution, index) comes from the fixed
vocabularies below. No external lookup and no model is involved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

SUFFIXES = re.compile(r"\b(limited|ltd\.?|ltd|corporation|corp\.?|company|co\.?|india|industries|enterprises|"
                      r"holdings|& co)\b", re.I)

VOCAB: dict[str, dict[str, str]] = {
    "COUNTRY": {r"\bindia(n)?\b": "IN", r"\b(united states|america(n)?)\b": "US", r"\bchina|chinese\b": "CN",
                r"\bjapan(ese)?\b": "JP", r"\brussia(n)?\b": "RU", r"\bukrain(e|ian)\b": "UA", r"\bisrael(i)?\b": "IL",
                r"\biran(ian)?\b": "IR", r"\bsaudi\b": "SA", r"\b(uae|emirates)\b": "AE",
                r"\b(european union|eu|europe(an)?)\b": "EU", r"\b(uk|britain|british)\b": "GB",
                r"\bpakistan\b": "PK", r"\btaiwan\b": "TW", r"\b(south korea|korea(n)?)\b": "KR"},
    "CURRENCY": {r"\b(rupee|inr)\b": "INR", r"\b(dollar|usd|greenback)\b": "USD", r"\b(yuan|renminbi|cny)\b": "CNY",
                 r"\beuro\b": "EUR", r"\byen\b": "JPY"},
    "COMMODITY": {r"\b(crude|brent|wti|oil prices?)\b": "CRUDE_OIL", r"\b(natural gas|lng)\b": "NATURAL_GAS",
                  r"\bgold\b": "GOLD", r"\bsilver\b": "SILVER", r"\bsteel\b": "STEEL", r"\biron ore\b": "IRON_ORE",
                  r"\bcopper\b": "COPPER", r"\balumin(i)?um\b": "ALUMINIUM", r"\bcoal\b": "COAL",
                  r"\bsugar\b": "SUGAR", r"\bwheat\b": "WHEAT", r"\bpalm oil\b": "PALM_OIL", r"\bcotton\b": "COTTON"},
    "INSTITUTION": {r"\b(reserve bank of india|rbi)\b": "RBI", r"\bsebi\b": "SEBI",
                    r"\b(federal reserve|the fed|fomc)\b": "FED", r"\b(ecb|european central bank)\b": "ECB",
                    r"\b(pboc|people's bank of china)\b": "PBOC", r"\bopec\+?\b": "OPEC", r"\bimf\b": "IMF",
                    r"\b(finance ministry|ministry of finance|government of india|centre)\b": "GOVT_IN",
                    r"\b(white house|us administration|trump administration|ustr|us commerce department)\b": "US_ADMIN"},
    "INDEX": {r"\b(nifty ?50|nifty)\b": "^NSEI", r"\bsensex\b": "^BSESN", r"\bbank nifty\b": "^NSEBANK",
              r"\b(s&p 500|dow jones|nasdaq|wall street)\b": "US_EQUITIES"},
    "SECTOR": {r"\b(it services|software services|it (stocks|companies|firms))\b": "Technology",
               r"\b(banks|lenders|banking|nbfcs?|psu banks)\b": "Financial Services",
               r"\b(pharma(ceutical)?s?|drugmakers)\b": "Healthcare",
               r"\b(automakers|carmakers|auto (stocks|sector|makers)|two-wheeler)\b": "Consumer Cyclical",
               r"\b(fmcg|consumer staples)\b": "Consumer Defensive",
               r"\b(metal stocks|metals|cement|chemicals)\b": "Basic Materials",
               r"\b(capital goods|infrastructure|engineering|defence stocks)\b": "Industrials",
               r"\b(oil marketing companies|omcs?|refiners)\b": "Energy",
               r"\b(power utilities|power stocks)\b": "Utilities", r"\btelecom\b": "Communication Services",
               r"\b(realty|real estate|developers)\b": "Real Estate"},
}


# Matched against the original (case-preserved) text: these are ordinary
# words in lower case ("us", "fed", "it").
CASE_SENSITIVE: dict[str, dict[str, str]] = {
    "COUNTRY": {r"\b(US|U\.S\.|USA)\b": "US"},
    "INSTITUTION": {r"\bFed\b": "FED"},
    "SECTOR": {r"\bIT (stocks|services|companies|firms|majors)\b": "Technology"},
}


@dataclass(frozen=True)
class Entity:
    entity_type: str
    entity_key: str
    symbol: Optional[str] = None
    matched: str = ""


class CompanyIndex:
    """Name and ticker patterns for a set of companies (symbol, name)."""

    def __init__(self, companies: Iterable[tuple[str, Optional[str]]]):
        self.names: list[tuple[re.Pattern, str]] = []
        self.tickers: dict[str, str] = {}
        for symbol, name in companies:
            base = symbol.split(".")[0].upper()
            if len(base) >= 3:
                self.tickers[base] = symbol
            for alias in {name or "", SUFFIXES.sub("", name or "").strip(" ,.&")}:
                alias = re.sub(r"\s+", " ", alias).strip()
                if len(alias) >= 4 and len(alias.split()) >= 1 and alias.lower() not in {"india", "bank", "steel"}:
                    self.names.append((re.compile(r"\b" + re.escape(alias.lower()) + r"\b"), symbol))

    def find(self, text: str) -> list[Entity]:
        low = text.lower()
        hits: dict[str, str] = {}
        for pat, sym in self.names:
            m = pat.search(low)
            if m:
                hits.setdefault(sym, m.group(0))
        for word in set(re.findall(r"\b[A-Z][A-Z&]{2,}\b", text)):
            sym = self.tickers.get(word)
            if sym:
                hits.setdefault(sym, word)
        return [Entity("COMPANY", s, s, m) for s, m in sorted(hits.items())]


def extract(text: str, companies: Optional[CompanyIndex] = None) -> list[Entity]:
    low = text.lower()
    out: dict[tuple[str, str], Entity] = {}
    for etype, vocab in VOCAB.items():
        for pattern, key in vocab.items():
            m = re.search(pattern, low)
            if m:
                out.setdefault((etype, key), Entity(etype, key, None, m.group(0)))
    for etype, vocab in CASE_SENSITIVE.items():
        for pattern, key in vocab.items():
            m = re.search(pattern, text)
            if m:
                out.setdefault((etype, key), Entity(etype, key, None, m.group(0)))
    if companies is not None:
        for e in companies.find(text):
            out.setdefault((e.entity_type, e.entity_key), e)
    return list(out.values())
