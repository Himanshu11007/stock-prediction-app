"""
catalysts/sources.py — source reliability (a priori tiers, reviewable).

PRIMARY  the originator of the information: regulators, central banks,
         governments, exchanges, the company itself.
TIER1    global and Indian wire services with editorial standards.
TIER2    established national financial and general press.
TIER3    aggregators, blogs, social and unknown sites (and anything not listed).

Credibility weight per tier is used by the impact assessment. Tiers are a
starting hypothesis to be reviewed against corrections and retractions.
"""
from __future__ import annotations

TIERS: dict[str, tuple[str, ...]] = {
    "PRIMARY": ("rbi.org.in", "sebi.gov.in", "federalreserve.gov", "pib.gov.in", "finmin.nic.in", "mospi.gov.in",
                "nseindia.com", "bseindia.com", "ustr.gov", "whitehouse.gov", "treasury.gov", "bls.gov", "ecb.europa.eu",
                "stats.gov.cn", "mofcom.gov.cn", "commerce.gov.in", "dgft.gov.in", "irdai.gov.in", "pfrda.org.in"),
    "TIER1": ("reuters.com", "bloomberg.com", "apnews.com", "ptinews.com", "afp.com", "ft.com", "wsj.com",
              "bloombergquint.com", "ndtvprofit.com"),
    "TIER2": ("economictimes.indiatimes.com", "livemint.com", "business-standard.com", "thehindubusinessline.com",
              "financialexpress.com", "moneycontrol.com", "cnbctv18.com", "thehindu.com", "hindustantimes.com",
              "indianexpress.com", "timesofindia.indiatimes.com", "cnbc.com", "marketwatch.com", "nikkei.com",
              "scmp.com", "bbc.co.uk", "bbc.com", "theguardian.com", "nytimes.com", "economist.com"),
}
CREDIBILITY = {"PRIMARY": 1.0, "TIER1": 0.85, "TIER2": 0.65, "TIER3": 0.3, "UNKNOWN": 0.3}


def tier(domain: str) -> str:
    d = (domain or "").lower().removeprefix("www.")
    for name, domains in TIERS.items():
        if any(d == x or d.endswith("." + x) for x in domains):
            return name
    return "TIER3" if d else "UNKNOWN"


def is_primary(domain: str) -> bool:
    return tier(domain) == "PRIMARY"
