"""
fqvf/service.py — Fundamental Quality & Value Framework (FQVF).

A FIXED 18-check investment evaluation framework. The checks, their order
and their meaning are defined here once; nothing else in the backend or the
mobile app recomputes them. Full definitions: docs/FQVF.md.

Statuses
  PASS            meets the preferred threshold
  WARNING         tolerable / acceptable band, or a mixed result (see `grade`)
  FAIL            does not meet the threshold, or the business fact
                  disqualifies it (e.g. negative earnings for PE)
  NOT_AVAILABLE   the data needed is missing; this is NOT a failure

Checks 3 (Industry Classification) and 4 (CMP) are informational: they report
data availability and are excluded from the FQVF score.

FQVF score = 100 x (PASS + 0.5 x WARNING) / (scored checks that were
evaluated). Coverage = evaluated scored checks / 16. A low coverage means
the score rests on little data; consumers must show coverage with the score.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import FQVF_ENGINE_VERSION, MARKET_DATA_STALE_DAYS

PASS, WARNING, FAIL, NOT_AVAILABLE = "PASS", "WARNING", "FAIL", "NOT_AVAILABLE"
STATUSES = (PASS, WARNING, FAIL, NOT_AVAILABLE)

# Fixed framework thresholds (docs/FQVF.md). Values the specification gives
# explicitly are marked [spec]; the others are documented project defaults.
THRESHOLDS: dict[str, float] = {
    "stable_min_years": 3,                # fiscal years of net income needed
    "eps_progression_min_years": 4,       # 5 requested; provider supplies 4 for most NSE stocks
    "eps_progression_requested_years": 5,
    "eps_growth_pass_cagr": 0.10,         # >= 10% CAGR PASS, 0-10% WARNING, < 0 FAIL
    "pe_pass_max": 25.0,
    "pe_warn_max": 40.0,
    "industry_pe_min_peers": 3,
    "relative_pe_warn_premium": 0.20,     # up to 20% above the benchmark = WARNING
    "historical_pe_min_years": 5,         # [spec] 5-7 years
    "intrinsic_warn_premium": 0.20,
    "peg_pass_max": 1.0,                  # [spec] <= 1 preferred
    "peg_warn_max": 2.0,
    "pb_pass_max": 1.0,                   # [spec] < 1 preferred
    "pb_tolerable_max": 3.0,              # [spec] up to 3 tolerable
    "roe_pass_min": 0.15,                 # [spec] > 15%
    "de_strong_max": 0.5,                 # [spec] strong < 0.5
    "de_acceptable_max": 2.0,             # [spec] acceptable < 2
    "psr_pass_max": 1.5,                  # [spec] < 1.5 preferred
    "psr_tolerable_max": 2.0,             # [spec] around 2 tolerable
    "fcf_conversion_pass_min": 0.50,      # [spec] > 50% (FCF / operating cash flow)
    "roce_pass_min": 0.15,
    "dividend_yield_pass_min": 0.01,
    "dividend_payout_max": 0.75,
}

CHECKS: tuple[tuple[int, str, bool], ...] = (
    # (id, name, scored)
    (1, "Stable Stock / Business Stability", True),
    (2, "5-Year EPS Progression", True),
    (3, "Industry Classification", False),
    (4, "Current Market Price (CMP)", False),
    (5, "EPS Growth", True),
    (6, "PE Ratio", True),
    (7, "Industry PE", True),
    (8, "Historical Average PE (5-7 years)", True),
    (9, "Intrinsic Value", True),
    (10, "PEG Ratio", True),
    (11, "Price-to-Book (PB) Ratio", True),
    (12, "Return on Equity (ROE)", True),
    (13, "Debt-to-Equity (D/E)", True),
    (14, "Price-to-Sales (PSR)", True),
    (15, "Free Cash Flow Strength", True),
    (16, "Return on Capital Employed (ROCE)", True),
    (17, "Dividend Strength", True),
    (18, "Sector Outlook", True),
)
SCORED_CHECK_COUNT = sum(1 for _, _, s in CHECKS if s)


@dataclass
class FQVFInputs:
    """Everything FQVF needs. Every field may be None (= not available)."""
    price: Optional[float] = None
    price_as_of: Optional[str] = None
    price_age_days: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    annual: list[dict] = field(default_factory=list)       # newest first (provider.normalise)
    trailing_pe: Optional[float] = None
    trailing_eps: Optional[float] = None
    price_to_book: Optional[float] = None
    book_value_per_share: Optional[float] = None
    price_to_sales: Optional[float] = None
    debt_to_equity: Optional[float] = None
    provider_peg: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    industry_pe_median: Optional[float] = None
    industry_peer_count: int = 0
    historical_avg_pe: Optional[float] = None
    historical_pe_years: int = 0
    sector_outlook: Optional[str] = None
    sector_outlook_notes: Optional[str] = None
    sector_outlook_updated_at: Optional[str] = None
    fundamentals_fetched_at: Optional[str] = None
    fiscal_period_end: Optional[str] = None


@dataclass
class FQVFCheck:
    id: int
    name: str
    status: str
    value: Optional[Any]
    threshold: str
    explanation: str
    grade: Optional[str] = None
    scored: bool = True
    source: Optional[str] = None
    source_timestamp: Optional[str] = None
    data_available: bool = True


@dataclass
class FQVFResult:
    version: str
    calculated_at: str
    checks: list[FQVFCheck]
    counts: dict[str, int]
    score: Optional[float]
    coverage: float
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def _num(x: Optional[float], nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def _series(annual: list[dict], key: str) -> list[tuple[str, float]]:
    """(fiscal_year_end, value) oldest -> newest, available values only."""
    pts = [(r["fiscal_year_end"], r.get(key)) for r in annual if r.get(key) is not None]
    return sorted(pts)


def _latest(annual: list[dict], *keys: str) -> tuple[Optional[str], dict]:
    """Newest fiscal year in which all `keys` are present."""
    for r in annual:  # newest first
        if all(r.get(k) is not None for k in keys):
            return r["fiscal_year_end"], r
    return None, {}


def eps_share_basis_break(annual: list[dict], tolerance: float = 0.35) -> Optional[str]:
    """
    Provider EPS history is NOT restated for splits/bonus issues. Detect a
    share-count change: between consecutive fiscal years, the EPS ratio must
    track the net-income ratio (EPS = NI / shares). If they differ by more than
    `tolerance`, the EPS series is not comparable across years; return the
    reason, else None. Requires positive net income in both years.
    """
    rows = sorted((r for r in annual if r.get("diluted_eps") and r.get("net_income")),
                  key=lambda r: r["fiscal_year_end"])
    for a, b in zip(rows, rows[1:]):
        if a["net_income"] <= 0 or b["net_income"] <= 0 or a["diluted_eps"] <= 0 or b["diluted_eps"] <= 0:
            continue
        implied_share_change = (b["net_income"] / a["net_income"]) / (b["diluted_eps"] / a["diluted_eps"])
        if abs(implied_share_change - 1) > tolerance:
            return (f"EPS history is not comparable across years: between FY {a['fiscal_year_end']} and "
                    f"FY {b['fiscal_year_end']} the implied share count changed {implied_share_change:.2f}x "
                    f"(likely a split/bonus issue; the provider does not restate historical EPS)")
    return None


def eps_cagr(annual: list[dict]) -> tuple[Optional[float], Optional[str], int]:
    """(CAGR, reason-if-None, number of points). Needs >= 3 EPS points, a
    positive starting EPS (a CAGR from <= 0 is undefined) and a comparable
    share basis (see eps_share_basis_break)."""
    pts = _series(annual, "diluted_eps")
    basis_break = eps_share_basis_break(annual)
    if basis_break:
        return None, basis_break, len(pts)
    if len(pts) < 3:
        return None, f"only {len(pts)} fiscal year(s) of EPS available; 3 needed", len(pts)
    start, end = pts[0][1], pts[-1][1]
    years = len(pts) - 1
    if start <= 0:
        return None, f"starting EPS ({_num(start)}) is not positive; growth rate undefined", len(pts)
    if end <= 0:
        return -1.0, None, len(pts)  # sentinel: earnings turned negative
    return (end / start) ** (1 / years) - 1, None, len(pts)


def _relative(value: float, benchmark: float, premium: float) -> tuple[str, str]:
    if value <= benchmark:
        return PASS, "at or below benchmark"
    if value <= benchmark * (1 + premium):
        return WARNING, f"within {premium:.0%} above benchmark"
    return FAIL, f"more than {premium:.0%} above benchmark"


# ── the 18 checks ────────────────────────────────────────────────────────────

def evaluate(inp: FQVFInputs, now: Optional[datetime] = None) -> FQVFResult:
    t = THRESHOLDS
    now = now or datetime.now(timezone.utc)
    fund_ts = inp.fundamentals_fetched_at
    stmt_src = "annual financial statements (Yahoo Finance)"
    info_src = "provider summary (Yahoo Finance)"
    out: list[FQVFCheck] = []

    def add(cid, status, value, threshold, explanation, grade=None, source=None, ts=None):
        name, scored = next((n, s) for i, n, s in CHECKS if i == cid)
        out.append(FQVFCheck(
            id=cid, name=name, status=status, value=value, threshold=threshold,
            explanation=explanation, grade=grade, scored=scored, source=source,
            source_timestamp=ts, data_available=status != NOT_AVAILABLE))

    # 1. Stable Stock / Business Stability
    ni = _series(inp.annual, "net_income")
    thr = f"net profit in every available fiscal year (>= {t['stable_min_years']:.0f} years)"
    if len(ni) < t["stable_min_years"]:
        add(1, NOT_AVAILABLE, None, thr,
            f"Only {len(ni)} fiscal year(s) of net income available.", source=stmt_src, ts=fund_ts)
    else:
        losses = [fy for fy, v in ni if v <= 0]
        value = f"{len(ni) - len(losses)} of {len(ni)} fiscal years profitable"
        if not losses:
            add(1, PASS, value, thr, f"Profitable in all {len(ni)} fiscal years ({ni[0][0]} to {ni[-1][0]}).",
                source=stmt_src, ts=ni[-1][0])
        elif len(losses) == 1 and losses[0] != ni[-1][0]:
            add(1, WARNING, value, thr, f"One loss year ({losses[0]}); latest year profitable.",
                source=stmt_src, ts=ni[-1][0])
        else:
            add(1, FAIL, value, thr, f"Loss years: {', '.join(losses)}.", source=stmt_src, ts=ni[-1][0])

    # 2. 5-Year EPS Progression
    eps = _series(inp.annual, "diluted_eps")
    req, minimum = int(t["eps_progression_requested_years"]), int(t["eps_progression_min_years"])
    thr = f"EPS rising year on year over {req} fiscal years (evaluated with >= {minimum})"
    basis_break = eps_share_basis_break(inp.annual)
    if len(eps) < minimum:
        add(2, NOT_AVAILABLE, None, thr,
            f"Only {len(eps)} fiscal year(s) of EPS available; at least {minimum} needed.",
            source=stmt_src, ts=fund_ts)
    elif basis_break:
        add(2, NOT_AVAILABLE, " -> ".join(_num(v) for _, v in eps), thr, basis_break + ".",
            source=stmt_src, ts=fund_ts)
    else:
        path = " -> ".join(_num(v) for _, v in eps)
        declines = sum(1 for (_, a), (_, b) in zip(eps, eps[1:]) if b < a)
        note = "" if len(eps) >= req else f" Based on {len(eps)} fiscal years ({req} requested; provider supplied {len(eps)})."
        if declines == 0:
            add(2, PASS, path, thr, f"EPS did not decline in any year.{note}", source=stmt_src, ts=eps[-1][0])
        elif declines == 1 and eps[-1][1] > eps[0][1]:
            add(2, WARNING, path, thr, f"One annual decline; EPS higher than {len(eps) - 1} years earlier.{note}",
                source=stmt_src, ts=eps[-1][0])
        else:
            add(2, FAIL, path, thr, f"EPS declined in {declines} year(s).{note}", source=stmt_src, ts=eps[-1][0])

    # 3. Industry Classification (informational)
    thr = "sector and industry known"
    if inp.sector and inp.industry:
        add(3, PASS, f"{inp.sector} / {inp.industry}", thr, "Classification available for peer comparison.",
            source=info_src, ts=fund_ts)
    else:
        add(3, NOT_AVAILABLE, inp.sector or inp.industry, thr,
            "Sector and/or industry not reported by the provider.", source=info_src, ts=fund_ts)

    # 4. Current Market Price (informational)
    thr = f"valid price no older than {MARKET_DATA_STALE_DAYS} days"
    if inp.price is None or inp.price <= 0:
        add(4, NOT_AVAILABLE, None, thr, "No valid market price available.", source="daily price history")
    elif inp.price_age_days is not None and inp.price_age_days > MARKET_DATA_STALE_DAYS:
        add(4, WARNING, inp.price, thr, f"Last price is {inp.price_age_days} days old (stale).",
            grade="STALE", source="daily price history", ts=inp.price_as_of)
    else:
        add(4, PASS, inp.price, thr, f"Closing price as of {inp.price_as_of}.",
            source="daily price history", ts=inp.price_as_of)

    # 5. EPS Growth
    cagr, reason, n_eps = eps_cagr(inp.annual)
    thr = f"EPS CAGR >= {t['eps_growth_pass_cagr']:.0%} PASS, 0-{t['eps_growth_pass_cagr']:.0%} WARNING, < 0 FAIL"
    if cagr is None:
        add(5, NOT_AVAILABLE, None, thr, reason, source=stmt_src, ts=fund_ts)
    elif cagr == -1.0:
        add(5, FAIL, None, thr, "Latest fiscal-year EPS is negative.", source=stmt_src, ts=eps[-1][0])
    else:
        st = PASS if cagr >= t["eps_growth_pass_cagr"] else WARNING if cagr >= 0 else FAIL
        add(5, st, round(cagr, 4), thr, f"EPS CAGR {_pct(cagr)} over {n_eps - 1} year(s).",
            source=stmt_src, ts=eps[-1][0])

    # 6. PE Ratio
    pe = inp.trailing_pe
    thr = f"0 < PE <= {t['pe_pass_max']:.0f} PASS, <= {t['pe_warn_max']:.0f} WARNING"
    if pe is None:
        if inp.trailing_eps is not None and inp.trailing_eps <= 0:
            add(6, FAIL, None, thr, f"Trailing EPS {_num(inp.trailing_eps)} is not positive; PE not meaningful.",
                source=info_src, ts=fund_ts)
        else:
            add(6, NOT_AVAILABLE, None, thr, "Trailing PE not reported.", source=info_src, ts=fund_ts)
    elif pe <= 0:
        add(6, FAIL, round(pe, 2), thr, "Negative PE (loss-making).", source=info_src, ts=fund_ts)
    else:
        st = PASS if pe <= t["pe_pass_max"] else WARNING if pe <= t["pe_warn_max"] else FAIL
        add(6, st, round(pe, 2), thr, f"Trailing PE {_num(pe)}.", source=info_src, ts=fund_ts)

    # 7. Industry PE
    thr = (f"PE <= industry median PASS, up to {t['relative_pe_warn_premium']:.0%} above WARNING "
           f"(>= {t['industry_pe_min_peers']:.0f} peers)")
    if pe is None or pe <= 0:
        add(7, NOT_AVAILABLE, None, thr, "Stock has no positive PE to compare.", source="peer median", ts=fund_ts)
    elif inp.industry_pe_median is None or inp.industry_peer_count < t["industry_pe_min_peers"]:
        add(7, NOT_AVAILABLE, None, thr,
            f"Industry median needs >= {t['industry_pe_min_peers']:.0f} peers with a positive PE; "
            f"{inp.industry_peer_count} available.", source="peer median", ts=fund_ts)
    else:
        st, how = _relative(pe, inp.industry_pe_median, t["relative_pe_warn_premium"])
        add(7, st, {"pe": round(pe, 2), "industry_median_pe": round(inp.industry_pe_median, 2),
                    "peers": inp.industry_peer_count}, thr,
            f"PE {_num(pe)} vs {inp.industry} median {_num(inp.industry_pe_median)} "
            f"({inp.industry_peer_count} peers): {how}.", source="peer median of analysed stocks", ts=fund_ts)

    # 8. Historical Average PE (5-7 years)
    thr = (f"PE <= own {t['historical_pe_min_years']:.0f}-7 year average PE PASS, "
           f"up to {t['relative_pe_warn_premium']:.0%} above WARNING")
    if pe is None or pe <= 0:
        add(8, NOT_AVAILABLE, None, thr, "Stock has no positive PE to compare.", source=stmt_src, ts=fund_ts)
    elif inp.historical_avg_pe is None or inp.historical_pe_years < t["historical_pe_min_years"]:
        add(8, NOT_AVAILABLE, None, thr,
            f"Historical average PE needs >= {t['historical_pe_min_years']:.0f} fiscal years of EPS and "
            f"year-end prices; {max(inp.historical_pe_years, len(eps))} year(s) of EPS available.",
            source=stmt_src, ts=fund_ts)
    else:
        st, how = _relative(pe, inp.historical_avg_pe, t["relative_pe_warn_premium"])
        add(8, st, {"pe": round(pe, 2), "historical_avg_pe": round(inp.historical_avg_pe, 2),
                    "years": inp.historical_pe_years}, thr,
            f"PE {_num(pe)} vs {inp.historical_pe_years}-year average {_num(inp.historical_avg_pe)}: {how}.",
            source=stmt_src, ts=fund_ts)

    # 9. Intrinsic Value (Graham number)
    eps_for_iv = inp.trailing_eps if inp.trailing_eps is not None else (eps[-1][1] if eps else None)
    bvps = inp.book_value_per_share
    thr = (f"CMP <= Graham number sqrt(22.5 x EPS x BVPS) PASS, "
           f"up to {t['intrinsic_warn_premium']:.0%} above WARNING")
    if eps_for_iv is None or bvps is None or inp.price is None:
        add(9, NOT_AVAILABLE, None, thr, "EPS, book value per share or price not available.",
            source=info_src, ts=fund_ts)
    elif eps_for_iv <= 0 or bvps <= 0:
        add(9, FAIL, None, thr, "Graham number is undefined for non-positive earnings or book value.",
            source=info_src, ts=fund_ts)
    else:
        iv = math.sqrt(22.5 * eps_for_iv * bvps)
        st, how = _relative(inp.price, iv, t["intrinsic_warn_premium"])
        add(9, st, {"intrinsic_value": round(iv, 2), "price": round(inp.price, 2),
                    "price_to_value": round(inp.price / iv, 3)}, thr,
            f"Graham number {_num(iv)} vs CMP {_num(inp.price)}: {how}.", source=info_src, ts=fund_ts)

    # 10. PEG Ratio
    thr = f"PEG <= {t['peg_pass_max']:.0f} PASS [spec], <= {t['peg_warn_max']:.0f} WARNING"
    if pe is not None and pe > 0 and cagr is not None and cagr != -1.0:
        if cagr <= 0:
            add(10, FAIL, None, thr, f"EPS growth {_pct(cagr)} is not positive; PEG not meaningful.",
                source=stmt_src, ts=fund_ts)
        else:
            peg = pe / (cagr * 100)
            st = PASS if peg <= t["peg_pass_max"] else WARNING if peg <= t["peg_warn_max"] else FAIL
            add(10, st, round(peg, 2), thr, f"PE {_num(pe)} / EPS CAGR {_pct(cagr)} = {_num(peg)}.",
                source="PE and statement EPS CAGR", ts=fund_ts)
    elif inp.provider_peg is not None and inp.provider_peg > 0 and not basis_break:
        peg = inp.provider_peg
        st = PASS if peg <= t["peg_pass_max"] else WARNING if peg <= t["peg_warn_max"] else FAIL
        add(10, st, round(peg, 2), thr, f"Provider-reported PEG {_num(peg)} (EPS history insufficient to compute).",
            source=info_src, ts=fund_ts)
    else:
        add(10, NOT_AVAILABLE, None, thr, "Needs a positive PE and an EPS growth rate.", source=stmt_src, ts=fund_ts)

    # 11. PB
    pb = inp.price_to_book
    thr = f"PB < {t['pb_pass_max']:.0f} PREFERRED (PASS), <= {t['pb_tolerable_max']:.0f} TOLERABLE (WARNING) [spec]"
    if pb is None:
        add(11, NOT_AVAILABLE, None, thr, "Price-to-book not reported.", source=info_src, ts=fund_ts)
    elif pb <= 0:
        add(11, FAIL, round(pb, 2), thr, "Non-positive book value.", source=info_src, ts=fund_ts)
    elif pb < t["pb_pass_max"]:
        add(11, PASS, round(pb, 2), thr, f"PB {_num(pb)}: preferred.", grade="PREFERRED", source=info_src, ts=fund_ts)
    elif pb <= t["pb_tolerable_max"]:
        add(11, WARNING, round(pb, 2), thr, f"PB {_num(pb)}: tolerable.", grade="TOLERABLE",
            source=info_src, ts=fund_ts)
    else:
        add(11, FAIL, round(pb, 2), thr, f"PB {_num(pb)} above tolerable range.", source=info_src, ts=fund_ts)

    # 12. ROE
    fy, row = _latest(inp.annual, "net_income", "stockholders_equity")
    thr = f"ROE > {_pct(t['roe_pass_min'], 0)} [spec]"
    if not fy:
        add(12, NOT_AVAILABLE, None, thr, "Net income or shareholders' equity not available.",
            source=stmt_src, ts=fund_ts)
    elif row["stockholders_equity"] <= 0:
        add(12, FAIL, None, thr, "Non-positive shareholders' equity.", source=stmt_src, ts=fy)
    else:
        roe = row["net_income"] / row["stockholders_equity"]
        add(12, PASS if roe > t["roe_pass_min"] else FAIL, round(roe, 4), thr,
            f"ROE {_pct(roe)} (net income / equity, FY {fy}).", source=stmt_src, ts=fy)

    # 13. Debt-to-Equity
    thr = (f"D/E < {t['de_strong_max']} STRONG (PASS), < {t['de_acceptable_max']:.0f} ACCEPTABLE (PASS), "
           f"else FAIL [spec]")
    de, de_src, de_ts = inp.debt_to_equity, info_src, fund_ts
    if de is None:
        fy, row = _latest(inp.annual, "total_debt", "stockholders_equity")
        if fy and row["stockholders_equity"] > 0:
            de, de_src, de_ts = row["total_debt"] / row["stockholders_equity"], stmt_src, fy
    if de is None:
        add(13, NOT_AVAILABLE, None, thr, "Debt or equity not reported (common for banks/financials).",
            source=info_src, ts=fund_ts)
    elif de < 0:
        add(13, FAIL, round(de, 3), thr, "Negative D/E (negative equity).", source=de_src, ts=de_ts)
    elif de < t["de_strong_max"]:
        add(13, PASS, round(de, 3), thr, f"D/E {_num(de)}: strong.", grade="STRONG", source=de_src, ts=de_ts)
    elif de < t["de_acceptable_max"]:
        add(13, PASS, round(de, 3), thr, f"D/E {_num(de)}: acceptable.", grade="ACCEPTABLE",
            source=de_src, ts=de_ts)
    else:
        add(13, FAIL, round(de, 3), thr, f"D/E {_num(de)}: high leverage.", source=de_src, ts=de_ts)

    # 14. Price-to-Sales
    psr = inp.price_to_sales
    thr = f"PSR < {t['psr_pass_max']} PREFERRED (PASS), <= {t['psr_tolerable_max']:.0f} TOLERABLE (WARNING) [spec]"
    if psr is None:
        add(14, NOT_AVAILABLE, None, thr, "Price-to-sales not reported.", source=info_src, ts=fund_ts)
    elif psr <= 0:
        add(14, FAIL, round(psr, 2), thr, "Invalid (non-positive) price-to-sales.", source=info_src, ts=fund_ts)
    elif psr < t["psr_pass_max"]:
        add(14, PASS, round(psr, 2), thr, f"PSR {_num(psr)}: preferred.", grade="PREFERRED",
            source=info_src, ts=fund_ts)
    elif psr <= t["psr_tolerable_max"]:
        add(14, WARNING, round(psr, 2), thr, f"PSR {_num(psr)}: tolerable.", grade="TOLERABLE",
            source=info_src, ts=fund_ts)
    else:
        add(14, FAIL, round(psr, 2), thr, f"PSR {_num(psr)} above tolerable range.", source=info_src, ts=fund_ts)

    # 15. Free Cash Flow Strength (FCF conversion = FCF / operating cash flow)
    fy, row = _latest(inp.annual, "free_cash_flow", "operating_cash_flow")
    thr = f"FCF / operating cash flow > {_pct(t['fcf_conversion_pass_min'], 0)} PASS [spec], > 0 WARNING"
    if not fy:
        add(15, NOT_AVAILABLE, None, thr, "Free cash flow or operating cash flow not available "
            "(not reported for most banks/financials).", source=stmt_src, ts=fund_ts)
    elif row["operating_cash_flow"] <= 0:
        add(15, FAIL, None, thr, f"Operating cash flow is not positive (FY {fy}).", source=stmt_src, ts=fy)
    else:
        conv = row["free_cash_flow"] / row["operating_cash_flow"]
        st = PASS if conv > t["fcf_conversion_pass_min"] else WARNING if conv > 0 else FAIL
        add(15, st, round(conv, 4), thr, f"FCF conversion {_pct(conv)} (FY {fy}).", source=stmt_src, ts=fy)

    # 16. ROCE = EBIT / (total assets - current liabilities)
    fy, row = _latest(inp.annual, "ebit", "total_assets", "current_liabilities")
    thr = f"ROCE > {_pct(t['roce_pass_min'], 0)}"
    if not fy:
        add(16, NOT_AVAILABLE, None, thr, "EBIT or capital employed not available (not reported for most "
            "banks/financials).", source=stmt_src, ts=fund_ts)
    else:
        ce = row["total_assets"] - row["current_liabilities"]
        if ce <= 0:
            add(16, NOT_AVAILABLE, None, thr, "Capital employed is not positive; ROCE undefined.",
                source=stmt_src, ts=fy)
        else:
            roce = row["ebit"] / ce
            add(16, PASS if roce > t["roce_pass_min"] else FAIL, round(roce, 4), thr,
                f"ROCE {_pct(roce)} (FY {fy}).", source=stmt_src, ts=fy)

    # 17. Dividend Strength
    dy, payout = inp.dividend_yield, inp.payout_ratio
    thr = (f"yield >= {_pct(t['dividend_yield_pass_min'], 0)} and payout <= "
           f"{_pct(t['dividend_payout_max'], 0)} PASS; any dividend WARNING; none FAIL")
    if dy is None:
        add(17, NOT_AVAILABLE, None, thr, "Dividend data not reported.", source=info_src, ts=fund_ts)
    elif dy <= 0:
        add(17, FAIL, 0.0, thr, "No dividend paid.", source=info_src, ts=fund_ts)
    else:
        payout_ok = payout is None or payout <= t["dividend_payout_max"]
        value = {"yield": round(dy, 4), "payout_ratio": round(payout, 4) if payout is not None else None}
        if dy >= t["dividend_yield_pass_min"] and payout_ok:
            add(17, PASS, value, thr, f"Yield {_pct(dy, 2)}, payout {_pct(payout)}.", source=info_src, ts=fund_ts)
        else:
            why = "low yield" if dy < t["dividend_yield_pass_min"] else "high payout ratio"
            add(17, WARNING, value, thr, f"Yield {_pct(dy, 2)}, payout {_pct(payout)}: {why}.",
                source=info_src, ts=fund_ts)

    # 18. Sector Outlook (admin-assessed, Sector Master)
    thr = "POSITIVE PASS, NEUTRAL WARNING, NEGATIVE FAIL (set by an administrator in the Sector Master)"
    if not inp.sector:
        add(18, NOT_AVAILABLE, None, thr, "Sector unknown.", source="Sector Master")
    elif not inp.sector_outlook:
        add(18, NOT_AVAILABLE, None, thr, f"No outlook has been assessed for the {inp.sector} sector.",
            source="Sector Master")
    else:
        st = {"POSITIVE": PASS, "NEUTRAL": WARNING, "NEGATIVE": FAIL}[inp.sector_outlook]
        note = f" {inp.sector_outlook_notes}" if inp.sector_outlook_notes else ""
        add(18, st, inp.sector_outlook, thr, f"{inp.sector}: {inp.sector_outlook.lower()} outlook.{note}",
            source="Sector Master (administrator)", ts=inp.sector_outlook_updated_at)

    counts = {s: sum(1 for c in out if c.status == s) for s in STATUSES}
    scored = [c for c in out if c.scored and c.status != NOT_AVAILABLE]
    score = (round(100 * sum(1.0 if c.status == PASS else 0.5 if c.status == WARNING else 0.0
                             for c in scored) / len(scored), 1) if scored else None)
    coverage = round(len(scored) / SCORED_CHECK_COUNT, 3)
    summary = (f"{counts[PASS]} pass, {counts[WARNING]} warning, {counts[FAIL]} fail, "
               f"{counts[NOT_AVAILABLE]} not available")
    return FQVFResult(version=FQVF_ENGINE_VERSION, calculated_at=now.isoformat(), checks=out,
                      counts=counts, score=score, coverage=coverage, summary=summary)
