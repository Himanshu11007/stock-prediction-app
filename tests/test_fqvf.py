"""
tests/test_fqvf.py — Fundamental Quality & Value Framework (fqvf/service.py)
and provider normalisation (fundamentals/provider.py).

Pure, deterministic tests: no network, no database.
"""
import math

import numpy as np
import pandas as pd
import pytest

from fqvf import CHECKS, FAIL, NOT_AVAILABLE, PASS, WARNING, FQVFInputs, evaluate
from fundamentals.provider import clean_number, normalise

EXPECTED_NAMES = [
    "Stable Stock / Business Stability", "5-Year EPS Progression", "Industry Classification",
    "Current Market Price (CMP)", "EPS Growth", "PE Ratio", "Industry PE",
    "Historical Average PE (5-7 years)", "Intrinsic Value", "PEG Ratio", "Price-to-Book (PB) Ratio",
    "Return on Equity (ROE)", "Debt-to-Equity (D/E)", "Price-to-Sales (PSR)", "Free Cash Flow Strength",
    "Return on Capital Employed (ROCE)", "Dividend Strength", "Sector Outlook",
]


def _annual(eps, ni=None, **latest):
    """Fiscal years newest first; eps/ni given oldest -> newest."""
    years = [f"{2026 - i}-03-31" for i in range(len(eps))][::-1]
    ni = ni or [e * 1e9 for e in eps]
    rows = [{"fiscal_year_end": y, "diluted_eps": e, "net_income": n} for y, e, n in zip(years, eps, ni)]
    rows[-1].update(latest)
    return rows[::-1]


def check(result, cid):
    return next(c for c in result.checks if c.id == cid)


# ── framework shape ──────────────────────────────────────────────────────────

def test_exactly_18_fixed_checks_in_order():
    r = evaluate(FQVFInputs())
    assert [c.id for c in r.checks] == list(range(1, 19))
    assert [c.name for c in r.checks] == EXPECTED_NAMES
    assert [n for _, n, _ in CHECKS] == EXPECTED_NAMES


def test_no_data_is_not_available_never_fail():
    r = evaluate(FQVFInputs())
    assert all(c.status == NOT_AVAILABLE for c in r.checks)
    assert all(not c.data_available and c.explanation for c in r.checks)
    assert r.score is None and r.coverage == 0.0
    assert r.counts[FAIL] == 0


def test_every_check_has_threshold_and_explanation():
    r = evaluate(FQVFInputs(price=100, price_age_days=1, annual=_annual([1, 2, 3, 4])))
    for c in r.checks:
        assert c.threshold and c.explanation and c.status in (PASS, WARNING, FAIL, NOT_AVAILABLE)


def test_score_and_coverage_formula_excludes_informational_checks():
    inp = FQVFInputs(price=100, price_as_of="2026-10-01", price_age_days=1, sector="S", industry="I",
                     price_to_book=0.8, price_to_sales=1.8, debt_to_equity=2.5)
    r = evaluate(inp)
    assert check(r, 3).scored is False and check(r, 4).scored is False
    # scored evaluated: PB PASS, PSR WARNING, D/E FAIL -> (1 + 0.5 + 0) / 3
    assert r.score == round(100 * 1.5 / 3, 1)
    assert r.coverage == round(3 / 16, 3)


# ── individual checks ────────────────────────────────────────────────────────

@pytest.mark.parametrize("pb,status,grade", [(0.8, PASS, "PREFERRED"), (2.5, WARNING, "TOLERABLE"),
                                             (3.0, WARNING, "TOLERABLE"), (3.5, FAIL, None), (-1, FAIL, None)])
def test_pb_preferred_vs_tolerable(pb, status, grade):
    c = check(evaluate(FQVFInputs(price_to_book=pb)), 11)
    assert (c.status, c.grade) == (status, grade)


@pytest.mark.parametrize("de,status,grade", [(0.3, PASS, "STRONG"), (1.5, PASS, "ACCEPTABLE"),
                                             (2.0, FAIL, None), (3.0, FAIL, None)])
def test_debt_to_equity_strong_acceptable(de, status, grade):
    c = check(evaluate(FQVFInputs(debt_to_equity=de)), 13)
    assert (c.status, c.grade) == (status, grade)


def test_debt_to_equity_falls_back_to_statements_and_is_na_without_data():
    r = evaluate(FQVFInputs(annual=_annual([1, 1, 1], total_debt=40.0, stockholders_equity=100.0)))
    assert check(r, 13).status == PASS and check(r, 13).value == 0.4
    assert check(evaluate(FQVFInputs()), 13).status == NOT_AVAILABLE


@pytest.mark.parametrize("psr,status", [(1.0, PASS), (1.8, WARNING), (2.0, WARNING), (2.5, FAIL)])
def test_psr(psr, status):
    assert check(evaluate(FQVFInputs(price_to_sales=psr)), 14).status == status


@pytest.mark.parametrize("ni,eq,status", [(20, 100, PASS), (15, 100, FAIL), (10, 100, FAIL), (10, -5, FAIL)])
def test_roe_preferred_above_15_percent(ni, eq, status):
    r = evaluate(FQVFInputs(annual=[{"fiscal_year_end": "2026-03-31", "net_income": ni, "stockholders_equity": eq}]))
    assert check(r, 12).status == status


@pytest.mark.parametrize("fcf,ocf,status", [(60, 100, PASS), (30, 100, WARNING), (-10, 100, FAIL), (5, -20, FAIL)])
def test_fcf_conversion(fcf, ocf, status):
    r = evaluate(FQVFInputs(annual=[{"fiscal_year_end": "2026-03-31", "free_cash_flow": fcf,
                                     "operating_cash_flow": ocf}]))
    assert check(r, 15).status == status


def test_fcf_not_available_for_missing_cash_flow():
    assert check(evaluate(FQVFInputs(annual=_annual([1, 2, 3]))), 15).status == NOT_AVAILABLE


def test_roce():
    row = {"fiscal_year_end": "2026-03-31", "ebit": 30, "total_assets": 200, "current_liabilities": 50}
    assert check(evaluate(FQVFInputs(annual=[row])), 16).status == PASS      # 20%
    row["ebit"] = 10
    assert check(evaluate(FQVFInputs(annual=[row])), 16).status == FAIL      # 6.7%


def test_eps_progression_with_four_years_discloses_partial_history():
    c = check(evaluate(FQVFInputs(annual=_annual([10, 11, 12, 13]))), 2)
    assert c.status == PASS and "5 requested; provider supplied 4" in c.explanation


def test_eps_progression_insufficient_history_is_not_available():
    assert check(evaluate(FQVFInputs(annual=_annual([10, 11, 12]))), 2).status == NOT_AVAILABLE


def test_eps_progression_one_decline_warning_two_declines_fail():
    assert check(evaluate(FQVFInputs(annual=_annual([10, 9, 12, 13]))), 2).status == WARNING
    assert check(evaluate(FQVFInputs(annual=_annual([10, 9, 8, 13]))), 2).status == FAIL


def test_split_or_bonus_makes_eps_history_not_comparable():
    # EPS halves while net income rises: a 1:1 bonus issue, not an earnings collapse.
    annual = _annual([88.0, 44.0, 45.0, 46.0], ni=[4.4e11, 4.6e11, 4.7e11, 4.8e11])
    r = evaluate(FQVFInputs(annual=annual, trailing_pe=15.0))
    for cid in (2, 5):
        assert check(r, cid).status == NOT_AVAILABLE and "split/bonus" in check(r, cid).explanation


@pytest.mark.parametrize("eps,status", [([10, 12, 14, 16], PASS), ([10, 10.2, 10.4, 10.6], WARNING),
                                        ([16, 14, 12, 10], FAIL)])
def test_eps_growth(eps, status):
    assert check(evaluate(FQVFInputs(annual=_annual(eps))), 5).status == status


def test_eps_growth_undefined_from_non_positive_start():
    c = check(evaluate(FQVFInputs(annual=_annual([-1, 2, 3, 4], ni=[-1e9, 2e9, 3e9, 4e9]))), 5)
    assert c.status == NOT_AVAILABLE and "undefined" in c.explanation


@pytest.mark.parametrize("pe,status", [(15, PASS), (30, WARNING), (50, FAIL), (-5, FAIL)])
def test_pe(pe, status):
    assert check(evaluate(FQVFInputs(trailing_pe=pe)), 6).status == status


def test_pe_loss_making_is_fail_and_missing_is_not_available():
    assert check(evaluate(FQVFInputs(trailing_eps=-2.0)), 6).status == FAIL
    assert check(evaluate(FQVFInputs()), 6).status == NOT_AVAILABLE


def test_industry_pe_needs_peers():
    assert check(evaluate(FQVFInputs(trailing_pe=20, industry="X", industry_pe_median=25,
                                     industry_peer_count=2)), 7).status == NOT_AVAILABLE
    r = evaluate(FQVFInputs(trailing_pe=20, industry="X", industry_pe_median=25, industry_peer_count=5))
    assert check(r, 7).status == PASS
    r = evaluate(FQVFInputs(trailing_pe=28, industry="X", industry_pe_median=25, industry_peer_count=5))
    assert check(r, 7).status == WARNING


def test_historical_average_pe_needs_five_years():
    assert check(evaluate(FQVFInputs(trailing_pe=20, historical_avg_pe=25,
                                     historical_pe_years=4)), 8).status == NOT_AVAILABLE
    assert check(evaluate(FQVFInputs(trailing_pe=20, historical_avg_pe=25,
                                     historical_pe_years=6)), 8).status == PASS


def test_intrinsic_value_graham_number():
    c = check(evaluate(FQVFInputs(price=100, trailing_eps=10, book_value_per_share=50)), 9)
    iv = math.sqrt(22.5 * 10 * 50)
    assert c.value["intrinsic_value"] == round(iv, 2) and c.status == PASS
    assert check(evaluate(FQVFInputs(price=100, trailing_eps=-1, book_value_per_share=50)), 9).status == FAIL
    assert check(evaluate(FQVFInputs(price=100, trailing_eps=10)), 9).status == NOT_AVAILABLE


def test_peg_preferred_at_most_one():
    # CAGR of 10 -> 13.31 over 3 years = 10%; PE 9 -> PEG 0.9
    r = evaluate(FQVFInputs(trailing_pe=9, annual=_annual([10, 11, 12.1, 13.31])))
    assert check(r, 10).status == PASS
    r = evaluate(FQVFInputs(trailing_pe=15, annual=_annual([10, 11, 12.1, 13.31])))
    assert check(r, 10).status == WARNING
    r = evaluate(FQVFInputs(trailing_pe=15, annual=_annual([13, 12, 11, 10])))
    assert check(r, 10).status == FAIL


@pytest.mark.parametrize("dy,payout,status", [(None, None, NOT_AVAILABLE), (0.0, None, FAIL),
                                              (0.02, 0.4, PASS), (0.005, 0.2, WARNING), (0.03, 0.9, WARNING)])
def test_dividend(dy, payout, status):
    assert check(evaluate(FQVFInputs(dividend_yield=dy, payout_ratio=payout)), 17).status == status


@pytest.mark.parametrize("outlook,status", [(None, NOT_AVAILABLE), ("POSITIVE", PASS),
                                            ("NEUTRAL", WARNING), ("NEGATIVE", FAIL)])
def test_sector_outlook(outlook, status):
    assert check(evaluate(FQVFInputs(sector="Energy", sector_outlook=outlook)), 18).status == status


def test_cmp_stale_and_missing():
    assert check(evaluate(FQVFInputs(price=10, price_age_days=10)), 4).status == WARNING
    assert check(evaluate(FQVFInputs(price=10, price_age_days=1)), 4).status == PASS
    assert check(evaluate(FQVFInputs()), 4).status == NOT_AVAILABLE


def test_stability():
    assert check(evaluate(FQVFInputs(annual=_annual([1, 2]))), 1).status == NOT_AVAILABLE
    assert check(evaluate(FQVFInputs(annual=_annual([1, 2, 3]))), 1).status == PASS
    assert check(evaluate(FQVFInputs(annual=_annual([1, 2, 3], ni=[-1, 2, 3]))), 1).status == WARNING
    assert check(evaluate(FQVFInputs(annual=_annual([1, 2, 3], ni=[1, 2, -3]))), 1).status == FAIL


# ── provider normalisation ───────────────────────────────────────────────────

def test_clean_number_never_substitutes():
    assert clean_number(float("nan")) is None and clean_number(float("inf")) is None
    assert clean_number("abc") is None and clean_number(None) is None and clean_number(True) is None
    assert clean_number("1.5") == 1.5


def test_normalise_units_and_invalid_values():
    info = {"debtToEquity": 46.3, "dividendRate": 5.5, "currentPrice": 1100.0, "trailingPE": "bad",
            "priceToBook": float("inf"), "sector": " Energy ", "industry": ""}
    stmt = pd.DataFrame({pd.Timestamp("2026-03-31"): [59.7, np.nan], pd.Timestamp("2025-03-31"): [51.5, 7e11]},
                        index=["Diluted EPS", "Net Income"])
    d = normalise(info, {"income_stmt": stmt})
    assert d["debt_to_equity"] == pytest.approx(0.463)
    assert d["dividend_yield"] == pytest.approx(0.005)
    assert d["trailing_pe"] is None and d["price_to_book"] is None
    assert any("trailingPE" in i for i in d["issues"]) and any("priceToBook" in i for i in d["issues"])
    assert d["sector"] == "Energy" and d["industry"] is None
    assert d["annual"][0] == {"fiscal_year_end": "2026-03-31", "diluted_eps": 59.7, "net_income": None}


def test_normalise_dividend_yield_percent_fallback():
    d = normalise({"dividendYield": 0.51}, {})
    assert d["dividend_yield"] == pytest.approx(0.0051) and d["dividend_yield_basis"] == "dividendYield / 100"
    assert normalise({}, {})["dividend_yield"] is None
