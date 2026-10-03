"""
tests/test_ranking_validation.py — ranking v1.0 validation harness
(evaluation/ranking_validation.py), prospective tracking (ranking/tracking.py)
and the v1.0 freeze (docs/RANKING_VALIDATION_V1.md).
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import engine_runs.service as runs
from config import RANKING_ENGINE_STATUS, RANKING_ENGINE_VERSION
from db.models.market import EngineRun, RankingOutcome, RankingSnapshot
from db.models.stock import Company, StockUniverseMember
from evaluation import ranking_validation as rv
from ranking import tracking
from ranking.service import DEFAULT_RULES, DEFAULT_WEIGHTS, StockRankingService

D = pd.Timestamp("2024-07-31")


def _annual(years=(2024, 2023), eps0=10.0):
    return [{"fiscal_year_end": f"{y}-03-31", "diluted_eps": eps0 + (y - 2023), "net_income": (eps0 + y - 2023) * 1e9,
             "revenue": 2e10, "stockholders_equity": 5e10, "total_debt": 1e10, "ebit": 3e9, "total_assets": 9e10,
             "current_liabilities": 1e10, "free_cash_flow": 2e9, "operating_cash_flow": 4e9} for y in years]


def _prices(seed, start="2021-06-01", end="2026-10-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(600_000, 2_000_000, len(idx)).astype(float)}, index=idx)


def _stocks(n=14):
    return {f"S{i:02d}.NS": {"name": f"S{i}", "sector": "S", "industry": "I" if i % 2 else "J",
                             "annual": _annual(eps0=5.0 + i), "prices": _prices(i),
                             "dividends": pd.Series(dtype=float)} for i in range(n)}


# ── chronological split ──────────────────────────────────────────────────────

SPLIT = rv.Split(pd.Timestamp("2023-07-01"), pd.Timestamp("2024-12-31"),
                 pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30"))


def test_split_is_chronological_and_dev_outcomes_are_embargoed():
    assert SPLIT.segment(pd.Timestamp("2024-12-31")) == "DEV"
    assert SPLIT.segment(pd.Timestamp("2025-01-31")) == "FINAL"
    assert SPLIT.segment(pd.Timestamp("2023-06-30")) is None
    assert SPLIT.dev_end < SPLIT.final_start
    # A DEV ranking whose forward window reaches into FINAL is not usable.
    assert SPLIT.dev_usable(pd.Timestamp("2024-11-29"), pd.Timestamp("2024-12-31"))
    assert not SPLIT.dev_usable(pd.Timestamp("2024-11-29"), pd.Timestamp("2025-02-28"))
    assert not SPLIT.dev_usable(pd.Timestamp("2024-11-29"), None)


# ── no look-ahead ────────────────────────────────────────────────────────────

def test_statements_are_usable_only_after_publication_lag():
    annual = _annual(years=(2024, 2023))
    fy_end = pd.Timestamp("2024-03-31")
    before = rv.available_statements(annual, fy_end + pd.Timedelta(days=rv.STATEMENT_LAG_DAYS - 1))
    after = rv.available_statements(annual, fy_end + pd.Timedelta(days=rv.STATEMENT_LAG_DAYS))
    assert [r["fiscal_year_end"] for r in before] == ["2023-03-31"]
    assert [r["fiscal_year_end"] for r in after] == ["2024-03-31", "2023-03-31"]


def test_point_in_time_inputs_use_only_past_prices_and_recompute_ratios():
    p = _prices(1)
    inp, status = rv.point_in_time_inputs(_annual(), "S", "I", p, pd.Series(dtype=float), D)
    close = float(p.loc[:D, "Close"].iloc[-1])
    assert status == "OK" and inp.price == close and inp.price_as_of <= D.date().isoformat()
    assert inp.trailing_eps == 11.0 and inp.trailing_pe == pytest.approx(close / 11.0)
    shares = 11e9 / 11.0
    assert inp.book_value_per_share == pytest.approx(5e10 / shares)
    assert inp.price_to_sales == pytest.approx(close * shares / 2e10)
    assert all(r["fiscal_year_end"] <= "2024-03-31" for r in inp.annual)


def test_no_price_before_ranking_date_is_a_leakage_error():
    with pytest.raises(rv.LeakageError):
        rv.point_in_time_inputs(_annual(), "S", "I", _prices(1, start="2024-08-01"), pd.Series(dtype=float), D)


def test_future_prices_and_statements_do_not_change_the_ranking():
    stocks = _stocks()
    base = rv.rank_universe(stocks, D, DEFAULT_WEIGHTS)
    for i, s in enumerate(stocks.values()):
        p = s["prices"].copy()
        p.loc[p.index > D, ["Open", "High", "Low", "Close"]] *= 3 + i     # rewrite the future
        s["prices"] = p
        s["annual"] = _annual(years=(2026, 2025, 2024, 2023), eps0=50.0 + i)[:2] + s["annual"]
        s["dividends"] = pd.Series([99.0], index=[D + pd.Timedelta(days=10)])
    after = rv.rank_universe(stocks, D, DEFAULT_WEIGHTS)
    strip = lambda rows: [(r["symbol"], r["stockai_score"], r["rank"], r["eligible"]) for r in rows]  # noqa: E731
    assert strip(base) == strip(after)


def test_ranking_is_reproducible_and_uses_the_production_service():
    stocks = _stocks()
    a = rv.rank_universe(stocks, D, DEFAULT_WEIGHTS)
    b = rv.rank_universe(stocks, D, DEFAULT_WEIGHTS)
    assert [(r["symbol"], r["stockai_score"], r["rank"]) for r in a] == \
           [(r["symbol"], r["stockai_score"], r["rank"]) for r in b]
    assert set(a[0]["components"]) == set(DEFAULT_WEIGHTS)
    assert a[0]["components"]["ml_signal"]["score"] is None          # ML not used (weight 0)
    assert isinstance(StockRankingService(DEFAULT_WEIGHTS, DEFAULT_RULES), StockRankingService)


# ── missing data ─────────────────────────────────────────────────────────────

def test_missing_fundamentals_and_dividends_stay_missing():
    inp, status = rv.point_in_time_inputs([], "S", "I", _prices(2), pd.Series(dtype=float), D)
    assert status == "UNAVAILABLE"
    assert inp.trailing_pe is None and inp.price_to_book is None and inp.dividend_yield is None
    stocks = _stocks()
    stocks["S00.NS"]["annual"] = []
    r = next(x for x in rv.rank_universe(stocks, D, DEFAULT_WEIGHTS) if x["symbol"] == "S00.NS")
    assert r["components"]["quality"]["score"] is None          # not zero, not neutral
    assert not r["eligible"]
    # sector outlook has no historical source: always NOT_AVAILABLE, never FAIL
    inp2, _ = rv.point_in_time_inputs(_annual(), "S", "I", _prices(2), pd.Series(dtype=float), D)
    from fqvf import evaluate
    assert evaluate(inp2).checks[17].status == "NOT_AVAILABLE"


def test_trailing_dividends_only_count_past_events():
    p = _prices(3)
    divs = pd.Series([2.0, 3.0, 50.0], index=pd.to_datetime(["2023-06-01", "2024-06-01", "2024-09-01"]))
    inp, _ = rv.point_in_time_inputs(_annual(), "S", "I", p, divs, D)
    assert inp.dividend_yield == pytest.approx(3.0 / inp.price)


# ── buckets, forward returns, benchmark ──────────────────────────────────────

def test_bucket_construction_by_rank_percentile():
    ranked = pd.DataFrame({"rank": range(1, 21)})
    counts = rv.assign_buckets(ranked).value_counts()
    assert counts.to_dict() == {"Top 10%": 2, "10-25%": 3, "25-50%": 5, "50-75%": 5, "Bottom 25%": 5}
    assert rv.assign_buckets(ranked).iloc[0] == "Top 10%" and rv.assign_buckets(ranked).iloc[-1] == "Bottom 25%"


def test_forward_return_uses_benchmark_sessions_and_never_fabricates():
    cal = pd.bdate_range("2024-01-01", periods=40)
    close = pd.Series(np.arange(100.0, 140.0), index=cal)
    assert rv.forward_return(close, cal, cal[5], 21) == pytest.approx(close.iloc[26] / close.iloc[5] - 1)
    assert rv.forward_return(close, cal, cal[30], 21) is None                  # horizon not elapsed
    gappy = close.drop(cal[15:30])
    assert rv.forward_return(gappy, cal, cal[5], 21) is None                   # no close near exit
    assert rv.forward_return(close, cal, cal[5] + pd.Timedelta(hours=12), 1) == pytest.approx(close.iloc[6] / close.iloc[5] - 1)


def test_benchmark_relative_portfolio_statistics():
    r = pd.Series([0.10, -0.05, 0.02, 0.03])
    b = pd.Series([0.05, -0.02, 0.02, 0.00])
    st = rv.portfolio_stats(r, b)
    assert st["cumulative_return"] == pytest.approx(1.10 * 0.95 * 1.02 * 1.03 - 1)
    assert st["max_drawdown"] == pytest.approx(-0.05)
    assert st["hit_rate_vs_benchmark"] == pytest.approx(0.5)               # 0.05, -0.03, 0, 0.03
    ex = r - b
    assert st["information_ratio"] == pytest.approx(ex.mean() / ex.std() * np.sqrt(12))


def test_newey_west_without_lags_is_the_classic_standard_error():
    x = np.array([0.1, -0.2, 0.05, 0.3, 0.0, 0.12])
    m, se, t = rv.newey_west_mean(x, 0)
    assert m == pytest.approx(x.mean()) and se == pytest.approx(x.std() / np.sqrt(len(x)))
    assert rv.newey_west_mean(x, 3)[1] != se


# ── prospective tracking ─────────────────────────────────────────────────────

def _bars(n=520, end=None, seed=0):
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    close = 100 + np.arange(n) * 0.1 + seed
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": np.full(n, 1e6)}, index=idx)


@pytest.fixture()
def eng(monkeypatch):
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    with Session(e) as s:
        for i in range(12):
            s.add(Company(symbol=f"T{i:02d}.NS", name=f"T{i}"))
            s.add(StockUniverseMember(symbol=f"T{i:02d}.NS", category="Large Cap"))
        s.commit()

    def fake_fundamentals(sym):
        from datetime import datetime, timezone
        return {"status": "OK", "error": None, "fiscal_period_end": "2026-03-31", "fetched_at": datetime.now(timezone.utc),
                "data": {"sector": "S", "industry": "I", "trailing_pe": 12.0, "price_to_book": 1.0,
                         "annual": _annual(years=(2026, 2025, 2024, 2023)), "issues": []}}

    monkeypatch.setattr(runs.provider, "fetch_price_history",
                        lambda symbols, period="2y": {s: _bars(seed=i) for i, s in enumerate(symbols)})
    monkeypatch.setattr(runs.provider, "fetch_fundamentals", fake_fundamentals)
    return e


def _run(e, **cfg):
    with Session(e) as s:
        run = runs.create_run(s, kind="RANKING", triggered_by=None,
                              config={"include_ml": False, "weights": None, "rules": None, **cfg})
        run_id = run.run_id
    runs.execute_run(e, run_id)
    return run_id


def test_engine_run_records_frozen_snapshots(eng):
    run_id = _run(eng)
    with Session(eng) as s:
        assert s.exec(select(EngineRun)).one().status == "COMPLETED"
        snaps = s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == run_id)).all()
        assert len(snaps) == 12
        top = min((x for x in snaps if x.rank), key=lambda x: x.rank)
        assert top.engine_version == RANKING_ENGINE_VERSION and top.market_regime is not None
        assert top.reference_price is not None and top.reference_date and top.benchmark_reference_price
        assert set(top.component_scores) == set(DEFAULT_WEIGHTS) and top.fqvf_summary["coverage"] > 0
        assert top.freshness["market_data_as_of"] == top.reference_date
        # idempotent: recording again adds nothing and changes nothing
        assert tracking.record_run_snapshots(s, run_id, {}, None, None) == 0


def test_custom_weights_are_tracked_under_a_separate_label(eng):
    run_id = _run(eng, weights={**DEFAULT_WEIGHTS, "quality": 50})
    with Session(eng) as s:
        snap = s.exec(select(RankingSnapshot).where(RankingSnapshot.run_id == run_id)).first()
        assert snap.engine_version == f"{RANKING_ENGINE_VERSION}+custom-config"
    assert tracking.tracked_engine_version(None, None) == RANKING_ENGINE_VERSION


def test_outcomes_only_for_elapsed_horizons_and_never_overwritten(eng):
    with Session(eng) as s:
        s.add(EngineRun(run_id="R-OLD", kind="RANKING", status="COMPLETED", engine_version="ranking-v1.0",
                        fqvf_version="fqvf-v1.0"))
        ref = _bars().index[-100].date().isoformat()          # ~100 sessions ago
        s.add(RankingSnapshot(run_id="R-OLD", symbol="T00.NS", engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0",
                              eligible=True, rank=1, stockai_score=80, reference_date=ref, reference_price=1.0))
        s.commit()
        prices = {"T00.NS": _bars(), tracking.BENCHMARK: _bars(seed=50)}
        res = tracking.record_outcomes(s, lambda symbols, period="2y": prices)
        outcomes = {o.horizon: o for o in s.exec(select(RankingOutcome)).all()}
        assert set(outcomes) == {"1M", "3M"} and res["inserted"] == 2          # 6M/12M not elapsed
        o = outcomes["3M"]
        assert o.stock_return == pytest.approx(o.outcome_price / o.start_price - 1)
        assert o.excess_return == pytest.approx(o.stock_return - o.benchmark_return)
        first = (o.id, o.stock_return)
        # later prices change (e.g. a restatement); recorded outcomes are not touched
        changed = {k: v.assign(Close=v["Close"] * 2) for k, v in prices.items()}
        assert tracking.record_outcomes(s, lambda symbols, period="2y": changed)["inserted"] == 0
        o2 = s.exec(select(RankingOutcome).where(RankingOutcome.horizon == "3M")).one()
        assert (o2.id, o2.stock_return) == first
        summary = tracking.summarise_outcomes(s)
        assert {r["portfolio"] for r in summary} == {"Top 10", "Top 20", "All eligible"}


# ── v1.0 freeze ──────────────────────────────────────────────────────────────

def test_ranking_engine_v1_is_frozen():
    assert RANKING_ENGINE_VERSION == "ranking-v1.0" and RANKING_ENGINE_STATUS == "FROZEN"
    assert DEFAULT_WEIGHTS == {"quality": 25, "valuation": 20, "financial_health": 15, "technical_trend": 10,
                               "momentum": 10, "risk": 10, "sector_outlook": 5, "market_regime": 5,
                               "ml_signal": 0}
    assert DEFAULT_RULES == {"min_score_coverage": 0.60, "min_fqvf_coverage": 0.50, "min_avg_volume_20d": 500_000,
                             "max_market_data_age_days": 4, "strong_component": 70, "weak_component": 30}
