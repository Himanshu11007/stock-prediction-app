"""
tests/test_ranking.py — StockRankingService (ranking/service.py) and the
analysis engine run (engine_runs/service.py) with a fake data provider.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import engine_runs.service as runs
from db.models.market import EngineRun, FundamentalSnapshot, Sector, StockAnalysisResult
from db.models.stock import Company, StockUniverseMember
from fqvf import FQVFInputs, evaluate
from ranking.service import DEFAULT_WEIGHTS, RankingInput, StockRankingService, validate_weights


def _fqvf(**kw):
    return evaluate(FQVFInputs(**kw)).to_dict()


GOOD = dict(price=100, price_age_days=1, sector="S", industry="I", trailing_pe=10, price_to_book=0.8,
            price_to_sales=1.0, debt_to_equity=0.3, dividend_yield=0.02, payout_ratio=0.3,
            annual=[{"fiscal_year_end": f"{y}-03-31", "diluted_eps": e, "net_income": e * 1e9,
                     "stockholders_equity": e * 5e9, "free_cash_flow": 6e9, "operating_cash_flow": 1e10,
                     "ebit": e * 2e9, "total_assets": e * 8e9, "current_liabilities": e * 1e9}
                    for y, e in ((2026, 16), (2025, 14), (2024, 12), (2023, 10))])


def _tech(i):
    return {"return_60d": 0.01 * i, "volatility_annual": 0.2 + 0.01 * i, "max_drawdown_1y": -0.1 - 0.01 * i,
            "trend_score": 0.3, "regime_score": 0.5, "regime": "Bullish", "regime_reason": "x",
            "avg_volume_20d": 1e6}


def _input(sym, i=0, fqvf=None, **kw):
    base = dict(symbol=sym, name=sym, sector="S", fqvf=fqvf if fqvf is not None else _fqvf(**GOOD),
                technical=_tech(i), ml=None, market_status="OK", market_age_days=1, fundamentals_status="OK")
    base.update(kw)
    return RankingInput(**base)


def _universe(n=12, **override):
    return [_input(f"S{i:02d}.NS", i, **override) for i in range(n)]


def test_weights_validation():
    assert validate_weights({}) == DEFAULT_WEIGHTS
    with pytest.raises(ValueError):
        validate_weights({"magic": 1})
    with pytest.raises(ValueError):
        validate_weights({"quality": -1})
    with pytest.raises(ValueError):
        validate_weights({k: 0 for k in DEFAULT_WEIGHTS})


def test_ml_signal_has_no_weight_by_default():
    assert DEFAULT_WEIGHTS["ml_signal"] == 0
    a = StockRankingService().rank(_universe())
    b = StockRankingService().rank([_input(i.symbol, n, ml={"available": True, "probability_up": 0.99})
                                    for n, i in enumerate(_universe())])
    assert [r["stockai_score"] for r in a] == [r["stockai_score"] for r in b]


def test_score_is_weighted_mean_of_available_components():
    r = StockRankingService().rank(_universe())[0]
    comps = r["components"]
    avail = {k: c for k, c in comps.items() if c["score"] is not None and c["weight"] > 0}
    expected = sum(c["weight"] * c["score"] for c in avail.values()) / sum(c["weight"] for c in avail.values())
    assert r["stockai_score"] == round(expected, 1)
    assert comps["sector_outlook"]["score"] is None          # no admin outlook -> excluded, not zero
    assert r["score_coverage"] == round(sum(c["weight"] for c in avail.values()) / 100, 3)


def test_missing_components_are_excluded_not_zeroed():
    r = StockRankingService().rank([_input("A.NS", technical={"avg_volume_20d": 1e6})])[0]
    for k in ("technical_trend", "momentum", "risk", "market_regime"):
        assert r["components"][k]["score"] is None


def test_percentiles_need_a_peer_pool():
    r = StockRankingService().rank(_universe(5))[0]
    assert r["components"]["momentum"]["score"] is None and r["components"]["risk"]["score"] is None
    r = StockRankingService().rank(_universe(12))[0]
    assert r["components"]["momentum"]["score"] is not None


def test_reference_universe_used_for_single_stock_percentiles():
    ref = {f"R{i}.NS": _tech(i) for i in range(15)}
    r = StockRankingService().rank([_input("A.NS", 7)], reference=ref)[0]
    assert r["components"]["momentum"]["score"] is not None


@pytest.mark.parametrize("override,reason", [
    ({"market_status": "STALE"}, "market data stale"),
    ({"market_age_days": 10}, "market data is 10 days old"),
    ({"fundamentals_status": "ERROR"}, "fundamentals error"),
    ({"company_active": False}, "inactive"),
    ({"company_tradable": False}, "non-tradable"),
])
def test_eligibility_rules(override, reason):
    r = StockRankingService().rank([_input("X.NS", 3, **override)] + _universe(11))[0]
    assert not r["eligible"] and r["rank"] is None
    assert any(reason in x for x in r["ineligible_reasons"])


def test_low_liquidity_and_low_coverage_are_ineligible():
    t = _tech(1)
    t["avg_volume_20d"] = 1000
    r = StockRankingService().rank([_input("X.NS", technical=t)])[0]
    assert any("liquidity" in x for x in r["ineligible_reasons"])
    r = StockRankingService().rank([_input("Y.NS", fqvf=_fqvf())])[0]
    assert any("FQVF coverage" in x for x in r["ineligible_reasons"])


def test_ranks_are_dense_over_eligible_only_and_deterministic():
    out = StockRankingService().rank(_universe(12) + [_input("Z.NS", 3, market_status="UNAVAILABLE")])
    ranked = sorted((r for r in out if r["eligible"]), key=lambda r: r["rank"])
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    scores = [r["stockai_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert StockRankingService().rank(_universe(12)) == StockRankingService().rank(_universe(12))


def test_explanations_present():
    r = StockRankingService().rank(_universe())[0]
    assert r["positives"] and r["engine_version"].startswith("ranking-")
    assert any("not be evaluated" in x for x in r["risks"])


# ── engine run with a fake provider ──────────────────────────────────────────

def _prices(seed, n=520):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = np.maximum(100 + np.cumsum(rng.normal(0, 1, n)), 5)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": rng.integers(800_000, 2_000_000, n).astype(float)}, index=idx)


@pytest.fixture()
def eng(monkeypatch):
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    with Session(e) as s:
        for i in range(12):
            sym = f"T{i:02d}.NS"
            s.add(Company(symbol=sym, name=f"Test {i}"))
            s.add(StockUniverseMember(symbol=sym, category="Large Cap"))
        s.add(Company(symbol="OFF.NS", name="Disabled", analysis_enabled=False))
        s.commit()

    def fake_prices(symbols, period="2y"):
        return {sym: (None if sym == "T11.NS" else _prices(hash(sym) % 1000)) for sym in symbols}

    def fake_fundamentals(sym):
        if sym == "T10.NS":
            raise RuntimeError("provider exploded")
        from datetime import datetime, timezone
        d = {k: GOOD.get(k) for k in ("sector", "industry", "trailing_pe", "price_to_book", "price_to_sales",
                                      "debt_to_equity", "dividend_yield", "payout_ratio")}
        d.update({"annual": GOOD["annual"], "issues": [], "trailing_eps": 16.0, "book_value_per_share": 80.0})
        return {"status": "OK", "error": None, "fiscal_period_end": "2026-03-31", "data": d,
                "fetched_at": datetime.now(timezone.utc)}

    monkeypatch.setattr(runs.provider, "fetch_price_history", fake_prices)
    monkeypatch.setattr(runs.provider, "fetch_fundamentals", fake_fundamentals)
    return e


def _run(e, **cfg):
    with Session(e) as s:
        run = runs.create_run(s, kind="RANKING", triggered_by=None,
                              config={"symbols": None, "limit": None, "include_ml": False,
                                      "refresh_fundamentals": False, "weights": None, "rules": None, **cfg})
        run_id = run.run_id
    runs.execute_run(e, run_id)
    with Session(e) as s:
        return s.exec(select(EngineRun).where(EngineRun.run_id == run_id)).one()


def test_engine_run_isolates_failures_and_records_counts(eng):
    run = _run(eng)
    assert run.status == "COMPLETED_WITH_ERRORS"
    assert run.total == run.processed == 12          # disabled stock excluded
    assert run.failed == 0 and run.succeeded + run.skipped == 12
    assert any(e["symbol"] == "T10.NS" and e["stage"] == "fundamentals" for e in run.errors)
    with Session(eng) as s:
        results = {r.symbol: r for r in s.exec(select(StockAnalysisResult)).all()}
        assert len(results) == 12
        # missing prices -> not eligible, explained, never fabricated
        assert not results["T11.NS"].eligible and "market data unavailable" in results["T11.NS"].ineligible_reasons
        assert results["T00.NS"].fqvf["checks"][17]["status"] == "NOT_AVAILABLE"
        assert s.get(Company, "T11.NS").data_status in ("PARTIAL", "UNAVAILABLE")
        assert s.exec(select(Sector)).first().name == "S"
        assert s.exec(select(FundamentalSnapshot).where(FundamentalSnapshot.symbol == "T10.NS")).first().status == "ERROR"


def test_concurrent_ranking_runs_are_refused(eng):
    with Session(eng) as s:
        runs.create_run(s, kind="RANKING", triggered_by=None, config={})
        with pytest.raises(runs.RunInProgressError):
            runs.create_run(s, kind="RANKING", triggered_by=None, config={})


def test_sector_outlook_flows_into_fqvf(eng):
    _run(eng)
    with Session(eng) as s:
        sector = s.exec(select(Sector)).first()
        sector.outlook = "POSITIVE"
        s.add(sector)
        s.commit()
    run = _run(eng)
    with Session(eng) as s:
        r = s.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == run.run_id,
                                                     StockAnalysisResult.symbol == "T00.NS")).one()
        assert r.fqvf["checks"][17]["status"] == "PASS"
