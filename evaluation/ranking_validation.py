"""
evaluation/ranking_validation.py — point-in-time validation of the production
StockLens ranking (docs/RANKING_VALIDATION_V1.md).

For a ranking date D everything is reconstructed from information available
at D and scored with the UNMODIFIED production code:
  technical metrics  ranking.technical.technical_snapshot(prices up to D)
  FQVF               fqvf.evaluate(point-in-time FQVFInputs)
  StockLens Score      ranking.service.StockRankingService(weights, rules).rank()

Point-in-time rules
  prices        daily bars dated <= D only
  statements    a fiscal year is usable only once fiscal_year_end + 75 days <= D
                (SEBI LODR: annual results within 60 days; +15 days buffer)
  ratios        recomputed at D from D's close and the usable statements:
                PE = close / EPS, BVPS = equity / shares, PB = close / BVPS,
                PSR = close x shares / revenue, shares = net income / EPS
  dividends     trailing-12-month dividend events <= D; no dividend events -> None
                (production's provider reports no yield for non-payers -> NOT_AVAILABLE)
  peers         industry PE medians from the other stocks' point-in-time PE at D
  sector outlook  None (no historical administrator outlook exists; production also NA)
Forward returns use split-adjusted closes (price return, dividends excluded)
for stocks and for the NIFTY 50 price index alike.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fqvf import FQVFInputs, evaluate
from ranking.service import DEFAULT_RULES, RankingInput, StockRankingService
from ranking.technical import technical_snapshot

STATEMENT_LAG_DAYS = 75
HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
BUCKETS = (("Top 10%", 0.0, 0.10), ("10-25%", 0.10, 0.25), ("25-50%", 0.25, 0.50),
           ("50-75%", 0.50, 0.75), ("Bottom 25%", 0.75, 1.0))


class LeakageError(AssertionError):
    pass


# ── point-in-time inputs ─────────────────────────────────────────────────────

def available_statements(annual: list[dict], date: pd.Timestamp, lag_days: int = STATEMENT_LAG_DAYS) -> list[dict]:
    """Fiscal years whose results were published by `date` (newest first)."""
    out = [r for r in annual if pd.Timestamp(r["fiscal_year_end"]) + pd.Timedelta(days=lag_days) <= date]
    return sorted(out, key=lambda r: r["fiscal_year_end"], reverse=True)


def point_in_time_inputs(annual: list[dict], sector: Optional[str], industry: Optional[str],
                         prices: pd.DataFrame, dividends: pd.Series, date: pd.Timestamp,
                         industry_pe: Optional[dict] = None, symbol: str = "") -> tuple[FQVFInputs, str]:
    """FQVFInputs using only data available at `date`; returns (inputs, fundamentals_status)."""
    hist = prices.loc[prices.index <= date]
    if hist.empty:
        raise LeakageError("no price on or before the ranking date")
    price_date = hist.index[-1]
    price = float(hist["Close"].iloc[-1])
    rows = available_statements(annual, date)
    for r in rows:
        if pd.Timestamp(r["fiscal_year_end"]) + pd.Timedelta(days=STATEMENT_LAG_DAYS) > date:
            raise LeakageError("statement used before its publication date")
    status = "OK" if rows else "UNAVAILABLE"
    latest = rows[0] if rows else {}
    eps, ni = latest.get("diluted_eps"), latest.get("net_income")
    shares = ni / eps if eps and ni and eps != 0 and (ni / eps) > 0 else None
    equity, revenue = latest.get("stockholders_equity"), latest.get("revenue")
    bvps = equity / shares if shares and equity is not None else None
    pe = price / eps if eps and eps > 0 else None
    pb = price / bvps if bvps and bvps > 0 else None
    psr = price * shares / revenue if shares and revenue and revenue > 0 else None
    div = dividends if dividends is not None and len(dividends) else pd.Series(dtype=float)
    if len(div):
        div = div.loc[(div.index > date - pd.Timedelta(days=365)) & (div.index <= date)]
    div_ttm = float(div.sum()) if len(div) else 0.0
    dy = div_ttm / price if div_ttm > 0 and price > 0 else None
    payout = div_ttm / eps if div_ttm > 0 and eps and eps > 0 else None
    peers = [pe_ for sym, pe_ in (industry_pe or {}).get(industry, []) if sym != symbol] if industry else []
    inputs = FQVFInputs(
        price=price, price_as_of=price_date.date().isoformat(), price_age_days=(date - price_date).days,
        sector=sector, industry=industry, annual=rows, trailing_pe=pe, trailing_eps=eps,
        price_to_book=pb, book_value_per_share=bvps, price_to_sales=psr, debt_to_equity=None,
        provider_peg=None, dividend_yield=dy, payout_ratio=payout,
        industry_pe_median=statistics.median(peers) if peers else None, industry_peer_count=len(peers),
        fundamentals_fetched_at=date.isoformat(), fiscal_period_end=latest.get("fiscal_year_end"))
    return inputs, status


def rank_universe(stocks: dict, date: pd.Timestamp, weights: dict, rules: Optional[dict] = None,
                  technicals: Optional[dict] = None) -> list[dict]:
    """Score and rank every stock at `date` with the production code.
    stocks[symbol] = {"name", "sector", "industry", "annual", "prices", "dividends"}.
    `technicals` (symbol -> technical_snapshot at date) may be precomputed."""
    # Pass 1: point-in-time PE for peer medians.
    pe_by_industry: dict[str, list[tuple[str, float]]] = {}
    pit = {}
    for sym, s in stocks.items():
        if s["prices"].loc[s["prices"].index <= date].empty:
            continue
        inp, status = point_in_time_inputs(s["annual"], s["sector"], s["industry"], s["prices"],
                                           s["dividends"], date, symbol=sym)
        pit[sym] = (inp, status)
        if s["industry"] and inp.trailing_pe and inp.trailing_pe > 0:
            pe_by_industry.setdefault(s["industry"], []).append((sym, inp.trailing_pe))
    inputs = []
    for sym, (inp, status) in pit.items():
        s = stocks[sym]
        peers = [pe for p, pe in pe_by_industry.get(s["industry"], []) if p != sym]
        inp.industry_pe_median = statistics.median(peers) if peers else None
        inp.industry_peer_count = len(peers)
        fq = evaluate(inp).to_dict()
        tech = (technicals or {}).get(sym)
        if tech is None:
            hist = s["prices"].loc[s["prices"].index <= date]
            tech = technical_snapshot(hist) if len(hist) >= 60 else {}
        age = inp.price_age_days
        inputs.append(RankingInput(
            symbol=sym, name=s.get("name", sym), sector=s["sector"], fqvf=fq, technical=tech, ml=None,
            market_status="OK" if age is not None and age <= 4 else "STALE", market_age_days=age,
            fundamentals_status=status))
    results = StockRankingService(weights, rules or DEFAULT_RULES).rank(inputs)
    for r in results:
        r["date"] = date
        r["fqvf_coverage"] = next(i.fqvf["coverage"] for i in inputs if i.symbol == r["symbol"])
    return results


# ── forward returns and benchmark ────────────────────────────────────────────

def forward_return(close: pd.Series, calendar: pd.DatetimeIndex, date: pd.Timestamp, sessions: int,
                   max_gap_days: int = 5) -> Optional[float]:
    """Price return from the last close <= date to the close at the
    benchmark-calendar session `sessions` after date. None if that session
    is beyond the data or the stock has no close within max_gap_days of it."""
    start = close.loc[close.index <= date]
    if start.empty:
        return None
    pos = calendar.searchsorted(date, side="right") - 1
    if pos < 0 or pos + sessions >= len(calendar):
        return None
    exit_date = calendar[pos + sessions]
    if exit_date <= date:
        raise LeakageError("exit date not after ranking date")
    end = close.loc[close.index <= exit_date]
    if end.empty or (exit_date - end.index[-1]).days > max_gap_days or end.index[-1] <= date:
        return None
    return float(end.iloc[-1] / start.iloc[-1] - 1)


def assign_buckets(ranked: pd.DataFrame) -> pd.Series:
    """Ranking buckets by rank percentile within one date (rank 1 = best)."""
    n = len(ranked)
    pct = (ranked["rank"] - 1) / n
    labels = pd.Series(index=ranked.index, dtype=object)
    for name, lo, hi in BUCKETS:
        labels[(pct >= lo) & (pct < hi)] = name
    return labels


# ── statistics ───────────────────────────────────────────────────────────────

def newey_west_mean(x: np.ndarray, lags: int) -> tuple[float, float, float]:
    """Mean, Newey-West standard error and t-stat of a time series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    m = x.mean()
    e = x - m
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * (e[lag:] @ e[:-lag]) / n
    se = math.sqrt(max(var, 0) / n)
    return float(m), se, float(m / se) if se > 0 else float("nan")


def spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10:
        return None
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min()) if len(eq) else float("nan")


def portfolio_stats(period_returns: pd.Series, bench_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Non-overlapping periodic portfolio returns -> risk metrics.
    Sharpe uses a 0% risk-free rate (documented); information ratio uses
    excess returns over the benchmark."""
    r = period_returns.dropna()
    b = bench_returns.reindex(r.index)
    ex = (r - b).dropna()
    downside = r[r < 0]
    ann = math.sqrt(periods_per_year)
    return {
        "periods": int(len(r)),
        "mean_period_return": float(r.mean()) if len(r) else None,
        "cumulative_return": float(np.prod(1 + r) - 1) if len(r) else None,
        "benchmark_cumulative": float(np.prod(1 + b.dropna()) - 1) if len(b.dropna()) else None,
        "annualised_volatility": float(r.std() * ann) if len(r) > 1 else None,
        "downside_volatility": float(np.sqrt((downside ** 2).mean()) * ann) if len(downside) else 0.0,
        "sharpe_rf0": float(r.mean() / r.std() * ann) if len(r) > 1 and r.std() > 0 else None,
        "information_ratio": float(ex.mean() / ex.std() * ann) if len(ex) > 1 and ex.std() > 0 else None,
        "max_drawdown": max_drawdown(r.to_numpy()) if len(r) else None,
        "hit_rate_vs_benchmark": float((ex > 0).mean()) if len(ex) else None,
    }


@dataclass(frozen=True)
class Split:
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    final_start: pd.Timestamp
    final_end: pd.Timestamp

    def segment(self, date: pd.Timestamp) -> Optional[str]:
        if self.dev_start <= date <= self.dev_end:
            return "DEV"
        if self.final_start <= date <= self.final_end:
            return "FINAL"
        return None

    def dev_usable(self, date: pd.Timestamp, exit_date: Optional[pd.Timestamp]) -> bool:
        """A DEV observation counts only if its forward window ends before
        FINAL starts (no outcome overlap with the final period)."""
        return exit_date is not None and exit_date < self.final_start
