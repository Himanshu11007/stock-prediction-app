"""
prediction_v2/backtest.py — chronological backtest of the v2 baseline rules.

For every session t (the cutoff), features use bars <= t only; the call is
evaluated from the close of t to the close of the h-th *market* session
after t (the NIFTY calendar), like a TOMORROW_EOD snapshot. A stock without
a bar on t or on that exact session gives no trade: a missing bar never
silently stretches the horizon (fixed in this version; earlier runs stepped
along each stock's own bars). Sessions are split chronologically into development,
validation and a final holdout, with an embargo of `embargo` sessions plus
the horizon at each boundary so no outcome window straddles two splits.

Compared strategies (same sessions, same costs):
  v2           the production rules (prediction_v2.rules.decide)
  always_neutral   makes no calls (the zero-risk baseline)
  previous_day     UP if the last session closed up, else DOWN
  sector           direction of the stock's sector basket over the last session
  random_matched   random UP/DOWN on the same number of calls per session as
                   v2, from the same stocks' pool (seeded, reproducible)
  v1_rank          optional: UP for v1 top-N stocks at the nearest month-end
                   ranking snapshot on or before t (from a supplied mapping)
  always_up        UP on every evaluated stock-session (market drift benchmark)

Every trade also records its gross return, its return in excess of NIFTY
over the same window (direction-adjusted), the stock's sector, the market
regime at t (NIFTY close vs its 50-session average, and 20-session realised
volatility vs its trailing one-year median - both from bars <= t) and a
liquidity bucket from the 20-session traded value at t, so results can be
broken down without re-running.

Nothing is tuned here: thresholds come from rules.THRESHOLDS. Results on a
handful of stocks prove nothing; the report states sample sizes.
"""
from __future__ import annotations

import bisect
import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from prediction_v2 import features as feat, rules
from prediction_v2.performance import mean_ci, wilson

STRATEGIES = ("v2", "always_neutral", "previous_day", "sector", "random_matched", "v1_rank", "always_up")


@dataclass
class Trade:
    strategy: str
    symbol: str
    cutoff: dt.date
    direction: str
    ret: float                # direction-adjusted, before costs
    split: str
    setup: str = ""
    excess: Optional[float] = None    # direction-adjusted return minus NIFTY over the same window
    sector: Optional[str] = None
    regime_trend: Optional[str] = None
    regime_vol: Optional[str] = None
    liquidity: Optional[str] = None


@dataclass
class BacktestResult:
    sessions: dict[str, list[dt.date]]
    trades: list[Trade] = field(default_factory=list)
    evaluated_stock_sessions: dict[str, int] = field(default_factory=dict)
    no_calls: dict[str, int] = field(default_factory=dict)

    def report(self, cost_bps: float = 20.0) -> dict:
        out: dict = {"cost_bps": cost_bps, "splits": {}}
        for split in self.sessions:
            rows = {strat: summarize_trades([t for t in self.trades if t.strategy == strat and t.split == split],
                                            cost_bps) for strat in STRATEGIES}
            evaluated = self.evaluated_stock_sessions.get(split, 0)
            rows["v2"]["coverage"] = rows["v2"]["calls"] / evaluated if evaluated else None
            rows["v2"]["no_call_rate"] = self.no_calls.get(split, 0) / evaluated if evaluated else None
            out["splits"][split] = {"sessions": len(self.sessions[split]),
                                    "first": self.sessions[split][0].isoformat() if self.sessions[split] else None,
                                    "last": self.sessions[split][-1].isoformat() if self.sessions[split] else None,
                                    "strategies": rows}
        return out

    def breakdown(self, by: str, split: str = "holdout", cost_bps: float = 20.0,
                  strategies: tuple[str, ...] = ("v2",)) -> dict:
        """Per-group statistics (`by` is a Trade attribute: setup, direction,
        sector, regime_trend, regime_vol or liquidity)."""
        out: dict = {}
        for strat in strategies:
            groups: dict[str, list[Trade]] = {}
            for t in self.trades:
                if t.strategy == strat and t.split == split:
                    groups.setdefault(str(getattr(t, by) or "UNKNOWN"), []).append(t)
            out[strat] = {k: summarize_trades(v, cost_bps, bootstrap=False) for k, v in sorted(groups.items())}
        return out


def summarize_trades(tr: list[Trade], cost_bps: float, bootstrap: bool = True, n_boot: int = 1000,
                     seed: int = 11) -> dict:
    """Calls, hit rate (gross direction-adjusted return > 0) with a Wilson
    interval, mean gross / net / excess return, a naive 95% interval that
    treats trades as independent and - when `bootstrap` - a 95% interval
    from resampling whole cutoff sessions (trades on the same day are
    correlated, so the naive interval overstates certainty)."""
    net = [t.ret - cost_bps / 1e4 for t in tr]
    hits = sum(1 for t in tr if t.ret > 0)
    ci = mean_ci(net)
    exc = [t.excess for t in tr if t.excess is not None]
    row = {"calls": len(tr), "sessions_with_calls": len({t.cutoff for t in tr}),
           "hit_rate": hits / len(tr) if tr else None, "hit_rate_ci95": wilson(hits, len(tr)),
           "net_hit_rate": sum(1 for x in net if x > 0) / len(net) if net else None,
           "mean_gross_return": sum(t.ret for t in tr) / len(tr) if tr else None,
           "mean_net_return": ci[0] if ci else None, "mean_net_return_ci95": ci[1:] if ci else None,
           "mean_excess_vs_nifty": sum(exc) / len(exc) if exc else None}
    if bootstrap:
        row["mean_net_return_ci95_clustered"] = cluster_bootstrap(tr, cost_bps, n_boot, seed)
    return row


def cluster_bootstrap(tr: list[Trade], cost_bps: float, n_boot: int = 1000,
                      seed: int = 11) -> Optional[tuple[float, float]]:
    """95% percentile interval of the mean net return per trade, resampling
    cutoff sessions with replacement (keeps each day's trades together)."""
    by_day: dict[dt.date, list[float]] = {}
    for t in tr:
        by_day.setdefault(t.cutoff, []).append(t.ret - cost_bps / 1e4)
    days = [(sum(v), len(v)) for v in by_day.values()]
    if len(days) < 10:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(len(days)):
            a, b = days[rng.randrange(len(days))]
            tot += a
            cnt += b
        means.append(tot / cnt)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def split_sessions(sessions: list[dt.date], fractions=(0.6, 0.2, 0.2), gap: int = 6) -> dict[str, list[dt.date]]:
    """Chronological development / validation / holdout with `gap` sessions
    dropped before each later split (embargo + horizon)."""
    n = len(sessions)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return {"development": sessions[:a], "validation": sessions[a + gap:b], "holdout": sessions[b + gap:]}


def _regimes(nifty: pd.DataFrame) -> dict[dt.date, tuple[Optional[str], Optional[str]]]:
    """Market regime at each session from NIFTY bars <= that session only."""
    c = nifty["Close"]
    sma50 = c.rolling(50).mean()
    vol20 = c.pct_change().rolling(20).std()
    vol_med = vol20.rolling(250, min_periods=120).median()
    out = {}
    for d, x, m, v, vm in zip(c.index, c, sma50, vol20, vol_med):
        trend = None if pd.isna(m) else ("ABOVE_SMA50" if x > m else "BELOW_SMA50")
        vol = None if pd.isna(v) or pd.isna(vm) else ("HIGH_VOL" if v > vm else "LOW_VOL")
        out[d.date()] = (trend, vol)
    return out


def _liquidity(traded_value: Optional[float]) -> Optional[str]:
    """Fixed a-priori buckets of 20-session mean traded value (INR)."""
    if traded_value is None:
        return None
    if traded_value < 2e8:
        return "<20cr"
    if traded_value < 1e9:
        return "20-100cr"
    return ">=100cr"


def run(histories: dict[str, pd.DataFrame], nifty: pd.DataFrame, sectors: dict[str, Optional[str]],
        horizon: int = 1, embargo: int = 5, seed: int = 7, min_history: int = feat.MIN_SESSIONS_FOR_CALL,
        v1_top: Optional[dict[dt.date, set[str]]] = None, decide: Callable = rules.decide) -> BacktestResult:
    calendar_ = [d.date() for d in nifty.index]
    cal_pos = {d: i for i, d in enumerate(calendar_)}
    usable = calendar_[min_history:len(calendar_) - horizon]
    splits = split_sessions(usable, gap=embargo + horizon)
    split_of = {d: k for k, v in splits.items() for d in v}
    rng = random.Random(seed)
    v1_dates = sorted(v1_top) if v1_top else []
    res = BacktestResult(sessions=splits)
    closes = {s: df["Close"] for s, df in histories.items()}
    pos = {s: {d.date(): i for i, d in enumerate(df.index)} for s, df in histories.items()}
    nclose = nifty["Close"]
    regimes = _regimes(nifty)

    def fwd(sym: str, t: dt.date) -> Optional[float]:
        """Close(t) -> close of the horizon-th market session after t; None
        when the stock has no bar on either of those sessions."""
        end = calendar_[cal_pos[t] + horizon]
        i, j = pos[sym].get(t), pos[sym].get(end)
        if i is None or j is None:
            return None
        c = closes[sym]
        return float(c.iloc[j] / c.iloc[i] - 1)

    for t in usable:
        split = split_of.get(t)
        if split is None:
            continue                                     # embargoed session
        k_t = cal_pos[t]
        n_ret = float(nclose.iloc[k_t + horizon] / nclose.iloc[k_t] - 1)
        trend, vol = regimes.get(t, (None, None))
        feats, flags = {}, {}
        for sym, df in histories.items():
            if t not in pos[sym]:
                continue
            feats[sym], flags[sym] = feat.compute(df, t, nifty, expected_session=t)
        feat.add_sector_relative(feats, sectors)
        sector_ret: dict[str, list[float]] = {}
        for sym, f in feats.items():
            if f.get("ret_1") is not None and sectors.get(sym):
                sector_ret.setdefault(sectors[sym], []).append(f["ret_1"])
        v2_calls = 0
        pool = []

        def trade(strategy: str, sym: str, s: int, r: float, setup: str = "") -> Trade:
            return Trade(strategy, sym, t, "UP" if s > 0 else "DOWN", s * r, split, setup, excess=s * (r - n_ret),
                         sector=sectors.get(sym), regime_trend=trend, regime_vol=vol,
                         liquidity=_liquidity(feats[sym].get("traded_value_20")))

        for sym, f in feats.items():
            r = fwd(sym, t)
            if r is None:
                continue
            pool.append((sym, r))
            res.evaluated_stock_sessions[split] = res.evaluated_stock_sessions.get(split, 0) + 1
            d = decide(f, flags[sym])
            if d["direction"] == "NO_CALL":
                res.no_calls[split] = res.no_calls.get(split, 0) + 1
            if d["direction"] in ("UP", "DOWN"):
                v2_calls += 1
                res.trades.append(trade("v2", sym, 1 if d["direction"] == "UP" else -1, r, d["setup_type"]))
            if f.get("ret_1") is not None and f["ret_1"] != 0:
                res.trades.append(trade("previous_day", sym, 1 if f["ret_1"] > 0 else -1, r))
            sr = sector_ret.get(sectors.get(sym) or "")
            if sr and len(sr) >= 3 and sum(sr) != 0:
                res.trades.append(trade("sector", sym, 1 if sum(sr) > 0 else -1, r))
            if v1_dates:
                k = bisect.bisect_right(v1_dates, t) - 1
                if k >= 0 and sym in v1_top[v1_dates[k]]:
                    res.trades.append(trade("v1_rank", sym, 1, r))
            res.trades.append(trade("always_up", sym, 1, r))
        for sym, r in rng.sample(pool, min(v2_calls, len(pool))):
            res.trades.append(trade("random_matched", sym, rng.choice((1, -1)), r))
    return res
