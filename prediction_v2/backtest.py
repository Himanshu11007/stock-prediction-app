"""
prediction_v2/backtest.py — chronological backtest of the v2 baseline rules.

For every session t (the cutoff), features use bars <= t only; the call is
evaluated over sessions t+1 .. t+h (entry at the close of t, like a
TOMORROW_EOD snapshot). Sessions are split chronologically into development,
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

STRATEGIES = ("v2", "always_neutral", "previous_day", "sector", "random_matched", "v1_rank")


@dataclass
class Trade:
    strategy: str
    symbol: str
    cutoff: dt.date
    direction: str
    ret: float                # direction-adjusted, before costs
    split: str
    setup: str = ""


@dataclass
class BacktestResult:
    sessions: dict[str, list[dt.date]]
    trades: list[Trade] = field(default_factory=list)
    evaluated_stock_sessions: dict[str, int] = field(default_factory=dict)
    no_calls: dict[str, int] = field(default_factory=dict)

    def report(self, cost_bps: float = 20.0) -> dict:
        out: dict = {"cost_bps": cost_bps, "splits": {}}
        for split in self.sessions:
            rows = {}
            for strat in STRATEGIES:
                tr = [t for t in self.trades if t.strategy == strat and t.split == split]
                net = [t.ret - cost_bps / 1e4 for t in tr]
                hits = sum(1 for t in tr if t.ret > 0)
                ci = mean_ci(net)
                rows[strat] = {"calls": len(tr), "hit_rate": hits / len(tr) if tr else None,
                               "hit_rate_ci95": wilson(hits, len(tr)), "mean_net_return": ci[0] if ci else None,
                               "mean_net_return_ci95": ci[1:] if ci else None}
            evaluated = self.evaluated_stock_sessions.get(split, 0)
            rows["v2"]["coverage"] = rows["v2"]["calls"] / evaluated if evaluated else None
            rows["v2"]["no_call_rate"] = self.no_calls.get(split, 0) / evaluated if evaluated else None
            out["splits"][split] = {"sessions": len(self.sessions[split]),
                                    "first": self.sessions[split][0].isoformat() if self.sessions[split] else None,
                                    "last": self.sessions[split][-1].isoformat() if self.sessions[split] else None,
                                    "strategies": rows}
        return out


def split_sessions(sessions: list[dt.date], fractions=(0.6, 0.2, 0.2), gap: int = 6) -> dict[str, list[dt.date]]:
    """Chronological development / validation / holdout with `gap` sessions
    dropped before each later split (embargo + horizon)."""
    n = len(sessions)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return {"development": sessions[:a], "validation": sessions[a + gap:b], "holdout": sessions[b + gap:]}


def run(histories: dict[str, pd.DataFrame], nifty: pd.DataFrame, sectors: dict[str, Optional[str]],
        horizon: int = 1, embargo: int = 5, seed: int = 7, min_history: int = feat.MIN_SESSIONS_FOR_CALL,
        v1_top: Optional[dict[dt.date, set[str]]] = None, decide: Callable = rules.decide) -> BacktestResult:
    calendar_ = [d.date() for d in nifty.index]
    usable = calendar_[min_history:len(calendar_) - horizon]
    splits = split_sessions(usable, gap=embargo + horizon)
    split_of = {d: k for k, v in splits.items() for d in v}
    rng = random.Random(seed)
    v1_dates = sorted(v1_top) if v1_top else []
    res = BacktestResult(sessions=splits)
    closes = {s: df["Close"] for s, df in histories.items()}
    pos = {s: {d.date(): i for i, d in enumerate(df.index)} for s, df in histories.items()}

    def fwd(sym: str, t: dt.date) -> Optional[float]:
        i = pos[sym].get(t)
        c = closes[sym]
        if i is None or i + horizon >= len(c):
            return None
        return float(c.iloc[i + horizon] / c.iloc[i] - 1)

    for t in usable:
        split = split_of.get(t)
        if split is None:
            continue                                     # embargoed session
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
                s = 1 if d["direction"] == "UP" else -1
                res.trades.append(Trade("v2", sym, t, d["direction"], s * r, split, d["setup_type"]))
            if f.get("ret_1") is not None and f["ret_1"] != 0:
                s = 1 if f["ret_1"] > 0 else -1
                res.trades.append(Trade("previous_day", sym, t, "UP" if s > 0 else "DOWN", s * r, split))
            sr = sector_ret.get(sectors.get(sym) or "")
            if sr and len(sr) >= 3 and sum(sr) != 0:
                s = 1 if sum(sr) > 0 else -1
                res.trades.append(Trade("sector", sym, t, "UP" if s > 0 else "DOWN", s * r, split))
            if v1_dates:
                k = bisect.bisect_right(v1_dates, t) - 1
                if k >= 0 and sym in v1_top[v1_dates[k]]:
                    res.trades.append(Trade("v1_rank", sym, t, "UP", r, split))
        for sym, r in rng.sample(pool, min(v2_calls, len(pool))):
            s = rng.choice((1, -1))
            res.trades.append(Trade("random_matched", sym, t, "UP" if s > 0 else "DOWN", s * r, split))
    return res
