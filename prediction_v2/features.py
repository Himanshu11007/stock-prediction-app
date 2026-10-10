"""
prediction_v2/features.py — short-horizon features from DAILY bars.

Cutoff safety: every feature is computed from bars dated on or before
`cutoff_date` only (the frame is sliced first; nothing later can leak in).
Missing inputs give None plus an explicit flag, never 0 and never an
estimate. Formulas (n = sessions, c = close, h = high, l = low, v = volume,
index -1 = the cutoff session):

  ret_n            c[-1] / c[-1-n] - 1                  n in 1, 3, 5, 20
  gap_pct          open[-1] / c[-2] - 1                 opening gap of the cutoff session
  range_pct        (h[-1] - l[-1]) / c[-2]
  close_location   (c[-1] - l[-1]) / (h[-1] - l[-1])    0 = closed at low, 1 = at high
  abnormal_volume  v[-1] / median(v[-21:-1])            needs 20 prior sessions, median > 0
  atr_pct          mean(true range, last 14) / c[-1]    TR = max(h-l, |h-c_prev|, |l-c_prev|)
  vol_20           stdev(daily returns, last 20) * sqrt(252)
  dist_high_20     c[-1] / max(h[-20:]) - 1             <= 0
  dist_low_20      c[-1] / min(l[-20:]) - 1             >= 0
  rs_nifty_n       ret_n(stock) - ret_n(NIFTY 50)       same cutoff, n in 5, 20
  sector_rs_5      ret_5 - mean(ret_5 of sector peers)  >= 3 peers, self excluded
  traded_value_20  mean(c * v, last 20)                 liquidity proxy (INR)

Lookback: at most 21 sessions for features, 61 sessions required for a call.
"""
from __future__ import annotations

import datetime as dt
import math
from statistics import median
from typing import Any, Optional

import pandas as pd

FEATURE_SET_VERSION = "v2-features-0.1"

MIN_SESSIONS_FOR_CALL = 61          # 60 returns of history (rules require it)
MIN_SESSIONS_FOR_BASICS = 21
LOW_LIQUIDITY_TRADED_VALUE = 5e7    # INR 5 crore per day (20-day mean)
SUSPECT_VOLUME_RATIO = 50.0         # far outside normal; likely a data error, verify before use
LIMIT_MOVE_PCT = 0.095              # >= ~10% close at the day's extreme: possible price band / circuit


def slice_to_cutoff(df: Optional[pd.DataFrame], cutoff_date: dt.date) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df[df.index.normalize() <= pd.Timestamp(cutoff_date)]
    out = out.dropna(subset=["Close"])
    return out if not out.empty else None


def _ret(c: pd.Series, n: int) -> Optional[float]:
    if len(c) <= n or c.iloc[-1 - n] <= 0:
        return None
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1)


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def compute(df: Optional[pd.DataFrame], cutoff_date: dt.date, nifty: Optional[pd.DataFrame] = None,
            expected_session: Optional[dt.date] = None) -> tuple[dict[str, Any], list[str]]:
    """Features of one stock as of the close of `cutoff_date`.

    `expected_session` is the session the cutoff should contain; if the
    stock's last bar is older, the price is STALE (no call)."""
    flags: list[str] = []
    f: dict[str, Any] = {"feature_set_version": FEATURE_SET_VERSION, "cutoff_date": cutoff_date.isoformat()}
    d = slice_to_cutoff(df, cutoff_date)
    if d is None:
        f["history_sessions"] = 0
        return f, ["NO_PRICE_DATA"]
    c, h, l, o, v = d["Close"], d["High"], d["Low"], d["Open"], d["Volume"].fillna(0)
    n = len(d)
    last_date = d.index[-1].date()
    f.update(history_sessions=n, last_session_date=last_date.isoformat(), close=_num(c.iloc[-1]),
             open=_num(o.iloc[-1]), high=_num(h.iloc[-1]), low=_num(l.iloc[-1]))
    if expected_session is not None and last_date < expected_session:
        flags.append("STALE_PRICE")
    if n < MIN_SESSIONS_FOR_BASICS:
        flags.append("INSUFFICIENT_HISTORY_20")
    if n < MIN_SESSIONS_FOR_CALL:
        flags.append("INSUFFICIENT_HISTORY_60")

    for k in (1, 3, 5, 20):
        f[f"ret_{k}"] = _ret(c, k)
    if n >= 2:
        prev = float(c.iloc[-2])
        f["gap_pct"] = _num(o.iloc[-1] / prev - 1) if prev > 0 else None
        f["range_pct"] = _num((h.iloc[-1] - l.iloc[-1]) / prev) if prev > 0 else None
    else:
        f["gap_pct"] = f["range_pct"] = None
    rng = float(h.iloc[-1] - l.iloc[-1])
    f["close_location"] = _num((c.iloc[-1] - l.iloc[-1]) / rng) if rng > 0 else None

    if float(v.iloc[-1]) <= 0:
        flags.append("ZERO_VOLUME_LAST")
    if n >= 21:
        base = median(float(x) for x in v.iloc[-21:-1])
        f["abnormal_volume"] = _num(v.iloc[-1] / base) if base > 0 else None
        if f["abnormal_volume"] is not None and f["abnormal_volume"] >= SUSPECT_VOLUME_RATIO:
            flags.append("SUSPECT_VOLUME_SPIKE")
    else:
        f["abnormal_volume"] = None

    if n >= 15:
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1).iloc[-14:]
        f["atr_pct"] = _num(tr.mean() / c.iloc[-1]) if c.iloc[-1] > 0 else None
    else:
        f["atr_pct"] = None
    if n >= 21:
        rets = c.pct_change().iloc[-20:]
        f["vol_20"] = _num(rets.std(ddof=1) * math.sqrt(252))
        f["dist_high_20"] = _num(c.iloc[-1] / h.iloc[-20:].max() - 1)
        f["dist_low_20"] = _num(c.iloc[-1] / l.iloc[-20:].min() - 1)
        f["traded_value_20"] = _num((c.iloc[-20:] * v.iloc[-20:]).mean())
        if f["traded_value_20"] is not None and f["traded_value_20"] < LOW_LIQUIDITY_TRADED_VALUE:
            flags.append("LOW_LIQUIDITY")
    else:
        f["vol_20"] = f["dist_high_20"] = f["dist_low_20"] = f["traded_value_20"] = None

    r1 = f["ret_1"]
    if r1 is not None and abs(r1) >= LIMIT_MOVE_PCT and rng > 0 and \
            (c.iloc[-1] >= h.iloc[-1] - 1e-9 or c.iloc[-1] <= l.iloc[-1] + 1e-9):
        flags.append("POSSIBLE_PRICE_BAND")

    nd = slice_to_cutoff(nifty, cutoff_date)
    if nd is None:
        flags.append("NO_BENCHMARK")
        f["rs_nifty_5"] = f["rs_nifty_20"] = None
    else:
        if nd.index[-1].date() != last_date:
            flags.append("BENCHMARK_MISALIGNED")
        for k in (5, 20):
            br, sr = _ret(nd["Close"], k), f[f"ret_{k}"]
            f[f"rs_nifty_{k}"] = _num(sr - br) if sr is not None and br is not None else None
    f["sector_rs_5"] = None            # filled by add_sector_relative()
    return f, flags


def add_sector_relative(features: dict[str, dict], sectors: dict[str, Optional[str]], min_peers: int = 3) -> None:
    """sector_rs_5 = own ret_5 - mean ret_5 of same-sector peers (self
    excluded), only when at least `min_peers` peers have a value."""
    by_sector: dict[str, list[tuple[str, float]]] = {}
    for sym, f in features.items():
        s, r = sectors.get(sym), f.get("ret_5")
        if s and r is not None:
            by_sector.setdefault(s, []).append((sym, r))
    for sym, f in features.items():
        s, r = sectors.get(sym), f.get("ret_5")
        peers = [x for p, x in by_sector.get(s or "", []) if p != sym]
        f["sector_peer_count"] = len(peers)
        f["sector_rs_5"] = (r - sum(peers) / len(peers)) if r is not None and len(peers) >= min_peers else None
