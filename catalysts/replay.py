"""
catalysts/replay.py — point-in-time historical replay of the news engine.

Input: an archive of articles that records, for each, when StockLens (or the
archive's collector) actually received it (`ingested_at`), plus optional
corrections received later. The replay never uses an article before its
ingestion time:

  for each cutoff T (chronological):
      ingest every not-yet-ingested article with ingested_at <= T, in
      ingestion order, with the pipeline clock set to that ingestion time
      (so classifications and entity links are dated correctly)
      assess every universe stock at T (catalysts.impact, price data
      truncated at the cutoff session)
      TOMORROW_EOD semantics: entry = close of the cutoff session, exit =
      close of the next market session; excess vs NIFTY; costs
      unscored when the stock has no bar on either session (recorded)

Baselines on the same stock-sessions: always neutral, previous-session
direction, and random direction matched to the number of news calls.
Everything runs in an isolated in-memory database; nothing persistent is
touched.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from catalysts import impact, pipeline
from catalysts.providers import RawArticle
from db.models.stock import Company
from prediction_v2 import features as feat
from prediction_v2.performance import mean_ci, wilson
from utils.market_session import IST


@dataclass(frozen=True)
class ArchivedArticle:
    article: RawArticle
    ingested_at: dt.datetime
    provider: str = "archive"


@dataclass
class ReplayCall:
    cutoff: dt.date
    symbol: str
    direction: str
    ret: Optional[float]                 # direction-adjusted next-session return (None = unscored)
    excess: Optional[float]
    score: float
    events: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReplayResult:
    calls: list[ReplayCall] = field(default_factory=list)
    neutral: int = 0
    no_news: int = 0
    no_data: int = 0
    unscored: int = 0
    baselines: dict[str, list[float]] = field(default_factory=dict)
    sessions: list[str] = field(default_factory=list)

    def summary(self, cost_bps: float = 25.0) -> dict:
        scored = [c for c in self.calls if c.ret is not None]
        hits = sum(1 for c in scored if c.ret > 0)
        net = [c.ret - cost_bps / 1e4 for c in scored]
        ci = mean_ci(net)
        out = {"sessions": len(self.sessions), "news_calls": len(self.calls), "scored": len(scored),
               "unscored": self.unscored, "neutral_with_news": self.neutral, "no_news": self.no_news,
               "no_data": self.no_data, "hit_rate": hits / len(scored) if scored else None,
               "hit_ci95": wilson(hits, len(scored)), "mean_net": ci[0] if ci else None,
               "mean_net_ci95": ci[1:] if ci else None,
               "median_net": sorted(net)[len(net) // 2] if net else None,
               "mean_excess_vs_nifty": (sum(c.excess for c in scored if c.excess is not None) / len(scored))
               if scored else None, "cost_bps": cost_bps}
        for name, rets in self.baselines.items():
            n = [r - cost_bps / 1e4 for r in rets]
            out[f"baseline_{name}"] = {"calls": len(n), "hit_rate": sum(1 for r in rets if r > 0) / len(rets) if rets else None,
                                       "mean_net": sum(n) / len(n) if n else None}
        return out


def _close_on(df: Optional[pd.DataFrame], day: dt.date) -> Optional[float]:
    if df is None:
        return None
    s = df["Close"][df.index.normalize() == pd.Timestamp(day)]
    return float(s.iloc[0]) if len(s) else None


def run(archive: Iterable[ArchivedArticle], cutoffs: list[dt.datetime], companies: list[tuple[str, str, Optional[str], Optional[str]]],
        prices: dict[str, pd.DataFrame], nifty: pd.DataFrame, seed: int = 7) -> ReplayResult:
    """`cutoffs`: aware datetimes after the close of each Indian session (IST
    date = the cutoff session). `companies`: (symbol, name, sector, industry)."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        for sym, name, sec, ind in companies:
            s.add(Company(symbol=sym, name=name, sector=sec, industry=ind))
        s.commit()
    pending = sorted(archive, key=lambda a: a.ingested_at)
    cal = [d.date() for d in nifty.index]
    res = ReplayResult(baselines={"previous_day": [], "random_matched": []})
    rng = random.Random(seed)
    k = 0
    for cutoff in sorted(cutoffs):
        day = cutoff.astimezone(IST).date()
        if day not in cal or cal.index(day) + 1 >= len(cal):
            continue
        nxt = cal[cal.index(day) + 1]
        while k < len(pending) and pending[k].ingested_at <= cutoff:          # never before ingestion
            batch_time = pending[k].ingested_at
            batch = []
            while k < len(pending) and pending[k].ingested_at == batch_time:
                batch.append(pending[k])
                k += 1
            with Session(eng) as s:
                pipeline.ingest(s, _ListProvider([b.article for b in batch], batch[0].provider),
                                batch_time - dt.timedelta(days=30), batch_time, now=batch_time)
        res.sessions.append(day.isoformat())
        n0, n1 = _close_on(nifty, day), _close_on(nifty, nxt)
        nret = (n1 / n0 - 1) if n0 and n1 else None
        news_calls = 0
        pool = []
        with Session(eng) as s:
            for sym, _, _, _ in companies:
                df = prices.get(sym)
                c0, c1 = _close_on(df, day), _close_on(df, nxt)
                d = feat.slice_to_cutoff(df, day) if df is not None else None
                if d is None or c0 is None:
                    res.no_data += 1
                    continue
                f, _ = feat.compute(df, day, nifty, expected_session=day)
                a = impact.assess_stock(s, sym, cutoff, _move(df, nifty, day, f.get("atr_pct")))
                r = (c1 / c0 - 1) if c1 else None
                if r is not None:
                    pool.append(r)
                    if f.get("ret_1"):
                        res.baselines["previous_day"].append((1 if f["ret_1"] > 0 else -1) * r)
                if a.status == "NO_NEWS":
                    res.no_news += 1
                    continue
                if a.status != "DIRECTIONAL":
                    res.neutral += 1
                    continue
                sign = 1 if a.direction == "UP" else -1
                news_calls += 1
                if r is None:
                    res.unscored += 1
                res.calls.append(ReplayCall(day, sym, a.direction, None if r is None else sign * r,
                                            None if r is None or nret is None else sign * (r - nret), a.score,
                                            [e.event_id for e in a.evidence], a.reasons[0] if a.reasons else ""))
        for r in rng.sample(pool, min(news_calls, len(pool))):
            res.baselines["random_matched"].append(rng.choice((1, -1)) * r)
    return res


def _move(df, nifty, day, atr):
    def move(since: dt.datetime):
        if not atr:
            return None
        local = since.astimezone(IST)
        base = local.date() if local.time() >= dt.time(15, 30) else local.date() - dt.timedelta(days=1)
        d0 = df[df.index.normalize() <= pd.Timestamp(base)]
        n0 = nifty[nifty.index.normalize() <= pd.Timestamp(base)]
        c, nc = _close_on(df, day), _close_on(nifty, day)
        if d0.empty or n0.empty or not c or not nc:
            return None
        return (c / float(d0["Close"].iloc[-1]) - 1) - (nc / float(n0["Close"].iloc[-1]) - 1), float(atr)
    return move


class _ListProvider:
    def __init__(self, items, name):
        self.items, self.name = items, name

    def fetch(self, since, until):
        return list(self.items)
