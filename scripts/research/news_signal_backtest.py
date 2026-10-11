"""
scripts/research/news_signal_backtest.py — does news improve StockLens
predictions? Point-in-time, chronological comparison (RESEARCH ONLY; the
v2 engine stays in shadow; Ranking Engine v1.0 is only read).

  python scripts/research/news_signal_backtest.py [--months 12] [--cost 20] [--slippage 10]
                                                  [--providers google_news,gdelt,stocklens_archive]
                                                  [--cache DIR] [--out PATH]

Universe: the 20 large caps of catalysts.benchmark.INDIAN_ALIASES (chosen in
advance for unambiguous headline names, not for performance).
Arms on identical stock-sessions (see catalysts/signal_eval.py): v1_rank,
v2, news_<provider>, news_combined, v2_plus_news and the baselines.

Point in time:
  * StockLens polls hourly: a backfilled article becomes usable at the next
    full hour after its publication time (date-only items: end of that day,
    IST, then the next hour) - never earlier.
  * GDELT: its first-seen time replaces the publication time.
  * StockLens's own archive: the recorded ingestion time.
  * catalysts.replay never uses an article before it was usable, nor a
    corrected article before its correction; duplicates are grouped by the
    pipeline; cutoffs are 19:30 IST, outcomes close(t) -> close(t+1).
Known leakage risk, stated in the report: Google News is not a
point-in-time archive (headlines can be edited later and results are ranked
by Google today); its arm is optimistic by construction.

Output: scripts/research/output/prediction_v2/news_signal_backtest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from catalysts import adapters as ad, benchmark as bm, providers as prov, replay, signal_eval as se  # noqa: E402
from prediction_v2 import backtest  # noqa: E402

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "news_signal_backtest.json"
DB = Path(r"C:\Python\Git\stock-prediction-app\storage\app.db")
VALIDATION_CSV = ROOT / "scripts" / "research" / "output" / "ranking_validation" / "ranking_snapshots_production.csv"
MACRO_QUERIES = ("RBI repo rate", "crude oil prices India", "rupee dollar", "India tariff")
WINDOW_DAYS = 14


def ceil_hour(t: dt.datetime) -> dt.datetime:
    base = t.replace(minute=0, second=0, microsecond=0)
    return base if base == t else base + dt.timedelta(hours=1)


def usable_at(a: prov.RawArticle) -> dt.datetime:
    t = a.first_seen_at or a.published_at
    if a.time_precision == "date":
        local = t.astimezone(IST)
        t = dt.datetime.combine(local.date(), dt.time(23, 59), tzinfo=IST).astimezone(UTC)
    return ceil_hour(t + dt.timedelta(seconds=1))


def _ser(a: prov.RawArticle) -> dict:
    return {"id": a.provider_article_id, "url": a.url, "title": a.title, "published_at": a.published_at.isoformat(),
            "excerpt": a.excerpt, "source_name": a.source_name, "source_domain": a.source_domain,
            "time_precision": a.time_precision, "first_seen_at": a.first_seen_at.isoformat() if a.first_seen_at else None}


def _de(d: dict) -> prov.RawArticle:
    return prov.RawArticle(d["id"], d["url"], d["title"], dt.datetime.fromisoformat(d["published_at"]), d.get("excerpt") or "",
                           d.get("source_name"), time_precision=d.get("time_precision") or "exact",
                           source_domain=d.get("source_domain"),
                           first_seen_at=dt.datetime.fromisoformat(d["first_seen_at"]) if d.get("first_seen_at") else None)


def windows(start: dt.datetime, end: dt.datetime, days: int = WINDOW_DAYS):
    t = start
    while t < end:
        yield t, min(end, t + dt.timedelta(days=days))
        t += dt.timedelta(days=days)


def collect(provider: str, start: dt.datetime, end: dt.datetime, cache: Path) -> tuple[list[prov.RawArticle], dict]:
    """Backfill a provider's archive for the universe (cached on disk; the
    cache holds third-party headlines and stays OUTSIDE the repository)."""
    f = cache / f"news_backfill_{provider}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return [_de(x) for x in d["articles"]], {**d["stats"], "from_cache": True}
    queries = [f'"{al.split("|")[0]}"' for al in bm.INDIAN_ALIASES.values()]
    stats = {"requests": 0, "ok": 0, "rate_limited": 0, "errors": 0, "not_configured": None}
    arts: dict[str, prov.RawArticle] = {}
    if provider == "gdelt":
        start = max(start, end - dt.timedelta(days=85))                    # DOC API history ~3 months
    for lo, hi in windows(start, end):
        for q in queries + (list(MACRO_QUERIES) if provider == "google_news" else []):
            stats["requests"] += 1
            try:
                if provider == "google_news":
                    time.sleep(2.0)
                    got = ad.GoogleNewsProvider(queries=[q], sleep=lambda s: None).fetch(lo, hi)
                elif provider == "gdelt":
                    time.sleep(6.5)
                    got = prov.GdeltProvider(query=f"{q} sourcelang:english", sleep=time.sleep).fetch(lo, hi)
                else:
                    raise ValueError(provider)
                for a in got:
                    if lo - dt.timedelta(days=1) <= a.published_at <= hi:
                        arts.setdefault(a.url, a)
                stats["ok"] += 1
            except prov.NotConfigured as e:
                stats["not_configured"] = str(e)[:120]
                break
            except prov.RateLimited:
                stats["rate_limited"] += 1
            except prov.ProviderError:
                stats["errors"] += 1
    out = sorted(arts.values(), key=lambda a: a.published_at)
    stats["articles"] = len(out)
    f.write_text(json.dumps({"stats": stats, "articles": [_ser(a) for a in out]}), encoding="utf-8")
    return out, stats


def own_archive() -> tuple[list[replay.ArchivedArticle], dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT provider_article_id, url, title, published_at, excerpt, source_name, ingested_at, "
                           "corrected_at, provider FROM news_articles").fetchall()
    except sqlite3.OperationalError as e:
        return [], {"status": "UNAVAILABLE", "detail": str(e)[:120]}
    out = []
    for pid, url, title, pub, exc, src, ing, cor, p in rows:
        t = lambda s: dt.datetime.fromisoformat(str(s)).replace(tzinfo=UTC) if s else None  # noqa: E731
        out.append(replay.ArchivedArticle(prov.RawArticle(pid, url, title, t(pub), exc or "", src, corrected_at=t(cor)),
                                          t(ing), p))
    return out, {"status": "OK", "articles": len(out),
                 "first_ingested": min((a.ingested_at for a in out), default=None),
                 "last_ingested": max((a.ingested_at for a in out), default=None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--cost", type=float, default=20.0, help="round-trip cost, bps")
    ap.add_argument("--slippage", type=float, default=10.0, help="round-trip slippage, bps")
    ap.add_argument("--providers", default="google_news,gdelt,stocklens_archive,marketaux,alphavantage")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "stocklens_news_cache"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-note", action="append", default=[], metavar="PROVIDER=REASON",
                    help="why a provider was not run (recorded in the report)")
    args = ap.parse_args(argv)
    args.skip_note = dict(x.split("=", 1) for x in args.skip_note)
    t0 = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    syms = [f"{s}.NS" for s in bm.INDIAN_ALIASES]
    meta = {r[0]: r[1:] for r in con.execute("SELECT symbol, name, sector, industry FROM companies") if r[0] in syms}
    companies = [(s, *meta.get(s, (s, None, None))) for s in syms]
    sectors = {s: m[1] for s, m in meta.items()}

    from fundamentals.provider import fetch_price_history
    data = fetch_price_history(syms + ["^NSEI"], period="2y")
    nifty = data.pop("^NSEI")
    hist = {s: df for s, df in data.items() if df is not None and len(df) > 80}
    end = dt.datetime.combine(nifty.index[-1].date(), dt.time(19, 30), tzinfo=IST).astimezone(UTC)
    start = end - dt.timedelta(days=30 * args.months)

    v1_top = None
    if VALIDATION_CSV.exists():
        v = pd.read_csv(VALIDATION_CSV)
        v = v[(v["eligible"] == True) & (v["rank"] <= 20)]  # noqa: E712
        v1_top = {dt.date.fromisoformat(d): set(g["symbol"]) for d, g in v.groupby("date")}
    base = backtest.run(hist, nifty, sectors, horizon=1, embargo=5, seed=7, v1_top=v1_top)

    # evaluation window = the news window; re-split it chronologically (embargo 5 + horizon 1)
    cal = [d.date() for d in nifty.index]
    days = [d for d in cal[:-1] if start.astimezone(IST).date() <= d <= end.astimezone(IST).date()]
    splits = backtest.split_sessions(days, gap=6)
    split_of = {d: k for k, v in splits.items() for d in v}
    nxt = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
    evaluated = {k: 0 for k in splits}
    idx = {s: set(df.index.date) for s, df in hist.items()}
    for d, k in split_of.items():
        evaluated[k] += sum(1 for s in hist if d in idx[s] and nxt[d] in idx[s])
    trades = [backtest.Trade(**{**t.__dict__, "split": split_of[t.cutoff]}) for t in base.trades if t.cutoff in split_of]

    archives: dict[str, list[replay.ArchivedArticle]] = {}
    archive_info: dict[str, dict] = {}
    wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
    for p in wanted:
        if p in ("google_news", "gdelt"):
            arts, st = collect(p, start, end, cache)
            archives[p] = [replay.ArchivedArticle(a, usable_at(a), p) for a in arts]
            archive_info[p] = {**st, "status": "OK" if arts else "EMPTY",
                               "point_in_time": "NO (published time; headlines may be revised; ranked results)"
                               if p == "google_news" else "first-seen times (GDELT); ~3 months only"}
        elif p == "stocklens_archive":
            arch, st = own_archive()
            archives[p] = [a for a in arch if start <= a.ingested_at <= end]
            archive_info[p] = {**st, "in_window": len(archives[p]), "point_in_time": "YES (recorded ingestion times)"}
        elif p in ("marketaux", "alphavantage"):
            var = {"marketaux": "MARKETAUX_API_TOKEN", "alphavantage": "ALPHAVANTAGE_API_KEY"}[p]
            archive_info[p] = {"status": "NOT_CONFIGURED" if not os.environ.get(var) else "NOT_RUN",
                               "detail": f"{var} is not set: arm not tested" if not os.environ.get(var) else
                               "free quotas (100 / 25 requests a day) cannot backfill 20 stocks x 12 months in one run; "
                               "collect over several days or use a paid plan"}
    for p in ("google_news", "gdelt", "stocklens_archive", "marketaux", "alphavantage"):
        archive_info.setdefault(p, {"status": "NOT_RUN", "detail": args.skip_note.get(p, "not requested in this run")})
    cutoffs = [dt.datetime.combine(d, dt.time(19, 30), tzinfo=IST) for d in days]
    news_arms = []
    available = {p: a for p, a in archives.items() if a}
    runs = list(available.items()) + ([("combined", [a for v in available.values() for a in v])] if len(available) > 1 else [])
    for p, arch in runs:
        print(f"replaying {p}: {len(arch)} articles", flush=True)
        r0 = time.time()
        res = replay.run(arch, cutoffs, companies, hist, nifty)
        arm = f"news_{p}"
        trades += se.trades_from_replay(res.calls, arm, split_of, sectors)
        news_arms.append(arm)
        archive_info.setdefault(p, {})["replay"] = {"calls": len(res.calls), "neutral_with_news": res.neutral,
                                                    "no_news": res.no_news, "unscored": res.unscored,
                                                    "seconds": round(time.time() - r0, 1)}
    if len(available) == 1:                  # one archive: the combined arm IS that arm (not replayed twice)
        only_arm = f"news_{next(iter(available))}"
        trades += [backtest.Trade(**{**t.__dict__, "strategy": "news_combined"}) for t in trades if t.strategy == only_arm]
        news_arms.append("news_combined")
        archive_info["combined"] = {"status": "SAME_AS", "detail": f"only one archive available: identical to {only_arm}"}
    v2 = [t for t in trades if t.strategy == "v2"]
    if "news_combined" in news_arms:
        trades += se.overlay(v2, [t for t in trades if t.strategy == "news_combined"], "v2_plus_news")
    total = args.cost + args.slippage
    cmp = se.compare(trades, splits, evaluated, total, category_arms=tuple(news_arms))
    sens = {f"{c}bps": {k: {a: {x: r[x] for x in ("calls", "hit_rate", "mean_net_return", "max_drawdown")}
                            for a, r in v["arms"].items()} for k, v in se.compare(trades, splits, evaluated, c)["splits"].items()}
            for c in (10.0, 50.0)}
    verdicts = [se.verdict(cmp, a, b) for a in news_arms + ["v2_plus_news", "v2", "v1_rank"]
                for b in ("v2", "always_up", "random_matched") if a != b]
    report = {
        "generated_at": dt.datetime.now(UTC).isoformat(), "research_only": True,
        "universe": syms, "universe_with_prices": sorted(hist), "window": {"start": start.isoformat(), "end": end.isoformat()},
        "costs_bps": {"cost": args.cost, "slippage": args.slippage, "total_round_trip": total},
        "archives": archive_info, "comparison": cmp, "cost_sensitivity": sens, "verdicts": verdicts,
        "limitations": [
            "20 large caps only: results do not generalise to mid / small caps.",
            "Google News is not point in time (edited headlines, ranking today): its arm is optimistic.",
            "GDELT covers about 3 months and is rate-limited from this network.",
            "StockLens's own archive is only as long as ingestion has run (scheduler not active).",
            "Marketaux / Alpha Vantage arms are untested without keys; free quotas cannot backfill a year.",
            "NSE / BSE filings are not used: exchange terms prohibit automated collection without permission.",
            "Daily Yahoo bars; v1 baseline from month-end research snapshots; thresholds fixed a priori.",
        ],
        "runtime_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    hold = cmp["splits"]["holdout"]["arms"]
    for a, r in sorted(hold.items()):
        print(f"{a:22s} calls={r['calls']:5d} hit={r['hit_rate']} net={r['mean_net_return']} dd={r['max_drawdown']} {r['status']}")
    for v in verdicts:
        print(v["arm"], "vs", v["against"], v["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
