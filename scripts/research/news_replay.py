"""
scripts/research/news_replay.py — replay the news engine over the archive
StockLens has collected itself (news_articles, with their real ingestion
times), and score next-session outcomes. Read-only on the source database.

  python scripts/research/news_replay.py --db sqlite:///path/app.db [--cost 25] [--out PATH]

The archive only starts when scheduled ingestion is switched on; until then
it is empty or a few days long, and the script says so instead of reporting
statistics. Minimum before any conclusion: MIN_SESSIONS sessions and
MIN_CALLS scored calls (the provisional promotion gate is 200 holdout events
per setup).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlmodel import Session, create_engine, select  # noqa: E402

from catalysts import replay  # noqa: E402
from catalysts.providers import RawArticle  # noqa: E402
from db.models.news import NewsArticle  # noqa: E402
from db.models.stock import Company  # noqa: E402
from utils.market_session import IST  # noqa: E402

MIN_SESSIONS = 20
MIN_CALLS = 50
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "news_replay.json"


def archive_from_db(url: str) -> tuple[list[replay.ArchivedArticle], list[tuple]]:
    eng = create_engine(url)
    with Session(eng) as s:
        arts = s.exec(select(NewsArticle).order_by(NewsArticle.ingested_at)).all()
        comps = [(c.symbol, c.name, c.sector, c.industry) for c in s.exec(select(Company)).all()]
    out = []
    for a in arts:
        ing = a.ingested_at if a.ingested_at.tzinfo else a.ingested_at.replace(tzinfo=dt.timezone.utc)
        pub = a.published_at if a.published_at.tzinfo else a.published_at.replace(tzinfo=dt.timezone.utc)
        out.append(replay.ArchivedArticle(RawArticle(a.provider_article_id, a.url, a.title, pub, a.excerpt or "",
                                                     a.source_name), ing, a.provider))
    return out, comps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cost", type=float, default=25.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    arch, comps = archive_from_db(args.db)
    report = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "articles": len(arch)}
    if not arch:
        report["status"] = "NO_ARCHIVE"
        report["detail"] = "no ingested news yet: enable scheduled ingestion to start the point-in-time archive"
    else:
        first, last = min(a.ingested_at for a in arch), max(a.ingested_at for a in arch)
        report.update(first_ingested=first.isoformat(), last_ingested=last.isoformat())
        from fundamentals.provider import fetch_price_history
        syms = [c[0] for c in comps]
        prices = {}
        for k in range(0, len(syms), 50):
            prices.update(fetch_price_history(syms[k:k + 50], period="6mo"))
        prices = {s: df for s, df in prices.items() if df is not None}
        nifty = fetch_price_history(["^NSEI"], period="6mo")["^NSEI"]
        days = [d.date() for d in nifty.index if first.astimezone(IST).date() <= d.date() <= last.astimezone(IST).date()]
        cutoffs = [dt.datetime.combine(d, dt.time(19, 30), tzinfo=IST) for d in days]
        res = replay.run(arch, cutoffs, comps, prices, nifty)
        summ = res.summary(args.cost)
        enough = summ["sessions"] >= MIN_SESSIONS and summ["scored"] >= MIN_CALLS
        report.update(status="OK" if enough else "INSUFFICIENT_SAMPLE", summary=summ,
                      note=None if enough else f"needs >= {MIN_SESSIONS} sessions and >= {MIN_CALLS} scored calls "
                                               "before any statistic is interpreted")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "articles")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
