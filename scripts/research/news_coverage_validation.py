"""
scripts/research/news_coverage_validation.py — does a provider actually cover
real, dated market events, how fast, with what timestamp quality, and how
much duplication reaches the pipeline? Read-only research (network).

  python scripts/research/news_coverage_validation.py [--out PATH]

Ground truth comes from authoritative, dated records (nothing typed in by
hand):
  corporate   NSE filings (dissemination time to the second) of large companies:
              one material filing (results, board outcome, press release,
              business update) per company in two windows
  tariffs     Federal Register presidential documents matching tariff / duty
              (publication date; signing usually earlier)
  sanctions   Federal Register presidential documents matching sanctions
              (geopolitical policy actions)
  fed         changes of the FOMC target range (FRED DFEDTARU)
  crude       Brent daily moves >= 2 sd of the prior 60 days (Yahoo BZ=F)
  yields      US 10-year Treasury daily changes >= 12 bp (FRED DGS10)

Providers tested historically: google_news (search windows). Live-only
feeds (official RSS, publisher RSS) cannot be tested on past events and are
reported as such; NewsAPI and Alpha Vantage need keys (not set); GDELT is
probed once.

Per event: found (a matching title in the window), earliest exact-time
article and its delay vs the official time (corporate only: both exact),
share of date-only timestamps, results capped at the provider's limit, and
the pipeline's canonical-event count for the matched articles (duplicates =
articles beyond one event).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from catalysts import adapters as ad, pipeline, providers as prov  # noqa: E402
from db.models.news import NewsArticle  # noqa: E402

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "news_coverage_validation.json"
COMPANIES = ["TRENT", "TITAN", "ITC", "CANBK", "HCLTECH", "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK", "LT",
             "SBIN", "BAJFINANCE", "MARUTI", "ASIANPAINT", "TATASTEEL", "SUNPHARMA", "BHARTIARTL", "AXISBANK", "WIPRO"]
WINDOWS = {"recent": (dt.date(2026, 9, 25), dt.date(2026, 10, 9)), "2025_q1_results": (dt.date(2025, 7, 10), dt.date(2025, 7, 31))}
RESULT_WORDS = re.compile(r"result|profit|revenue|earning|quarter|\bq[1-4]\b|update|sales|board|dividend", re.I)
from catalysts.benchmark import INDIAN_ALIASES as ALIASES  # noqa: E402
NOT_PREVIEW = re.compile(r"^(?!.*\b(preview|expect(ed|s)?|likely|ahead of|estimates for|what to watch|may report)\b)", re.I)
MATERIAL = re.compile(r"financial result|outcome of board meeting|press release|business update|update for q", re.I)
SUFFIX = re.compile(r"\b(limited|ltd\.?|company|industries|corporation|of india|india)\b", re.I)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockLens-research/0.1"}


def gnews(query: str, start: dt.date, end: dt.date) -> list[prov.RawArticle]:
    p = ad.GoogleNewsProvider(queries=[query], sleep=lambda s: None)
    since = dt.datetime.combine(start, dt.time(0), tzinfo=UTC)
    until = dt.datetime.combine(end, dt.time(23, 59), tzinfo=UTC)
    try:
        items = list(p.fetch(since, until))
    except prov.ProviderError as e:
        return [prov.RawArticle("ERR", "", f"ERROR {e}", since)]
    time.sleep(2.0)
    return items


def gdelt(query: str, start: dt.date, end: dt.date) -> list[prov.RawArticle]:
    p = prov.GdeltProvider(query=f"{query} sourcelang:english", sleep=lambda s: None)
    since = dt.datetime.combine(start, dt.time(0), tzinfo=UTC)
    until = dt.datetime.combine(end, dt.time(23, 59), tzinfo=UTC)
    time.sleep(6.5)                                   # GDELT: at most one request per 5 seconds
    try:
        return list(p.fetch(since, until))
    except prov.ProviderError as e:
        return [prov.RawArticle("ERR", "", f"ERROR {type(e).__name__}", since)]


def dedup_events(items: list[prov.RawArticle]) -> int:
    """Canonical events the pipeline creates from these articles (fresh DB)."""
    if not items:
        return 0
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)

    class L:
        name = "validation"

        def fetch(self, since, until):
            return items
    now = max(i.published_at for i in items) + dt.timedelta(hours=1)
    with Session(eng) as s:
        pipeline.ingest(s, L(), now - dt.timedelta(days=60), now, now=now)
        return len({a.event_id for a in s.exec(select(NewsArticle)).all() if a.event_id})


def evaluate(event: dict, items: list[prov.RawArticle], must: list[re.Pattern]) -> dict:
    matched = [i for i in items if all(m.search(i.title) for m in must)]
    exact = [i for i in matched if i.time_precision == "exact"]
    errors = [i.title for i in items if i.provider_article_id == "ERR"]
    if errors:
        return {**event, "error": errors[0]}
    out = {**event, "provider_results": len(items), "matched": len(matched), "found": bool(matched),
           "capped": len(items) >= 95, "date_only_share": round(1 - len(exact) / len(matched), 2) if matched else None,
           "canonical_events_from_matched": dedup_events(matched)}
    if event.get("official_time") and exact:
        official = dt.datetime.fromisoformat(event["official_time"])
        after = [i for i in exact if i.published_at >= official]
        out["related_exact_before_official"] = len(exact) - len(after)
        if after:
            first = min(after, key=lambda i: i.published_at)
            out["delay_minutes_first_exact"] = round((first.published_at - official).total_seconds() / 60, 1)
            out["first_title"] = first.title[:100]
    elif matched:
        out["first_title"] = min(matched, key=lambda i: i.published_at).title[:100]
    return out


def corporate_events() -> list[dict]:
    nse = ad.NseAnnouncementsProvider(sleep=lambda s: None)
    events = []
    for wname, (a, b) in WINDOWS.items():
        since, until = dt.datetime.combine(a, dt.time(0), tzinfo=IST), dt.datetime.combine(b, dt.time(23, 59), tzinfo=IST)
        for sym in COMPANIES:
            try:
                rows = nse._rows(since, until, sym)
            except prov.ProviderError as e:
                events.append({"kind": "corporate", "window": wname, "symbol": sym, "error": str(e)[:100]})
                continue
            time.sleep(1.5)
            mat = [r for r in rows if MATERIAL.search(f"{r.get('desc')} {r.get('attchmntText')}")]
            if not mat:
                continue
            r = sorted(mat, key=lambda r: r.get("exchdisstime") or "")[0]
            t = dt.datetime.strptime(r["exchdisstime"], "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
            name = SUFFIX.sub("", r.get("sm_name") or sym).strip(" ,.&")
            short = name.split()[0] if len(name.split()[0]) > 3 or len(name.split()) == 1 else " ".join(name.split()[:2])
            events.append({"kind": "corporate", "window": wname, "symbol": sym, "name": short,
                           "official_time": t.astimezone(UTC).isoformat(), "filing": (r.get("desc") or "")[:60],
                           "detail": (r.get("attchmntText") or "")[:120]})
    return events


def federal_register_events(term: str, kind: str, n: int = 10) -> list[dict]:
    q = (f"{ad.FederalRegisterProvider.URL}?conditions[term]={term}&conditions[type][]=PRESDOCU&per_page={n}"
         "&order=newest&fields[]=title&fields[]=publication_date&fields[]=document_number")
    data = json.loads(prov.fetch(q, prov.http_get))
    return [{"kind": kind, "date": d["publication_date"], "title": d["title"][:140], "document_number": d["document_number"]}
            for d in data.get("results", [])]


def fred(series: str) -> pd.Series:
    body = prov.fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", prov.http_get)
    df = pd.read_csv(io.BytesIO(body))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index(pd.to_datetime(df.dropna()["date"]))["value"]


def salient(title: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]{4,}", title)
             if w.lower() not in {"adjusting", "imports", "united", "states", "into", "with", "regarding", "certain",
                                  "further", "amendment", "amendments", "tariff", "tariffs", "duties", "presidential",
                                  "proclamation", "executive", "order", "sanctions", "national", "emergency",
                                  "respect", "additional", "modifying", "continuation", "imposing"}]
    return words[0] if words else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", default="", help="comma-separated kinds (corporate,tariffs,sanctions,fed,crude,yields)")
    args = ap.parse_args()
    only = {k for k in args.only.split(",") if k}
    results: list[dict] = []
    # corporate: explicit aliases (short names collide: 'Tata', 'Titan', 'State', 'Bharti'), result wording,
    # previews excluded
    for ev in (corporate_events() if not only or "corporate" in only else []):
        if "error" in ev:
            results.append(ev)
            continue
        d = dt.datetime.fromisoformat(ev["official_time"]).date()
        alias = ALIASES[ev["symbol"]]
        items = gnews(f'"{alias.split("|")[0]}"', d - dt.timedelta(days=1), d + dt.timedelta(days=2))
        must = [re.compile(r"\b(" + alias + r")\b", re.I), RESULT_WORDS, NOT_PREVIEW]
        results.append(evaluate({**ev, "alias": alias, "provider": "google_news"}, items, must))
        if dt.datetime.now(UTC).date() - d <= dt.timedelta(days=85):
            results.append(evaluate({**ev, "alias": alias, "provider": "gdelt"},
                                    gdelt(f'"{alias.split("|")[0]}"', d - dt.timedelta(days=1), d + dt.timedelta(days=2)), must))
    if only and not (only - {"corporate"}):
        _finish(results, args.out, {"provider": "gdelt", "status": "not probed"})
        return
    # US policy: tariffs and sanctions (Federal Register)
    for term, kind in (("tariff", "tariffs"), ("sanctions", "sanctions")):
        for ev in federal_register_events(term, kind):
            d = dt.date.fromisoformat(ev["date"])
            key = salient(ev["title"])
            must = [re.compile(r"tariff|dut(y|ies)" if kind == "tariffs" else r"sanction", re.I)]
            if key:
                must.append(re.compile(re.escape(key), re.I))
            items = gnews(f"{kind[:-1] if kind == 'tariffs' else 'sanctions'} {key}".strip(), d - dt.timedelta(days=10), d + dt.timedelta(days=1))
            row = evaluate({**ev, "keyword": key, "provider": "google_news"}, items, must)
            hit = [i for i in items if all(m.search(i.title) for m in must)]
            if hit:
                row["news_days_before_federal_register"] = (d - min(i.published_at for i in hit).date()).days
            results.append(row)
    # Fed decisions (target range changes, last 3 years)
    up = fred("DFEDTARU")
    ch = up[up.diff().fillna(0) != 0]
    for d, v in ch[ch.index >= "2023-10-01"].items():
        items = gnews("Federal Reserve interest rate", (d - pd.Timedelta(days=2)).date(), d.date())
        results.append(evaluate({"kind": "fed", "date": d.date().isoformat(), "new_upper_bound": float(v)}, items,
                                [re.compile(r"\bfed\b|federal reserve|fomc|powell", re.I), re.compile(r"rate", re.I)]))
    # crude shocks (Brent) and US 10y yield changes
    from fundamentals.provider import fetch_price_history
    brent = fetch_price_history(["BZ=F"], period="3y")["BZ=F"]["Close"]
    r = brent.pct_change()
    shocks = r[(r.abs() >= 2 * r.rolling(60).std().shift(1))].dropna()
    for d, x in list(shocks.items())[-15:]:
        items = gnews("crude oil prices", d.date(), (d + pd.Timedelta(days=1)).date())
        results.append(evaluate({"kind": "crude", "date": d.date().isoformat(), "move": round(float(x), 4)}, items,
                                [re.compile(r"\boil\b|crude|brent", re.I)]))
    y = fred("DGS10")
    dy = y.diff()
    big = dy[dy.abs() >= 0.12].dropna()
    for d, x in list(big.items())[-15:]:
        items = gnews("Treasury yields", d.date(), (d + pd.Timedelta(days=1)).date())
        results.append(evaluate({"kind": "yields", "date": d.date().isoformat(), "change_bp": round(float(x) * 100, 1)},
                                items, [re.compile(r"yield|treasur", re.I)]))
    # GDELT probe (one request)
    gd = {"provider": "gdelt"}
    try:
        list(prov.GdeltProvider(sleep=lambda s: None).fetch(dt.datetime.now(UTC) - dt.timedelta(days=1), dt.datetime.now(UTC)))
        gd["status"] = "OK"
    except prov.ProviderError as e:
        gd["status"], gd["detail"] = type(e).__name__, str(e)[:120]
    _finish(results, args.out, gd)


def _finish(results: list[dict], out_path: str, gd: dict) -> None:
    summary = {}
    for kind, provider in sorted({(r_["kind"], r_.get("provider", "google_news")) for r_ in results}):
        rows = [r_ for r_ in results if r_["kind"] == kind and r_.get("provider", "google_news") == provider
                and "error" not in r_]
        found = [r_ for r_ in rows if r_["found"]]
        delays = [r_["delay_minutes_first_exact"] for r_ in found if r_.get("delay_minutes_first_exact") is not None]
        summary[f"{kind}:{provider}"] = {"events": len(rows), "found": len(found), "missing": len(rows) - len(found),
                         "coverage": round(len(found) / len(rows), 2) if rows else None,
                         "errors": sum(1 for r_ in results if r_["kind"] == kind and r_.get("provider", "google_news") == provider
                                       and "error" in r_),
                         "median_days_news_before_fr": median([r_["news_days_before_federal_register"] for r_ in found
                                                               if "news_days_before_federal_register" in r_] or [None]),
                         "median_delay_minutes": median(delays) if delays else None,
                         "delays_measured": len(delays),
                         "median_date_only_share": median([r_["date_only_share"] for r_ in found]) if found else None,
                         "mean_articles_per_event": round(sum(r_["matched"] for r_ in found) / len(found), 1) if found else None,
                         "mean_canonical_events_per_event": round(sum(r_["canonical_events_from_matched"] for r_ in found) / len(found), 2) if found else None,
                         "results_capped": sum(1 for r_ in rows if r_.get("capped"))}
    if any(r_["kind"] == "corporate" for r_ in results):
        for w in WINDOWS:
            rows = [r_ for r_ in results if r_["kind"] == "corporate" and r_.get("window") == w and "error" not in r_
                    and r_.get("provider") == "google_news"]
            summary[f"corporate_{w}"] = {"events": len(rows), "found": sum(r_["found"] for r_ in rows),
                                         "median_delay_minutes": median([r_["delay_minutes_first_exact"] for r_ in rows
                                                                         if r_.get("delay_minutes_first_exact") is not None] or [None])}
    report = {"generated_at": dt.datetime.now(UTC).isoformat(), "method": __doc__.split("Ground truth")[1].strip(),
              "summary": summary, "gdelt_probe": gd, "events": results,
              "not_testable_historically": {"official_rss": "feeds hold only the latest items",
                                            "publisher_rss": "about 2 days of items",
                                            "newsapi": "no key configured; free plan 31 days back",
                                            "alphavantage": "no key configured"}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    print("gdelt:", gd)


if __name__ == "__main__":
    main()
