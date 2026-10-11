"""
scripts/research/news_provider_benchmark.py — reproducible news-provider
benchmark on independently verified, timestamped market events (RESEARCH
ONLY; network; read-only on StockLens data).

  python scripts/research/news_provider_benchmark.py [--out PATH] [--max-per-kind N]
  python scripts/research/news_provider_benchmark.py --events-from REPORT --providers marketaux,alphavantage
  python scripts/research/news_provider_benchmark.py --resummarize          (no network)

--events-from reuses the stored ground truth with IDENTICAL event
definitions (query, candidate and verification patterns, windows); only the
listed providers are queried and their outcomes replace the stored ones,
every other provider's stored outcomes are kept. NSE / BSE ground truth is
only rebuilt live when the exchange's written permission is recorded
(NSE_AUTOMATED_ACCESS_PERMISSION / BSE_AUTOMATED_ACCESS_PERMISSION): their
terms prohibit automated collection otherwise.

Ground truth (built at run time from authoritative records; evidence URL and
official time stored with every event):
  nse_filing                NSE corporate announcements: dissemination time to the
                            second (recent window and a 2025 results window)
  bse_filing                BSE announcements (attempted; the API refuses
                            automated access intermittently -> recorded)
  rbi, sebi                 official RSS releases with exact times (material only)
  india_policy              PIB (Government of India) releases (material only)
  us_policy_whitehouse      White House presidential actions / news on tariffs,
                            sanctions and trade (exact times)
  us_policy_federal_register Federal Register presidential documents on tariffs /
                            sanctions (date only)
  fed                       Federal Reserve press releases (exact) and FOMC target
                            changes (FRED DFEDTARU, date only)
  crude, gold, currency     Brent, gold and USD/INR daily moves >= 2 sd of the
                            prior 60 days (Yahoo; date only)

Providers: google_news, gdelt, marketaux, alphavantage, newsapi,
publisher_rss, official_rss. Keyed providers use MARKETAUX_API_TOKEN,
ALPHAVANTAGE_API_KEY, NEWSAPI_KEY from the environment (never printed);
without a key they are UNTESTABLE_NOT_CONFIGURED - never "missed". The
exposed NewsAPI key from Git history is never used.

Output: scripts/research/output/prediction_v2/news_provider_benchmark.json
(served by GET /admin/news/benchmark and the web Labs dashboard).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from catalysts import adapters as ad, benchmark as bm, entities as ent, pipeline, providers as prov, registry  # noqa: E402
from db.models.news import NewsArticle  # noqa: E402

UTC = dt.timezone.utc
IST = bm.IST
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "news_provider_benchmark.json"
DB = Path(r"C:\Python\Git\stock-prediction-app\storage\app.db")
STOP = set("""the of and to in for on a an is with by as at from its it this that are be will has have new
press release releases statement issues issued announces announced update updates notice regarding under
about into over amid after before india indian united states government reserve bank federal board securities
exchange commission department ministry president presidential executive order proclamation fact sheet
remarks donald trump certain excluding modifying scope products regulatory risk working open agencies seek
general matter order orders settlement dated paper data gaps level""".split())
WINDOWS = {"recent": 25, "2025_q1_results": None}       # days back for recent; fixed window for 2025


class Counter:
    """Wraps a provider HTTP get to count requests, failures and latency."""

    def __init__(self, get=prov.http_get):
        self.get, self.stats = get, {"requests": 0, "ok": 0, "http_errors": 0, "rate_limited": 0, "exceptions": 0,
                                     "seconds": 0.0}

    def __call__(self, url, timeout=20.0, headers=None):
        t0 = time.monotonic()
        self.stats["requests"] += 1
        try:
            st, body = self.get(url, timeout, headers) if headers else self.get(url, timeout)
        except OSError:
            self.stats["exceptions"] += 1
            raise
        finally:
            self.stats["seconds"] += time.monotonic() - t0
        key = "ok" if st == 200 else "rate_limited" if st in (402, 429, 503) else "http_errors"
        self.stats[key] += 1
        return st, body


TOPICS = ("tariff", "tariffs", "sanction", "sanctions", "repo", "foreign exchange", "forex", "liquidity", "inflation",
          "oil marketing", "crude", "gold", "rupee", "steel", "aluminum", "copper", "semiconductor", "pharmaceutical",
          "canada", "china", "mexico", "russia", "iran", "venezuela", "brazil", "japan", "european union",
          "derivatives", "mutual fund", "ipo", "minutes", "interest rate", "monetary policy", "fomc", "vrrr", "crr")


def salient(title: str, n: int = 2) -> list[str]:
    """Search / verification terms: topic words found in the title first,
    then the title's first content words (titles lead with their subject)."""
    low = title.lower()
    topics = [t for t in TOPICS if re.search(r"\b" + re.escape(t) + r"\b", low)]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title) if w.lower() not in STOP and w.lower() not in topics]
    return list(dict.fromkeys(topics + words))[:n]


def rss_items(url: str) -> tuple[list[tuple[str, str, dt.datetime, str]], int]:
    """(title, link, time, precision) for dated items, and the number of
    undated items (which cannot serve as timestamped ground truth)."""
    body = prov._as_utf8(prov.fetch(url, prov.http_get, sleep=lambda s: None))
    out, undated = [], 0
    for it in ET.fromstring(body).iter("item"):
        t, precision = prov.parse_pub_date(it.findtext("pubDate"), prov.naive_tz_for(url))
        if t is None:
            undated += 1
            continue
        out.append((prov.tx.sanitize(it.findtext("title"), 300), (it.findtext("link") or "").strip(), t, precision))
    return out, undated


def nse_events(now: dt.datetime, notes: dict) -> list[bm.Event]:
    if not os.environ.get(ad.NSE_PERMISSION_ENV, "").strip():
        notes["nse_filing"] = {"status": "NOT_PERMITTED", "events": 0,
                               "detail": f"NSE Terms of Use prohibit automated collection; {ad.NSE_PERMISSION_ENV} not set"}
        return []
    nse = ad.NseAnnouncementsProvider(sleep=lambda s: None)
    material = re.compile(r"financial result|outcome of board meeting|press release|business update|update for q", re.I)
    windows = {"recent": (now - dt.timedelta(days=25), now - dt.timedelta(days=2)),
               "2025_q1_results": (dt.datetime(2025, 7, 10, tzinfo=IST), dt.datetime(2025, 7, 31, 23, 59, tzinfo=IST))}
    events, errors = [], 0
    for wname, (a, b) in windows.items():
        for sym, alias in bm.INDIAN_ALIASES.items():
            try:
                rows = nse._rows(a, b, sym)
            except prov.ProviderError:
                errors += 1
                continue
            time.sleep(1.2)
            mat = [r for r in rows if material.search(f"{r.get('desc')} {r.get('attchmntText')}")
                   and ad.filings.classify(r.get("desc"), r.get("attchmntText")).keep]
            if not mat:
                continue
            r = sorted(mat, key=lambda r: dt.datetime.strptime(r["exchdisstime"], "%d-%b-%Y %H:%M:%S"))[0]
            t = dt.datetime.strptime(r["exchdisstime"], "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).astimezone(UTC)
            events.append(bm.Event(
                id=f"nse:{sym}:{r.get('seq_id')}", kind="nse_filing", title=f"{r.get('sm_name')}: {r.get('desc')}"[:160],
                official_time=t, precision="exact", source=f"NSE filing ({wname})", evidence_url=r.get("attchmntFile"),
                query=f'"{alias.split("|")[0]}"', candidate=r"\b(" + alias + r")\b",
                must=[r"result|profit|revenue|earning|quarter|\bq[1-4]\b|update|sales|board|dividend|order|fund"],
                expected_symbols=(f"{sym}.NS",)))
    notes["nse_filing"] = {"status": "OK" if events else "UNAVAILABLE", "events": len(events), "request_errors": errors}
    return events


# BSE rows carry every filing (audio recordings, transcripts, newspaper copies...): keep material ones only
BSE_MATERIAL = re.compile(r"financial result|outcome of board meeting|press release|business update|update for q|"
                          r"dividend|order|acquisition|fund raising", re.I)
BSE_NOT_MATERIAL = re.compile(r"audio|recording|transcript|newspaper|publication|intimation of (analyst|investor) meet", re.I)


def bse_events(now: dt.datetime, notes: dict) -> list[bm.Event]:
    if not os.environ.get(ad.BSE_PERMISSION_ENV, "").strip():
        notes["bse_filing"] = {"status": "NOT_PERMITTED", "events": 0,
                               "detail": f"BSE website terms: personal, non-commercial use; {ad.BSE_PERMISSION_ENV} not set"}
        return []
    day = (now - dt.timedelta(days=3)).astimezone(IST)
    url = ("https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno=1&strCat=-1"
           f"&strPrevDate={day:%Y%m%d}&strScrip=&strSearch=P&strToDate={day:%Y%m%d}&strType=C&subcategory=-1")
    st, body = prov.http_get(url, 20.0, {"Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com",
                                         "User-Agent": ad.BROWSER_UA})
    if st != 200:
        notes["bse_filing"] = {"status": "UNAVAILABLE", "detail": f"BSE API HTTP {st} (automated access refused)", "events": 0}
        return []
    rows = (json.loads(body or b"{}").get("Table") or [])
    events = []
    for r in rows:
        name = r.get("SLONGNAME") or ""
        hit = next(((s, al) for s, al in bm.INDIAN_ALIASES.items() if re.search(r"\b(" + al + r")\b", name, re.I)), None)
        subject = r.get("NEWSSUB") or ""
        if not hit or not r.get("DT_TM") or not BSE_MATERIAL.search(subject) or BSE_NOT_MATERIAL.search(subject):
            continue
        try:
            t = dt.datetime.fromisoformat(r["DT_TM"][:19]).replace(tzinfo=IST).astimezone(UTC)
        except ValueError:
            continue
        sym, alias = hit
        events.append(bm.Event(id=f"bse:{r.get('NEWSID')}", kind="bse_filing", title=f"{name}: {r.get('NEWSSUB')}"[:160],
                               official_time=t, precision="exact", source="BSE filing", evidence_url=None,
                               query=f'"{alias.split("|")[0]}"', candidate=r"\b(" + alias + r")\b",
                               must=[r"result|profit|revenue|quarter|update|board|order|dividend"],
                               expected_symbols=(f"{sym}.NS",)))
    notes["bse_filing"] = {"status": "OK", "events": len(events), "rows_seen": len(rows)}
    return events


OFFICIAL = {
    # kind: (feeds, material filter, candidate regex)
    "rbi": (("https://www.rbi.org.in/pressreleases_rss.xml", "https://www.rbi.org.in/notifications_rss.xml"),
            r"monetary|repo|policy|rate|liquidity|foreign exchange|forex|penalt|regulat|framework|special window|crr|slr|inflation",
            r"\bRBI\b|Reserve Bank"),
    "sebi": (("https://www.sebi.gov.in/sebirss.xml",),
             r"^(?!.*in the matter of).*(framework|circular|regulations?|derivative|f&o|ipo|mutual fund|consultation|board meeting)",
             r"\bSEBI\b|Securities and Exchange Board"),
    "india_policy": (("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3&reg=3",),
                     r"cabinet|approves|gst|tax|budget|export|import|tariff|scheme|infrastructure|policy|inflation|gdp|trade",
                     r"India|Centre|government|cabinet|ministry|minister"),
    "us_policy_whitehouse": (("https://www.whitehouse.gov/presidential-actions/feed/", "https://www.whitehouse.gov/news/feed/"),
                             r"tariff|sanction|trade|dut(y|ies)|import|export|proclamation|executive order",
                             r"Trump|White House|tariff|sanction|executive order|proclamation"),
    "fed": (("https://www.federalreserve.gov/feeds/press_all.xml",), r"fomc|monetary policy|interest rate|statement|minutes",
            r"\bFed\b|Federal Reserve|FOMC|Powell"),
}


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower())[:70].strip()


def official_events(now: dt.datetime, notes: dict, max_per_kind: int) -> list[bm.Event]:
    events = []
    for kind, (feeds, material, cand) in OFFICIAL.items():
        rows, errs, undated = [], [], 0
        for f in feeds:
            try:
                got, und = rss_items(f)
                rows += got
                undated += und
            except (prov.ProviderError, ET.ParseError) as e:
                errs.append(f"{prov.tx.domain(f)}: {e}"[:120])
        seen_titles: set[str] = set()
        rows = [r for r in sorted(rows, key=lambda r: r[2], reverse=True)
                if re.search(material, r[0], re.I) and r[2] <= now - dt.timedelta(hours=12)
                and not (_norm(r[0]) in seen_titles or seen_titles.add(_norm(r[0])))][:max_per_kind]
        for title, link, t, precision in rows:
            keys = salient(title)
            if not keys:
                continue
            events.append(bm.Event(id=f"{kind}:{link[-60:]}", kind=kind, title=title[:160], official_time=t, precision=precision,
                                   source=f"official RSS ({prov.tx.domain(link)})", evidence_url=link,
                                   query=" ".join(keys + ([cand.split('|')[0].strip('\\b')] if kind in ("rbi", "sebi", "fed") else [])),
                                   candidate=cand, must=[re.escape(keys[0])]))      # the most specific term
        status = "OK" if rows else ("UNAVAILABLE" if errs else
                                    "UNAVAILABLE_NO_TIMESTAMPS" if undated and not rows else "NO_MATERIAL_ITEMS")
        notes[kind] = {"status": status, "events": sum(1 for e in events if e.kind == kind), "errors": errs,
                       "undated_items": undated, "history": "live feed only: the latest items, no archive"}
    return events


def federal_register_events(notes: dict, max_per_kind: int) -> list[bm.Event]:
    events = []
    for term in ("tariff", "sanctions"):
        q = (f"{ad.FederalRegisterProvider.URL}?conditions[term]={term}&conditions[type][]=PRESDOCU&per_page={max_per_kind}"
             "&order=newest&fields[]=title&fields[]=publication_date&fields[]=document_number&fields[]=html_url")
        data = json.loads(prov.fetch(q, prov.http_get, sleep=lambda s: None))
        for d in data.get("results", []):
            keys = salient(d["title"])
            if not keys:
                continue
            day = dt.date.fromisoformat(d["publication_date"])
            events.append(bm.Event(id=f"fr:{d['document_number']}", kind="us_policy_federal_register", title=d["title"][:160],
                                   official_time=dt.datetime.combine(day, dt.time(12), tzinfo=UTC), precision="date",
                                   source="Federal Register (publication date; signing is usually days earlier)",
                                   evidence_url=d.get("html_url"), query=f"{term} {keys[0]}",
                                   candidate=r"tariff|dut(y|ies)" if term == "tariff" else r"sanction",
                                   must=["|".join(re.escape(k) for k in keys)],
                                   window_before=dt.timedelta(days=10)))
    notes["us_policy_federal_register"] = {"status": "OK", "events": len(events), "history": "full (date only)"}
    return events


def fred_series(series: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(prov.fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", prov.http_get)))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return df.set_index(pd.to_datetime(df["date"]))["value"]


OTHER_RUPEES = r"^(?!.*\b(pakistan|pakistani|sri lanka|sri lankan|nepal|nepalese|indonesia|indonesian|mauritius|mauritian)\b)"


def market_events(now: dt.datetime, notes: dict, max_per_kind: int) -> list[bm.Event]:
    from fundamentals.provider import fetch_price_history
    events = []
    up = fred_series("DFEDTARU")
    ch = up[up.diff().fillna(0) != 0]
    for d, v in ch[ch.index >= "2024-06-01"].items():
        events.append(bm.Event(id=f"fed_rate:{d.date()}", kind="fed", title=f"FOMC target range upper bound -> {v:.2f}%",
                               official_time=dt.datetime.combine(d.date(), dt.time(12), tzinfo=UTC), precision="date",
                               source="FRED DFEDTARU (effective date)", evidence_url="https://fred.stlouisfed.org/series/DFEDTARU",
                               query="Federal Reserve interest rate decision", candidate=r"\bFed\b|Federal Reserve|FOMC|Powell",
                               must=[r"rate|cut|hike|hold|basis points|bps"], window_before=dt.timedelta(days=2)))
    px = fetch_price_history(["BZ=F", "GC=F", "INR=X"], period="2y")
    spec = {"crude": ("BZ=F", "crude oil price", r"\boil\b|crude|brent"), "gold": ("GC=F", "gold price", r"\bgold\b"),
            "currency": ("INR=X", "rupee dollar", r"rupee|\bINR\b")}
    move = r"surge|jump|rise|rises|rose|fall|falls|fell|drop|plunge|slump|spike|tumble|soar|gain|record|slide|climb|weak|strong"
    for kind, (sym, query, cand) in spec.items():
        df = px.get(sym)
        if df is None or df.empty:
            notes[kind] = {"status": "UNAVAILABLE", "events": 0}
            continue
        r = df["Close"].pct_change()
        shocks = r[(r.abs() >= 2 * r.rolling(60).std().shift(1)) & (r.index >= pd.Timestamp(now.date() - dt.timedelta(days=365)))].dropna()
        for d, x in list(shocks.items())[-max_per_kind:]:
            events.append(bm.Event(id=f"{kind}:{d.date()}", kind=kind, title=f"{sym} daily move {x:+.2%}",
                                   official_time=dt.datetime.combine(d.date(), dt.time(12), tzinfo=UTC), precision="date",
                                   source=f"Yahoo Finance {sym} close-to-close (>= 2 sd of prior 60 days)", evidence_url=None,
                                   query=query, candidate=cand,
                                   must=[move] + ([OTHER_RUPEES] if kind == "currency" else [])))
        notes[kind] = {"status": "OK", "events": sum(1 for e in events if e.kind == kind)}
    notes["fed"] = {**notes.get("fed", {}), "fred_rate_changes": int((ch.index >= "2024-06-01").sum())}
    return events


def nse_holidays() -> list[str]:
    from fundamentals.provider import fetch_price_history
    n = fetch_price_history(["^NSEI"], period="2y")["^NSEI"]
    have = {d.date() for d in n.index}
    days = pd.bdate_range(min(have), max(have))
    return [d.date().isoformat() for d in days if d.date() not in have]          # weekdays without a NIFTY bar


def canonical_count(items: list[prov.RawArticle]) -> int:
    if not items:
        return 0
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)

    class L:
        name = "benchmark"

        def fetch(self, since, until):
            return items
    now = max(i.published_at for i in items) + dt.timedelta(hours=1)
    with Session(eng) as s:
        pipeline.ingest(s, L(), now - dt.timedelta(days=90), now, now=now)
        return len({a.event_id for a in s.exec(select(NewsArticle)).all() if a.event_id})


def event_to_dict(e: bm.Event) -> dict:
    return {"id": e.id, "kind": e.kind, "title": e.title, "official_time": e.official_time.isoformat(),
            "precision": e.precision, "source": e.source, "evidence_url": e.evidence_url,
            "expected_symbols": list(e.expected_symbols), "segment": bm.SEGMENT_OF.get(e.kind),
            "definition": {"query": e.query, "candidate": e.candidate, "must": list(e.must),
                           "window_before_s": e.window_before.total_seconds(),
                           "window_after_s": e.window_after.total_seconds()}}


NSE_MUST = r"result|profit|revenue|earning|quarter|\bq[1-4]\b|update|sales|board|dividend|order|fund"
BSE_MUST = r"result|profit|revenue|quarter|update|board|order|dividend"
MOVE = r"surge|jump|rise|rises|rose|fall|falls|fell|drop|plunge|slump|spike|tumble|soar|gain|record|slide|climb|weak|strong"
MARKET_SPEC = {"crude": ("crude oil price", r"\boil\b|crude|brent"), "gold": ("gold price", r"\bgold\b"),
               "currency": ("rupee dollar", r"rupee|\bINR\b")}


def legacy_definition(d: dict) -> dict:
    """Definition of an event stored before definitions were saved (report
    of 11 Oct 2026), rebuilt with the rules that created it. Federal Register
    documents record no search term: tariff when the title names tariffs,
    duties or imports, else sanctions."""
    kind, title, eid = d["kind"], d["title"], d["id"]
    dflt = {"window_before_s": 6 * 3600.0, "window_after_s": 2 * 86400.0}
    if kind in ("nse_filing", "bse_filing"):
        sym = eid.split(":")[1] if kind == "nse_filing" else d["expected_symbols"][0].split(".")[0]
        alias = bm.INDIAN_ALIASES[sym]
        return {**dflt, "query": f'"{alias.split("|")[0]}"', "candidate": r"\b(" + alias + r")\b",
                "must": [NSE_MUST if kind == "nse_filing" else BSE_MUST]}
    if eid.startswith("fed_rate:"):
        return {**dflt, "window_before_s": 2 * 86400.0, "query": "Federal Reserve interest rate decision",
                "candidate": r"\bFed\b|Federal Reserve|FOMC|Powell", "must": [r"rate|cut|hike|hold|basis points|bps"]}
    if kind in MARKET_SPEC:
        q, cand = MARKET_SPEC[kind]
        return {**dflt, "query": q, "candidate": cand, "must": [MOVE] + ([OTHER_RUPEES] if kind == "currency" else [])}
    keys = salient(title)
    if kind == "us_policy_federal_register":
        term = "tariff" if re.search(r"tariff|dut(y|ies)|import", title, re.I) else "sanctions"
        return {**dflt, "window_before_s": 10 * 86400.0, "query": f"{term} {keys[0]}",
                "candidate": r"tariff|dut(y|ies)" if term == "tariff" else r"sanction",
                "must": ["|".join(re.escape(k) for k in keys)]}
    cand = OFFICIAL[kind][2]
    return {**dflt, "query": " ".join(keys + ([cand.split('|')[0].strip('\\b')] if kind in ("rbi", "sebi", "fed") else [])),
            "candidate": cand, "must": [re.escape(keys[0])]}


def event_from_dict(d: dict) -> bm.Event:
    df = d.get("definition") or legacy_definition(d)
    return bm.Event(id=d["id"], kind=d["kind"], title=d["title"], official_time=dt.datetime.fromisoformat(d["official_time"]),
                    precision=d["precision"], source=d["source"], evidence_url=d.get("evidence_url"), query=df["query"],
                    candidate=df["candidate"], must=list(df["must"]), expected_symbols=tuple(d.get("expected_symbols") or ()),
                    window_before=dt.timedelta(seconds=df["window_before_s"]),
                    window_after=dt.timedelta(seconds=df["window_after_s"]))


INDIAN_KINDS = {"nse_filing", "bse_filing", "rbi", "sebi", "india_policy"}
# kinds StockLens already receives at the official time from its own active
# source (official_rss). NSE filings are NOT included: the NSE adapter is
# research-only until its licence is decided.
COVERED_BY_OWN_SOURCES = tuple(OFFICIAL)


def derive(outcomes: list[bm.Outcome], events: Optional[list[dict]] = None) -> dict:
    """Every evidence section computed from the outcomes alone, so a stored
    report can be re-derived (--resummarize) without re-querying providers."""
    summary = bm.summarize(outcomes)
    rated = [o for o in outcomes if o.provider != "official_rss"]
    value = bm.value_over_sources(rated, COVERED_BY_OWN_SOURCES)
    ev_by_id = {e["id"]: e for e in events or []}
    return {"providers": summary, "providers_indian_events": bm.summarize([o for o in outcomes if o.kind in INDIAN_KINDS]),
            "segments": bm.by_segment(outcomes), "audit": bm.audit(outcomes, ev_by_id),
            "value_over_existing_sources": value,
            "recommendation": bm.recommend({k: v for k, v in summary.items() if k != "official_rss"},
                                           {"official_rss": True}, value=value)}


def examples(outcomes: list[bm.Outcome], ev_by_id: dict[str, bm.Event]) -> dict:
    missed = [{"provider": o.provider, "event_id": o.event_id, "kind": o.kind, "segment": bm.SEGMENT_OF.get(o.kind),
               "title": ev_by_id[o.event_id].title, "official_time": ev_by_id[o.event_id].official_time.isoformat(),
               "candidates": o.candidates, "false_matches": o.false_matches} for o in outcomes if o.status == bm.MISSED]
    ambiguous = [{"provider": o.provider, "event_id": o.event_id, "kind": o.kind, "title": ev_by_id[o.event_id].title,
                  "first_title": o.first_title, "reason": o.ambiguous_reason} for o in outcomes if o.status == bm.AMBIGUOUS]
    false_matches = [{"provider": o.provider, "event": ev_by_id[o.event_id].title, "kind": o.kind, "example": fm}
                     for o in outcomes for fm in o.false_matches[:1]][:40]
    wrong_maps = [{"provider": o.provider, "event": ev_by_id[o.event_id].title,
                   "expected": list(ev_by_id[o.event_id].expected_symbols), "also_tagged": o.wrong_symbols}
                  for o in outcomes if o.wrong_symbols][:30]
    return {"missed_events": missed, "ambiguous_events": ambiguous, "false_match_examples": false_matches,
            "wrong_mappings": wrong_maps}


def resummarize(path: str) -> None:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    outcomes = [bm.Outcome(**o) for o in report["outcomes"]]
    report.update(derive(outcomes, report["events"]))
    report.update(examples(outcomes, {e["id"]: event_from_dict(e) for e in report["events"]}))
    report["resummarized_at"] = dt.datetime.now(UTC).isoformat()
    Path(path).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"recommended": report["recommendation"]["recommended"],
                      "value": report["value_over_existing_sources"]}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-per-kind", type=int, default=8)
    ap.add_argument("--resummarize", action="store_true", help="re-derive the metrics of an existing report (no network)")
    ap.add_argument("--events-from", help="reuse the ground truth (identical definitions) of a stored report")
    ap.add_argument("--providers", help="comma list: query only these providers (with --events-from: merge)")
    args = ap.parse_args()
    if args.resummarize:
        return resummarize(args.out)
    now = dt.datetime.now(UTC)
    notes: dict = {}
    base = None
    if args.events_from:
        base = json.loads(Path(args.events_from).read_text(encoding="utf-8"))
        events = [event_from_dict(e) for e in base["events"]]
        notes = base["ground_truth"]["sources"]
    else:
        events = nse_events(now, notes) + bse_events(now, notes) + official_events(now, notes, args.max_per_kind)
        events += (federal_register_events(notes, max(3, args.max_per_kind // 2))
                   + market_events(now, notes, max(3, args.max_per_kind // 2)))
    only = {p.strip() for p in (args.providers or "").split(",") if p.strip()} or None
    holidays = nse_holidays()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    index = ent.CompanyIndex(list(con.execute("SELECT symbol, name FROM companies")))
    counters = {p: Counter() for p in ("google_news", "gdelt", "marketaux", "alphavantage", "newsapi", "publisher_rss")}
    env = os.environ
    caps = {
        "google_news": bm.Capability("google_news", True),
        "gdelt": bm.Capability("gdelt", True, max_age=dt.timedelta(days=88)),
        "marketaux": bm.Capability("marketaux", bool(env.get("MARKETAUX_API_TOKEN"))),
        "alphavantage": bm.Capability("alphavantage", bool(env.get("ALPHAVANTAGE_API_KEY")),
                                      max_age=now - dt.datetime(2022, 3, 1, tzinfo=UTC)),
        "newsapi": bm.Capability("newsapi", bool(env.get("NEWSAPI_KEY")), max_age=dt.timedelta(days=30)),
    }
    pub_items: list[prov.RawArticle] = []
    try:
        pub_items = list(ad.PublisherRssProvider(get=counters["publisher_rss"], sleep=lambda s: None)
                         .fetch(now - dt.timedelta(days=30), now))
    except prov.ProviderError:
        pass
    caps["publisher_rss"] = bm.Capability("publisher_rss", True,
                                          live_since=min((a.published_at for a in pub_items), default=now))
    caps["official_rss"] = bm.Capability("official_rss", True, live_since=now - dt.timedelta(days=30),
                                         of_record_kinds=tuple(OFFICIAL))

    def query(pname: str, ev: bm.Event) -> list[prov.RawArticle]:
        lo, hi = ev.window
        if pname == "google_news":
            time.sleep(2.0)
            return list(ad.GoogleNewsProvider(queries=[ev.query], get=counters[pname], sleep=lambda s: None).fetch(lo, hi))
        if pname == "gdelt":
            time.sleep(6.5)
            return list(prov.GdeltProvider(query=f"{ev.query} sourcelang:english", get=counters[pname], sleep=lambda s: None).fetch(lo, hi))
        if pname == "marketaux":
            return list(ad.MarketauxProvider(search=ev.query.replace('"', ""), countries="in,us", get=counters[pname],
                                             sleep=lambda s: None, max_pages=1).fetch(lo, hi))
        if pname == "alphavantage":
            tick = [s.replace(".NS", ".BSE") for s in ev.expected_symbols]
            time.sleep(12.5)
            return list(ad.AlphaVantageNewsProvider(tickers=tick or None, get=counters[pname], sleep=lambda s: None).fetch(lo, hi))
        if pname == "newsapi":
            return list(prov.NewsApiProvider(query=ev.query, get=counters[pname], sleep=lambda s: None).fetch(lo, hi))
        if pname == "publisher_rss":
            return pub_items
        return []

    outcomes: list[bm.Outcome] = []
    if base is not None and only:
        outcomes = [bm.Outcome(**o) for o in base["outcomes"] if o["provider"] not in only]
    refused: dict[str, str] = {}
    for ev in events:
        for pname, cap in caps.items():
            if only and pname not in only:
                continue
            st = bm.testability(cap, ev, now)
            if st or pname == "official_rss":
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, st or bm.NO_HISTORY,
                                           detail=None if st else "live feed: official items only"))
                continue
            if pname in refused:                 # plan / permission refusal: never re-asked this run
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, bm.NOT_CONFIGURED, detail=refused[pname]))
                continue
            try:
                items = query(pname, ev)
            except prov.NotPermitted as e:
                refused[pname] = str(e)[:120]
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, bm.NOT_CONFIGURED, detail=refused[pname]))
                continue
            except prov.NotConfigured as e:
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, bm.NOT_CONFIGURED, detail=str(e)[:120]))
                continue
            except prov.RateLimited as e:
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, bm.RATE_LIMITED, detail=str(e)[:120]))
                continue
            except prov.ProviderError as e:
                outcomes.append(bm.Outcome(pname, ev.id, ev.kind, bm.OUTAGE, detail=str(e)[:120]))
                continue
            outcomes.append(bm.evaluate(pname, ev, items, index=index, holidays=holidays, canonical=canonical_count))
    ev_dicts = [event_to_dict(e) for e in events]
    derived = derive(outcomes, ev_dicts)
    summary, rec = derived["providers"], derived["recommendation"]
    health = {}
    for name in ("official_rss", "federal_register", "nse_announcements", "nse_rss", "google_news", "publisher_rss",
                 "gdelt", "marketaux", "alphavantage", "newsapi"):
        try:
            health[name] = registry.health(name)
        except Exception as e:  # noqa: BLE001 - report, never crash the benchmark on a health probe
            health[name] = {"provider": name, "status": "ERROR", "detail": type(e).__name__}
    stats = {k: v.stats for k, v in counters.items() if not only or k in only}
    if base is not None and only:
        stats = {**{k: v for k, v in (base.get("request_stats") or {}).items() if k not in only}, **stats}
    costs = {n: {"cost": s.cost, "licence": s.licence, "daily_limit": s.daily_limit,
                 "requests_this_run": stats[n]["requests"] if n in stats else None}
             for n, s in registry.SPECS.items() if n in caps or n in ("federal_register", "nse_announcements", "nse_rss")}
    report = {
        "generated_at": now.isoformat(), "research_only": True,
        "ground_truth_from": args.events_from and Path(args.events_from).name,
        "ground_truth_generated_at": base["generated_at"] if base else now.isoformat(),
        "providers_queried_this_run": sorted(only) if only else sorted(caps),
        "method": __doc__.split("Ground truth")[1].split("Output:")[0].strip(),
        "status_definitions": bm.__doc__.split("Matching,")[0].split("the outcome is exactly one of:")[1].strip(),
        "ground_truth": {"events": len(events), "by_kind": {k: sum(1 for e in events if e.kind == k) for k in sorted({e.kind for e in events})},
                         "sources": notes, "nse_holidays_inferred": len(holidays)},
        "events": ev_dicts,
        **derived, **examples(outcomes, {e.id: e for e in events}),
        "source_health": health, "request_stats": stats,
        "costs": costs,
        "outcomes": [o.__dict__ for o in outcomes],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"events": report["ground_truth"], "providers": {k: {x: v[x] for x in ("tested", "detected", "recall")}
                                                                        for k, v in summary.items()},
                      "recommended": rec["recommended"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
