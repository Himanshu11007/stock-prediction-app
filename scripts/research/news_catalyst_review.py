"""
scripts/research/news_catalyst_review.py — why are the experimental engine's
news calls (Prediction Engine v2, setup NEWS_CATALYST) wrong so often?

Read-only. Lists every NEWS_CATALYST call with the news behind it (headline,
category, how the stock is linked: named in the story or inferred through a
sector / commodity link), the price move BEFORE the call (was the news already
in the price?) and the result after 1 / 3 / 5 sessions. Then groups the calls
to show which kind of news call fails.

Run on the server (the env file supplies DATABASE_URL; it is never printed):

  sudo -u stocklens bash -c 'set -a; . /etc/stocklens/stocklens.env; set +a; \
    cd /opt/stocklens/app && /opt/stocklens/venv/bin/python scripts/research/news_catalyst_review.py'

Options: --days N (default 30), --limit N calls listed (default 40), --json FILE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_env_file(path: Path) -> None:
    """KEY=value lines, taken literally (a password may contain & $ ;), as run-job.sh does."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            os.environ[key] = value


ap = argparse.ArgumentParser()
ap.add_argument("--env-file", type=Path)
ap.add_argument("--days", type=int, default=30)
ap.add_argument("--limit", type=int, default=40)
ap.add_argument("--json", type=Path)
ARGS = ap.parse_args() if __name__ == "__main__" else None
if ARGS and ARGS.env_file:
    load_env_file(ARGS.env_file)                                   # before db.session reads DATABASE_URL

from sqlmodel import Session, select  # noqa: E402

from db.models.news import EventEntity, utcnow  # noqa: E402
from db.models.prediction import (EventClassification, MarketEvent, Prediction,  # noqa: E402
                                  PredictionOutcome, PredictionRun)
from db.session import engine  # noqa: E402

HORIZONS = (1, 3, 5)


def _pct(x):
    return "   -  " if x is None else f"{x * 100:+5.1f}%"


def _links(session: Session, event_ids: list, symbol: str) -> list[dict]:
    out = []
    for eid in event_ids or []:
        ev = session.get(MarketEvent, eid)
        if ev is None:
            continue
        cls = session.exec(select(EventClassification).where(EventClassification.event_id == eid)
                           .order_by(EventClassification.id.desc())).first()
        ent = session.exec(select(EventEntity).where(EventEntity.event_id == eid,
                                                     EventEntity.symbol == symbol)).first()
        out.append({"event_id": eid, "title": ev.title[:140], "type": ev.event_type, "source": ev.source,
                    "published_at": ev.published_at.isoformat() if ev.published_at else None,
                    "direction": cls.direction if cls else None,
                    "link": ("not linked" if ent is None else
                             f"inferred ({ent.hypothesis_id or ent.mechanism or 'sector'})" if ent.inferred
                             else f"named ({ent.relation.lower()})"),
                    "inferred": bool(ent and ent.inferred)})
    return out


def collect(session: Session, days: int) -> list[dict]:
    since = (utcnow() - timedelta(days=days)).date().isoformat()
    rows = session.exec(select(Prediction, PredictionRun).join(PredictionRun, PredictionRun.run_id == Prediction.run_id)
                        .where(Prediction.setup_type == "NEWS_CATALYST", PredictionRun.trading_date >= since)
                        .order_by(PredictionRun.trading_date.desc())).all()
    calls = []
    for p, run in rows:
        outcomes = {o.horizon_sessions: o for o in session.exec(
            select(PredictionOutcome).where(PredictionOutcome.prediction_id == p.prediction_id)).all()}
        f = p.features or {}
        sign = 1 if p.direction == "UP" else -1
        before = f.get("ret_1")
        links = _links(session, p.event_ids, p.symbol)
        calls.append({
            "date": run.trading_date, "run_type": run.run_type, "symbol": p.symbol, "call": p.direction,
            "move_before_1d": before, "move_before_5d": f.get("ret_5"), "gap": f.get("gap_pct"),
            # the stock had already moved >= 2% the call's way on the cutoff session
            "already_moved": before is not None and before * sign >= 0.02,
            "technical": next((r for r in (p.reasons or []) if isinstance(r, str) and "technical setup" in r), None),
            "reasons": [r for r in (p.reasons or []) if isinstance(r, str)][:4],
            "news": links,
            "inferred_only": bool(links) and all(n["inferred"] for n in links),
            "results": {h: {"status": outcomes[h].outcome_status, "stock_return": outcomes[h].stock_return,
                            "directional_return": outcomes[h].directional_return, "hit": outcomes[h].hit}
                        for h in HORIZONS if h in outcomes},
        })
    return calls


def _summary(calls: list[dict], label) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for c in calls:
        groups[label(c)].append(c)
    out = {}
    for key, items in sorted(groups.items()):
        row = {"calls": len(items)}
        for h in HORIZONS:
            done = [c["results"][h] for c in items if h in c["results"] and c["results"][h]["hit"] is not None]
            if done:
                rets = [r["directional_return"] for r in done if r["directional_return"] is not None]
                row[f"{h}d"] = {"evaluated": len(done), "right": sum(1 for r in done if r["hit"]),
                                "avg_return_in_call_direction": sum(rets) / len(rets) if rets else None}
        out[key] = row
    return out


def main() -> None:
    args = ARGS
    with Session(engine) as session:
        calls = collect(session, args.days)
    if not calls:
        print(f"No NEWS_CATALYST calls in the last {args.days} days.")
        return

    print(f"NEWS CALLS, last {args.days} days: {len(calls)} (showing {min(args.limit, len(calls))})")
    print("move before = the stock's move on the session the call was made; result = move after, in the call's"
          " direction (+ = the call was right)\n")
    for c in calls[:args.limit]:
        res = "  ".join(f"{h}d {_pct(c['results'][h]['directional_return'])}" if h in c["results"] else f"{h}d   -  "
                        for h in HORIZONS)
        print(f"{c['date']} {c['run_type'][:13]:<13} {c['symbol']:<14} {c['call']:<4} before {_pct(c['move_before_1d'])}"
              f" (5d {_pct(c['move_before_5d'])})   result {res}")
        for n in c["news"][:3]:
            print(f"    - [{n['direction'] or '?'} | {n['type']} | {n['link']}] {n['title']}")
        if c["technical"]:
            print(f"    - {c['technical']}")

    groupings = {
        "by call direction": lambda c: c["call"],
        "by news link (named in the story vs inferred through a sector / commodity link)":
            lambda c: "inferred link only" if c["inferred_only"] else "named in the story",
        "by price before the call (already moved >= 2% the call's way?)":
            lambda c: "already moved" if c["already_moved"] else "not yet moved",
        "by technical picture": lambda c: (c["technical"] or "unknown").split(" (")[0],
        "by run": lambda c: c["run_type"],
    }
    summaries = {}
    for title, fn in groupings.items():
        summaries[title] = s = _summary(calls, fn)
        print(f"\n== {title}")
        for key, row in s.items():
            parts = [f"{key:<28} {row['calls']:>3} calls"]
            for h in HORIZONS:
                r = row.get(f"{h}d")
                if r:
                    parts.append(f"{h}d right {r['right']}/{r['evaluated']} avg {_pct(r['avg_return_in_call_direction'])}")
            print("  " + "   ".join(parts))
    if args.json:
        args.json.write_text(json.dumps({"calls": calls, "summaries": summaries}, indent=2, default=str),
                             encoding="utf-8")
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
