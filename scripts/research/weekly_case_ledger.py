"""
scripts/research/weekly_case_ledger.py — reproducible ledger for the
5-9 October 2026 case review (read-only; writes JSON + Markdown).

For each case symbol and session it records the daily move, NIFTY move,
abnormal volume, the v1 rank in the live 3 Oct run (local database copy,
read-only), and what the Prediction v2 baseline WOULD HAVE decided at the
previous close (a TOMORROW_EOD simulation using only bars up to that close).
No original v2 snapshot existed that week, so the v2 column is a historical
simulation, not a recovered prediction. Event publication times are not
available (no event provider), so catalyst timing stays BLOCKED.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yfinance as yf  # noqa: E402

from prediction_v2 import features as feat, rules  # noqa: E402

CASES = ["CANBK.NS", "ITC.NS", "TRENT.NS", "TITAN.NS", "HCLTECH.NS", "CHENNPETRO.NS", "MRPL.NS", "IOC.NS",
         "BPCL.NS", "HINDPETRO.NS", "KANOHAR.NS", "MONEYVIEW.NS"]
SESSIONS = ["2026-10-01", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]
LIVE_RUN = "RANKING-20261003-091616-7eb2ae"
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2"


def main() -> None:
    raw = yf.download(CASES + ["^NSEI"], start="2026-04-01", end="2026-10-11", interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    frames = {}
    for s in CASES + ["^NSEI"]:
        try:
            df = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        frames[s] = df
    nifty = frames["^NSEI"]
    con = sqlite3.connect(f"file:{ROOT / 'storage' / 'app.db'}?mode=ro", uri=True)
    v1 = {s: (r, sc, e) for s, r, sc, e in con.execute(
        "SELECT symbol, rank, stockai_score, eligible FROM stock_analysis_results WHERE run_id = ?", (LIVE_RUN,))}
    members = {r[0] for r in con.execute("SELECT symbol FROM stock_universe")}
    rows = []
    for sym in CASES:
        df = frames.get(sym)
        for day in SESSIONS:
            d = dt.date.fromisoformat(day)
            row = {"symbol": sym, "session": day, "in_v1_universe": sym in members,
                   "v1_rank_3oct": v1.get(sym, (None,))[0], "v1_score_3oct": v1.get(sym, (None, None))[1]}
            if df is None or d not in {x.date() for x in df.index}:
                row["note"] = "no bar"
                rows.append(row)
                continue
            i = [x.date() for x in df.index].index(d)
            if i == 0:
                rows.append({**row, "note": "first listed session"})
                continue
            prev = df.index[i - 1].date()
            c, pc = float(df["Close"].iloc[i]), float(df["Close"].iloc[i - 1])
            ni = [x.date() for x in nifty.index]
            nret = float(nifty["Close"].iloc[ni.index(d)] / nifty["Close"].iloc[ni.index(d) - 1] - 1) if d in ni else None
            base = df["Volume"].iloc[max(0, i - 20):i].median()
            f, flags = feat.compute(df, prev, nifty, expected_session=prev)
            decision = rules.decide(f, flags)
            row.update({"ret_pct": round((c / pc - 1) * 100, 2), "nifty_ret_pct": round(nret * 100, 2) if nret is not None else None,
                        "gap_pct": round((float(df["Open"].iloc[i]) / pc - 1) * 100, 2),
                        "volume_x_median20": round(float(df["Volume"].iloc[i] / base), 1) if base else None,
                        "v2_sim_at_prev_close": decision["direction"], "v2_sim_setup": decision["setup_type"],
                        "v2_sim_flags": [x for x in flags if x in rules.BLOCKING_FLAGS],
                        "history_sessions_at_prev_close": f.get("history_sessions")})
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weekly_case_ledger.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    lines = ["| Symbol | Session | Move % | NIFTY % | Gap % | Vol x | v1 rank (3 Oct) | v2 sim (prev close) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "ret_pct" not in r:
            continue
        v1r = "not in universe" if not r["in_v1_universe"] else (r["v1_rank_3oct"] if r["v1_rank_3oct"] else "ineligible")
        sim = r["v2_sim_at_prev_close"] + (f" ({', '.join(r['v2_sim_flags'])})" if r["v2_sim_flags"] else
                                          f" ({r['v2_sim_setup']})" if r["v2_sim_setup"] not in ("NONE",) else "")
        lines.append(f"| {r['symbol']} | {r['session']} | {r['ret_pct']:+.2f} | {r['nifty_ret_pct']:+.2f} | "
                     f"{r['gap_pct']:+.2f} | {r['volume_x_median20']} | {v1r} | {sim} |")
    (OUT / "weekly_case_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    calls = [r for r in rows if r.get("v2_sim_at_prev_close") in ("UP", "DOWN")]
    print(f"{len(rows)} rows; simulated v2 directional calls: {[(r['symbol'], r['session'], r['v2_sim_at_prev_close'], r['ret_pct']) for r in calls]}")


if __name__ == "__main__":
    main()
