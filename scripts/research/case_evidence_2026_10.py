"""
scripts/research/case_evidence_2026_10.py — reproducible evidence for the
5-9 October 2026 case review (read-only; writes JSON + Markdown).

  python scripts/research/case_evidence_2026_10.py [--cache PATH]

Sections:
  1. Canara Bank forensics: the live v1 selection (run, timestamps, every
     component, reference price and its source), the month-end rank history,
     price path from the frozen reference and from the first tradable open,
     NIFTY / Bank Nifty and two baselines (the rest of the same run's top 10,
     and the top 10 by the run's own momentum component), the first
     qualifying move under a rule fixed in advance, and a classification
     from rules fixed in advance (below).
  2. Intraday reaction timing (Yahoo 5-minute bars, IST): when each case
     stock repriced - the market-side timestamp evidence. Filing / news
     publication and ingestion times stay BLOCKED (no event source).
  3. Session audit for 5-9 Oct over the v1 universe plus known omissions:
     material movers (|close-to-close| >= 5%), whether v1 Top Candidates or
     a v2 TOMORROW_EOD *simulation* at the previous close surfaced them,
     per-session counts, data exclusions, Top Candidates staleness.

Rules fixed before looking at outcomes (no tuning; nothing here feeds v2):
  QUALIFYING_MOVE  cumulative return from the reference >= +3% AND >= +2
                   points above Bank Nifty, within 5 sessions.
  Canara class     "Verified short-term early-signal success" needs a frozen
                   directional prediction made before the move: v1 makes none,
                   so it is impossible. Otherwise: excess vs Bank Nifty > 0
                   and absolute > 0.25% at 5 sessions -> "Long-term
                   quality/value selection followed by a positive move";
                   absolute > 0 but excess <= 0 -> "Coincidental positive
                   outcome" (explained by the sector); missing data ->
                   "Insufficient evidence"; otherwise "No positive move".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from prediction_v2 import features as feat, rules  # noqa: E402
from ranking.presenter import ranking_freshness  # noqa: E402
from utils.market_session import IST  # noqa: E402

LIVE_RUN = "RANKING-20261003-091616-7eb2ae"
WEEK = ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]
HOLIDAYS = ["2026-10-02"]                       # NSE closed (no bars on any symbol); see F-07
OUTSIDE_V1 = ["CHENNPETRO.NS", "MRPL.NS", "KANOHAR.NS", "MONEYVIEW.NS"]
INTRADAY_CASES = ["CANBK.NS", "TRENT.NS", "ITC.NS", "TITAN.NS", "CHENNPETRO.NS", "MRPL.NS", "IOC.NS", "BPCL.NS",
                  "HINDPETRO.NS", "HCLTECH.NS"]
INDEXES = ["^NSEI", "^NSEBANK"]
MOVER = 0.05
OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2"
VALIDATION_CSV = ROOT / "scripts" / "research" / "output" / "ranking_validation" / "ranking_snapshots_production.csv"


def _ist(ts: str) -> str:
    t = pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)
    return t.tz_convert(IST).isoformat()


def _daily(symbols: list[str], start: str = "2026-04-01", end: str = "2026-10-10") -> dict[str, pd.DataFrame]:
    raw = yf.download(symbols, start=start, end=end, interval="1d", auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out = {}
    for s in symbols:
        try:
            df = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out[s] = df
    return out


def _ret(df: pd.DataFrame, start_price: float, start_date: dt.date, n: int) -> float | None:
    after = df[df.index.date > start_date]
    return None if len(after) < n else float(after["Close"].iloc[n - 1] / start_price - 1)


def canara(con, daily: dict[str, pd.DataFrame]) -> dict:
    run = dict(zip(("run_id", "kind", "status", "started_at", "finished_at", "engine_version", "total", "processed",
                    "skipped"), con.execute("SELECT run_id, kind, status, started_at, finished_at, engine_version, "
                                            "total, processed, skipped FROM engine_runs WHERE run_id = ?",
                                            (LIVE_RUN,)).fetchone()))
    rows = {r[0]: r for r in con.execute(
        "SELECT symbol, rank, stockai_score, score_coverage, eligible, components, freshness, computed_at "
        "FROM stock_analysis_results WHERE run_id = ?", (LIVE_RUN,))}
    sym = "CANBK.NS"
    _, rank, score, cov, elig, comps, fresh, computed = rows[sym]
    comps, fresh = json.loads(comps), json.loads(fresh)
    eligible_n = sum(1 for r in rows.values() if r[4])
    snap = con.execute("SELECT count(*) FROM ranking_snapshots WHERE run_id = ?", (LIVE_RUN,)).fetchone()[0]
    ms = con.execute("SELECT fetched_at, source, as_of_date, close FROM market_snapshots WHERE symbol = ? "
                     "ORDER BY fetched_at DESC LIMIT 1", (sym,)).fetchone()
    ref_price, ref_date = float(ms[3]), dt.date.fromisoformat(ms[2])
    df, bank, nifty = daily[sym], daily["^NSEBANK"], daily["^NSEI"]
    first_open_day = df[df.index.date > dt.date(2026, 10, 3)].index[0].date()     # selection was published on a Saturday
    first_open = float(df.loc[pd.Timestamp(first_open_day), "Open"])

    def path(start_price: float, start_date: dt.date, bench_from_open: bool = False) -> dict:
        out = {}
        for n in (1, 3, 5):
            r = _ret(df, start_price, start_date if not bench_from_open else start_date - dt.timedelta(days=1), n)
            b = {}
            for name, idx in (("nifty", nifty), ("bank_nifty", bank)):
                if bench_from_open:
                    o = float(idx.loc[pd.Timestamp(first_open_day), "Open"])
                    b[name] = _ret(idx, o, start_date - dt.timedelta(days=1), n)
                else:
                    c0 = float(idx[idx.index.date <= start_date]["Close"].iloc[-1])
                    b[name] = _ret(idx, c0, start_date, n)
            out[f"{n}s"] = {"return": r, **b, "excess_vs_bank_nifty": None if r is None or b["bank_nifty"] is None
                            else r - b["bank_nifty"]}
        win = df[df.index.date > (start_date if not bench_from_open else start_date - dt.timedelta(days=1))].iloc[:5]
        out["mfe_5s"] = float(win["High"].max() / start_price - 1)
        out["mae_5s"] = float(win["Low"].min() / start_price - 1)
        return out

    from_ref = path(ref_price, ref_date)
    from_open = path(first_open, first_open_day, bench_from_open=True)
    # first qualifying move (rule in the module docstring), from the frozen reference
    qualifying = None
    after = df[df.index.date > ref_date].iloc[:5]
    b0 = float(bank[bank.index.date <= ref_date]["Close"].iloc[-1])
    for d, c in after["Close"].items():
        cum = float(c / ref_price - 1)
        bcum = float(bank.loc[d, "Close"] / b0 - 1) if d in bank.index else None
        if cum >= 0.03 and bcum is not None and cum - bcum >= 0.02:
            qualifying = {"date": d.date().isoformat(), "cumulative": cum, "excess_vs_bank_nifty": cum - bcum}
            break
    daily_moves = [{"date": d.date().isoformat(), "close": float(c), "ret": float(c / p - 1)}
                   for (d, c), p in zip(after["Close"].items(), [ref_price] + list(after["Close"].iloc[:-1]))]
    # baselines from the same live run
    eligible = sorted((r for r in rows.values() if r[4] and r[1]), key=lambda r: r[1])
    top10 = [r[0] for r in eligible[:10]]

    def mom(r):
        m = json.loads(r[5]).get("momentum", {}).get("score")
        return -1 if m is None else m
    mom10 = [r[0] for r in sorted(eligible, key=mom, reverse=True)[:10]]
    ref_closes = {s: float(daily[s][daily[s].index.date <= ref_date]["Close"].iloc[-1]) for s in set(top10 + mom10)
                  if s in daily}

    def basket(syms):
        rs = [_ret(daily[s], ref_closes[s], ref_date, 5) for s in syms if s in daily and s in ref_closes]
        rs = [r for r in rs if r is not None]
        return {"members": syms, "with_data": len(rs), "mean_5s": sum(rs) / len(rs) if rs else None}
    r5 = from_ref["5s"]
    if r5["return"] is None or r5["bank_nifty"] is None:
        cls = "Insufficient evidence"
    elif r5["excess_vs_bank_nifty"] > 0 and r5["return"] > 0.0025:
        cls = "Long-term quality/value selection followed by a positive move"
    elif r5["return"] > 0:
        cls = "Coincidental positive outcome"
    else:
        cls = "No positive move"
    hist = []
    if VALIDATION_CSV.exists():
        v = pd.read_csv(VALIDATION_CSV)
        n_by_date = v[v["eligible"] == True].groupby("date").size().to_dict()  # noqa: E712
        for _, r in v[v["symbol"] == sym].sort_values("date").iterrows():
            hist.append({"date": r["date"], "rank": None if pd.isna(r["rank"]) else int(r["rank"]),
                         "eligible_count": int(n_by_date.get(r["date"], 0)), "score": r["stockai_score"],
                         "quality": r["c_quality"], "valuation": r["c_valuation"],
                         "financial_health": r["c_financial_health"], "technical_trend": r["c_technical_trend"],
                         "momentum": r["c_momentum"], "risk": r["c_risk"]})
    top10_count = sum(1 for h in hist if h["rank"] and h["rank"] <= 10)
    return {
        "live_selection": {
            "run": {**run, "started_at_ist": _ist(run["started_at"]), "finished_at_ist": _ist(run["finished_at"])},
            "computed_at_ist": _ist(computed), "rank": rank, "eligible_in_run": eligible_n, "score": score,
            "score_coverage": cov, "components": {k: {"score": c.get("score"), "weight": c.get("weight"),
                                                      "contribution": c.get("contribution"), "basis": c.get("basis")}
                                                  for k, c in comps.items()},
            "freshness": fresh, "frozen_ranking_snapshot_rows_for_run": snap,
            "reference_price": ref_price, "reference_date": ref_date.isoformat(),
            "reference_price_source": "market_snapshot_at_analysis (fallback: the run has no ranking_snapshots rows, F-09)",
            "first_tradable_session_after_publication": first_open_day.isoformat(), "first_tradable_open": first_open,
            "selection_visible_to_users_from_ist": _ist(run["finished_at"]),
            "note": "published Saturday 3 Oct after the 1 Oct close (2 Oct was an NSE holiday); "
                    "a user could first act at the 5 Oct open"},
        "rank_history_month_end_reproductions": hist,
        "top10_in_month_end_reproductions": f"{top10_count} of {len(hist)}",
        "returns_from_reference_close": from_ref, "returns_from_first_tradable_open": from_open,
        "daily_moves_after_reference": daily_moves, "first_qualifying_move": qualifying,
        "baselines_5s_from_reference": {"same_run_top10_equal_weight": basket(top10),
                                        "same_run_top10_by_momentum_component": basket(mom10)},
        "drivers": "quality, valuation and financial health at or near maximum; technical trend bearish; "
                   "sector outlook not available (F-08); no event input exists (F-05)",
        "classification": cls,
        "early_signal_success_possible": False,
        "why": "v1 is a long-term ranking with no frozen directional prediction or horizon, so a short-term "
               "early-signal success cannot be verified for any stock",
    }


def intraday(symbols: list[str]) -> dict:
    """Reaction timing from 5-minute bars (IST)."""
    raw = yf.download(symbols + ["^NSEI"], start="2026-10-01", end="2026-10-10", interval="5m",
                      group_by="ticker", progress=False, threads=True, auto_adjust=False)
    out = {}
    for s in symbols:
        try:
            df = raw[s].dropna(subset=["Close"])
        except KeyError:
            out[s] = {"error": "no intraday data"}
            continue
        if df.empty:
            out[s] = {"error": "no intraday data"}
            continue
        df.index = df.index.tz_convert(IST)
        days = sorted({d.date() for d in df.index})
        rows = []
        for i, d in enumerate(days):
            if d.isoformat() not in WEEK:
                continue
            day = df[df.index.date == d]
            prev = df[df.index.date == days[i - 1]] if i else None
            if prev is None or prev.empty or day.empty:
                continue
            pc = float(prev["Close"].iloc[-1])
            o, c = float(day["Open"].iloc[0]), float(day["Close"].iloc[-1])
            by945 = day[day.index.time <= dt.time(9, 45)]
            vol = float(day["Volume"].sum())
            hi_t, lo_t = day["High"].idxmax(), day["Low"].idxmin()
            total = c / pc - 1
            rows.append({"session": d.isoformat(), "first_bar_ist": day.index[0].strftime("%H:%M"),
                         "open_gap": o / pc - 1, "ret_by_0945": float(by945["Close"].iloc[-1] / pc - 1) if len(by945) else None,
                         "ret_close": total,
                         "share_of_day_move_by_0945": (float(by945["Close"].iloc[-1] / pc - 1) / total)
                         if len(by945) and abs(total) > 0.005 else None,
                         "first_30min_volume_share": float(day[day.index.time < dt.time(9, 45)]["Volume"].sum() / vol)
                         if vol else None,
                         "high_at_ist": hi_t.strftime("%H:%M"), "low_at_ist": lo_t.strftime("%H:%M")})
        out[s] = {"sessions": rows}
    return out


def audit(con, daily: dict[str, pd.DataFrame], nifty: pd.DataFrame, sectors: dict) -> dict:
    v1 = {s: (r, e) for s, r, e in con.execute("SELECT symbol, rank, eligible FROM stock_analysis_results "
                                               "WHERE run_id = ?", (LIVE_RUN,))}
    members = {r[0] for r in con.execute("SELECT symbol FROM stock_universe")}
    top20 = {s for s, (r, e) in v1.items() if e and r and r <= 20}
    sessions, movers = [], []
    for day in WEEK:
        d = dt.date.fromisoformat(day)
        feats, flags, rets = {}, {}, {}
        for s, df in daily.items():
            if s.startswith("^"):
                continue
            idx = [x.date() for x in df.index]
            if d not in idx or idx.index(d) == 0:
                continue
            i = idx.index(d)
            prev = idx[i - 1]
            rets[s] = float(df["Close"].iloc[i] / df["Close"].iloc[i - 1] - 1)
            feats[s], flags[s] = feat.compute(df, prev, nifty, expected_session=prev)
        feat.add_sector_relative(feats, sectors)
        dec = {s: rules.decide(feats[s], flags[s]) for s in feats}
        counts = {k: sum(1 for x in dec.values() if x["direction"] == k) for k in ("UP", "DOWN", "NEUTRAL", "NO_CALL")}
        calls = [(s, x["direction"], rets[s]) for s, x in dec.items() if x["direction"] in ("UP", "DOWN")]
        sim_hits = sum(1 for _, dr, r in calls if (r > 0) == (dr == "UP"))
        day_movers = []
        for s, r in rets.items():
            if abs(r) < MOVER:
                continue
            rank, elig = v1.get(s, (None, None))
            m = {"session": day, "symbol": s, "ret": r, "in_v1_universe": s in members, "v1_rank_3oct": rank,
                 "v1_eligible": bool(elig) if elig is not None else None, "in_v1_top20": s in top20,
                 "v2_sim_prev_close": dec[s]["direction"], "v2_sim_setup": dec[s]["setup_type"],
                 "v2_sim_flags": [f for f in flags[s] if f in rules.BLOCKING_FLAGS]}
            m["surfaced_by_v1_top20"] = m["in_v1_top20"]
            m["v2_sim_right_direction"] = (dec[s]["direction"] == "UP") == (r > 0) if dec[s]["direction"] in ("UP", "DOWN") else None
            day_movers.append(m)
        now = dt.datetime.combine(d, dt.time(18, 0), tzinfo=IST)
        fresh = ranking_freshness("2026-10-01", HOLIDAYS, now)
        sessions.append({"session": day, "stocks_with_bar": len(rets),
                         "v2_sim_counts_at_prev_close": counts, "v2_sim_directional_calls": len(calls),
                         "v2_sim_calls_right_direction_next_session": sim_hits,
                         "material_movers": len(day_movers),
                         "material_movers_in_v1_top20": sum(1 for m in day_movers if m["in_v1_top20"]),
                         "material_movers_v1_ineligible": sum(1 for m in day_movers if m["v1_eligible"] is False),
                         "material_movers_outside_v1_universe": sum(1 for m in day_movers if not m["in_v1_universe"]),
                         "top_candidates_ranking_date": "2026-10-01",
                         "top_candidates_freshness_at_1800_ist": {"status": fresh["status"],
                                                                  "sessions_behind": fresh["sessions_behind"]},
                         "v1_full_runs_that_day": 0})
        movers += day_movers
    v1_symbols = sorted(members)
    no_data = sorted(s for s in v1_symbols if s not in daily)
    return {"sessions": sessions, "movers": movers,
            "summary": {"stock_days": len(movers), "stocks": len({m["symbol"] for m in movers}),
                        "up": sum(1 for m in movers if m["ret"] > 0), "down": sum(1 for m in movers if m["ret"] < 0),
                        "in_v1_top20": sum(1 for m in movers if m["in_v1_top20"]),
                        "v1_ineligible": sum(1 for m in movers if m["v1_eligible"] is False),
                        "outside_v1_universe": sorted({m["symbol"] for m in movers if not m["in_v1_universe"]}),
                        "v2_sim_called_right_direction": sum(1 for m in movers if m["v2_sim_right_direction"]),
                        "v2_sim_called_wrong_direction": sum(1 for m in movers if m["v2_sim_right_direction"] is False)},
            "data_exclusions": {"v1_symbols_without_yahoo_data": no_data, "count": len(no_data)},
            "v1_runs_in_week_local_db": [r for r in con.execute(
                "SELECT run_id, kind, status, started_at FROM engine_runs WHERE started_at >= '2026-10-04'")],
            "job_ledger": "scheduled_job_runs does not exist in the local database copy (older schema): no job "
                          "records for the week exist locally; production run history is BLOCKED (F-13)",
            "formal_classification": "Formal false positives / false negatives need frozen directional predictions "
                                     "made before each session. None existed (v1 ranks are not directional; no v2 "
                                     "snapshot ran that week), so none can be classified. The v2 columns are a "
                                     "historical SIMULATION at the previous close, not recovered predictions."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "storage" / "price_cache" / "backtest_v2_3y.pkl"))
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{ROOT / 'storage' / 'app.db'}?mode=ro", uri=True)
    sectors = dict(con.execute("SELECT symbol, sector FROM companies"))
    data, nifty = pickle.loads(Path(args.cache).read_bytes())
    daily = {s: df for s, df in data.items() if df is not None and not df.empty}
    extra = _daily(OUTSIDE_V1 + INDEXES)
    daily.update({s: df for s, df in extra.items() if not s.startswith("^")})
    idx = {s: df for s, df in extra.items() if s.startswith("^")}
    report = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sources": {"daily": "Yahoo Finance daily bars (engine provider; delayed, adjusted; not exchange data)",
                          "intraday": "Yahoo Finance 5-minute bars, IST (unlicensed, delayed; research only)",
                          "v1": f"local database copy, run {LIVE_RUN} (read-only)",
                          "rank_history": str(VALIDATION_CSV.relative_to(ROOT))},
              "canara": canara(con, {**daily, **idx}),
              "intraday_reaction": intraday(INTRADAY_CASES),
              "audit": audit(con, daily, nifty, sectors),
              "catalyst_timestamps": "BLOCKED: no filings or news source with publication timestamps is connected "
                                     "(F-05); nothing was ingested that week, so ingestion times do not exist."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case_evidence_2026_10.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    c = report["canara"]
    print("canara:", c["classification"], c["returns_from_reference_close"]["5s"], c["first_qualifying_move"])
    print("audit:", report["audit"]["summary"])


if __name__ == "__main__":
    main()
