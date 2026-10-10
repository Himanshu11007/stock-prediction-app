"""
scripts/research/backtest_v2.py — chronological backtest of the Prediction v2
baseline rules on the v1 universe (read-only research; writes a JSON report).

  python scripts/research/backtest_v2.py [--period 3y] [--horizon 1] [--out PATH]

Data: daily bars from the engine's provider (Yahoo Finance via
fundamentals.provider.fetch_price_history); sectors from the local companies
table (read-only). v1 baseline: top-20 of each month-end validation snapshot
(scripts/research/output/ranking_validation/ranking_snapshots_production.csv).

Limitations (also written into the report): survivorship (today's universe
list is used for the whole period; delisted/renamed symbols are missing),
provider data quality, no corporate-event inputs, no intraday data. The rules
are NOT tuned here: thresholds are prediction_v2.rules.THRESHOLDS as written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fundamentals.provider import fetch_price_history  # noqa: E402
from prediction_v2 import backtest, rules  # noqa: E402
from prediction_v2.features import FEATURE_SET_VERSION  # noqa: E402

VALIDATION_CSV = ROOT / "scripts" / "research" / "output" / "ranking_validation" / "ranking_snapshots_production.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "scripts" / "research" / "output" / "prediction_v2" / "backtest_report.json"))
    args = ap.parse_args()
    t0 = time.time()
    con = sqlite3.connect(f"file:{ROOT / 'storage' / 'app.db'}?mode=ro", uri=True)
    symbols = sorted({r[0] for r in con.execute("SELECT symbol FROM stock_universe")})
    sectors = dict(con.execute("SELECT symbol, sector FROM companies"))
    data = {}
    for i in range(0, len(symbols), 50):
        data.update(fetch_price_history(symbols[i:i + 50], period=args.period))
    nifty = fetch_price_history(["^NSEI"], period=args.period)["^NSEI"]
    hist = {s: df for s, df in data.items() if df is not None and len(df) > 80}
    missing = sorted(set(symbols) - set(hist))
    v1_top = None
    if VALIDATION_CSV.exists():
        v = pd.read_csv(VALIDATION_CSV)
        v = v[(v["eligible"] == True) & (v["rank"] <= 20)]  # noqa: E712
        v1_top = {dt.date.fromisoformat(d): set(g["symbol"]) for d, g in v.groupby("date")}
    res = backtest.run(hist, nifty, sectors, horizon=args.horizon, embargo=5, seed=7, v1_top=v1_top)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "period": args.period, "horizon_sessions": args.horizon,
        "rule_version": rules.RULE_VERSION, "thresholds": rules.THRESHOLDS, "feature_set_version": FEATURE_SET_VERSION,
        "universe_requested": len(symbols), "universe_with_data": len(hist), "symbols_without_data": missing,
        "nifty_sessions": len(nifty), "first_session": nifty.index[0].date().isoformat(),
        "last_session": nifty.index[-1].date().isoformat(),
        "costs": {name: res.report(cost_bps=c) for name, c in (("10bps", 10), ("15bps", 15), ("20bps", 20))},
        "v2_setups": {s: sum(1 for t in res.trades if t.strategy == "v2" and t.setup == s)
                      for s in sorted({t.setup for t in res.trades if t.strategy == "v2"})},
        "limitations": [
            "Survivorship: today's universe list is applied to the whole period; delisted or renamed symbols are absent.",
            "Daily Yahoo Finance bars (delayed, adjusted); not exchange data.",
            "No corporate-event, consensus or intraday inputs; catalyst setups are not tested.",
            "v1 baseline uses month-end research snapshots, not live production runs.",
            "Thresholds are a-priori hypotheses; nothing was tuned on these results.",
        ],
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    hold = report["costs"]["20bps"]["splits"]["holdout"]["strategies"]
    print(f"wrote {out} in {report['runtime_seconds']}s; holdout @20bps:")
    for k, v in hold.items():
        print(f"  {k:15s} calls={v['calls']:6d} hit={v['hit_rate']} net={v['mean_net_return']}")


if __name__ == "__main__":
    main()
