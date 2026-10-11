"""
scripts/export_web_news_fixtures.py — write the REAL news benchmark API
response (GET /admin/news/benchmark, produced from the stored research
report, probe and signal backtest) as a StockLens.Web test fixture, so the
dashboard tests render actual backend results.

  python scripts/export_web_news_fixtures.py --web ../StockLens.Web

The candidates / status fixtures come from tests/test_news_e2e.py
(tests/contracts/), which runs the full ingestion -> API flow.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", required=True, help="path of the StockLens.Web repository")
    args = ap.parse_args(argv)
    from api.routes.news import news_benchmark
    dest = Path(args.web) / "tests" / "StockLens.Web.Tests" / "TestData"
    body = news_benchmark(include_outcomes=False)
    (dest / "news_benchmark.json").write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
    for name in ("news_candidates_tomorrow.json", "news_status.json", "predictions_tomorrow.json",
                 "prediction_detail.json"):
        shutil.copyfile(ROOT / "tests" / "contracts" / name, dest / name)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
