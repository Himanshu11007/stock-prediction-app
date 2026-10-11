"""
scripts/research/news_provider_probe.py — live validation of Marketaux and
Alpha Vantage (connectivity, authentication, rate limits, schema,
timestamps, history depth, Indian coverage). RESEARCH ONLY.

  python scripts/research/news_provider_probe.py [--providers marketaux,alphavantage]

Keys come from MARKETAUX_API_TOKEN / ALPHAVANTAGE_API_KEY in the environment;
they are never printed or written. Without a key the provider is reported as
NOT_CONFIGURED (not tested), never as failed. Budget: Marketaux 6 requests,
Alpha Vantage 5 (free quotas: about 100 and 25 a day).

Output: scripts/research/output/prediction_v2/news_provider_probe.json
(merged into the benchmark dashboard).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catalysts import probe  # noqa: E402

OUT = ROOT / "scripts" / "research" / "output" / "prediction_v2" / "news_provider_probe.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="marketaux,alphavantage")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    results = {p: probe.probe(p) for p in [x.strip() for x in args.providers.split(",") if x.strip()]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"research_only": True, "providers": results}, indent=1, default=str),
                              encoding="utf-8")
    for p, r in results.items():
        print(f"{p}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
