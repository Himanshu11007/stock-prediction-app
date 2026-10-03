"""
scripts/research/common.py — shared setup for the Phase 12–15 research scripts.

Uses the Phase 10 price snapshot (scripts/audit/.price_snapshot, sha256 in
scripts/audit/output/walk_forward_benchmark/full/price_manifest.json) and the
same 27-symbol universe as the clean benchmark. Prediction tables are cached
(gitignored) under scripts/research/.cache; summaries are committed under
scripts/research/output.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Single-threaded numerics per process; parallelism is across symbols.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

SNAPSHOT = ROOT / "scripts" / "audit" / ".price_snapshot"
CACHE = ROOT / "scripts" / "research" / ".cache"
OUTPUT = ROOT / "scripts" / "research" / "output"

# The 27 symbols the clean benchmark evaluates (30 configured, 3 without data).
UNIVERSE = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BPCL.NS",
    "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AIAENG.NS", "AJANTPHARM.NS",
    "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUBANK.NS",
    "AARTIIND.NS", "AAVAS.NS", "AETHER.NS", "ALEMBICLTD.NS", "AMBER.NS",
    "ANGELONE.NS", "APARINDS.NS",
]


def price_files() -> dict[str, str]:
    files = {s: str(SNAPSHOT / f"{s.replace('.', '_')}.csv") for s in UNIVERSE}
    missing = [s for s, p in files.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing price snapshot for {missing}; run "
                         "scripts/audit/walk_forward_benchmark.py --config full first.")
    return files


def write_json(name: str, obj) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
