"""
scripts/audit/compare_benchmarks.py — Phase 11A pre-fix vs post-fix comparison.

Reads two walk-forward benchmark summaries (default: the preserved Phase 10
pre-fix run and the current `full` run) and writes a side-by-side JSON and
Markdown comparison. Pure reporting: no recomputation, no new metrics.

Usage:
    python scripts/audit/compare_benchmarks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "output" / "walk_forward_benchmark"
BEFORE = ROOT / "phase10_prefix_full" / "summary.json"
AFTER = ROOT / "phase11a_full" / "summary.json"
OUT = ROOT / "phase11a_full"


def _by_h(rows: list[dict]) -> dict[int, dict]:
    return {r["horizon_days"]: r for r in rows}


def _fmt(r: dict | None) -> str | None:
    if not r or r.get("rate_pct") is None:
        return None
    return f"{r['rate_pct']}% ({r['ci95_low_pct']}–{r['ci95_high_pct']}, n={r['n']})"


def compare(before: dict, after: dict) -> dict:
    b_raw, a_raw = before["raw_model_benchmark"], after["raw_model_benchmark"]
    horizons = after["horizons_trading_days"]
    direction = []
    for h in horizons:
        direction.append({
            "horizon_days": h,
            "phase10_model": _fmt(_by_h(b_raw["as_deployed_direction"]).get(h)),
            "phase11a_model": _fmt(_by_h(a_raw["as_deployed_direction"]).get(h)),
            "phase10_majority_baseline": _fmt(_by_h(b_raw["baseline_majority_class"]).get(h)),
            "phase11a_majority_baseline": _fmt(_by_h(a_raw["baseline_majority_class"]).get(h)),
            "phase10_prev_direction_baseline": _fmt(_by_h(b_raw["baseline_previous_direction"]).get(h)),
            "phase11a_prev_direction_baseline": _fmt(_by_h(a_raw["baseline_previous_direction"]).get(h)),
        })

    def _signals(s, key):
        return {(r["horizon_days"], r["signal"]): r
                for r in s["production_recommendation_benchmark"][key]}

    b_sig, a_sig = _signals(before, "signals"), _signals(after, "signals")
    signals = []
    for key in sorted(set(b_sig) | set(a_sig)):
        signals.append({
            "horizon_days": key[0], "signal": key[1],
            "phase10": _fmt(b_sig.get(key)), "phase11a": _fmt(a_sig.get(key)),
        })

    ti_b, ti_a = before["temporal_integrity"], after["temporal_integrity"]
    return {
        "before": BEFORE.relative_to(ROOT.parents[3]).as_posix(),
        "after": AFTER.relative_to(ROOT.parents[3]).as_posix(),
        "population": {
            "phase10": {k: before["population"][k] for k in
                        ("n_predictions", "n_skipped", "skip_reasons", "n_production_recommendations")},
            "phase11a": {k: after["population"][k] for k in
                         ("n_predictions", "n_skipped", "skip_reasons", "n_production_recommendations")},
        },
        "prediction_row_in_training_set_pct": {
            "phase10": ti_b["prediction_row_in_training_set_pct"],
            "phase11a": ti_a["prediction_row_in_training_set_pct"],
        },
        "prediction_equals_realized_last_move": {
            "phase10": _fmt(ti_b["as_deployed_prediction_equals_realized_issue_move"]),
            "phase11a": _fmt(ti_a["as_deployed_prediction_equals_realized_issue_move"]),
        },
        "ensemble_walk_forward_cv_mean": {
            "phase10": before["ensemble_walk_forward_cv"]["mean_ensemble_accuracy"],
            "phase11a": after["ensemble_walk_forward_cv"]["mean_ensemble_accuracy"],
        },
        "direction_accuracy": direction,
        "filtered_signal_success": signals,
    }


def render(c: dict) -> str:
    lines = [
        "# Phase 10 (pre-fix) vs Phase 11A (post-fix) — walk-forward benchmark\n",
        f"Before: `{c['before']}` · After: `{c['after']}`\n",
        f"- Prediction row inside its own training set: {c['prediction_row_in_training_set_pct']['phase10']}% → "
        f"{c['prediction_row_in_training_set_pct']['phase11a']}%",
        f"- Prediction == already-realised D-1→D move: {c['prediction_equals_realized_last_move']['phase10']} → "
        f"{c['prediction_equals_realized_last_move']['phase11a']}",
        f"- Population: {c['population']['phase10']} → {c['population']['phase11a']}\n",
        "## Raw model direction accuracy\n",
        "| Horizon | Phase 10 model | Phase 11A model | Majority baseline (10 / 11A) | Previous-direction baseline (10 / 11A) |",
        "|---|---|---|---|---|",
    ]
    for r in c["direction_accuracy"]:
        lines.append(
            f"| {r['horizon_days']}D | {r['phase10_model']} | {r['phase11a_model']} | "
            f"{r['phase10_majority_baseline']} / {r['phase11a_majority_baseline']} | "
            f"{r['phase10_prev_direction_baseline']} / {r['phase11a_prev_direction_baseline']} |"
        )
    lines += ["", "## Filtered recommendations: signal success\n",
              "| Horizon | Signal | Phase 10 | Phase 11A |", "|---|---|---|---|"]
    for r in c["filtered_signal_success"]:
        lines.append(f"| {r['horizon_days']}D | {r['signal']} | {r['phase10']} | {r['phase11a']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    after = json.loads(AFTER.read_text(encoding="utf-8"))
    c = compare(before, after)
    (OUT / "phase10_vs_phase11a.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (OUT / "phase10_vs_phase11a.md").write_text(render(c), encoding="utf-8")
    print(f"Wrote {OUT / 'phase10_vs_phase11a.md'}")


if __name__ == "__main__":
    main()
