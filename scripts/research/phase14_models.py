"""
scripts/research/phase14_models.py — Phase 14: model comparison.

E14: production LR / RF / XGBoost, the production 20/30/50 blend, an
equal-weight blend (research only), and three standard tabular candidates
with library defaults and a fixed seed (no tuning):
  hgb  sklearn HistGradientBoostingClassifier
  et   sklearn ExtraTreesClassifier
  gb   sklearn GradientBoostingClassifier
All are fitted in the SAME research run: identical refit dates, windows,
features (27 production features), target (1-day direction) and rows.

Per model, DEV and FINAL, 1/3/5/10D: accuracy (Wilson + date-clustered CI),
balanced accuracy, precision, recall, F1, AUC, Brier, log loss, ECE,
Platt-calibrated Brier (Phase 12 protocol), paired bootstrap vs the
production blend, and return diagnostics.

Pre-registered candidate rule (docs/MODEL_COMPARISON.md): among non-
production models, the lowest DEV Platt-calibrated 1D Brier with DEV AUC
above the production blend's; it is then checked on FINAL. Promotion also
needs the remaining criteria in docs/ML_EXPERIMENT_REGISTRY.md.

Usage:
    python scripts/research/phase14_models.py
"""
from __future__ import annotations

import common
from evaluation import research as rs
from evaluation import research_metrics as rm
from evaluation.walk_forward import HORIZONS

MODELS = ("lr", "rf", "xgb", "hgb", "et", "gb")
CANDIDATES = ("hgb", "et", "gb", "ens_equal")


def platt_brier(df, p_col, h, fit_segments, eval_segment):
    y_col = f"y_{h}d"
    start = rs.SEGMENTS[eval_segment][0]
    pool = df[df["segment"].isin(fit_segments) & df[y_col].notna()]
    fit = rm.fit_rows_before(pool, h, start)
    ev = df[(df["segment"] == eval_segment) & df[y_col].notna()]
    q = rm.Calibrator("platt").fit(fit[p_col], fit[y_col].astype(int)).predict(ev[p_col])
    m = rm.classification(ev[y_col].astype(int), q)
    return {k: m[k] for k in ("brier", "log_loss", "ece", "auc")}


def main() -> None:
    files = common.price_files()
    cfg = rs.ResearchConfig(name="models", models=MODELS)
    df = rs.run(cfg, files, cache_dir=str(common.CACHE))
    cols = {k: f"p_{k}" for k in MODELS} | {"ens": "p_ens", "ens_equal": "p_ens_equal"}

    results = {}
    for name, col in cols.items():
        results[name] = {}
        for seg, fit_segs in (("DEV", ("CAL",)), ("FINAL", ("CAL", "DEV"))):
            s = df[df["segment"] == seg]
            results[name][seg] = {}
            for h in HORIZONS:
                d = s[s[f"y_{h}d"].notna()]
                m = rm.classification(d[f"y_{h}d"].astype(int), d[col])
                m.pop("accuracy_ci95", None)
                m["accuracy_ci95_date_clustered"] = rm.accuracy_ci_clustered(d, f"y_{h}d", col)
                m["platt_calibrated"] = platt_brier(df, col, h, fit_segs, seg)
                if col != "p_ens":
                    m["paired_vs_production_blend"] = rm.paired_bootstrap(d, f"y_{h}d", "p_ens", col)
                if h in (1, 5):
                    m["returns"] = rm.return_diagnostics(d, col, h)
                results[name][seg][f"{h}d"] = m

    prod_dev_auc = results["ens"]["DEV"]["1d"]["auc"]
    eligible = [c for c in CANDIDATES if results[c]["DEV"]["1d"]["auc"] > prod_dev_auc]
    selected = min(eligible, key=lambda c: results[c]["DEV"]["1d"]["platt_calibrated"]["brier"],
                   default=None)
    confirmation = None
    if selected:
        f = results[selected]["FINAL"]
        confirmation = {f"{h}d": {"auc": f[f"{h}d"]["auc"], "prod_auc": results["ens"]["FINAL"][f"{h}d"]["auc"],
                                  "paired": f[f"{h}d"]["paired_vs_production_blend"]} for h in HORIZONS}

    common.write_json("phase14_models.json", {
        "experiment": "E14", "config_key": cfg.key(), "n_predictions": len(df),
        "fingerprint": rs.fingerprint(df), "models": MODELS,
        "candidate_rule": "lowest DEV Platt-calibrated 1D Brier among candidates with DEV 1D AUC > production blend",
        "eligible_on_dev": eligible, "selected_on_dev": selected,
        "final_confirmation": confirmation, "results": results,
    })
    for name in cols:
        r = results[name]
        print(f"{name:10s} DEV auc {r['DEV']['1d']['auc']} acc {r['DEV']['1d']['accuracy']} | "
              f"FINAL auc {[r['FINAL'][f'{h}d']['auc'] for h in HORIZONS]} "
              f"acc {[r['FINAL'][f'{h}d']['accuracy'] for h in HORIZONS]}")
    print("eligible", eligible, "selected", selected)


if __name__ == "__main__":
    main()
