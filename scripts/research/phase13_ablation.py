"""
scripts/research/phase13_ablation.py — Phase 13: feature and component ablation.

E13.G.<group>: leave one production feature group out (evaluation.research.
FEATURE_GROUPS), everything else identical to the baseline (E-BASE).
E13.C.<variant>: component ablation from the baseline run (no retraining):
LR / RF / XGBoost alone, the 20/30/50 blend, an equal-weight blend and
leave-one-model-out blends (weights renormalised). Research variants only.

Pre-registered reading (docs/FEATURE_ABLATION.md):
  * Compare each variant with the baseline on the SAME rows with a paired,
    date-clustered bootstrap of ΔBrier and Δaccuracy, separately for DEV
    and FINAL and for 1/3/5/10D.
  * "Evidence of incremental value" for a group: removing it worsens Brier
    on DEV with a CI excluding 0 AND worsens it on FINAL in the same
    direction, at >= 2 horizons. "Evidence it hurts": the mirror image.
    Anything else: "no reproducible evidence either way".
  * Per-stock consistency: number of symbols whose DEV accuracy improves.

Usage:
    python scripts/research/phase13_ablation.py
"""
from __future__ import annotations

import common
from evaluation import research as rs
from evaluation import research_metrics as rm
from evaluation.walk_forward import HORIZONS
from models.trainer import ENSEMBLE_WEIGHTS
from utils.helpers import FEATURE_COLS

W = {"lr": ENSEMBLE_WEIGHTS["Logistic Regression"], "rf": ENSEMBLE_WEIGHTS["Random Forest"],
     "xgb": ENSEMBLE_WEIGHTS["XGBoost"]}


def compare(base, other, p_base="p_ens", p_other="p_ens") -> dict:
    """Paired comparison on rows present in both runs."""
    keys = ["symbol", "date"]
    m = base[keys + ["segment", "majority", "prev_dir"] + [f"y_{h}d" for h in HORIZONS]
             + [f"ret_{h}d" for h in HORIZONS] + [p_base]].rename(columns={p_base: "p_a"})
    m = m.merge(other[keys + [p_other]].rename(columns={p_other: "p_b"}), on=keys)
    out = {}
    for seg in ("DEV", "FINAL"):
        s = m[m["segment"] == seg]
        out[seg] = {}
        for h in HORIZONS:
            d = s[s[f"y_{h}d"].notna()]
            y = d[f"y_{h}d"].astype(int)
            mb = rm.classification(y, d["p_b"])
            out[seg][f"{h}d"] = {
                "variant": {k: mb[k] for k in ("n", "accuracy", "balanced_accuracy", "precision",
                                                "recall", "f1", "auc", "brier", "log_loss", "ece")},
                "paired_vs_baseline": rm.paired_bootstrap(d, f"y_{h}d", "p_a", "p_b"),
                "returns": rm.return_diagnostics(d.rename(columns={"p_b": "p"}), "p", h),
            }
        dev = s[s["y_1d"].notna()]
        per_stock = dev.groupby("symbol").apply(
            lambda g: ((g["p_b"] > 0.5) == g["y_1d"]).mean() - ((g["p_a"] > 0.5) == g["y_1d"]).mean(),
            include_groups=False)
        out[seg]["symbols_improved_1d"] = int((per_stock > 0).sum())
        out[seg]["symbols_worse_1d"] = int((per_stock < 0).sum())
    return out


def verdict(cmp: dict) -> str:
    """Pre-registered rule on ΔBrier (variant − baseline); for an ablation,
    a positive Δ means removing the group made forecasts worse."""
    def sig(seg, h, sign):
        lo, hi = cmp[seg][f"{h}d"]["paired_vs_baseline"]["brier_diff_ci95"]
        return lo > 0 if sign > 0 else hi < 0

    def direction(seg, h):
        return cmp[seg][f"{h}d"]["paired_vs_baseline"]["brier_diff"]

    helps = sum(1 for h in HORIZONS if sig("DEV", h, +1) and direction("FINAL", h) > 0)
    hurts = sum(1 for h in HORIZONS if sig("DEV", h, -1) and direction("FINAL", h) < 0)
    if helps >= 2:
        return "evidence of incremental value (removing it worsens Brier on DEV, confirmed on FINAL)"
    if hurts >= 2:
        return "evidence it hurts (removing it improves Brier on DEV, confirmed on FINAL)"
    return "no reproducible evidence either way"


def component_verdict(cmp: dict) -> str:
    """Same rule as verdict(), worded for a variant that REPLACES the blend:
    a negative ΔBrier means the variant forecasts better than the blend."""
    v = verdict(cmp)
    if v.startswith("evidence of incremental value"):
        return "higher Brier than the production blend (DEV, confirmed on FINAL)"
    if v.startswith("evidence it hurts"):
        return "lower Brier than the production blend (DEV, confirmed on FINAL)"
    return v


def main() -> None:
    files = common.price_files()
    base = rs.run(rs.ResearchConfig(name="base"), files, cache_dir=str(common.CACHE))

    groups = {}
    for group, cols in rs.FEATURE_GROUPS.items():
        feats = tuple(c for c in FEATURE_COLS if c not in cols)
        cfg = rs.ResearchConfig(name=f"drop_{group}", features=feats)
        df = rs.run(cfg, files, cache_dir=str(common.CACHE))
        c = compare(base, df)
        groups[group] = {"removed": cols, "config_key": cfg.key(), "n_predictions": len(df),
                         "comparison": c, "verdict": verdict(c)}
        print(f"{group:20s} {groups[group]['verdict']}")

    variants = {
        "lr_only": base["p_lr"], "rf_only": base["p_rf"], "xgb_only": base["p_xgb"],
        "equal_weight": base["p_ens_equal"],
        "no_lr": (W["rf"] * base["p_rf"] + W["xgb"] * base["p_xgb"]) / (W["rf"] + W["xgb"]),
        "no_rf": (W["lr"] * base["p_lr"] + W["xgb"] * base["p_xgb"]) / (W["lr"] + W["xgb"]),
        "no_xgb": (W["lr"] * base["p_lr"] + W["rf"] * base["p_rf"]) / (W["lr"] + W["rf"]),
    }
    components = {}
    for name, p in variants.items():
        other = base[["symbol", "date"]].assign(p_ens=p)
        c = compare(base, other)
        components[name] = {"comparison": c, "verdict": component_verdict(c)}
        print(f"{name:20s} {components[name]['verdict']}")

    common.write_json("phase13_ablation.json", {
        "experiment": "E13", "baseline": "E-BASE (all 27 features, LR/RF/XGB 20/30/50, 1-day target)",
        "feature_groups": groups, "components": components,
    })


if __name__ == "__main__":
    main()
