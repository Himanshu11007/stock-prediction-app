"""
scripts/research/phase15_targets.py — Phase 15a: target / horizon research.

E15.T<h>: the production pipeline (27 features, LR/RF/XGB, 20/30/50 blend)
trained on a different binary target, Close[t+h] > Close[t], h in 1/3/5/10.
Each target is its own experiment; labels are never mixed. Training rows
for target h at refit date R are those with t+h <= R (label known at R).

Measured per target, on DEV and FINAL, at its own horizon: class balance
(training and realised), majority and previous-direction baselines,
accuracy, balanced accuracy, precision, recall, F1, AUC, Brier, log loss,
Platt-calibrated Brier (Phase 12 protocol), return diagnostics, and a
paired comparison with the 1-day-target model evaluated at the same horizon.

Note: training labels count raw bars (as production's `Up` does, including
Yahoo holiday placeholder bars); outcomes count real sessions. Over the
period this differs on ~5 holiday bars per year.

Usage:
    python scripts/research/phase15_targets.py
"""
from __future__ import annotations

import common
from evaluation import research as rs
from evaluation import research_metrics as rm
from evaluation.walk_forward import HORIZONS
from phase14_models import platt_brier


def main() -> None:
    files = common.price_files()
    base = rs.run(rs.ResearchConfig(name="base"), files, cache_dir=str(common.CACHE))
    out = {}
    for h in HORIZONS:
        cfg = rs.ResearchConfig(name=f"target_{h}d", target_horizon=h)
        df = base if h == 1 else rs.run(cfg, files, cache_dir=str(common.CACHE))
        y_col = f"y_{h}d"
        res = {"config_key": cfg.key(), "n_predictions": len(df)}
        merged = df[["symbol", "date", "segment", "majority", "prev_dir", y_col, f"ret_{h}d",
                     f"outcome_date_{h}d", "p_ens", "train_up_rate"]].merge(
            base[["symbol", "date", "p_ens"]].rename(columns={"p_ens": "p_base"}),
            on=["symbol", "date"])
        for seg, fit_segs in (("DEV", ("CAL",)), ("FINAL", ("CAL", "DEV"))):
            d = merged[(merged["segment"] == seg) & merged[y_col].notna()]
            y = d[y_col].astype(int)
            m = rm.classification(y, d["p_ens"])
            m["accuracy_ci95_date_clustered"] = rm.accuracy_ci_clustered(d, y_col, "p_ens")
            m["train_up_rate_mean"] = round(float(d["train_up_rate"].mean()), 4)
            m["baseline_majority_accuracy"] = round(float((d["majority"] == y).mean()), 4)
            m["baseline_prev_direction_accuracy"] = round(float((d["prev_dir"] == y).mean()), 4)
            m["platt_calibrated"] = platt_brier(merged, "p_ens", h, fit_segs, seg)
            m["returns"] = rm.return_diagnostics(d.rename(columns={"p_ens": "p"}), "p", h)
            m["spread_ci"] = rm.spread_ci(d, "p_ens", h)
            m["spread_ci_1d_target_model"] = rm.spread_ci(d, "p_base", h)
            if h != 1:
                m["paired_vs_1d_target_model"] = rm.paired_bootstrap(d, y_col, "p_base", "p_ens")
            res[seg] = m
        out[f"{h}d"] = res
        print(f"target {h}d: DEV acc {res['DEV']['accuracy']} auc {res['DEV']['auc']} | "
              f"FINAL acc {res['FINAL']['accuracy']} auc {res['FINAL']['auc']} "
              f"up% {res['FINAL']['actual_up_pct']} maj {res['FINAL']['baseline_majority_accuracy']}")
    common.write_json("phase15_targets.json", {"experiment": "E15.T", "by_target": out})


if __name__ == "__main__":
    main()
