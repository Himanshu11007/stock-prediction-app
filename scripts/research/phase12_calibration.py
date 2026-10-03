"""
scripts/research/phase12_calibration.py — Phase 12: confidence calibration.

Experiment E12: can the production ensemble probability (20/30/50 blend,
1-day target, all 27 features) be calibrated, per horizon?

Protocol (pre-registered; see docs/CONFIDENCE_CALIBRATION.md):
  * Base models are always out-of-sample (research harness).
  * DEV evaluation: calibrator fitted on CAL rows whose h-day outcome was
    realised before DEV starts; applied unchanged to DEV.
  * FINAL evaluation: calibrator fitted on CAL+DEV rows whose outcome was
    realised before FINAL starts; applied unchanged to FINAL.
  * Method selection per horizon = lowest DEV Brier. FINAL is reported for
    every method but used only to confirm the DEV choice.
  * Methods: none, platt (logistic on log-odds), isotonic, and the
    constant forecast of the fit-period up-rate (a no-skill reference).

Also records the clean research baseline (E-BASE) and verifies
reproducibility by running it twice without cache.

Usage:
    python scripts/research/phase12_calibration.py [--verify]
"""
from __future__ import annotations

import argparse

import numpy as np

import common
from evaluation import research as rs
from evaluation import research_metrics as rm
from evaluation.walk_forward import HORIZONS

METHODS = ("none", "platt", "isotonic", "constant_base_rate")


def baseline_config() -> rs.ResearchConfig:
    return rs.ResearchConfig(name="base")


def segment_summary(df, p_col="p_ens") -> dict:
    out = {}
    for seg in rs.SEGMENTS:
        s = df[df["segment"] == seg]
        out[seg] = {}
        for h in HORIZONS:
            d = s[s[f"y_{h}d"].notna()]
            y = d[f"y_{h}d"].astype(int)
            m = rm.classification(y, d[p_col])
            m["accuracy_ci95_date_clustered"] = rm.accuracy_ci_clustered(d, f"y_{h}d", p_col)
            m["baseline_majority_accuracy"] = round(float((d["majority"] == y).mean()), 4)
            m["baseline_prev_direction_accuracy"] = round(float((d["prev_dir"] == y).mean()), 4)
            m["n_symbols"] = int(d["symbol"].nunique())
            m["first_date"], m["last_date"] = str(d["date"].min().date()), str(d["date"].max().date())
            out[seg][f"{h}d"] = m
    return out


def calibrate(df, h: int, fit_segments: tuple, eval_segment: str) -> dict:
    y_col = f"y_{h}d"
    start = rs.SEGMENTS[eval_segment][0]
    pool = df[df["segment"].isin(fit_segments) & df[y_col].notna()]
    fit = rm.fit_rows_before(pool, h, start)
    ev = df[(df["segment"] == eval_segment) & df[y_col].notna()].copy()
    assert fit["date"].max() < ev["date"].min()
    y_fit, y_ev = fit[y_col].astype(int).to_numpy(), ev[y_col].astype(int).to_numpy()

    res = {"fit_rows": len(fit), "fit_first": str(fit["date"].min().date()),
           "fit_last": str(fit["date"].max().date()),
           "fit_last_outcome": str(fit[f"outcome_date_{h}d"].max().date()),
           "eval_rows": len(ev), "eval_first": str(ev["date"].min().date()),
           "eval_last": str(ev["date"].max().date()), "methods": {}}
    for method in METHODS:
        if method == "constant_base_rate":
            q = np.full(len(ev), y_fit.mean())
        else:
            q = rm.Calibrator(method).fit(fit["p_ens"], y_fit).predict(ev["p_ens"])
        ev[f"q_{method}"] = q
        m = rm.classification(y_ev, q)
        m["reliability"] = rm.reliability(y_ev, q)
        m["confidence_buckets"] = rm.confidence_buckets(y_ev, q)
        if method != "none":
            m["vs_none"] = rm.paired_bootstrap(ev, y_col, "q_none", f"q_{method}")
        res["methods"][method] = m
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-run without cache and compare")
    args = ap.parse_args()
    files = common.price_files()

    cfg = baseline_config()
    df = rs.run(cfg, files, cache_dir=str(common.CACHE))
    repro = {"fingerprint": rs.fingerprint(df)}
    if args.verify:
        again = rs.run(cfg, files, cache_dir=None, use_cache=False)
        repro["fingerprint_rerun"] = rs.fingerprint(again)
        repro["identical"] = repro["fingerprint"] == repro["fingerprint_rerun"]

    baseline = {
        "config": cfg.__dict__, "config_key": cfg.key(), "reproducibility": repro,
        "n_predictions": len(df), "n_symbols": int(df["symbol"].nunique()),
        "segments": rs.SEGMENTS, "by_segment": segment_summary(df),
        "components": {seg: {k: segment_summary(df, k)[seg] for k in
                             ("p_lr", "p_rf", "p_xgb", "p_ens")} for seg in ("DEV", "FINAL")},
    }
    common.write_json("baseline.json", baseline)

    cal = {}
    for h in HORIZONS:
        dev = calibrate(df, h, ("CAL",), "DEV")
        final = calibrate(df, h, ("CAL", "DEV"), "FINAL")
        chosen = min(("none", "platt", "isotonic"), key=lambda m: dev["methods"][m]["brier"])
        cal[f"{h}d"] = {"dev": dev, "final": final, "selected_on_dev": chosen}
    common.write_json("phase12_calibration.json", {
        "experiment": "E12", "base_config_key": cfg.key(), "methods": METHODS,
        "selection_rule": "lowest DEV Brier among none/platt/isotonic, per horizon",
        "by_horizon": cal,
    })
    for h in HORIZONS:
        c = cal[f"{h}d"]
        line = " ".join(f"{m}:{c['final']['methods'][m]['brier']}" for m in METHODS)
        print(f"{h}d selected={c['selected_on_dev']} FINAL brier {line}")
    print("reproducibility", repro)


if __name__ == "__main__":
    main()
