"""
evaluation/research_metrics.py — metrics, calibration and uncertainty for
the research harness (evaluation/research.py).

All functions are pure; nothing here touches production code paths.
Return-based diagnostics are historical research measurements on
close-to-close prices with no costs, slippage or position sizing. They are
not trading returns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from evaluation.walk_forward import wilson_interval

EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# Classification / probability metrics
# ══════════════════════════════════════════════════════════════════════════════

def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on P(up)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def classification(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    n = len(y)
    if n == 0:
        return {"n": 0}
    pred = (p > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc_k = tp + tn
    lo, hi = wilson_interval(acc_k, n)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else 0.0 if precision is not None and recall is not None else None)
    bal = (recall + tnr) / 2 if recall is not None and tnr is not None else None
    try:
        auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    except ValueError:
        auc = None
    return {
        "n": n,
        "accuracy": round(acc_k / n, 4),
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": round(bal, 4) if bal is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "auc": round(auc, 4) if auc is not None else None,
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
        "ece": round(ece(y, p), 4),
        "pred_up_pct": round(pred.mean() * 100, 1),
        "actual_up_pct": round(y.mean() * 100, 1),
        "mean_p": round(float(p.mean()), 4),
    }


def baseline_brier(y) -> float:
    """Brier of always predicting the sample base rate (an in-sample
    reference, not a usable forecast)."""
    y = np.asarray(y, dtype=float)
    return round(float(np.mean((y.mean() - y) ** 2)), 5)


def reliability(y, p, bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson_interval(k, n)
        out.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": n,
            "mean_predicted": round(float(p[m].mean()), 4) if n else None,
            "observed_up_rate": round(k / n, 4) if n else None,
            "observed_ci95": [lo, hi],
        })
    return out


def confidence_buckets(y, p) -> list[dict]:
    """Production-style confidence = max(p, 1-p)*100; success = direction right."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p) * 100
    correct = ((p > 0.5).astype(int) == y).astype(int)
    out = []
    for lo in range(50, 100, 5):
        m = (conf >= lo) & ((conf < lo + 5) | ((lo + 5 == 100) & (conf <= 100)))
        k, n = int(correct[m].sum()), int(m.sum())
        a, b = wilson_interval(k, n)
        out.append({"bucket": f"{lo}-{lo + 5}", "n": n,
                    "success_rate": round(k / n, 4) if n else None, "ci95": [a, b]})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Uncertainty: paired, date-clustered bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def paired_bootstrap(df: pd.DataFrame, y_col: str, p_a: str, p_b: str,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """
    Difference B − A in accuracy and Brier on the SAME rows, resampling
    whole dates (predictions on one date are correlated across stocks).
    Brier: negative difference means B is better.
    """
    d = df[[y_col, p_a, p_b, "date"]].dropna()
    y = d[y_col].to_numpy(int)
    pa, pb = d[p_a].to_numpy(float), d[p_b].to_numpy(float)
    acc_diff = ((pb > 0.5) == y).astype(float) - ((pa > 0.5) == y).astype(float)
    brier_diff = (pb - y) ** 2 - (pa - y) ** 2
    codes, uniq = pd.factorize(d["date"])
    k = len(uniq)
    sums = np.zeros((k, 3))
    np.add.at(sums, codes, np.c_[acc_diff, brier_diff, np.ones_like(acc_diff)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    tot = sums[draws].sum(axis=1)
    acc_b = tot[:, 0] / tot[:, 2]
    brier_b = tot[:, 1] / tot[:, 2]
    return {
        "n": len(d), "n_dates": int(k),
        "accuracy_diff": round(float(acc_diff.mean()), 4),
        "accuracy_diff_ci95": [round(float(np.percentile(acc_b, 2.5)), 4),
                               round(float(np.percentile(acc_b, 97.5)), 4)],
        "brier_diff": round(float(brier_diff.mean()), 5),
        "brier_diff_ci95": [round(float(np.percentile(brier_b, 2.5)), 5),
                            round(float(np.percentile(brier_b, 97.5)), 5)],
    }


def accuracy_ci_clustered(df: pd.DataFrame, y_col: str, p_col: str,
                          n_boot: int = 1000, seed: int = 0) -> list[float]:
    """Date-clustered bootstrap CI for accuracy (wider, more honest than Wilson)."""
    d = df[[y_col, p_col, "date"]].dropna()
    correct = ((d[p_col].to_numpy() > 0.5) == d[y_col].to_numpy(int)).astype(float)
    codes, uniq = pd.factorize(d["date"])
    sums = np.zeros((len(uniq), 2))
    np.add.at(sums, codes, np.c_[correct, np.ones_like(correct)])
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    tot = sums[draws].sum(axis=1)
    acc = tot[:, 0] / tot[:, 1]
    return [round(float(np.percentile(acc, 2.5)), 4), round(float(np.percentile(acc, 97.5)), 4)]


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def fit_rows_before(df: pd.DataFrame, h: int, segment_start) -> pd.DataFrame:
    """Rows whose h-day outcome was realised strictly before `segment_start` —
    the only rows a calibrator applied from that date may learn from."""
    return df[df[f"outcome_date_{h}d"] < pd.Timestamp(segment_start)]


class Calibrator:
    def __init__(self, method: str):
        if method not in ("none", "platt", "isotonic"):
            raise ValueError(method)
        self.method = method
        self.model = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        y = np.asarray(y, dtype=int)
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                np.asarray(p, dtype=float), y)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return p
        if self.method == "platt":
            return self.model.predict_proba(self._logit(p))[:, 1]
        return self.model.predict(p)


# ══════════════════════════════════════════════════════════════════════════════
# Return diagnostics (research only)
# ══════════════════════════════════════════════════════════════════════════════

def max_drawdown(returns_pct: np.ndarray) -> float:
    equity = np.cumprod(1 + np.asarray(returns_pct) / 100)
    peak = np.maximum.accumulate(equity)
    return round(float(((equity - peak) / peak).min() * 100), 2) if len(equity) else 0.0


def return_diagnostics(df: pd.DataFrame, p_col: str, h: int, threshold: float = 0.5) -> dict:
    """
    Forward h-day returns split by the model's call, on every row with an
    outcome. For h == 1 also a hypothetical equal-weight long-only daily
    series (each date: mean 1D return of names called up), its max drawdown
    and turnover. No costs; overlapping h>1 returns are not compounded.
    """
    d = df[df[f"ret_{h}d"].notna()]
    up = d[p_col] > threshold
    r_up, r_dn = d.loc[up, f"ret_{h}d"], d.loc[~up, f"ret_{h}d"]
    out = {
        "horizon_days": h,
        "n_up_calls": int(up.sum()), "n_down_calls": int((~up).sum()),
        "mean_ret_up_calls": round(float(r_up.mean()), 4) if len(r_up) else None,
        "median_ret_up_calls": round(float(r_up.median()), 4) if len(r_up) else None,
        "mean_ret_down_calls": round(float(r_dn.mean()), 4) if len(r_dn) else None,
        "median_ret_down_calls": round(float(r_dn.median()), 4) if len(r_dn) else None,
        "spread_up_minus_down": (round(float(r_up.mean() - r_dn.mean()), 4)
                                 if len(r_up) and len(r_dn) else None),
        "hit_rate_up_calls": round(float((r_up > 0).mean()), 4) if len(r_up) else None,
        "downside_rate_up_calls": round(float((r_up < 0).mean()), 4) if len(r_up) else None,
        "mean_ret_all": round(float(d[f"ret_{h}d"].mean()), 4),
    }
    if h == 1 and len(d):
        long_daily = d[up].groupby("date")[f"ret_{h}d"].mean()
        all_dates = pd.Index(sorted(d["date"].unique()))
        daily = long_daily.reindex(all_dates).fillna(0.0)
        bench = d.groupby("date")[f"ret_{h}d"].mean().reindex(all_dates)
        pos = d.assign(up=up.astype(int)).pivot_table(index="date", columns="symbol", values="up")
        changes = pos.diff().abs().sum(axis=1).iloc[1:]
        held = pos.notna().sum(axis=1).iloc[1:]
        out.update({
            "long_only_days": int(len(daily)),
            "long_only_mean_daily_pct": round(float(daily.mean()), 4),
            "long_only_total_pct": round(float((np.prod(1 + daily / 100) - 1) * 100), 2),
            "long_only_max_drawdown_pct": max_drawdown(daily.to_numpy()),
            "equal_weight_all_total_pct": round(float((np.prod(1 + bench / 100) - 1) * 100), 2),
            "equal_weight_all_max_drawdown_pct": max_drawdown(bench.to_numpy()),
            "turnover_per_day": round(float((changes / held.replace(0, np.nan)).mean()), 4),
        })
    return out


def spread_ci(df: pd.DataFrame, p_col: str, h: int, threshold: float = 0.5,
              n_boot: int = 1000, seed: int = 0) -> dict:
    """
    Mean h-day return of up-calls minus down-calls, with a date-clustered
    bootstrap CI. Dates are resampled as whole blocks; overlapping h-day
    returns of consecutive dates remain correlated, so for h > 1 the CI is
    still somewhat optimistic.
    """
    d = df[["date", p_col, f"ret_{h}d"]].dropna()
    up = (d[p_col] > threshold).to_numpy()
    r = d[f"ret_{h}d"].to_numpy(float)
    codes, uniq = pd.factorize(d["date"])
    sums = np.zeros((len(uniq), 4))
    np.add.at(sums, codes, np.c_[r * up, up, r * ~up, ~up])
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    tot = sums[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = tot[:, 0] / tot[:, 1] - tot[:, 2] / tot[:, 3]
    point = r[up].mean() - r[~up].mean() if up.any() and (~up).any() else float("nan")
    return {"spread_pp": round(float(point), 4),
            "ci95": [round(float(np.nanpercentile(boot, 2.5)), 4),
                     round(float(np.nanpercentile(boot, 97.5)), 4)]}


def wilson_pct(k: int, n: int):
    return wilson_interval(k, n)


def safe_round(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)
