"""
evaluation/research.py — walk-forward research harness (Phases 12–15).

Purpose: measure how much out-of-sample signal the production feature set,
models and targets carry, with enough predictions to say something
statistically. It is NOT the production replay (that is
evaluation/walk_forward.py, which runs the full decision engine on every
5th session). This harness covers ML only:

  For every symbol, every refit date R (every `refit_every` sessions):
    window   = raw daily bars in (R − 1 year, R]          (production window)
    features = features.engineer.compute_features(window) (production code)
    label_h  = Close[t+h] > Close[t]  over raw bars          (h=1 == production `Up`)
    train on rows with complete features and a label known at R (t+h <= R)
  For every session P in [R, next refit):
    features at P from (P − 1 year, P]; predict with the models fitted at R.
  Outcomes: Close of the 1st/3rd/5th/10th trading session after P
  (holiday placeholder bars excluded), entry = Close[P].

Every prediction is out-of-sample: the model only saw labels realised at
or before R <= P. assert_row_integrity() fails the run otherwise.

Chronological segments (all phases):
  CAL    2024-07-01 .. 2025-03-31   calibrator fitting
  DEV    2025-04-01 .. 2025-09-30   comparisons and every selection decision
  FINAL  2025-10-01 .. 2026-09-30   confirmation of pre-registered candidates
                                    (same period as the Phase 10/11 benchmark)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from evaluation.walk_forward import DAILY_LOOKBACK, HORIZONS, session_closes
from features.engineer import compute_features
from models.trainer import ENSEMBLE_WEIGHTS, _make_candidates
from utils.helpers import FEATURE_COLS

SEGMENTS = {
    "CAL":   ("2024-07-01", "2025-03-31"),
    "DEV":   ("2025-04-01", "2025-09-30"),
    "FINAL": ("2025-10-01", "2026-09-30"),
}

# Logical groups of the 27 production features (utils/helpers.FEATURE_COLS).
FEATURE_GROUPS = {
    "price_level":       ["Close"],
    "returns_momentum":  ["Price_Change", "Momentum"],
    "moving_averages":   ["MA_5", "MA_10", "MA_Diff", "EMA_20", "EMA_50", "EMA_Cross", "Price_vs_EMA20"],
    "rsi":               ["RSI"],
    "volatility_atr":    ["Volatility", "ATR", "ATR_Pct"],
    "bollinger":         ["BB_Width", "BB_Position"],
    "volume":            ["Volume", "Volume_Change", "Volume_MA", "Volume_Ratio", "Vol_Breakout"],
    "macd":              ["MACD", "MACD_Hist", "MACD_Cross"],
    "trend_strength_adx": ["ADX", "Plus_DI", "Minus_DI"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURE_COLS)

PRODUCTION_MODELS = ("lr", "rf", "xgb")
_PRODUCTION_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


def _production_model(key: str):
    """The exact production pipeline, single-threaded (process-level
    parallelism is used instead; thread count does not change results)."""
    for name, model in _make_candidates():
        if name == _PRODUCTION_NAMES[key]:
            for step in model.named_steps.values():
                if "n_jobs" in step.get_params():
                    step.set_params(n_jobs=1)
            return model
    raise KeyError(key)


def _candidate_model(key: str):
    # Phase 14 candidates: library defaults, fixed seed, no tuning.
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  HistGradientBoostingClassifier)
    if key == "hgb":
        return HistGradientBoostingClassifier(random_state=42)
    if key == "et":
        return ExtraTreesClassifier(random_state=42, n_jobs=1)
    if key == "gb":
        return GradientBoostingClassifier(random_state=42)
    raise KeyError(key)


def make_model(key: str):
    return _production_model(key) if key in PRODUCTION_MODELS else _candidate_model(key)


@dataclass(frozen=True)
class ResearchConfig:
    name: str
    features: tuple = tuple(FEATURE_COLS)
    target_horizon: int = 1
    models: tuple = PRODUCTION_MODELS
    start: str = SEGMENTS["CAL"][0]
    end: str = SEGMENTS["FINAL"][1]
    refit_every: int = 5

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class ResearchIntegrityError(AssertionError):
    pass


def forward_label(close: pd.Series, h: int) -> pd.Series:
    """Close[t+h] > Close[t] over consecutive raw bars; NaN where unknown.
    For h=1 this is exactly compute_features()'s `Up`."""
    nxt = close.shift(-h)
    return pd.Series(np.where(nxt.isna(), np.nan, (nxt > close).astype(float)), index=close.index)


def _window(raw: pd.DataFrame, end) -> pd.DataFrame:
    return raw.loc[(raw.index > end - DAILY_LOOKBACK) & (raw.index <= end)]


def segment_of(date) -> str | None:
    for name, (a, b) in SEGMENTS.items():
        if pd.Timestamp(a) <= date <= pd.Timestamp(b):
            return name
    return None


def run_symbol(symbol: str, raw: pd.DataFrame, cfg: ResearchConfig) -> list[dict]:
    cols = list(cfg.features)
    sessions = session_closes(raw)
    s_idx = sessions.index
    dates = s_idx[(s_idx >= pd.Timestamp(cfg.start)) & (s_idx <= pd.Timestamp(cfg.end))]
    rows: list[dict] = []

    for block_start in range(0, len(dates), cfg.refit_every):
        refit = dates[block_start]
        win = _window(raw, refit)
        feats = compute_features(win)
        y_all = forward_label(win["Close"], cfg.target_horizon)
        usable = feats.drop(columns="Up").notna().all(axis=1) & y_all.notna()
        X, y = feats.loc[usable, cols], y_all[usable].astype(int)
        if len(X) < 60 or y.nunique() < 2:
            continue
        label_end = win.index[win.index.get_loc(X.index[-1]) + cfg.target_horizon]

        models = {k: make_model(k).fit(X, y) for k in cfg.models}
        majority = int(y.mean() > 0.5)

        for p_date in dates[block_start:block_start + cfg.refit_every]:
            fp = compute_features(_window(raw, p_date))
            if p_date not in fp.index or not fp.drop(columns="Up").loc[p_date].notna().all():
                continue
            row_X = fp.loc[[p_date], cols]
            pos = s_idx.get_loc(p_date)
            if pos == 0:
                continue
            rec = {
                "symbol": symbol, "date": p_date, "segment": segment_of(p_date),
                "refit_date": refit, "train_end": X.index[-1], "label_end": label_end,
                "n_train": len(X), "train_up_rate": float(y.mean()),
                "majority": majority,
                "prev_dir": int(sessions.iloc[pos] > sessions.iloc[pos - 1]),
                "entry": float(sessions.iloc[pos]),
            }
            for k, m in models.items():
                rec[f"p_{k}"] = float(m.predict_proba(row_X)[0, 1])
            for h in HORIZONS:
                if pos + h < len(sessions):
                    exit_ = float(sessions.iloc[pos + h])
                    rec[f"outcome_date_{h}d"] = s_idx[pos + h]
                    rec[f"ret_{h}d"] = (exit_ - rec["entry"]) / rec["entry"] * 100
                    rec[f"y_{h}d"] = int(exit_ > rec["entry"])
                else:
                    rec[f"outcome_date_{h}d"] = pd.NaT
                    rec[f"ret_{h}d"] = np.nan
                    rec[f"y_{h}d"] = np.nan
            assert_row_integrity(rec)
            rows.append(rec)
    return rows


def assert_row_integrity(rec: dict) -> None:
    d = rec["date"]
    problems = []
    if not rec["train_end"] < d:
        problems.append("train_end >= prediction date")
    if not rec["label_end"] <= rec["refit_date"] <= d:
        problems.append("training label realised after the refit/prediction date")
    for h in HORIZONS:
        od = rec[f"outcome_date_{h}d"]
        if not pd.isna(od) and not od > d:
            problems.append(f"outcome_date_{h}d not after prediction date")
    if problems:
        raise ResearchIntegrityError(f"{rec['symbol']} {d:%Y-%m-%d}: " + "; ".join(problems))


def add_ensembles(df: pd.DataFrame) -> pd.DataFrame:
    """Production blend (20/30/50) and, as a research variant only, equal weights."""
    if all(f"p_{k}" in df for k in PRODUCTION_MODELS):
        w = ENSEMBLE_WEIGHTS
        df["p_ens"] = (df["p_lr"] * w["Logistic Regression"] + df["p_rf"] * w["Random Forest"]
                       + df["p_xgb"] * w["XGBoost"])
        df["p_ens_equal"] = (df["p_lr"] + df["p_rf"] + df["p_xgb"]) / 3
    return df


def _run_symbol_from_csv(symbol: str, path: str, cfg: ResearchConfig) -> list[dict]:
    raw = pd.read_csv(path, index_col="Date", parse_dates=True, float_precision="round_trip")
    return run_symbol(symbol, raw, cfg)


def run(cfg: ResearchConfig, price_files: dict[str, str], cache_dir: str | None = None,
        n_jobs: int = 18, use_cache: bool = True) -> pd.DataFrame:
    """Run (or load) one experiment. Output is sorted and deterministic."""
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{cfg.name}_{cfg.key()}.csv.gz")
        if use_cache and os.path.exists(cache):
            df = pd.read_csv(cache, parse_dates=["date", "refit_date", "train_end", "label_end"]
                             + [f"outcome_date_{h}d" for h in HORIZONS])
            return add_ensembles(df)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_symbol_from_csv)(s, p, cfg) for s, p in sorted(price_files.items()))
    df = pd.DataFrame([r for rows in results for r in rows])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    if cache:
        df.to_csv(cache, index=False)
    return add_ensembles(df)


def fingerprint(df: pd.DataFrame) -> str:
    """Hash of the prediction table, for reproducibility checks."""
    cols = sorted(c for c in df.columns if c.startswith("p_") or c in ("symbol", "date"))
    blob = df[cols].round(12).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()
