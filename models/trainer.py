from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# Minimum rows required for a training fold and a test fold respectively.
_MIN_TRAIN_ROWS = 60
_MIN_TEST_ROWS  = 10

# Production ensemble blend weights — the single authoritative definition,
# used by ensemble_predict() and walk_forward_validate_ensemble(). Sums to
# 1.0; no normalisation is applied. When XGBoost is unavailable its weight
# is applied to the Random Forest probability (RF effectively 0.80).
ENSEMBLE_WEIGHTS = {
    "Logistic Regression": 0.20,
    "Random Forest":       0.30,
    "XGBoost":             0.50,
}


def _make_candidates():
    """Return a fresh list of (name, Pipeline) pairs on every call."""
    candidates = [
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, random_state=42)),
        ])),
        ("Random Forest", Pipeline([
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ])),
    ]
    if _XGB_AVAILABLE:
        candidates.append(("XGBoost", Pipeline([
            ("xgb", XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )),
        ])))
    return candidates


def _walk_forward_splits(n_samples, n_splits=5):
    """
    Compute (train_end, test_end) index pairs for expanding-window walk-forward.

    The first fold trains on at least 60 % of the data (or _MIN_TRAIN_ROWS,
    whichever is larger); each subsequent fold adds one equal-sized step.
    Ensures every test fold has at least _MIN_TEST_ROWS samples.
    """
    min_train = max(_MIN_TRAIN_ROWS, int(n_samples * 0.6))
    remaining = n_samples - min_train

    if remaining < _MIN_TEST_ROWS:
        return []

    actual_splits = min(n_splits, remaining // _MIN_TEST_ROWS)
    step = remaining // actual_splits

    splits = []
    for i in range(actual_splits):
        train_end = min_train + i * step
        test_end  = min(train_end + step, n_samples)
        if test_end - train_end < _MIN_TEST_ROWS:
            break
        splits.append((train_end, test_end))

    return splits


def walk_forward_validate(X, y, n_splits=5):
    """
    Time-series–safe expanding-window walk-forward cross-validation.

    Each fold:
      - trains on  X[:train_end]  /  y[:train_end]   (past only)
      - evaluates on X[train_end:test_end]  (future unseen block)

    No shuffling, no future leakage.

    Note: the per-fold score is the MEAN of each individual model's
    accuracy, not the accuracy of the blended production ensemble. Kept
    unchanged because train_model() returns it and production filters
    consume it; walk_forward_validate_ensemble() reports the actual
    ensemble's accuracy on the same folds.

    Returns:
        float: weighted mean accuracy across all valid folds (weight = fold size).
               Falls back to 0.5 when data is too short for even one fold.
    """
    splits = _walk_forward_splits(len(X), n_splits)
    if not splits:
        return 0.5

    fold_results = []  # (mean_accuracy, test_size)

    for train_end, test_end in splits:
        X_tr, y_tr = X.iloc[:train_end],        y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        # Skip folds where training data has only one class — can't fit classifiers.
        if len(set(y_tr)) < 2:
            continue

        fold_accs = []
        for _, model in _make_candidates():
            try:
                model.fit(X_tr, y_tr)
                fold_accs.append(model.score(X_te, y_te))
            except Exception:
                continue

        if fold_accs:
            fold_results.append((sum(fold_accs) / len(fold_accs), len(y_te)))

    if not fold_results:
        return 0.5

    total_weight = sum(w for _, w in fold_results)
    return sum(acc * w for acc, w in fold_results) / total_weight


def walk_forward_validate_ensemble(X, y, n_splits=5) -> dict:
    """
    Walk-forward evaluation of the ACTUAL production ensemble.

    Uses the same folds as walk_forward_validate() (_walk_forward_splits),
    the same candidates (_make_candidates) and the same blend
    (ensemble_proba / ENSEMBLE_WEIGHTS). Every model in a fold is fit on
    rows [0, train_end) only and scored on rows [train_end, test_end).

    Reporting only — not used by train_model() or any production filter.

    Returns a dict with:
        ensemble_accuracy               test-size-weighted ensemble accuracy
        mean_individual_model_accuracy  identical to walk_forward_validate()
        per_model_accuracy              test-size-weighted, per model
        n_folds, n_test_rows, folds     per-fold detail
    Accuracies are None when no fold could be evaluated.
    """
    folds = []
    for train_end, test_end in _walk_forward_splits(len(X), n_splits):
        X_tr, y_tr = X.iloc[:train_end],        y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        if len(set(y_tr)) < 2:
            continue
        models, per_model = {}, {}
        for name, model in _make_candidates():
            try:
                model.fit(X_tr, y_tr)
                per_model[name] = model.score(X_te, y_te)
                models[name] = model
            except Exception:
                continue
        if not per_model:
            continue
        fold = {
            "train_start": 0, "train_end": train_end, "test_end": test_end,
            "n_test": len(y_te),
            "per_model_accuracy": per_model,
            "mean_individual_accuracy": sum(per_model.values()) / len(per_model),
            "ensemble_accuracy": None,
        }
        if "Logistic Regression" in models and "Random Forest" in models:
            preds = (ensemble_proba(models, X_te) > 0.5).astype(int)
            fold["ensemble_accuracy"] = float((preds == y_te.to_numpy()).mean())
        folds.append(fold)

    def _weighted(key):
        vals = [(f[key], f["n_test"]) for f in folds if f[key] is not None]
        total = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / total if total else None

    per_model_acc = {}
    for name in ENSEMBLE_WEIGHTS:
        vals = [(f["per_model_accuracy"][name], f["n_test"])
                for f in folds if name in f["per_model_accuracy"]]
        total = sum(w for _, w in vals)
        per_model_acc[name] = sum(v * w for v, w in vals) / total if total else None

    return {
        "ensemble_accuracy":              _weighted("ensemble_accuracy"),
        "mean_individual_model_accuracy": _weighted("mean_individual_accuracy"),
        "per_model_accuracy":             per_model_acc,
        "n_folds":                        len(folds),
        "n_test_rows":                    sum(f["n_test"] for f in folds),
        "folds":                          folds,
    }


def _fast_accuracy(X, y) -> float:
    """
    Single held-out split accuracy — used during background scans where speed
    matters more than a perfectly calibrated CV score.

    Trains on the first 80 % of rows (time-ordered), evaluates on the last 20 %.
    Returns 0.5 if the split is too small.
    """
    split = int(len(X) * 0.8)
    if split < _MIN_TRAIN_ROWS or (len(X) - split) < _MIN_TEST_ROWS:
        return 0.5

    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    if len(set(y_tr)) < 2:
        return 0.5

    accs = []
    for _, model in _make_candidates():
        try:
            model.fit(X_tr, y_tr)
            accs.append(model.score(X_te, y_te))
        except Exception:
            continue

    return sum(accs) / len(accs) if accs else 0.5


def train_model(X, y, fast: bool = False):
    """
    Validate then retrain on the full dataset.

    Args:
        X, y  : full feature matrix and label series.
        fast  : if True, use a single 80/20 split instead of full walk-forward
                CV.  ~5× faster — use this in background scans.

    Returns:
        trained_models (dict[str, Pipeline]): name → fitted Pipeline
        wf_accuracy    (float): validation accuracy in [0, 1]
    """
    wf_acc = _fast_accuracy(X, y) if fast else walk_forward_validate(X, y)

    trained_models = {}
    for name, model in _make_candidates():
        model.fit(X, y)
        trained_models[name] = model

    return trained_models, wf_acc


def component_probabilities(models, X) -> dict:
    """P(up) per row of X for each ensemble member. Missing XGBoost falls
    back to the Random Forest probability, exactly as the blend does."""
    lr_prob = models["Logistic Regression"].predict_proba(X)[:, 1]
    rf_prob = models["Random Forest"].predict_proba(X)[:, 1]
    if "XGBoost" in models:
        xgb_prob = models["XGBoost"].predict_proba(X)[:, 1]
    else:
        xgb_prob = rf_prob
    return {"Logistic Regression": lr_prob, "Random Forest": rf_prob, "XGBoost": xgb_prob}


def ensemble_proba(models, X):
    """Production blended P(up) for every row of X."""
    p = component_probabilities(models, X)
    return (
        p["Logistic Regression"] * ENSEMBLE_WEIGHTS["Logistic Regression"] +
        p["Random Forest"] * ENSEMBLE_WEIGHTS["Random Forest"] +
        p["XGBoost"] * ENSEMBLE_WEIGHTS["XGBoost"]
    )


def ensemble_predict(models, latest_data):

    final_prob = ensemble_proba(models, latest_data)[0]

    pred       = 1 if final_prob > 0.5 else 0
    confidence = round(max(final_prob, 1 - final_prob) * 100, 2)

    return pred, confidence, final_prob