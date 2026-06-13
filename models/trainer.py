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


def ensemble_predict(models, latest_data):

    lr_prob = models["Logistic Regression"] \
        .predict_proba(latest_data)[0][1]

    rf_prob = models["Random Forest"] \
        .predict_proba(latest_data)[0][1]

    if "XGBoost" in models:
        xgb_prob = models["XGBoost"] \
            .predict_proba(latest_data)[0][1]
    else:
        xgb_prob = rf_prob

    final_prob = (
        lr_prob * 0.20 +
        rf_prob * 0.30 +
        xgb_prob * 0.50
    )

    pred       = 1 if final_prob > 0.5 else 0
    confidence = round(max(final_prob, 1 - final_prob) * 100, 2)

    return pred, confidence, final_prob