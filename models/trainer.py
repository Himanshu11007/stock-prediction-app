from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def train_model(X_train, X_test, y_train, y_test):
    """
    Train multiple models and return the best one wrapped in a Pipeline.
    Using Pipeline ensures the scaler is bundled with the model, so
    model.predict() always receives correctly-scaled data.
    """
    candidates = [
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, random_state=42)),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ]),
        ),
    ]

    if _XGB_AVAILABLE:
        candidates.append((
            "XGBoost",
            Pipeline([
                ("xgb", XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1,
                )),
            ]),
        ))

    best_name, best_model, best_acc = None, None, -1.0

    for name, model in candidates:
        model.fit(X_train, y_train)
        acc = float(model.score(X_test, y_test))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    return best_model, best_acc, best_name
