"""
tests/test_production_temporal_integrity.py — Phase 11A regression coverage
for the production inference contract (docs/PRODUCTION_TEMPORAL_INTEGRITY.md):

    data through bar D -> train only on rows whose label is known (<= D-1)
                       -> predict from bar D's features (never trained on)
                       -> D+1.. outcomes observed later

Deterministic synthetic OHLCV only; no network, price cache or tracker.db.
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import walk_forward as wf
from features.engineer import compute_features, create_features
from models.trainer import ENSEMBLE_WEIGHTS, _make_candidates, ensemble_predict, ensemble_proba, train_model
from utils.helpers import prepare_data, prepare_inference_data


def _make_ohlcv(n: int = 600, seed: int = 21, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = np.maximum(100 + np.cumsum(rng.normal(0, 1.2, size=n)), 5.0)
    return pd.DataFrame({
        "Open":   close + rng.uniform(-0.7, 0.7, size=n),
        "High":   close + rng.uniform(0.1, 1.5, size=n),
        "Low":    close - rng.uniform(0.1, 1.5, size=n),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 3_000_000, size=n).astype(float),
    }, index=dates)


@pytest.fixture(scope="module")
def full():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def issue_date(full):
    return full.index[-20]


@pytest.fixture(scope="module")
def history(full, issue_date):
    """What production downloads at T: a 1-year window ending at bar D."""
    daily, _ = wf.production_inputs(full, issue_date)
    return daily


@pytest.fixture(scope="module")
def inf(history):
    return prepare_inference_data(history)


# ══════════════════════════════════════════════════════════════════════════════
# Training rows vs prediction row
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingVsPredictionRow:
    def test_prediction_row_is_the_latest_bar(self, inf, history):
        assert list(inf.X_pred.index) == [history.index[-1]]

    def test_prediction_row_is_not_in_training(self, inf, history):
        assert history.index[-1] not in inf.X.index
        assert history.index[-1] not in inf.y.index
        assert inf.X.index.max() < inf.X_pred.index[0]

    def test_only_rows_with_known_labels_are_trained_on(self, inf, history):
        next_close = history["Close"].shift(-1)
        expected = (next_close > history["Close"]).astype(int).loc[inf.y.index]
        assert next_close.loc[inf.y.index].notna().all()
        pd.testing.assert_series_equal(inf.y, expected, check_names=False)

    def test_training_set_is_unchanged_by_the_fix(self, inf, history):
        """Same rows, same labels, same features as before Phase 11A."""
        _, X_old, y_old, *_ = prepare_data(history.copy())
        pd.testing.assert_frame_equal(inf.X, X_old)
        pd.testing.assert_series_equal(inf.y, y_old)

    def test_create_features_is_compute_features_minus_incomplete_rows(self, history):
        pd.testing.assert_frame_equal(create_features(history), compute_features(history).dropna())

    def test_prediction_features_match_the_feature_definition_at_D(self, inf, history):
        feats = compute_features(history)
        pd.testing.assert_frame_equal(inf.X_pred, feats.loc[[history.index[-1]], inf.X.columns])

    def test_decision_data_ends_at_D(self, inf, history):
        assert inf.data.index[-1] == history.index[-1]
        assert np.isnan(inf.data["Up"].iloc[-1])

    def test_incomplete_latest_bar_yields_no_prediction_row(self, history):
        broken = history.copy()
        broken.iloc[-1, broken.columns.get_loc("Volume")] = np.nan  # invalid input at D
        inf = prepare_inference_data(broken)
        assert inf.X_pred is None
        assert "Volume_Change" in inf.X_pred_invalid


# ══════════════════════════════════════════════════════════════════════════════
# Information cutoff
# ══════════════════════════════════════════════════════════════════════════════

class TestInformationCutoff:
    def test_future_prices_do_not_change_features_at_D(self, full, issue_date, inf):
        later = full.loc[full.index <= full.index[full.index.get_loc(issue_date) + 5]]
        poisoned = later.copy()
        after = poisoned.index > issue_date
        poisoned.loc[after, ["Open", "High", "Low", "Close"]] *= 2.5
        poisoned.loc[after, "Volume"] *= 0.2
        feats = compute_features(poisoned.loc[poisoned.index > issue_date - wf.DAILY_LOOKBACK])
        pd.testing.assert_frame_equal(feats.loc[[issue_date], inf.X.columns], inf.X_pred)

    def test_future_prices_do_not_change_the_prediction(self, full, issue_date):
        base = wf.replay_production_prediction(full, issue_date)
        poisoned = full.copy()
        poisoned.loc[poisoned.index > issue_date, "Close"] *= 0.3
        again = wf.replay_production_prediction(poisoned, issue_date)
        for k in ("ensemble_probability", "predicted_direction", "confidence", "signal", "confluence"):
            assert again[k] == base[k], k

    def test_news_cannot_change_the_ml_prediction(self, full, issue_date, history, monkeypatch):
        """News only enters the confluence News pillar; the ML inputs and
        output are identical whatever headlines are returned."""
        from scanner import engine

        _, weekly = wf.production_inputs(full, issue_date)
        monkeypatch.setattr(engine, "load_multi_timeframe_data",
                            lambda s: {"weekly": weekly, "daily": history})
        monkeypatch.setattr(engine, "passes_quality_filters", lambda *a, **k: True)
        monkeypatch.setattr(engine, "get_sector", lambda s: None)
        monkeypatch.setattr(engine, "fetch_news", lambda s: ["headline"])

        results = []
        for score in (-1.0, 1.0):
            monkeypatch.setattr(engine, "analyze_overall_sentiment",
                                lambda h, s=score: ("x", s, [], {}))
            results.append(engine._scan_one("SYN.NS", {}, lambda s: history.copy()))
        a, b = results
        assert a["confidence"] == b["confidence"]
        assert a["accuracy"] == b["accuracy"]
        assert a["pillar_scores"]["ML Direction"] == b["pillar_scores"]["ML Direction"]
        assert a["pillar_scores"]["News Sentiment"] != b["pillar_scores"]["News Sentiment"]

    def test_benchmark_never_fetches_news(self, full, issue_date, monkeypatch):
        import news.api

        def _boom(*a, **k):
            raise AssertionError("historical replay must not fetch live news")
        monkeypatch.setattr(news.api, "fetch_news", _boom)
        rec = wf.replay_production_prediction(full, issue_date)
        assert rec["news_score"] == 0.0 and rec["news_status"] == wf.NEWS_STATUS


# ══════════════════════════════════════════════════════════════════════════════
# The Phase 10 bug
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase10Regression:
    def test_prediction_no_longer_reproduces_the_known_last_move(self, full):
        """Before Phase 11A production predicted X.iloc[-1:] — bar D-1, a
        training row whose label is the realised D-1->D move — and reproduced
        that move almost always. With the same fitted models, the old row
        reproduces it; the new row (bar D, never trained on) does not."""
        dates = full.index[-60:-20:2]
        old_agree = new_agree = 0
        for d in dates:
            daily, _ = wf.production_inputs(full, d)
            inf = prepare_inference_data(daily)
            models, _ = train_model(inf.X, inf.y, fast=True)
            realised = int(daily["Close"].iloc[-1] > daily["Close"].iloc[-2])
            old_pred, *_ = ensemble_predict(models, inf.X.iloc[-1:])
            new_pred, *_ = ensemble_predict(models, inf.X_pred)
            old_agree += int(old_pred == realised)
            new_agree += int(new_pred == realised)
        n = len(dates)
        assert old_agree / n >= 0.9          # the bug, reproduced
        assert new_agree / n <= 0.75         # gone with the fix

    def test_scanner_predicts_bar_D_and_stores_close_D(self, full, issue_date, history, monkeypatch):
        from scanner import engine

        seen = []
        real = engine.ensemble_predict
        monkeypatch.setattr(engine, "ensemble_predict",
                            lambda models, row: (seen.append(row.index[-1]), real(models, row))[1])
        _, weekly = wf.production_inputs(full, issue_date)
        monkeypatch.setattr(engine, "fetch_news", lambda s: [])
        monkeypatch.setattr(engine, "load_multi_timeframe_data",
                            lambda s: {"weekly": weekly, "daily": history})
        monkeypatch.setattr(engine, "passes_quality_filters", lambda *a, **k: True)
        monkeypatch.setattr(engine, "get_sector", lambda s: None)

        result = engine._scan_one("SYN.NS", {}, lambda s: history.copy())
        assert seen == [issue_date]
        assert result["close"] == round(float(history["Close"].iloc[-1]), 2)

    def test_api_analyze_predicts_bar_D_and_stores_close_D(self, full, issue_date, history, monkeypatch):
        from api import services

        seen, stored = [], {}
        real = services.ensemble_predict
        monkeypatch.setattr(services, "ensemble_predict",
                            lambda models, row: (seen.append(row.index[-1]), real(models, row))[1])
        _, weekly = wf.production_inputs(full, issue_date)
        monkeypatch.setattr(services, "load_data", lambda s: history.copy())
        monkeypatch.setattr(services, "get_company_names", lambda s: "Synthetic")
        monkeypatch.setattr(services, "fetch_news", lambda s: [])
        monkeypatch.setattr(services, "load_multi_timeframe_data",
                            lambda s: {"weekly": weekly, "daily": history})
        monkeypatch.setattr(services, "save_signal", lambda *a, **k: None)
        monkeypatch.setattr(services, "upsert_recommendation", lambda **k: stored.update(k))
        monkeypatch.setattr(services, "get_sector", lambda s: None)

        services.analyze_stock("SYN.NS")
        assert seen == [issue_date]
        assert stored["cmp"] == float(history["Close"].iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Unchanged model definition, determinism, entry convention
# ══════════════════════════════════════════════════════════════════════════════

class TestUnchangedAndDeterministic:
    def test_ensemble_weights_unchanged(self):
        assert ENSEMBLE_WEIGHTS == {"Logistic Regression": 0.20, "Random Forest": 0.30, "XGBoost": 0.50}

    def test_xgboost_fallback_unchanged(self, inf):
        models = {n: m.fit(inf.X, inf.y) for n, m in _make_candidates() if n != "XGBoost"}
        lr = models["Logistic Regression"].predict_proba(inf.X_pred)[0][1]
        rf = models["Random Forest"].predict_proba(inf.X_pred)[0][1]
        assert ensemble_proba(models, inf.X_pred)[0] == lr * 0.20 + rf * 0.30 + rf * 0.50

    def test_production_prediction_is_deterministic(self, full, issue_date):
        a = wf.replay_production_prediction(full, issue_date)
        b = wf.replay_production_prediction(full, issue_date)
        assert a == b

    def test_entry_price_and_outcome_dates(self, full, issue_date):
        rec = wf.build_prediction_record("SYN.NS", full, issue_date)
        pos = full.index.get_loc(issue_date)
        assert rec["prediction_price"] == float(full["Close"].iloc[pos])
        assert rec["production_cmp"] == round(float(full["Close"].iloc[pos]), 2)
        for h in wf.HORIZONS:
            assert rec[f"outcome_date_{h}d"] == full.index[pos + h]
            assert rec[f"legacy_cmp_return_{h}d"] == pytest.approx(rec[f"return_{h}d"], abs=0.01)
