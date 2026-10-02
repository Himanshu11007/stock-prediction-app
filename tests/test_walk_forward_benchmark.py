"""
tests/test_walk_forward_benchmark.py — Phase 10 regression coverage for the
clean walk-forward benchmark (evaluation/walk_forward.py) and the ensemble
reporting added to models/trainer.py.

All data is deterministic synthetic OHLCV (fixed seeds); nothing here
touches the network, the price cache or storage/tracker.db.
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import walk_forward as wf
from models import trainer
from models.trainer import (
    ENSEMBLE_WEIGHTS,
    _make_candidates,
    _walk_forward_splits,
    ensemble_predict,
    ensemble_proba,
    walk_forward_validate,
    walk_forward_validate_ensemble,
)
from storage.recommendation_validation import calculate_success
from utils.helpers import prepare_data


def _make_ohlcv(n: int = 600, seed: int = 11, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = np.maximum(100 + np.cumsum(rng.normal(0, 1.2, size=n)), 5.0)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    open_ = close + rng.uniform(-0.7, 0.7, size=n)
    volume = rng.integers(1_000_000, 3_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def full():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def issue_date(full):
    return full.index[-20]


@pytest.fixture(scope="module")
def record(full, issue_date):
    return wf.build_prediction_record("SYN.NS", full, issue_date)


@pytest.fixture(scope="module")
def xy():
    _, X, y, *_ = prepare_data(_make_ohlcv(n=320, seed=3))
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble definition and walk-forward reporting (Tasks 2, 3)
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleDefinition:
    def test_weights_are_exact_and_sum_to_one(self):
        assert ENSEMBLE_WEIGHTS == {
            "Logistic Regression": 0.20, "Random Forest": 0.30, "XGBoost": 0.50,
        }
        assert sum(ENSEMBLE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_ensemble_predict_is_bit_identical_to_the_original_blend(self, xy):
        X, y = xy
        models = {name: m.fit(X, y) for name, m in _make_candidates()}
        row = X.iloc[-1:]
        lr = models["Logistic Regression"].predict_proba(row)[0][1]
        rf = models["Random Forest"].predict_proba(row)[0][1]
        xgb = models["XGBoost"].predict_proba(row)[0][1] if "XGBoost" in models else rf
        original = (lr * 0.20 + rf * 0.30 + xgb * 0.50)

        pred, conf, prob = ensemble_predict(models, row)
        assert prob == original
        assert pred == (1 if original > 0.5 else 0)
        assert conf == round(max(original, 1 - original) * 100, 2)

    def test_vectorised_blend_equals_row_by_row_predictions(self, xy):
        X, y = xy
        models = {name: m.fit(X, y) for name, m in _make_candidates()}
        batch = ensemble_proba(models, X.iloc[-15:])
        single = [ensemble_predict(models, X.iloc[i:i + 1])[2] for i in range(len(X) - 15, len(X))]
        # Batched BLAS can differ from single-row calls in the last ulp only.
        assert list(batch) == pytest.approx(single, rel=1e-12, abs=1e-15)

    def test_missing_xgboost_falls_back_to_random_forest(self, xy):
        X, y = xy
        models = {name: m.fit(X, y) for name, m in _make_candidates() if name != "XGBoost"}
        row = X.iloc[-1:]
        lr = models["Logistic Regression"].predict_proba(row)[0][1]
        rf = models["Random Forest"].predict_proba(row)[0][1]
        assert ensemble_proba(models, row)[0] == lr * 0.20 + rf * 0.30 + rf * 0.50


class TestEnsembleWalkForward:
    def test_legacy_value_is_reproduced_and_unchanged(self, xy):
        X, y = xy
        result = walk_forward_validate_ensemble(X, y)
        assert result["mean_individual_model_accuracy"] == pytest.approx(walk_forward_validate(X, y))

    def test_ensemble_accuracy_is_the_blended_prediction_on_each_fold(self, xy):
        X, y = xy
        result = walk_forward_validate_ensemble(X, y)
        correct = total = 0
        for train_end, test_end in _walk_forward_splits(len(X)):
            models = {n: m.fit(X.iloc[:train_end], y.iloc[:train_end]) for n, m in _make_candidates()}
            preds = (ensemble_proba(models, X.iloc[train_end:test_end]) > 0.5).astype(int)
            correct += int((preds == y.iloc[train_end:test_end].to_numpy()).sum())
            total += test_end - train_end
        assert result["ensemble_accuracy"] == pytest.approx(correct / total)
        assert result["n_test_rows"] == total

    def test_every_fold_trains_strictly_before_it_tests(self, xy):
        X, y = xy
        folds = walk_forward_validate_ensemble(X, y)["folds"]
        assert folds
        for f in folds:
            assert f["train_start"] == 0 < f["train_end"] < f["test_end"] <= len(X)

    def test_train_model_still_returns_the_legacy_score(self, xy, monkeypatch):
        X, y = xy
        _, acc = trainer.train_model(X, y)
        assert acc == pytest.approx(walk_forward_validate(X, y))


# ══════════════════════════════════════════════════════════════════════════════
# Trading-day horizons (Task 6)
# ══════════════════════════════════════════════════════════════════════════════

class TestHorizons:
    def _close(self):
        idx = pd.bdate_range("2026-03-02", periods=15)  # Mon..; weekends excluded
        return pd.Series(np.arange(100.0, 115.0), index=idx)

    def test_outcome_is_exactly_n_trading_days_ahead_across_weekends(self):
        close = self._close()
        friday = pd.Timestamp("2026-03-06")
        out = wf.forward_outcomes(close, friday)
        assert out["outcome_date_1d"] == pd.Timestamp("2026-03-09")   # Monday
        assert out["outcome_date_5d"] == pd.Timestamp("2026-03-13")
        assert out["actual_price_3d"] == close.iloc[close.index.get_loc(friday) + 3]
        assert out["prediction_price"] == close[friday]

    def test_insufficient_future_is_null_never_substituted(self):
        close = self._close()
        d = close.index[-4]  # only 3 future sessions exist
        out = wf.forward_outcomes(close, d)
        assert out["outcome_date_3d"] == close.index[-1]
        for h in (5, 10):
            assert out[f"outcome_date_{h}d"] is None
            assert out[f"actual_price_{h}d"] is None
            assert out[f"return_{h}d"] is None
            assert out[f"outcome_{h}d"] is None

    def test_holiday_placeholder_bar_is_not_a_trading_session(self):
        df = _make_ohlcv(n=10, start="2026-03-02")
        holiday = df.index[3]
        df.loc[holiday, ["Open", "High", "Low", "Close"]] = df.loc[df.index[2], "Close"]
        df.loc[holiday, "Volume"] = 0
        sessions = wf.session_closes(df)
        assert holiday not in sessions.index
        out = wf.forward_outcomes(sessions, df.index[2])
        assert out["outcome_date_1d"] == df.index[4]

    def test_return_and_direction_use_entry_at_T(self):
        close = self._close()
        out = wf.forward_outcomes(close, close.index[0])
        assert out["return_1d"] == pytest.approx((101.0 - 100.0) / 100.0 * 100, abs=1e-4)
        assert out["outcome_1d"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Point-in-time integrity (Tasks 4, 16, 18)
# ══════════════════════════════════════════════════════════════════════════════

_PREDICTION_FIELDS = (
    "training_end_timestamp", "n_training_rows", "logistic_probability", "rf_probability",
    "xgb_probability", "ensemble_probability", "predicted_direction", "confidence",
    "forward_row_ensemble_probability", "signal", "confluence", "weighted_score",
    "market_regime", "timeframe_score", "passes_quality_filters", "production_cmp",
)


class TestPointInTime:
    def test_replay_is_invariant_to_any_change_after_T(self, full, issue_date, record):
        poisoned = full.copy()
        after = poisoned.index > issue_date
        poisoned.loc[after, ["Open", "High", "Low", "Close"]] *= 3.0
        poisoned.loc[after, "Volume"] *= 0.1
        replay = wf.replay_production_prediction(poisoned, issue_date)
        for field in _PREDICTION_FIELDS:
            assert replay[field] == record[field], field

    def test_record_timestamps_respect_T(self, full, issue_date, record):
        t = issue_date
        prev_bar = full.index[full.index.get_loc(t) - 1]
        assert record["prediction_timestamp"] == t
        assert record["training_end_timestamp"] == prev_bar < t
        assert record["training_label_end_timestamp"] == t
        assert record["daily_window_end"] == t
        assert record["weekly_window_end"] <= t
        for h in wf.HORIZONS:
            assert record[f"outcome_date_{h}d"] > t

    def test_issue_row_features_do_not_depend_on_the_next_bar(self, full, issue_date):
        daily, _ = wf.production_inputs(full, issue_date)
        _, X, *_ = prepare_data(daily.copy())
        cols = list(X.columns)
        via_placeholder = wf.issue_row_features(daily, cols)

        with_real_next = full.loc[full.index > issue_date - wf.DAILY_LOOKBACK].iloc[: len(daily) + 1]
        _, X_next, *_ = prepare_data(with_real_next.copy())
        pd.testing.assert_frame_equal(via_placeholder, X_next.loc[[issue_date], cols])

    def test_assertion_fails_when_training_reaches_T(self, record):
        bad = dict(record, training_end_timestamp=record["prediction_timestamp"])
        with pytest.raises(wf.TemporalIntegrityError):
            wf.assert_temporal_integrity(bad)

    def test_assertion_fails_when_an_outcome_is_not_after_T(self, record):
        bad = dict(record, outcome_date_5d=record["prediction_timestamp"])
        with pytest.raises(wf.TemporalIntegrityError):
            wf.assert_temporal_integrity(bad)

    def test_assertion_fails_when_inputs_extend_past_T(self, record):
        bad = dict(record, daily_window_end=record["prediction_timestamp"] + pd.Timedelta(days=1))
        with pytest.raises(wf.TemporalIntegrityError):
            wf.assert_temporal_integrity(bad)

    def test_news_is_flagged_unavailable_and_uses_the_no_headlines_path(self, record):
        assert record["news_status"] == wf.NEWS_STATUS
        assert record["news_score"] == 0.0

    def test_previous_direction_baseline_uses_only_bars_up_to_T(self, full, issue_date, record):
        pos = full.index.get_loc(issue_date)
        expected = int(full["Close"].iloc[pos] > full["Close"].iloc[pos - 1])
        assert record["realized_issue_move"] == expected


# ══════════════════════════════════════════════════════════════════════════════
# Production parity (Tasks 1, 2, 10, 11)
# ══════════════════════════════════════════════════════════════════════════════

class TestProductionParity:
    def test_replay_matches_scanner_scan_one(self, full, issue_date, record, monkeypatch):
        """The benchmark must produce exactly what scanner/engine.py:_scan_one
        produces for the same point-in-time inputs (news forced empty)."""
        from scanner import engine

        daily, weekly = wf.production_inputs(full, issue_date)
        monkeypatch.setattr(engine, "fetch_news", lambda symbol: [])
        monkeypatch.setattr(engine, "load_multi_timeframe_data",
                            lambda symbol: {"weekly": weekly, "daily": daily})
        monkeypatch.setattr(engine, "passes_quality_filters", lambda *a, **k: True)
        monkeypatch.setattr(engine, "get_sector", lambda symbol: None)

        result = engine._scan_one("SYN.NS", {}, lambda symbol: daily.copy())

        assert result is not None
        assert result["signal"] == record["signal"]
        assert result["score"] == round(record["confluence"], 4)
        assert result["confidence"] == round(record["confidence"], 2)
        assert result["accuracy"] == round(record["production_accuracy_fast"] * 100, 2)
        assert result["close"] == record["production_cmp"]
        assert result["regime"] == record["market_regime"]
        assert result["timeframe_score"] == round(record["timeframe_score"], 2)
        assert (result["pillar_scores"]["ML Direction"] > 0) == (record["predicted_direction"] == 1)

    def test_signal_success_uses_the_existing_rules(self, record):
        for h in wf.HORIZONS:
            assert record[f"signal_success_{h}d"] == calculate_success(
                record["signal"], record[f"return_{h}d"])

    def test_as_deployed_prediction_row_is_bar_T_and_not_in_training(self, record):
        """Phase 11A: production predicts the unseen bar at T, not its last
        training row (the Phase 10 finding)."""
        assert record["prediction_row_in_training_set"] is False
        assert record["feature_row_timestamp"] == record["prediction_timestamp"]
        assert record["training_end_timestamp"] < record["feature_row_timestamp"]

    def test_assertion_fails_if_prediction_row_is_in_training(self, record):
        bad = dict(record, prediction_row_in_training_set=True)
        with pytest.raises(wf.TemporalIntegrityError):
            wf.assert_temporal_integrity(bad)


# ══════════════════════════════════════════════════════════════════════════════
# Summaries (Tasks 12, 13)
# ══════════════════════════════════════════════════════════════════════════════

class TestSummaries:
    @pytest.mark.parametrize("conf,bucket", [
        (50.0, "50-55"), (54.99, "50-55"), (55.0, "55-60"), (94.99, "90-95"),
        (95.0, "95-100"), (100.0, "95-100"), (None, None),
    ])
    def test_confidence_bucket_boundaries(self, conf, bucket):
        assert wf.confidence_bucket(conf) == bucket

    def test_wilson_interval_brackets_the_rate(self):
        lo, hi = wf.wilson_interval(55, 100)
        assert lo < 55.0 < hi
        assert wf.wilson_interval(0, 0) == (None, None)

    def test_rate_ignores_missing_outcomes(self):
        r = wf.rate(pd.Series([1, 0, None, 1]))
        assert r["n"] == 3 and r["rate_pct"] == pytest.approx(66.7)
