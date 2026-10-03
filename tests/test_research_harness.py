"""
tests/test_research_harness.py — integrity of the Phase 12–15 research
harness (evaluation/research.py, evaluation/research_metrics.py).

Deterministic synthetic data; no network, no snapshot, no database.
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import research as rs
from evaluation import research_metrics as rm
from evaluation import walk_forward as wf
from features.engineer import compute_features
from models.trainer import _make_candidates
from utils.helpers import FEATURE_COLS, prepare_inference_data


def _make_ohlcv(n=900, seed=31, start="2023-06-01"):
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
def raw():
    return _make_ohlcv()


CFG = rs.ResearchConfig(name="t", start="2025-03-24", end="2025-04-18", refit_every=5)


@pytest.fixture(scope="module")
def rows(raw):
    return pd.DataFrame(rs.run_symbol("SYN", raw, CFG))


class TestHarnessIntegrity:
    def test_every_row_is_out_of_sample(self, rows):
        assert len(rows) > 0
        assert (rows["train_end"] < rows["date"]).all()
        assert (rows["label_end"] <= rows["refit_date"]).all()
        assert (rows["refit_date"] <= rows["date"]).all()
        for h in wf.HORIZONS:
            od = rows[f"outcome_date_{h}d"].dropna()
            assert (od > rows.loc[od.index, "date"]).all()

    def test_multi_horizon_labels_are_known_at_refit(self, raw):
        cfg = rs.ResearchConfig(name="t5", target_horizon=5, start="2025-03-24",
                                end="2025-04-04", refit_every=5)
        out = pd.DataFrame(rs.run_symbol("SYN", raw, cfg))
        assert (out["label_end"] <= out["refit_date"]).all()
        # label of the last training row needs the bar 5 raw bars later
        r0 = out.iloc[0]
        assert raw.index.get_loc(r0["label_end"]) - raw.index.get_loc(r0["train_end"]) == 5

    def test_h1_label_is_the_production_label(self, raw):
        win = raw.iloc[:300]
        pd.testing.assert_series_equal(
            rs.forward_label(win["Close"], 1), compute_features(win)["Up"], check_names=False)

    def test_matches_production_replay_at_a_refit_date(self, raw, rows):
        """At its refit date the harness prediction is the production prediction."""
        r = rows.iloc[0]
        assert r["date"] == r["refit_date"]
        replay = wf.replay_production_prediction(raw, r["date"])
        ens = rs.add_ensembles(rows.iloc[[0]].copy())["p_ens"].iloc[0]
        assert ens == pytest.approx(replay["ensemble_probability"], abs=1e-12)

    def test_production_models_unchanged_by_single_threading(self, raw):
        inf = prepare_inference_data(raw.iloc[:300])
        for (name, prod), key in zip(_make_candidates(), rs.PRODUCTION_MODELS):
            a = prod.fit(inf.X, inf.y).predict_proba(inf.X_pred)[0, 1]
            b = rs.make_model(key).fit(inf.X, inf.y).predict_proba(inf.X_pred)[0, 1]
            assert a == b, name

    def test_future_prices_cannot_change_a_prediction(self, raw, rows):
        d = rows["date"].iloc[7]
        poisoned = raw.copy()
        poisoned.loc[poisoned.index > d, ["Open", "High", "Low", "Close"]] *= 2.0
        again = pd.DataFrame(rs.run_symbol("SYN", poisoned, CFG))
        a = rows.set_index("date").loc[d, ["p_lr", "p_rf", "p_xgb"]]
        b = again.set_index("date").loc[d, ["p_lr", "p_rf", "p_xgb"]]
        pd.testing.assert_series_equal(a, b)

    def test_integrity_assertion_fires(self, rows):
        bad = rows.iloc[0].to_dict()
        bad["train_end"] = bad["date"]
        with pytest.raises(rs.ResearchIntegrityError):
            rs.assert_row_integrity(bad)

    def test_feature_groups_partition_the_production_features(self):
        flat = sum(rs.FEATURE_GROUPS.values(), [])
        assert sorted(flat) == sorted(FEATURE_COLS) and len(flat) == len(set(flat))


class TestSegmentsAndCalibrationSplit:
    def test_segments_are_ordered_and_disjoint(self):
        spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in rs.SEGMENTS.values()]
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            assert a1 <= b1 < a2 <= b2

    def test_calibration_rows_never_overlap_final_evaluation(self):
        dates = pd.bdate_range("2025-03-01", "2025-12-31")
        df = pd.DataFrame({"date": dates})
        for h in wf.HORIZONS:
            df[f"outcome_date_{h}d"] = dates + pd.tseries.offsets.BDay(h)
        df["segment"] = df["date"].map(rs.segment_of)
        for h in wf.HORIZONS:
            for seg in ("DEV", "FINAL"):
                start = pd.Timestamp(rs.SEGMENTS[seg][0])
                fit = rm.fit_rows_before(df, h, start)
                assert (fit["outcome_date_" + f"{h}d"] < start).all()
                assert set(fit["date"]).isdisjoint(set(df.loc[df["segment"] == seg, "date"]))

    def test_embargo_drops_rows_whose_outcome_lands_in_the_test_segment(self):
        start = pd.Timestamp(rs.SEGMENTS["DEV"][0])
        last_cal = pd.Timestamp(rs.SEGMENTS["CAL"][1])
        df = pd.DataFrame({"date": [last_cal], "outcome_date_10d": [start + pd.Timedelta(days=10)]})
        assert rm.fit_rows_before(df, 10, start).empty


class TestMetrics:
    def test_calibrators(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.3, 0.9, 2000)
        y = (rng.uniform(size=2000) < 0.5).astype(int)   # p is badly over-confident
        for method in ("platt", "isotonic"):
            q = rm.Calibrator(method).fit(p, y).predict(p)
            assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)          # monotone
            assert rm.classification(y, q)["brier"] < rm.classification(y, p)["brier"]
        np.testing.assert_array_equal(rm.Calibrator("none").fit(p, y).predict(p), p)

    def test_classification_known_values(self):
        m = rm.classification([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6])
        assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
        assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.36 + 0.36) / 4, abs=1e-5)

    def test_paired_bootstrap_of_identical_predictions_is_zero(self):
        df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2025-01-01", periods=20), 3),
                           "y": np.tile([0, 1, 1], 20), "a": 0.6})
        df["b"] = df["a"]
        r = rm.paired_bootstrap(df, "y", "a", "b")
        assert r["accuracy_diff"] == 0 and r["accuracy_diff_ci95"] == [0, 0]

    def test_max_drawdown(self):
        assert rm.max_drawdown(np.array([10.0, -50.0, 20.0])) == -50.0
