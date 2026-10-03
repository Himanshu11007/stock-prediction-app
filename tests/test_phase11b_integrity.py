"""
tests/test_phase11b_integrity.py — Phase 11B regression coverage:

  11B.1  zero-volume holiday bars no longer make Volume_Change infinite
  11B.2  persisted recommendations are tagged with the engine version
  11B.3  the bar a prediction was made from, and whether it was a finished
         session, are recorded

Deterministic synthetic data only.
"""
import datetime as dt
import sqlite3

import numpy as np
import pandas as pd
import pytest

from config import PRE_TEMPORAL_FIX_ENGINE_VERSIONS, RECOMMENDATION_ENGINE_VERSION
from evaluation import walk_forward as wf
from features.engineer import compute_features
from storage import tracker
from utils.helpers import prepare_inference_data
from utils.market_session import IST, is_daily_bar_complete


def _make_ohlcv(n=600, seed=5, start="2024-01-01"):
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


def _with_holiday(df, pos):
    """Turn bar `pos` into a Yahoo-style holiday placeholder."""
    df = df.copy()
    prev_close = df["Close"].iloc[pos - 1]
    for col in ("Open", "High", "Low", "Close"):
        df.iloc[pos, df.columns.get_loc(col)] = prev_close
    df.iloc[pos, df.columns.get_loc("Volume")] = 0.0
    return df


@pytest.fixture(scope="module")
def window():
    full = _make_ohlcv()
    daily, _ = wf.production_inputs(full, full.index[-20])
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# 11B.1 — zero-volume holiday bars
# ══════════════════════════════════════════════════════════════════════════════

class TestZeroVolumeHandling:
    def test_session_after_holiday_uses_last_traded_volume(self, window):
        df = _with_holiday(window, len(window) - 2)
        feats = compute_features(df)
        expected = df["Volume"].iloc[-1] / df["Volume"].iloc[-3] - 1
        assert feats["Volume_Change"].iloc[-1] == pytest.approx(expected, rel=1e-12)
        assert np.isfinite(feats["Volume_Change"]).iloc[1:].all()

    def test_holiday_bar_itself_is_minus_100_percent(self, window):
        df = _with_holiday(window, len(window) - 2)
        assert compute_features(df)["Volume_Change"].iloc[-2] == -1.0

    def test_normal_bars_are_identical_to_pct_change(self, window):
        feats = compute_features(window)
        pd.testing.assert_series_equal(
            feats["Volume_Change"], window["Volume"].pct_change(), check_names=False)

    def test_no_infinite_feature_after_holiday(self, window):
        df = _with_holiday(window, len(window) - 2)
        feats = compute_features(df).drop(columns="Up")
        assert not np.isinf(feats.to_numpy(dtype=float)).any()

    def test_prediction_row_exists_on_first_session_after_holiday(self, window):
        df = _with_holiday(window, len(window) - 2)
        inf = prepare_inference_data(df)
        assert inf.X_pred is not None
        assert inf.X_pred.index[0] == df.index[-1]

    def test_training_rows_after_holidays_are_kept(self, window):
        df = _with_holiday(window, len(window) - 40)
        inf = prepare_inference_data(df)
        assert df.index[len(df) - 39] in inf.X.index

    def test_invalid_feature_is_named_and_prediction_skipped(self, window):
        df = window.copy()
        df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
        inf = prepare_inference_data(df)
        assert inf.X_pred is None
        assert "Volume_Change" in inf.X_pred_invalid

    def test_later_volume_cannot_change_volume_change_at_D(self, window):
        full = _make_ohlcv()
        d = full.index[-20]
        base = compute_features(full.loc[:d])["Volume_Change"].iloc[-1]
        poisoned = full.copy()
        poisoned.loc[poisoned.index > d, "Volume"] = 0.0
        assert compute_features(poisoned)["Volume_Change"].loc[d] == base


def _patch_scanner(monkeypatch, daily):
    from scanner import engine
    monkeypatch.setattr(engine, "fetch_news", lambda s: [])
    monkeypatch.setattr(engine, "load_multi_timeframe_data",
                        lambda s: {"weekly": wf.weekly_bars(daily), "daily": daily})
    monkeypatch.setattr(engine, "passes_quality_filters", lambda *a, **k: True)
    monkeypatch.setattr(engine, "get_sector", lambda s: None)
    return engine


def _patch_services(monkeypatch, daily, stored):
    from api import services
    monkeypatch.setattr(services, "load_data", lambda s: daily.copy())
    monkeypatch.setattr(services, "get_company_names", lambda s: "Synthetic")
    monkeypatch.setattr(services, "fetch_news", lambda s: [])
    monkeypatch.setattr(services, "load_multi_timeframe_data",
                        lambda s: {"weekly": wf.weekly_bars(daily), "daily": daily})
    monkeypatch.setattr(services, "save_signal", lambda *a, **k: None)
    monkeypatch.setattr(services, "upsert_recommendation", lambda **k: stored.update(k))
    monkeypatch.setattr(services, "get_sector", lambda s: None)
    return services


class TestProductionPathsAfterHoliday:
    def test_scanner_predicts_after_holiday(self, window, monkeypatch):
        df = _with_holiday(window, len(window) - 2)
        engine = _patch_scanner(monkeypatch, df)
        result = engine._scan_one("SYN.NS", {}, lambda s: df.copy())
        assert result is not None
        assert result["prediction_bar_date"] == df.index[-1].date().isoformat()

    def test_scanner_skips_invalid_bar(self, window, monkeypatch):
        df = window.copy()
        df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
        engine = _patch_scanner(monkeypatch, df)
        assert engine._scan_one("SYN.NS", {}, lambda s: df.copy()) is None

    def test_analyze_stock_works_after_holiday(self, window, monkeypatch):
        df = _with_holiday(window, len(window) - 2)
        stored = {}
        services = _patch_services(monkeypatch, df, stored)
        services.analyze_stock("SYN.NS")
        assert stored["cmp"] == float(df["Close"].iloc[-1])
        assert stored["engine_version"] == RECOMMENDATION_ENGINE_VERSION
        assert stored["prediction_bar_date"] == df.index[-1].date().isoformat()
        assert stored["bar_complete"] is True  # synthetic bars are in the past

    def test_analyze_stock_names_invalid_features(self, window, monkeypatch):
        df = window.copy()
        df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
        services = _patch_services(monkeypatch, df, {})
        with pytest.raises(ValueError, match="Volume_Change"):
            services.analyze_stock("SYN.NS")


# ══════════════════════════════════════════════════════════════════════════════
# 11B.2 — engine version tagging
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "tracker.db"
    monkeypatch.setattr(tracker, "TRACKER_DB", path)
    return path


def _upsert(**kw):
    args = dict(symbol="ABC.NS", stock="ABC", signal="BUY", cmp=100.0,
                confluence_score=0.6, ml_confidence=70.0, news_score=0.0,
                accuracy=0.5, target=110.0, stop_loss=95.0,
                saved_date="2026-10-05", scan_id="SCAN-1")
    args.update(kw)
    return tracker.upsert_recommendation(**args)


def _row(db_path, cols):
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(f"SELECT {cols} FROM recommendation_validation").fetchall()
    finally:
        con.close()


class TestEngineVersion:
    def test_current_version_is_post_fix(self):
        assert RECOMMENDATION_ENGINE_VERSION not in PRE_TEMPORAL_FIX_ENGINE_VERSIONS
        assert tracker.is_pre_temporal_fix(None)
        assert tracker.is_pre_temporal_fix("v1.0")
        assert not tracker.is_pre_temporal_fix(RECOMMENDATION_ENGINE_VERSION)

    def test_new_rows_default_to_the_current_engine_version(self, db_path):
        _upsert()
        assert _row(db_path, "engine_version") == [(RECOMMENDATION_ENGINE_VERSION,)]

    def test_legacy_rows_keep_their_version_and_outcomes(self, db_path):
        con = sqlite3.connect(str(db_path))
        con.execute("""
            CREATE TABLE recommendation_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT, saved_date TEXT NOT NULL,
                symbol TEXT NOT NULL, stock TEXT NOT NULL, signal TEXT NOT NULL,
                cmp REAL NOT NULL, confluence_score REAL, ml_confidence REAL,
                news_score REAL, accuracy REAL, target REAL, stop_loss REAL,
                is_validated INTEGER DEFAULT 0, validation_date TEXT,
                validation_price REAL, return_pct REAL, success INTEGER,
                scan_id TEXT, engine_version TEXT)""")
        con.execute(
            "INSERT INTO recommendation_validation (saved_date, symbol, stock, signal, cmp, "
            "is_validated, return_pct, success, scan_id, engine_version) "
            "VALUES ('2026-09-01', 'OLD.NS', 'OLD', 'SELL', 50.0, 1, -2.5, 1, 'SCAN-0', 'v1.0')")
        con.commit()
        con.close()

        _upsert(symbol="NEW.NS")  # runs the migration

        rows = dict((r[0], r[1:]) for r in _row(
            db_path, "symbol, engine_version, is_validated, return_pct, success, prediction_bar_date"))
        assert rows["OLD.NS"] == ("v1.0", 1, -2.5, 1, None)
        assert rows["NEW.NS"][0] == RECOMMENDATION_ENGINE_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# 11B.3 — partial / intraday bars
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionCompleteness:
    @pytest.mark.parametrize("now,expected", [
        (dt.datetime(2026, 10, 5, 10, 0, tzinfo=IST), False),   # market open
        (dt.datetime(2026, 10, 5, 15, 45, tzinfo=IST), False),  # just after close
        (dt.datetime(2026, 10, 5, 16, 0, tzinfo=IST), True),
        (dt.datetime(2026, 10, 5, 22, 0, tzinfo=IST), True),
        (dt.datetime(2026, 10, 6, 9, 0, tzinfo=IST), True),     # next morning
    ])
    def test_today_bar_is_complete_only_after_the_close(self, now, expected):
        assert is_daily_bar_complete("2026-10-05", now) is expected

    def test_utc_clock_is_converted_to_ist(self):
        # 10:00 UTC = 15:30 IST (not final yet); 10:31 UTC = 16:01 IST
        assert not is_daily_bar_complete("2026-10-05", dt.datetime(2026, 10, 5, 10, 0, tzinfo=dt.timezone.utc))
        assert is_daily_bar_complete("2026-10-05", dt.datetime(2026, 10, 5, 10, 31, tzinfo=dt.timezone.utc))

    def test_naive_clock_is_rejected(self):
        with pytest.raises(ValueError):
            is_daily_bar_complete("2026-10-05", dt.datetime(2026, 10, 5, 12, 0))

    def test_bar_date_and_completeness_are_persisted(self, db_path):
        _upsert(prediction_bar_date="2026-10-05", bar_complete=False)
        assert _row(db_path, "saved_date, prediction_bar_date, bar_complete") == [
            ("2026-10-05", "2026-10-05", 0)]

    def test_update_refreshes_bar_metadata(self, db_path):
        _upsert(prediction_bar_date="2026-10-05", bar_complete=False)
        _upsert(prediction_bar_date="2026-10-05", bar_complete=True)
        assert _row(db_path, "bar_complete") == [(1,)]
