"""
tests/test_feature_engineering.py — Regression coverage for
features/engineer.py:create_features(), added during the Phase 9
recommendation-quality audit.

Context (see docs/RECOMMENDATION_QUALITY_AUDIT.md "Critical problems
discovered" for the full writeup): prior to commit 8997c5a
(2026-09-27), the last row's `Up` label — whose true next-day outcome is
unknowable at feature-creation time — was fabricated as False/0 instead of
being left NaN. Because dropna() only removes NaN rows, that fabricated
0.0 label survived into the training set on every single run, teaching
every per-stock model that "today's actual, current feature pattern" means
"tomorrow is down." The production database shows the resulting damage
directly: every recommendation's ML Direction pillar was bearish (-0.7,
never +0.7) across 1,473 consecutive scanned rows spanning 2026-07-12
through 2026-09-21, before the fix landed. These tests exist so that bug
can never silently return.
"""
import numpy as np
import pandas as pd
import pytest

from features.engineer import create_features


def _make_ohlcv(n: int = 40, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV data - a random walk with a
    fixed seed, so every test run sees exactly the same prices."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    steps = rng.normal(loc=0.0, scale=1.0, size=n)
    close = 100 + np.cumsum(steps)
    close = np.maximum(close, 1.0)  # keep prices positive
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    open_ = close + rng.uniform(-0.5, 0.5, size=n)
    volume = rng.integers(100_000, 1_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestNoFabricatedFinalRowLabel:
    """The exact regression this audit found: the final row's Up label
    must be unknowable (NaN) at creation time and therefore dropped, never
    coerced to a concrete 0/False value that would survive dropna()."""

    def test_last_input_row_is_never_present_in_the_output(self):
        raw = _make_ohlcv(n=40)
        result = create_features(raw)

        assert raw.index[-1] not in result.index, (
            "The last row's next-day close is unknown - it must be dropped, "
            "not retained with a fabricated label."
        )

    def test_output_is_shorter_than_input_by_at_least_one_row(self):
        raw = _make_ohlcv(n=40)
        result = create_features(raw)

        assert len(result) < len(raw)

    def test_up_column_has_no_null_or_nan_values_in_the_output(self):
        """dropna() must have actually removed every row where Up was
        unknown - none should leak through as NaN either."""
        raw = _make_ohlcv(n=40)
        result = create_features(raw)

        assert result["Up"].notna().all()
        assert not np.isinf(result["Up"]).any()

    def test_up_is_strictly_binary_zero_or_one(self):
        raw = _make_ohlcv(n=40)
        result = create_features(raw)

        assert set(result["Up"].unique()) <= {0.0, 1.0}


class TestTargetAlignment:
    """Up[t] must equal "did Close rise from t to t+1" - never off-by-one,
    never derived from anything other than the next row's Close."""

    def test_up_matches_next_day_direction_for_every_retained_row(self):
        raw = _make_ohlcv(n=60)
        result = create_features(raw)

        # Recompute the expected label directly from the ORIGINAL raw
        # Close series (not from any intermediate feature) and compare by
        # index, so this test is independent of create_features' own
        # internal implementation.
        expected_next_close = raw["Close"].shift(-1)
        expected_up = (expected_next_close > raw["Close"]).astype(float)

        aligned_expected = expected_up.loc[result.index]
        pd.testing.assert_series_equal(
            result["Up"], aligned_expected, check_names=False,
        )

    def test_a_known_up_day_is_labelled_one(self):
        raw = _make_ohlcv(n=40)
        raw = raw.copy()
        # Force a specific, known up-move: row -3 close rises into row -2.
        raw.iloc[-3, raw.columns.get_loc("Close")] = 100.0
        raw.iloc[-2, raw.columns.get_loc("Close")] = 105.0
        result = create_features(raw)

        assert result.loc[raw.index[-3], "Up"] == 1.0

    def test_a_known_down_day_is_labelled_zero(self):
        raw = _make_ohlcv(n=40)
        raw = raw.copy()
        raw.iloc[-3, raw.columns.get_loc("Close")] = 100.0
        raw.iloc[-2, raw.columns.get_loc("Close")] = 95.0
        result = create_features(raw)

        assert result.loc[raw.index[-3], "Up"] == 0.0


class TestInsufficientData:
    def test_too_few_rows_yields_an_empty_frame_not_an_error(self):
        raw = _make_ohlcv(n=5)
        result = create_features(raw)  # rolling(20)/MACD need more history than this
        assert result.empty
