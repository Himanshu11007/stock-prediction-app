"""
tests/test_risk.py — Unit tests for utils/risk.py

calculate_risk() requires a DataFrame with "Close" and "ATR" columns.
These tests verify the geometric relationships that must always hold for
a sane risk panel, rather than locking in specific multiplier constants.
"""
import pandas as pd

from utils.risk import calculate_risk


def _make_data(close: float = 100.0, atr: float = 2.0, rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "Close": [close] * rows,
        "ATR":   [atr] * rows,
    })


class TestLongSignals:
    def test_stop_loss_below_close_for_buy(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "BUY")
        assert risk["stop_loss"] < risk["close"]

    def test_target_above_close_for_buy(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "BUY")
        assert risk["target"] > risk["close"]

    def test_stop_loss_below_close_for_strong_buy(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "STRONG BUY")
        assert risk["stop_loss"] < risk["close"]
        assert risk["target"] > risk["close"]


class TestShortSignals:
    def test_stop_loss_above_close_for_sell(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "SELL")
        assert risk["stop_loss"] > risk["close"]

    def test_target_below_close_for_sell(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "SELL")
        assert risk["target"] < risk["close"]

    def test_strong_sell_levels_are_directionally_correct(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "STRONG SELL")
        assert risk["stop_loss"] > risk["close"]
        assert risk["target"] < risk["close"]


class TestHoldSignal:
    def test_hold_provides_symmetric_context_levels(self):
        """HOLD has no directional conviction but should still return levels."""
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "HOLD")
        assert risk["stop_loss"] < risk["close"] < risk["target"]


class TestRiskRewardRatio:
    def test_rr_ratio_is_positive_for_long(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "BUY")
        assert risk["rr_ratio"] > 0

    def test_rr_ratio_is_positive_for_short(self):
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "SELL")
        assert risk["rr_ratio"] > 0

    def test_rr_ratio_reflects_target_to_stop_distance_ratio(self):
        """
        Given the documented 2:1 minimum reward:risk design (target distance
        is 2x the stop distance), rr_ratio should be approximately 2.0 for
        a plain BUY (non-STRONG) signal.
        """
        data = _make_data(close=100.0, atr=2.0)
        risk = calculate_risk(data, "BUY")
        assert risk["rr_ratio"] == 2.0


class TestMissingDataFallback:
    def test_empty_dataframe_returns_safe_fallback(self):
        """An empty DataFrame must not raise — it should return a safe fallback dict."""
        data = pd.DataFrame({"Close": [], "ATR": []})
        risk = calculate_risk(data, "BUY")
        assert risk["close"] == 0.0
        assert risk["stop_loss"] is None
        assert risk["target"] is None
        assert risk["rr_ratio"] is None

    def test_missing_atr_column_returns_safe_fallback(self):
        """Missing the ATR column entirely must not crash calculate_risk."""
        data = pd.DataFrame({"Close": [100.0, 101.0]})
        risk = calculate_risk(data, "BUY")
        # Falls into the except branch — close should still be derivable
        assert risk["stop_loss"] is None
        assert risk["target"] is None