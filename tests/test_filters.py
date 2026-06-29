"""
tests/test_filters.py — Unit tests for scanner/filters.py

Tests import MIN_ACCURACY, MIN_CONFIDENCE, MIN_CONFLUENCE_SCORE directly
from config.py rather than hardcoding numbers, so these tests verify the
*behaviour* (BUY signals are gated by these thresholds) without locking
in specific threshold values that may be tuned later.
"""
import pandas as pd
import pytest

from config import MIN_ACCURACY, MIN_CONFIDENCE, MIN_CONFLUENCE_SCORE
from scanner.filters import passes_quality_filters


def _make_data(rows: int = 25) -> pd.DataFrame:
    """
    Minimal DataFrame with no Volume/RSI/Volatility columns, so only the
    accuracy/confidence/confluence-score gates are exercised — isolating
    the specific filter being tested in each case.
    """
    return pd.DataFrame({"Close": [100.0] * rows}, index=pd.RangeIndex(rows))


class TestAccuracyFilter:
    def test_buy_below_min_accuracy_is_rejected(self):
        data = _make_data()
        result = passes_quality_filters(
            data, signal="BUY",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=MIN_ACCURACY - 0.01,   # just below threshold
            score=MIN_CONFLUENCE_SCORE + 0.1,
        )
        assert result is False

    def test_buy_at_or_above_min_accuracy_passes_accuracy_gate(self):
        data = _make_data()
        result = passes_quality_filters(
            data, signal="BUY",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=MIN_ACCURACY + 0.05,
            score=MIN_CONFLUENCE_SCORE + 0.1,
        )
        assert result is True

    def test_strong_buy_below_min_accuracy_is_rejected(self):
        data = _make_data()
        result = passes_quality_filters(
            data, signal="STRONG BUY",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=MIN_ACCURACY - 0.01,
            score=MIN_CONFLUENCE_SCORE + 0.1,
        )
        assert result is False


class TestConfidenceFilter:
    def test_below_min_confidence_is_rejected_regardless_of_signal(self):
        data = _make_data()
        for signal in ("BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL"):
            result = passes_quality_filters(
                data, signal=signal,
                confidence=MIN_CONFIDENCE - 5,
                accuracy=MIN_ACCURACY + 0.1,
                score=MIN_CONFLUENCE_SCORE + 0.1,
            )
            assert result is False, f"{signal} should be rejected below MIN_CONFIDENCE"


class TestConfluenceScoreFilter:
    def test_buy_below_min_confluence_score_is_rejected(self):
        data = _make_data()
        result = passes_quality_filters(
            data, signal="BUY",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=MIN_ACCURACY + 0.1,
            score=MIN_CONFLUENCE_SCORE - 0.01,
        )
        assert result is False


class TestHoldAndSellAreNotBuyGated:
    def test_hold_is_not_rejected_by_buy_only_filters(self):
        """
        HOLD signals should not be rejected by the accuracy or confluence
        score gates — those only apply to bullish (_BULLISH) signals.
        Low accuracy / low score should NOT reject a HOLD.
        """
        data = _make_data()
        result = passes_quality_filters(
            data, signal="HOLD",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=0.01,            # deliberately terrible accuracy
            score=0.01,               # deliberately terrible score
        )
        assert result is True, "HOLD should not be gated by the BUY-only accuracy/score filters"

    def test_sell_is_not_rejected_by_buy_only_filters(self):
        """SELL/STRONG SELL should also bypass the accuracy and confluence gates."""
        data = _make_data()
        result = passes_quality_filters(
            data, signal="SELL",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=0.01,
            score=0.01,
        )
        assert result is True, "SELL should not be gated by the BUY-only accuracy/score filters"

    def test_strong_sell_is_not_rejected_by_buy_only_filters(self):
        data = _make_data()
        result = passes_quality_filters(
            data, signal="STRONG SELL",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=0.01,
            score=0.01,
        )
        assert result is True


class TestRsiGateOnlyAppliesToBullish:
    def test_rsi_outside_range_rejects_buy(self):
        from config import RSI_BUY_MIN, RSI_BUY_MAX
        data = _make_data()
        data["RSI"] = [RSI_BUY_MAX + 5] * len(data)  # overbought, outside range
        result = passes_quality_filters(
            data, signal="BUY",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=MIN_ACCURACY + 0.1,
            score=MIN_CONFLUENCE_SCORE + 0.1,
        )
        assert result is False

    def test_rsi_outside_range_does_not_reject_sell(self):
        from config import RSI_BUY_MAX
        data = _make_data()
        data["RSI"] = [RSI_BUY_MAX + 5] * len(data)
        result = passes_quality_filters(
            data, signal="SELL",
            confidence=MIN_CONFIDENCE + 10,
            accuracy=0.01,
            score=0.01,
        )
        assert result is True, "RSI gate must only apply to bullish signals"