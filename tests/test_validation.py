"""
tests/test_validation.py — Unit tests for storage/recommendation_validation.py

Covers calculate_return() and calculate_success() — the pure functions
that determine whether a recommendation was successful after 5 trading
days. These do not touch the database or the network.
"""
import pytest

from storage.recommendation_validation import calculate_return, calculate_success


class TestCalculateReturn:
    def test_positive_return_calculated_correctly(self):
        # ((110 - 100) / 100) * 100 = 10.0
        assert calculate_return(cmp=100.0, current_price=110.0) == 10.0

    def test_negative_return_calculated_correctly(self):
        # ((90 - 100) / 100) * 100 = -10.0
        assert calculate_return(cmp=100.0, current_price=90.0) == -10.0

    def test_zero_return_when_price_unchanged(self):
        assert calculate_return(cmp=100.0, current_price=100.0) == 0.0

    def test_zero_cmp_returns_zero_safely(self):
        """Guards against division by zero — must not raise."""
        assert calculate_return(cmp=0.0, current_price=50.0) == 0.0

    def test_return_matches_documented_example(self):
        """
        From the task's logging example:
            CMP: 1264, Current Price: 1293, Return: 2.29%
        """
        result = calculate_return(cmp=1264.0, current_price=1293.0)
        assert round(result, 2) == 2.29


class TestCalculateSuccessBuy:
    def test_buy_success_when_price_rises(self):
        ret = calculate_return(cmp=100.0, current_price=110.0)
        assert calculate_success("BUY", ret) == 1

    def test_buy_fails_when_price_falls(self):
        ret = calculate_return(cmp=100.0, current_price=90.0)
        assert calculate_success("BUY", ret) == 0

    def test_strong_buy_uses_same_rule_as_buy(self):
        ret = calculate_return(cmp=100.0, current_price=105.0)
        assert calculate_success("STRONG BUY", ret) == 1


class TestCalculateSuccessSell:
    def test_sell_success_when_price_falls(self):
        ret = calculate_return(cmp=100.0, current_price=90.0)
        assert calculate_success("SELL", ret) == 1

    def test_sell_fails_when_price_rises(self):
        ret = calculate_return(cmp=100.0, current_price=110.0)
        assert calculate_success("SELL", ret) == 0

    def test_strong_sell_uses_same_rule_as_sell(self):
        ret = calculate_return(cmp=100.0, current_price=95.0)
        assert calculate_success("STRONG SELL", ret) == 1


class TestCalculateSuccessHold:
    def test_hold_success_within_plus_3_percent(self):
        ret = calculate_return(cmp=100.0, current_price=102.5)  # +2.5%
        assert calculate_success("HOLD", ret) == 1

    def test_hold_success_within_minus_3_percent(self):
        ret = calculate_return(cmp=100.0, current_price=97.5)   # -2.5%
        assert calculate_success("HOLD", ret) == 1

    def test_hold_success_at_exactly_3_percent_boundary(self):
        ret = calculate_return(cmp=100.0, current_price=103.0)  # exactly +3.0%
        assert calculate_success("HOLD", ret) == 1

    def test_hold_fails_when_return_exceeds_positive_3_percent(self):
        ret = calculate_return(cmp=100.0, current_price=104.0)  # +4.0%
        assert calculate_success("HOLD", ret) == 0

    def test_hold_fails_when_return_exceeds_negative_3_percent(self):
        ret = calculate_return(cmp=100.0, current_price=96.0)   # -4.0%
        assert calculate_success("HOLD", ret) == 0


class TestSignalCaseInsensitivity:
    def test_lowercase_signal_is_handled(self):
        """calculate_success upper-cases the signal internally."""
        ret = calculate_return(cmp=100.0, current_price=110.0)
        assert calculate_success("buy", ret) == 1