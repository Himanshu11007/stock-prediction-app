"""
tests/test_decision_engine.py — Unit tests for utils/decision_engine.py

These tests exercise the real generate_signal() function against the real
pillar weights and signal thresholds defined in config.py. They do not
mock or override any threshold — if you tune a weight or a bucket boundary
later, these tests will tell you whether the qualitative behaviour they
describe (strong bullish input -> BUY/STRONG BUY, etc.) still holds.
"""
import pandas as pd
import pytest

from config import (
    W_ML_DIR, W_ML_CONF, W_TECH, W_NEWS,
    W_VOLUME, W_REGIME, W_TIMEFRAME, W_MOMENTUM,
)
from utils.decision_engine import generate_signal


def _make_data(rsi=55, macd_hist=0.0, macd_cross=0.0, ema_cross=0.0,
               price_vs_ema20=0.0, adx=20, bb_position=0.5,
               close=100.0, volume=1_000_000) -> pd.DataFrame:
    """
    Build a minimal single-row feature DataFrame with the columns
    _technical_score(), _volume_score(), and _momentum_score() read.
    All values default to neutral so a caller only needs to override the
    columns relevant to the scenario being tested.
    """
    return pd.DataFrame({
        "RSI":             [rsi],
        "MACD_Hist":       [macd_hist],
        "MACD_Cross":      [macd_cross],
        "EMA_Cross":       [ema_cross],
        "Price_vs_EMA20":  [price_vs_ema20],
        "ADX":             [adx],
        "BB_Position":     [bb_position],
        "Close":           [close],
        "Volume":          [volume],
        "Price_Change":    [0.01],
        "Vol_Breakout":    [0.0],
    })


class TestPillarWeights:
    def test_weights_sum_to_one(self):
        """The module asserts this at import time, but verify explicitly too."""
        total = (
            W_ML_DIR + W_ML_CONF + W_TECH + W_NEWS
            + W_VOLUME + W_REGIME + W_TIMEFRAME + W_MOMENTUM
        )
        assert abs(total - 1.0) < 1e-9, f"Pillar weights sum to {total}, expected 1.0"

    def test_module_imports_without_assertion_error(self):
        """
        utils/decision_engine.py asserts weight sum == 1.0 at import time.
        If config.py weights are ever changed to not sum to 1.0, importing
        the module itself will raise AssertionError — this test documents
        that contract explicitly.
        """
        import importlib
        import utils.decision_engine as de
        importlib.reload(de)  # re-run the module-level assertion


class TestBullishScenarios:
    def test_high_bullish_inputs_produce_buy_or_strong_buy(self):
        """
        Strong bullish across every pillar: correct prediction, high
        confidence, bullish technicals, positive news, bullish regime,
        bullish timeframe.
        """
        data = _make_data(
            rsi=65, macd_hist=0.5, macd_cross=1.0,
            ema_cross=1.0, price_vs_ema20=0.02, adx=35, bb_position=0.3,
        )
        signal, score, reason, factors = generate_signal(
            prediction=1,
            confidence=95.0,
            news_score=0.8,
            timeframe_score=1.0,
            data=data,
            regime_info={"regime": "Bullish", "regime_score": 1.0, "reason": "strong uptrend"},
        )
        assert signal in ("BUY", "STRONG BUY"), (
            f"Expected BUY or STRONG BUY for strong bullish inputs, got {signal} (score={score})"
        )
        assert 0.0 <= score <= 1.0
        assert isinstance(factors, list) and len(factors) > 0


class TestBearishScenarios:
    def test_strong_bearish_inputs_produce_sell_or_strong_sell(self):
        """Strong bearish across every pillar — mirror of the bullish case."""
        data = _make_data(
            rsi=25, macd_hist=-0.5, macd_cross=-1.0,
            ema_cross=-1.0, price_vs_ema20=-0.02, adx=35, bb_position=0.85,
        )
        signal, score, reason, factors = generate_signal(
            prediction=0,
            confidence=95.0,
            news_score=-0.8,
            timeframe_score=-1.0,
            data=data,
            regime_info={"regime": "Bearish", "regime_score": -1.0, "reason": "strong downtrend"},
        )
        assert signal in ("SELL", "STRONG SELL"), (
            f"Expected SELL or STRONG SELL for strong bearish inputs, got {signal} (score={score})"
        )


class TestMixedScenarios:
    def test_mixed_inputs_produce_hold(self):
        """
        Conflicting signals (bullish prediction but bearish technicals/news/
        regime/timeframe) should land in the middle HOLD bucket rather than
        a confident BUY or SELL.
        """
        data = _make_data(
            rsi=50, macd_hist=0.0, macd_cross=0.0,
            ema_cross=-1.0, price_vs_ema20=0.0, adx=15, bb_position=0.5,
        )
        signal, score, reason, factors = generate_signal(
            prediction=1,
            confidence=55.0,           # low-ish confidence
            news_score=0.0,            # neutral news
            timeframe_score=0.0,       # neutral timeframe
            data=data,
            regime_info={"regime": "Sideways", "regime_score": 0.0, "reason": "no clear trend"},
        )
        assert signal == "HOLD", (
            f"Expected HOLD for mixed/neutral inputs, got {signal} (score={score})"
        )


class TestReturnShape:
    def test_returns_expected_tuple_shape(self):
        data = _make_data()
        result = generate_signal(
            prediction=1, confidence=60.0, news_score=0.0,
            timeframe_score=0.0, data=data, regime_info=None,
        )
        assert len(result) == 4
        signal, score, reason, factors = result
        assert isinstance(signal, str)
        assert isinstance(score, float)
        assert isinstance(reason, str)
        assert isinstance(factors, list)

    def test_handles_missing_data_gracefully(self):
        """generate_signal must not crash when data=None (e.g. data load failed)."""
        signal, score, reason, factors = generate_signal(
            prediction=1, confidence=60.0, news_score=0.0,
            timeframe_score=0.0, data=None, regime_info=None,
        )
        assert signal in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")