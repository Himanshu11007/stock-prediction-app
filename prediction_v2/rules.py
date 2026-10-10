"""
prediction_v2/rules.py — transparent baseline setups (SHADOW, UNVALIDATED).

Every threshold below is a starting hypothesis, chosen a priori and recorded
with the rule version. None was fitted to any particular stock or to the
October 2026 case review. Changing a threshold creates a new RULE_VERSION;
old predictions keep the version they were made with.

Decision order:
  1. Data quality: any blocking flag -> NO_CALL (reason recorded).
  2. MOMENTUM_CONTINUATION -> UP: strong 5-session gain, outperforming NIFTY,
     close near the day's high, elevated volume, at/near the 20-session high.
  3. BREAKDOWN -> DOWN: the mirror image.
  4. Otherwise NEUTRAL (no setup). NEUTRAL is the expected common answer.

Levels are ATR-based hypotheses (to be tested in the backtest, not
defaults): stop 1.5 ATR, target 2 ATR, trailing 1 ATR from the best close.
Confidence stays None: these are rule matches, not calibrated probabilities.
"""
from __future__ import annotations

from typing import Any, Optional

RULE_VERSION = "baseline-v0.1"

THRESHOLDS: dict[str, float] = {
    "ret_5_min": 0.03,               # |5-session return| at least 3%
    "rs_nifty_5_min": 0.02,          # beats / lags NIFTY by at least 2 points over 5 sessions
    "close_location_up": 0.70,       # closed in the top 30% of the day's range
    "close_location_down": 0.30,
    "abnormal_volume_min": 1.5,      # at least 1.5x the 20-session median volume
    "near_extreme_20": 0.02,         # within 2% of the 20-session high / low
    "stop_atr": 1.5,
    "target_atr": 2.0,
    "trail_atr": 1.0,
}

BLOCKING_FLAGS = ("NO_PRICE_DATA", "STALE_PRICE", "INSUFFICIENT_HISTORY_60", "ZERO_VOLUME_LAST", "LOW_LIQUIDITY",
                  "NO_BENCHMARK", "BENCHMARK_MISALIGNED", "SUSPECT_VOLUME_SPIKE", "POSSIBLE_PRICE_BAND")

REQUIRED = ("close", "ret_5", "rs_nifty_5", "close_location", "abnormal_volume", "atr_pct", "dist_high_20",
            "dist_low_20")


def decide(features: dict[str, Any], flags: list[str], thresholds: Optional[dict] = None) -> dict[str, Any]:
    """Direction, setup, levels and reasons for one stock."""
    t = {**THRESHOLDS, **(thresholds or {})}
    blocking = [f for f in flags if f in BLOCKING_FLAGS]
    if blocking:
        return _no_call(f"data quality: {', '.join(blocking)}", blocking)
    missing = [k for k in REQUIRED if features.get(k) is None]
    if missing:
        return _no_call(f"missing features: {', '.join(missing)}", [f"MISSING:{m}" for m in missing])

    r5, rs5, cl, av = features["ret_5"], features["rs_nifty_5"], features["close_location"], features["abnormal_volume"]
    up = [r5 >= t["ret_5_min"], rs5 >= t["rs_nifty_5_min"], cl >= t["close_location_up"],
          av >= t["abnormal_volume_min"], features["dist_high_20"] >= -t["near_extreme_20"]]
    down = [r5 <= -t["ret_5_min"], rs5 <= -t["rs_nifty_5_min"], cl <= t["close_location_down"],
            av >= t["abnormal_volume_min"], features["dist_low_20"] <= t["near_extreme_20"]]
    if all(up):
        return _call("UP", "MOMENTUM_CONTINUATION", features, t, [
            f"5-session return {r5:+.1%}, {rs5:+.1%} vs NIFTY", f"closed at {cl:.0%} of the day's range",
            f"volume {av:.1f}x the 20-session median", "at or near the 20-session high"])
    if all(down):
        return _call("DOWN", "BREAKDOWN", features, t, [
            f"5-session return {r5:+.1%}, {rs5:+.1%} vs NIFTY", f"closed at {cl:.0%} of the day's range",
            f"volume {av:.1f}x the 20-session median", "at or near the 20-session low"])
    return {"direction": "NEUTRAL", "setup_type": "NONE", "reasons": ["no setup matched"], "quality_flags": [],
            "stop_loss": None, "target": None, "entry_condition": None, "trailing_stop_rule": None,
            "invalidation_condition": None}


def _no_call(reason: str, flags: list[str]) -> dict[str, Any]:
    return {"direction": "NO_CALL", "setup_type": "INSUFFICIENT_DATA", "reasons": [reason], "quality_flags": flags,
            "stop_loss": None, "target": None, "entry_condition": None, "trailing_stop_rule": None,
            "invalidation_condition": None}


def _call(direction: str, setup: str, f: dict, t: dict, reasons: list[str]) -> dict[str, Any]:
    ref, atr = f["close"], f["atr_pct"]
    sign = 1 if direction == "UP" else -1
    return {
        "direction": direction, "setup_type": setup, "reasons": reasons, "quality_flags": [],
        "stop_loss": round(ref * (1 - sign * t["stop_atr"] * atr), 2),
        "target": round(ref * (1 + sign * t["target_atr"] * atr), 2),
        "entry_condition": (f"next session opens within {t['stop_atr'] / 3:.1f} ATR of the reference close "
                            f"({'not gapping below' if sign > 0 else 'not gapping above'} the stop)"),
        "trailing_stop_rule": f"trail {t['trail_atr']:g} ATR from the best close after the target is reached",
        "invalidation_condition": ("close beyond the stop level, or the 5-session relative strength vs NIFTY "
                                   f"turns {'negative' if sign > 0 else 'positive'}"),
    }
