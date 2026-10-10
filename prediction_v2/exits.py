"""
prediction_v2/exits.py — shadow exit-state machine (versioned, no alerts).

States: HOLD -> TIGHTEN -> PARTIAL_EXIT -> EXIT (EXIT is terminal). Any
state may go straight to EXIT on a stop or invalidation. Legal transitions
are enforced; a repeated signal for the current state is not a transition.

`step()` evaluates one completed daily bar for one directional prediction.
It is pure (no I/O) so it can be replayed in backtests; `update_states()`
persists transitions in shadow mode. Nothing here notifies users or touches
holdings.

Order of checks within a bar (conservative: risk first):
  1. missing bar                      -> no change
  2. opening gap through the stop     -> EXIT STOP_LOSS at the open (gap risk)
  3. intraday breach of the stop      -> EXIT STOP_LOSS (or TRAILING_STOP) at the stop
  4. close beyond the reference bar's
     extreme (event-day low/high)     -> EXIT SUPPORT_BREAKDOWN
  5. external invalidation signals    -> EXIT THESIS_INVALIDATED / MACRO_REVERSAL
  6. target reached                   -> PARTIAL_EXIT TARGET_REACHED, then trail
  7. failed breakout (early close
     back through the reference)      -> EXIT FAILED_BREAKOUT
  8. catalyst exhaustion (after a
     >= 2 ATR day, weak close)        -> TIGHTEN CATALYST_EXHAUSTION
  9. distribution (heavy volume,
     weak close)                      -> TIGHTEN DISTRIBUTION
 10. relative-strength deterioration  -> TIGHTEN RELATIVE_STRENGTH_DETERIORATION
 11. holding period reached           -> EXIT TIME_STOP
All thresholds are hypotheses in EXIT_CONFIG and carry EXIT_RULE_VERSION.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

EXIT_RULE_VERSION = "exit-shadow-0.1"

EXIT_CONFIG = {
    "failed_breakout_sessions": 2,      # close back through the reference within N sessions
    "failed_breakout_atr": 0.5,
    "exhaustion_move_atr": 2.0,
    "exhaustion_close_location": 0.3,
    "distribution_volume": 2.0,
    "distribution_close_location": 0.3,
    "rs_deterioration": 0.02,           # underperforms NIFTY by 2 points in one session
    "max_holding_sessions": 5,
    "trail_atr": 1.0,
}

LEGAL = {"HOLD": {"TIGHTEN", "PARTIAL_EXIT", "EXIT"}, "TIGHTEN": {"PARTIAL_EXIT", "EXIT"},
         "PARTIAL_EXIT": {"EXIT"}, "EXIT": set()}

REASONS = ("STOP_LOSS", "TRAILING_STOP", "TARGET_REACHED", "SUPPORT_BREAKDOWN", "FAILED_BREAKOUT", "DISTRIBUTION",
           "RELATIVE_STRENGTH_DETERIORATION", "CATALYST_EXHAUSTION", "MACRO_REVERSAL", "THESIS_INVALIDATED",
           "TIME_STOP")


@dataclass(frozen=True)
class Position:
    direction: str                     # UP (long thesis) or DOWN (short thesis)
    reference: float                   # price at the prediction cutoff
    atr: float                         # ATR in price units at the cutoff
    stop: float
    target: Optional[float]
    ref_high: Optional[float] = None   # reference (cutoff) bar extremes = event-day levels
    ref_low: Optional[float] = None
    state: str = "HOLD"
    best_close: Optional[float] = None
    sessions: int = 0
    trailing: bool = False


@dataclass(frozen=True)
class Bar:
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume_ratio: Optional[float] = None       # volume / 20-session median
    nifty_return: Optional[float] = None
    prev_close: Optional[float] = None


@dataclass(frozen=True)
class Signals:
    thesis_invalidated: bool = False
    macro_reversal: bool = False


@dataclass(frozen=True)
class Transition:
    to_state: str
    reason: str
    price: Optional[float]


def _move(pos: Position, to: str, reason: str, price: Optional[float]) -> tuple[Position, Optional[Transition]]:
    if to == pos.state or to not in LEGAL[pos.state]:
        return pos, None
    return replace(pos, state=to), Transition(to, reason, price)


def step(pos: Position, bar: Optional[Bar], signals: Signals = Signals(),
         cfg: Optional[dict] = None) -> tuple[Position, Optional[Transition]]:
    c = {**EXIT_CONFIG, **(cfg or {})}
    if pos.state == "EXIT":
        return pos, None
    if bar is None or None in (bar.open, bar.high, bar.low, bar.close):
        return pos, None                                     # missing data: never guess
    s = 1 if pos.direction == "UP" else -1
    pos = replace(pos, sessions=pos.sessions + 1)
    stop_reason = "TRAILING_STOP" if pos.trailing else "STOP_LOSS"

    # 2-3: stop (gap first)
    if s * (bar.open - pos.stop) <= 0:
        return _move(pos, "EXIT", stop_reason, bar.open)
    breach = bar.low if s > 0 else bar.high
    if s * (breach - pos.stop) <= 0:
        return _move(pos, "EXIT", stop_reason, pos.stop)
    # 4: event-day reference level
    level = pos.ref_low if s > 0 else pos.ref_high
    if level is not None and s * (bar.close - level) < 0:
        return _move(pos, "EXIT", "SUPPORT_BREAKDOWN", bar.close)
    # 5: external signals
    if signals.thesis_invalidated:
        return _move(pos, "EXIT", "THESIS_INVALIDATED", bar.close)
    if signals.macro_reversal:
        return _move(pos, "EXIT", "MACRO_REVERSAL", bar.close)

    best = bar.close if pos.best_close is None else (max(pos.best_close, bar.close) if s > 0
                                                     else min(pos.best_close, bar.close))
    pos = replace(pos, best_close=best)
    if pos.trailing:                                         # ratchet the trailing stop
        trail = best - s * c["trail_atr"] * pos.atr
        if s * (trail - pos.stop) > 0:
            pos = replace(pos, stop=trail)
    # 6: target
    if pos.target is not None and not pos.trailing and s * ((bar.high if s > 0 else bar.low) - pos.target) >= 0:
        trail = best - s * c["trail_atr"] * pos.atr
        pos = replace(pos, trailing=True, stop=trail if s * (trail - pos.stop) > 0 else pos.stop)
        return _move(pos, "PARTIAL_EXIT", "TARGET_REACHED", pos.target)
    # 7: failed breakout
    if pos.sessions <= c["failed_breakout_sessions"] and \
            s * (bar.close - (pos.reference - s * c["failed_breakout_atr"] * pos.atr)) < 0:
        return _move(pos, "EXIT", "FAILED_BREAKOUT", bar.close)
    rng = bar.high - bar.low
    cl = (bar.close - bar.low) / rng if rng > 0 else None
    weak = None if cl is None else (cl if s > 0 else 1 - cl)       # 0 = closed against the thesis
    # 8: catalyst exhaustion
    if bar.prev_close and weak is not None and pos.sessions >= 2 and weak <= c["exhaustion_close_location"] \
            and s * (bar.prev_close - pos.reference) >= c["exhaustion_move_atr"] * pos.atr:
        return _move(pos, "TIGHTEN", "CATALYST_EXHAUSTION", bar.close)
    # 9: distribution
    if bar.volume_ratio is not None and weak is not None and bar.volume_ratio >= c["distribution_volume"] \
            and weak <= c["distribution_close_location"]:
        return _move(pos, "TIGHTEN", "DISTRIBUTION", bar.close)
    # 10: relative strength
    if bar.nifty_return is not None and bar.prev_close:
        own = bar.close / bar.prev_close - 1
        if s * (own - bar.nifty_return) <= -c["rs_deterioration"]:
            return _move(pos, "TIGHTEN", "RELATIVE_STRENGTH_DETERIORATION", bar.close)
    # 11: time stop
    if pos.sessions >= c["max_holding_sessions"]:
        return _move(pos, "EXIT", "TIME_STOP", bar.close)
    return pos, None


def simulate(pos: Position, bars: list[Optional[Bar]], cfg: Optional[dict] = None
             ) -> tuple[Position, list[tuple[int, Transition]]]:
    """Replay bars; returns the final position and (bar index, transition) list."""
    out = []
    for i, b in enumerate(bars):
        pos, t = step(pos, b, cfg=cfg)
        if t:
            out.append((i, t))
        if pos.state == "EXIT":
            break
    return pos, out
