"""
prediction_v2/outcomes.py — 1/3/5-session outcomes and shadow exit states.

Outcome window for horizon h: the h trading sessions starting with the
prediction's target session. start = reference price (close at the cutoff),
end = close of the h-th session. Returns are DIRECTION-ADJUSTED for UP/DOWN
calls (DOWN profits when the price falls). NEUTRAL and NO_CALL predictions
get NOT_APPLICABLE outcomes that still record the stock's return, so
coverage and "what we declined" can be analysed.

Costs: COST_BPS round-trip transaction cost + SLIPPAGE_BPS, subtracted from
directional returns (provisional; the backtest varies them 10-20 bps).

Idempotent: UNIQUE(prediction_id, horizon) - re-running inserts nothing new.
Outcomes are only written once all h sessions have a final daily bar.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Optional

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

import masters.service as masters
from db.models.prediction import ExitState, ExitTransition, Prediction, PredictionOutcome, PredictionRun
from db.models.stock import Company
from prediction_v2 import exits
from prediction_v2.features import slice_to_cutoff
from prediction_v2.service import NIFTY, _default_fetch
from utils.market_session import is_daily_bar_complete

EVALUATOR_VERSION = "outcomes-v0.1"
HORIZONS = (1, 3, 5)
COST_BPS = 15.0
SLIPPAGE_BPS = 5.0
LOOKBACK_DAYS = 30


def _window(df: Optional[pd.DataFrame], start: dt.date, h: int) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    w = df[df.index.normalize() >= pd.Timestamp(start)].iloc[:h]
    return w if len(w) == h else None


def _ret(w: Optional[pd.DataFrame], start_price: Optional[float]) -> Optional[float]:
    if w is None or not start_price:
        return None
    return float(w["Close"].iloc[-1] / start_price - 1)


def outcome_for(pred: Prediction, target: dt.date, h: int, df: Optional[pd.DataFrame], nifty: Optional[pd.DataFrame],
                sector_dfs: list[pd.DataFrame], now: dt.datetime) -> Optional[PredictionOutcome]:
    """Outcome row, or None when the window is not complete yet."""
    w = _window(df, target, h)
    if w is None or not is_daily_bar_complete(w.index[-1].date(), now):
        return None
    start = pred.reference_price
    base = dict(prediction_id=pred.prediction_id, horizon_sessions=h, evaluator_version=EVALUATOR_VERSION,
                start_date=pred.reference_date, end_date=w.index[-1].date().isoformat(), evaluated_at=now)
    if not start:
        return PredictionOutcome(**base, outcome_status="INSUFFICIENT_DATA", note="no reference price")
    r = _ret(w, start)
    nw = _window(nifty, target, h)
    nr = _ret(nw, float(slice_to_cutoff(nifty, target - dt.timedelta(days=1))["Close"].iloc[-1])) \
        if nw is not None and slice_to_cutoff(nifty, target - dt.timedelta(days=1)) is not None else None
    sector = []
    for sdf in sector_dfs:
        sw, prev = _window(sdf, target, h), slice_to_cutoff(sdf, target - dt.timedelta(days=1))
        if sw is not None and prev is not None:
            sector.append(float(sw["Close"].iloc[-1] / prev["Close"].iloc[-1] - 1))
    sr = sum(sector) / len(sector) if len(sector) >= 3 else None
    row = PredictionOutcome(**base, start_price=start, end_price=float(w["Close"].iloc[-1]), stock_return=r,
                            nifty_return=nr, sector_return=sr,
                            excess_return_nifty=(r - nr) if nr is not None else None,
                            cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS, outcome_status="NOT_APPLICABLE")
    if pred.direction in ("UP", "DOWN"):
        s = 1 if pred.direction == "UP" else -1
        hi, lo = float(w["High"].max()), float(w["Low"].min())
        row.directional_return = s * r
        row.mfe = (hi / start - 1) if s > 0 else (1 - lo / start)
        row.mae = (lo / start - 1) if s > 0 else (1 - hi / start)
        row.cost_adjusted_return = s * r - (COST_BPS + SLIPPAGE_BPS) / 1e4
        row.hit = row.directional_return > 0
        row.outcome_status = "EVALUATED"
        atr = (pred.features or {}).get("atr_pct")
        if atr and pred.stop_loss:
            feats = pred.features or {}
            pos = exits.Position(direction=pred.direction, reference=start, atr=atr * start, stop=pred.stop_loss,
                                 target=pred.target, ref_high=feats.get("high"), ref_low=feats.get("low"))
            final, trans = exits.simulate(pos, [exits.Bar(float(b.Open), float(b.High), float(b.Low), float(b.Close))
                                                for b in w.itertuples()])
            row.simulated_exit_reason = trans[-1][1].reason if final.state == "EXIT" else f"OPEN_{final.state}"
    return row


def evaluate_due(engine, now: Optional[dt.datetime] = None, fetch: Optional[Callable] = None) -> dict[str, int]:
    """Write every outcome whose window has completed. Safe to repeat."""
    now = now or dt.datetime.now(dt.timezone.utc)
    since = (now - dt.timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    with Session(engine) as s:
        preds = s.exec(select(Prediction, PredictionRun).join(PredictionRun, PredictionRun.run_id == Prediction.run_id)
                       .where(PredictionRun.status == "COMPLETED", PredictionRun.target_session_date >= since)).all()
        done = {(o.prediction_id, o.horizon_sessions) for o in s.exec(select(PredictionOutcome)).all()}
        sectors = {c.symbol: c.sector for c in s.exec(select(Company)).all()}
    todo = [(p, r) for p, r in preds if any((p.prediction_id, h) not in done for h in HORIZONS)]
    if not todo:
        return {"evaluated": 0, "pending": 0}
    symbols = sorted({p.symbol for p, _ in todo})
    peers: dict[str, list[str]] = {}
    for sym, sec in sectors.items():
        if sec:
            peers.setdefault(sec, []).append(sym)
    peer_syms = sorted({x for p, _ in todo for x in peers.get(sectors.get(p.symbol) or "", []) if x != p.symbol})
    data = (fetch or _default_fetch)(sorted(set(symbols) | set(peer_syms)) + [NIFTY])
    written = pending = 0
    with Session(engine) as s:
        for p, r in todo:
            target = dt.date.fromisoformat(r.target_session_date)
            sec_dfs = [data[x] for x in peers.get(sectors.get(p.symbol) or "", [])
                       if x != p.symbol and data.get(x) is not None]
            for h in HORIZONS:
                if (p.prediction_id, h) in done:
                    continue
                row = outcome_for(p, target, h, data.get(p.symbol), data.get(NIFTY), sec_dfs, now)
                if row is None:
                    pending += 1
                    continue
                s.add(row)
                try:
                    s.commit()
                    written += 1
                except IntegrityError:          # written concurrently by another worker
                    s.rollback()
    return {"evaluated": written, "pending": pending}


def update_exit_states(engine, now: Optional[dt.datetime] = None, fetch: Optional[Callable] = None) -> dict[str, int]:
    """Shadow mode: advance the exit state of recent directional predictions
    by each completed session since their target date. Writes transitions
    (unique per prediction/session/state); sends nothing to anyone."""
    now = now or dt.datetime.now(dt.timezone.utc)
    since = (now - dt.timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    with Session(engine) as s:
        rows = s.exec(select(Prediction, PredictionRun).join(PredictionRun, PredictionRun.run_id == Prediction.run_id)
                      .where(PredictionRun.status == "COMPLETED", PredictionRun.target_session_date >= since,
                             Prediction.direction.in_(("UP", "DOWN")))).all()
        states = {e.prediction_id: e for e in s.exec(select(ExitState)).all()}
    live = [(p, r) for p, r in rows if states.get(p.prediction_id) is None or states[p.prediction_id].state != "EXIT"]
    if not live:
        return {"updated": 0, "transitions": 0}
    data = (fetch or _default_fetch)(sorted({p.symbol for p, _ in live}) + [NIFTY])
    updated = transitions = 0
    with Session(engine) as s:
        for p, r in live:
            df = data.get(p.symbol)
            atr = (p.features or {}).get("atr_pct")
            if df is None or not atr or not p.reference_price or not p.stop_loss:
                continue
            est = s.exec(select(ExitState).where(ExitState.prediction_id == p.prediction_id)).first()
            target = dt.date.fromisoformat(r.target_session_date)
            bars = df[df.index.normalize() >= pd.Timestamp(target)]
            bars = bars[[is_daily_bar_complete(d.date(), now) for d in bars.index]]
            done_through = est.last_session_date if est else None
            feats = p.features or {}
            pos = exits.Position(direction=p.direction, reference=p.reference_price, atr=atr * p.reference_price,
                                 stop=p.stop_loss, target=p.target, ref_high=feats.get("high"),
                                 ref_low=feats.get("low"))
            prev_close = p.reference_price
            cur_state = "HOLD"
            for d, b in bars.iterrows():                        # replay from the start (deterministic)
                pos, t = exits.step(pos, exits.Bar(float(b.Open), float(b.High), float(b.Low), float(b.Close),
                                                   prev_close=prev_close))
                prev_close = float(b.Close)
                day = d.date().isoformat()
                if t and (done_through is None or day > done_through):
                    s.add(ExitTransition(prediction_id=p.prediction_id, session_date=day, from_state=cur_state,
                                         to_state=t.to_state, reason=t.reason, price=t.price,
                                         rule_version=exits.EXIT_RULE_VERSION))
                    transitions += 1
                if t:
                    cur_state = t.to_state
                if pos.state == "EXIT":
                    break
            if est is None:
                est = ExitState(prediction_id=p.prediction_id, rule_version=exits.EXIT_RULE_VERSION)
            est.state, est.stop_level, est.target_level = pos.state, pos.stop, pos.target
            est.peak_price, est.updated_at = pos.best_close, now
            last = s.exec(select(ExitTransition).where(ExitTransition.prediction_id == p.prediction_id)
                          .order_by(ExitTransition.session_date.desc())).first()
            est.reason = last.reason if last else None
            est.last_session_date = bars.index[-1].date().isoformat() if len(bars) else est.last_session_date
            s.add(est)
            try:
                s.commit()
                updated += 1
            except IntegrityError:
                s.rollback()
    return {"updated": updated, "transitions": transitions}
