"""
prediction_v2/service.py — immutable prediction snapshots.

Run types (all times Asia/Kolkata, trading days only):

  TODAY_PREOPEN    before 09:15. Data cutoff: the previous session's close.
                   Target: today.
  TODAY_CONFIRMED  09:45-15:30, only when an intraday feed is enabled
                   (PREDICTION_V2_INTRADAY_ENABLED) - otherwise SKIPPED, never
                   simulated. Daily features as of the previous close plus the
                   opening move up to the cutoff. Target: today.
  TOMORROW_EOD     after 16:00, once today's NIFTY bar exists. Data cutoff:
                   today's close. Target: the next trading day.

Guarantees:
  * one published run per (run_type, trading date): idempotency_key is
    UNIQUE; a duplicate trigger returns the existing run. A FAILED run keeps
    its record but releases the key, so a later trigger can retry;
  * one RUNNING run per run type across processes (partial unique index);
  * a run is computed fully in memory, then written in one transaction;
    COMPLETED runs and their predictions are never modified afterwards;
  * a run that could not evaluate at least MIN_USABLE_RATIO of its universe
    is FAILED and writes no predictions (no partial snapshot is published).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any, Callable, Optional

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

import masters.service as masters
from db.models.prediction import FeatureSnapshot, Prediction, PredictionRun
from db.models.stock import Company
from events.ingest import available_events
from prediction_v2 import ENGINE_VERSION, calendar, features as feat, rules
from prediction_v2.universe import members
from utils.logger import get_logger
from utils.market_session import DAILY_BAR_FINAL_AFTER, IST, NSE_CLOSE, NSE_OPEN, is_trading_day, now_ist

logger = get_logger(__name__)

NIFTY = "^NSEI"
MIN_USABLE_RATIO = 0.5
CONFIRMED_EARLIEST = dt.time(9, 45)
HISTORY_PERIOD = "6mo"            # > 61 sessions needed by the rules
BATCH = 50
STALE_RUNNING_AFTER = dt.timedelta(hours=2)


class RunSkipped(Exception):
    """The run should not happen now (calendar, timing, data, duplicate)."""


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _default_fetch(symbols: list[str]) -> dict[str, Optional[pd.DataFrame]]:
    from fundamentals.provider import fetch_price_history
    out: dict[str, Optional[pd.DataFrame]] = {}
    for i in range(0, len(symbols), BATCH):
        out.update(fetch_price_history(symbols[i:i + BATCH], period=HISTORY_PERIOD))
    return out


def plan(run_type: str, now: dt.datetime, holidays: list[str]) -> dict[str, Any]:
    """Trading date, target session and cutoff for a run, or RunSkipped."""
    local = now_ist(now)
    today = local.date()
    if not is_trading_day(today, holidays):
        raise RunSkipped(f"{today} is not a trading day")
    prev = calendar.previous_trading_day(today, holidays)
    if run_type == "TODAY_PREOPEN":
        if local.time() >= NSE_OPEN:
            raise RunSkipped("pre-open snapshot must be created before 09:15 IST")
        return {"trading_date": today, "target": today, "cutoff_session": prev, "expected_session": prev}
    if run_type == "TODAY_CONFIRMED":
        if not (CONFIRMED_EARLIEST <= local.time() < NSE_CLOSE):
            raise RunSkipped("confirmed snapshot is created between 09:45 and 15:30 IST")
        return {"trading_date": today, "target": today, "cutoff_session": prev, "expected_session": prev}
    if run_type == "TOMORROW_EOD":
        if local.time() < DAILY_BAR_FINAL_AFTER:
            raise RunSkipped(f"end-of-day snapshot needs final daily bars (after {DAILY_BAR_FINAL_AFTER:%H:%M} IST)")
        return {"trading_date": today, "target": calendar.next_trading_day(today, holidays), "cutoff_session": today,
                "expected_session": today}
    raise ValueError(f"unknown run type {run_type}")


def _config(run_type: str, holidays: list[str], intraday: bool) -> dict[str, Any]:
    return {"rule_version": rules.RULE_VERSION, "thresholds": rules.THRESHOLDS,
            "feature_set_version": feat.FEATURE_SET_VERSION, "blocking_flags": list(rules.BLOCKING_FLAGS),
            "holiday_calendar_configured": bool(holidays), "intraday_confirmation": intraday,
            "min_usable_ratio": MIN_USABLE_RATIO, "run_type": run_type, **_news_config()}


def _news_config() -> dict[str, Any]:
    from config import PREDICTION_V2_NEWS_ENABLED
    from catalysts import CLASSIFIER_VERSION, NEWS_RULE_VERSION
    from catalysts.transmission import TRANSMISSION_VERSION
    return {"news_enabled": PREDICTION_V2_NEWS_ENABLED, "news_rule_version": NEWS_RULE_VERSION,
            "news_classifier_version": CLASSIFIER_VERSION, "transmission_version": TRANSMISSION_VERSION}


def _price_move(df: Optional[pd.DataFrame], nifty: Optional[pd.DataFrame], cutoff_session: dt.date,
                atr_pct: Optional[float]) -> Callable[[dt.datetime], Optional[tuple[float, float]]]:
    """Excess return over NIFTY from the last close before a catalyst became
    available up to the cutoff close (for the news 'already priced' check)."""
    def move(since: dt.datetime) -> Optional[tuple[float, float]]:
        if df is None or nifty is None or not atr_pct:
            return None
        local = now_ist(since)
        base_day = local.date() if local.time() >= NSE_CLOSE else local.date() - dt.timedelta(days=1)
        d, n = feat.slice_to_cutoff(df, cutoff_session), feat.slice_to_cutoff(nifty, cutoff_session)
        if d is None or n is None:
            return None
        d0, n0 = d[d.index.normalize() <= pd.Timestamp(base_day)], n[n.index.normalize() <= pd.Timestamp(base_day)]
        if d0.empty or n0.empty:
            return None
        r = float(d["Close"].iloc[-1] / d0["Close"].iloc[-1] - 1)
        rn = float(n["Close"].iloc[-1] / n0["Close"].iloc[-1] - 1)
        return r - rn, float(atr_pct)
    return move


def _confirm(decision: dict, intraday: Optional[pd.DataFrame], prev_close: Optional[float],
             atr_pct: Optional[float], cutoff_at: dt.datetime) -> tuple[dict, dict]:
    """TODAY_CONFIRMED: keep a pre-open call only if the opening move agrees.
    Uses intraday bars up to the cutoff instant only."""
    info: dict[str, Any] = {}
    if intraday is None or intraday.empty or not prev_close or not atr_pct:
        if decision["direction"] in ("UP", "DOWN"):
            decision = {**decision, "direction": "NO_CALL", "setup_type": "INSUFFICIENT_DATA",
                        "reasons": ["no intraday data for confirmation"], "quality_flags": ["NO_INTRADAY_DATA"]}
        return decision, info
    cut = pd.Timestamp(cutoff_at)
    cut = cut.tz_convert(intraday.index.tz) if intraday.index.tz is not None else cut.tz_convert(None)
    bars = intraday[intraday.index <= cut]          # nothing after the cutoff instant
    if bars.empty:
        return decision, info
    info = {"open_gap_pct": float(bars["Open"].iloc[0] / prev_close - 1),
            "early_return": float(bars["Close"].iloc[-1] / prev_close - 1),
            "early_bars": int(len(bars))}
    if decision["direction"] in ("UP", "DOWN"):
        sign = 1 if decision["direction"] == "UP" else -1
        agrees = sign * info["early_return"] > 0
        not_extended = abs(info["open_gap_pct"]) <= atr_pct
        if not (agrees and not_extended):
            decision = {**decision, "direction": "NEUTRAL", "setup_type": "NOT_CONFIRMED",
                        "reasons": decision["reasons"] + ["opening move did not confirm" if not agrees
                                                          else "opening gap larger than 1 ATR"],
                        "stop_loss": None, "target": None}
    return decision, info


def run_predictions(engine, run_type: str, now: Optional[dt.datetime] = None,
                    fetch: Optional[Callable[[list[str]], dict]] = None,
                    fetch_intraday: Optional[Callable[[list[str], dt.datetime], dict]] = None,
                    intraday_enabled: Optional[bool] = None) -> dict[str, Any]:
    """Create one snapshot. Returns a result dict with status COMPLETED,
    SKIPPED or FAILED (and the run_id when a run exists)."""
    from config import PREDICTION_V2_INTRADAY_ENABLED, PREDICTION_V2_NEWS_ENABLED
    from catalysts import impact
    news_on = PREDICTION_V2_NEWS_ENABLED
    now = _utc(now or _now())
    intraday_enabled = PREDICTION_V2_INTRADAY_ENABLED if intraday_enabled is None else intraday_enabled
    with Session(engine) as session:
        holidays = masters.get_config(session, "market.holidays") or []
        try:
            p = plan(run_type, now, holidays)
            if run_type == "TODAY_CONFIRMED" and not (intraday_enabled and fetch_intraday):
                raise RunSkipped("no approved intraday data feed (PREDICTION_V2_INTRADAY_ENABLED is off)")
        except RunSkipped as e:
            return {"status": "SKIPPED", "reason": str(e)}
        key = f"{run_type}:{p['trading_date'].isoformat()}"
        existing = session.exec(select(PredictionRun).where(PredictionRun.idempotency_key == key)).first()
        if existing is not None:
            if existing.status == "RUNNING" and now - _utc(existing.started_at) > STALE_RUNNING_AFTER:
                existing.status, existing.completed_at = "FAILED", now
                existing.failure_reason = "abandoned (no completion within 2 hours); retry by a new trading day"
                session.add(existing)
                session.commit()
            return {"status": "SKIPPED", "reason": f"snapshot {key} already exists ({existing.status})",
                    "run_id": existing.run_id}
        cfg = _config(run_type, holidays, bool(intraday_enabled))
        run = PredictionRun(
            run_id=f"PRED-{run_type}-{p['trading_date']:%Y%m%d}-{uuid.uuid4().hex[:6]}", idempotency_key=key,
            engine_version=ENGINE_VERSION, rule_version=rules.RULE_VERSION,
            feature_set_version=feat.FEATURE_SET_VERSION, run_type=run_type,
            trading_date=p["trading_date"].isoformat(), target_session_date=p["target"].isoformat(),
            data_cutoff_at=now, config=cfg, config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
            .hexdigest()[:16], started_at=now)
        session.add(run)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return {"status": "SKIPPED", "reason": f"another {run_type} run is in progress or {key} exists"}
        run_id = run.run_id
        companies = members(session)
        sectors = {c.symbol: c.sector for c in companies}

    try:
        symbols = [c.symbol for c in companies]
        data = (fetch or _default_fetch)(symbols + [NIFTY])
        nifty = data.get(NIFTY)
        if p["cutoff_session"] == p["trading_date"] and run_type == "TOMORROW_EOD":
            nd = feat.slice_to_cutoff(nifty, p["cutoff_session"])
            if nd is None or nd.index[-1].date() != p["cutoff_session"]:
                raise RunSkipped("NIFTY 50 bar for today is not available yet")
        feats: dict[str, dict] = {}
        flags: dict[str, list[str]] = {}
        for sym in symbols:
            feats[sym], flags[sym] = feat.compute(data.get(sym), p["cutoff_session"], nifty, p["expected_session"])
        feat.add_sector_relative(feats, sectors)
        intraday = fetch_intraday(symbols, now) if (run_type == "TODAY_CONFIRMED" and fetch_intraday) else {}
        rows, counts = [], {d: 0 for d in ("UP", "DOWN", "NEUTRAL", "NO_CALL")}
        usable = 0
        with Session(engine) as session:
            for sym in symbols:
                decision = rules.decide(feats[sym], flags[sym])
                news = None
                if news_on:
                    news = impact.assess_stock(session, sym, now, _price_move(
                        data.get(sym), nifty, p["cutoff_session"], feats[sym].get("atr_pct")))
                    decision = impact.combine(decision, news, feats[sym],
                                              lambda d_, s_, f_, _t, r_: rules._call(d_, s_, f_, rules.THRESHOLDS, r_))
                extra: dict[str, Any] = {}
                if run_type == "TODAY_CONFIRMED":
                    decision, extra = _confirm(decision, intraday.get(sym), feats[sym].get("close"),
                                               feats[sym].get("atr_pct"), now)
                if decision["direction"] != "NO_CALL":
                    usable += 1
                counts[decision["direction"]] += 1
                events = available_events(session, sym, now)
                rows.append(Prediction(
                    prediction_id=uuid.uuid4().hex, run_id=run_id, symbol=sym, direction=decision["direction"],
                    setup_type=decision["setup_type"], horizon_sessions=1, confidence=None,
                    reference_price=feats[sym].get("close"), reference_date=feats[sym].get("last_session_date"),
                    entry_condition=decision["entry_condition"], stop_loss=decision["stop_loss"],
                    target=decision["target"], trailing_stop_rule=decision["trailing_stop_rule"],
                    invalidation_condition=decision["invalidation_condition"],
                    features={**feats[sym], **({"intraday": extra} if extra else {}),
                              **({"news": news.as_dict()} if news is not None else {})},
                    reasons=decision["reasons"], quality_flags=sorted(set(flags[sym] + decision["quality_flags"])),
                    event_ids=sorted({e.id for e in events} | {x.event_id for x in (news.evidence if news else [])}),
                    created_at=now))
            if symbols and usable < MIN_USABLE_RATIO * len(symbols):
                raise _QualityGate(f"only {usable} of {len(symbols)} stocks had usable data "
                                   f"(minimum {MIN_USABLE_RATIO:.0%}); snapshot not published")
            for r in rows:
                session.add(r)
            for sym in symbols:
                if feats[sym].get("history_sessions") and not session.exec(select(FeatureSnapshot.id).where(
                        FeatureSnapshot.symbol == sym, FeatureSnapshot.session_date == p["cutoff_session"].isoformat(),
                        FeatureSnapshot.feature_set_version == feat.FEATURE_SET_VERSION)).first():
                    session.add(FeatureSnapshot(symbol=sym, session_date=p["cutoff_session"].isoformat(),
                                                feature_set_version=feat.FEATURE_SET_VERSION, values=feats[sym],
                                                flags=flags[sym], data_cutoff_at=now))
            r = session.exec(select(PredictionRun).where(PredictionRun.run_id == run_id)).one()
            r.status, r.completed_at = "COMPLETED", _now()
            r.universe_count, r.eligible_count, r.prediction_count = len(symbols), usable, len(rows)
            r.counts = counts
            session.add(r)
            session.commit()
        logger.info("PREDICTION_RUN | %s | COMPLETED | %s", run_id, counts)
        return {"status": "COMPLETED", "run_id": run_id, "target_session_date": p["target"].isoformat(),
                "counts": counts, "universe": len(symbols)}
    except (RunSkipped, _QualityGate, Exception) as e:   # noqa: B014 - all paths end the run
        status = "SKIPPED" if isinstance(e, RunSkipped) else "FAILED"
        with Session(engine) as session:
            r = session.exec(select(PredictionRun).where(PredictionRun.run_id == run_id)).one()
            if isinstance(e, RunSkipped):
                # A data-availability skip must not block a later retry the same day.
                session.delete(r)
            else:
                # Keep the FAILED run as a record, but release the day's key so
                # a later trigger can retry (attempts are capped by the job slot).
                r.status, r.completed_at, r.failure_reason = "FAILED", _now(), f"{type(e).__name__}: {e}"[:500]
                r.idempotency_key = f"{r.idempotency_key}#FAILED-{r.run_id[-6:]}"
                session.add(r)
            session.commit()
        if status == "FAILED":
            logger.exception("PREDICTION_RUN | %s | FAILED", run_id)
        return {"status": status, "run_id": None if status == "SKIPPED" else run_id, "reason": str(e)}


class _QualityGate(Exception):
    pass


def latest_run(session: Session, run_types: tuple[str, ...], target_session_date: Optional[str] = None
               ) -> Optional[PredictionRun]:
    stmt = select(PredictionRun).where(PredictionRun.status == "COMPLETED", PredictionRun.run_type.in_(run_types))
    if target_session_date:
        stmt = stmt.where(PredictionRun.target_session_date == target_session_date)
    return session.exec(stmt.order_by(PredictionRun.target_session_date.desc(),
                                      PredictionRun.completed_at.desc())).first()
