"""
tests/test_prediction_v2.py — Prediction Engine v2 (shadow): calendar,
features, rules, snapshot service, outcomes, exits, events, performance,
backtest and universe. Synthetic data only; no network.
"""
import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
import masters.service as masters
from db.models.market import EngineRun, StockAnalysisResult
from db.models.prediction import (EventClassification, ExitState, ExitTransition, FeatureSnapshot, MarketEvent,
                                  Prediction, PredictionOutcome, PredictionRun, V2UniverseMember)
from db.models.stock import Company, StockUniverseMember
from events import ingest as ev
from prediction_v2 import backtest, calendar, exits, features as feat, outcomes, performance, rules, service, universe
from utils.market_session import IST

UTC = dt.timezone.utc


# ── synthetic market data ────────────────────────────────────────────────────

def bars(days: list[dt.date], closes, volume=1_000_000.0, spread=0.01, opens=None):
    closes = list(map(float, closes))
    o = opens or [c * (1 - spread / 4) for c in closes]
    df = pd.DataFrame({"Open": o, "High": [max(a, c) * (1 + spread / 2) for a, c in zip(o, closes)],
                       "Low": [min(a, c) * (1 - spread / 2) for a, c in zip(o, closes)], "Close": closes,
                       "Volume": volume if isinstance(volume, list) else [volume] * len(closes)},
                      index=pd.to_datetime(days))
    return df


def sessions(end: dt.date, n: int, holidays=()):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5 and d.isoformat() not in holidays:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


def walk(n, seed, drift=0.0, vol=0.01, start=100.0):
    rng = np.random.default_rng(seed)
    return list(start * np.cumprod(1 + drift + rng.normal(0, vol, n)))


MON = dt.date(2026, 10, 12)          # Monday
FRI = dt.date(2026, 10, 9)


# ── calendar ─────────────────────────────────────────────────────────────────

def test_calendar_skips_weekends_and_configured_holidays():
    assert calendar.next_trading_day(FRI) == MON
    assert calendar.next_trading_day(FRI, ["2026-10-12"]) == dt.date(2026, 10, 13)
    assert calendar.previous_trading_day(MON) == FRI
    assert calendar.previous_trading_day(dt.date(2026, 10, 5), ["2026-10-02"]) == dt.date(2026, 10, 1)
    assert calendar.add_trading_days(FRI, 3) == dt.date(2026, 10, 14)


# ── features ─────────────────────────────────────────────────────────────────

def test_feature_formulas_on_hand_checkable_data():
    days = sessions(FRI, 30)
    closes = [100.0] * 24 + [100, 101, 102, 103, 104, 110]
    vol = [1_000_000.0] * 29 + [3_000_000.0]
    df = bars(days, closes, vol)
    nifty = bars(days, [1000.0] * 25 + [1000, 1000, 1000, 1000, 1010])
    f, flags = feat.compute(df, FRI, nifty, expected_session=FRI)
    assert f["ret_1"] == pytest.approx(110 / 104 - 1)
    assert f["ret_5"] == pytest.approx(110 / 100 - 1)
    assert f["ret_20"] == pytest.approx(110 / 100 - 1)
    assert f["abnormal_volume"] == pytest.approx(3.0)
    assert f["rs_nifty_5"] == pytest.approx(0.10 - 0.01)
    assert f["dist_high_20"] <= 0 and f["dist_low_20"] >= 0
    assert 0 <= f["close_location"] <= 1
    assert f["atr_pct"] > 0 and f["vol_20"] > 0
    assert f["gap_pct"] == pytest.approx(df["Open"].iloc[-1] / 104 - 1)
    assert "INSUFFICIENT_HISTORY_60" in flags and "INSUFFICIENT_HISTORY_20" not in flags


def test_features_never_use_data_after_the_cutoff():
    days = sessions(dt.date(2026, 10, 30), 120)
    df = bars(days, walk(120, 1))
    nifty = bars(days, walk(120, 2, start=1000))
    cutoff = days[90]
    before, _ = feat.compute(df.iloc[:91], cutoff, nifty.iloc[:91])
    with_future, _ = feat.compute(df, cutoff, nifty)
    shocked = df.copy()
    shocked.iloc[91:, :4] *= 3                                        # wild future prices
    after_shock, _ = feat.compute(shocked, cutoff, nifty)
    assert before == with_future == after_shock


def test_missing_history_gives_none_and_flags_never_zero():
    days = sessions(FRI, 6)                                           # e.g. a fresh listing
    f, flags = feat.compute(bars(days, [55, 51, 50, 60, 58, 62]), FRI, None)
    assert f["ret_20"] is None and f["abnormal_volume"] is None and f["atr_pct"] is None
    assert f["rs_nifty_5"] is None
    assert {"INSUFFICIENT_HISTORY_20", "INSUFFICIENT_HISTORY_60", "NO_BENCHMARK"} <= set(flags)
    f2, flags2 = feat.compute(None, FRI, None)
    assert flags2 == ["NO_PRICE_DATA"] and f2["history_sessions"] == 0


def test_stale_suspect_volume_and_price_band_flags():
    days = sessions(FRI, 70)
    df = bars(days[:-1], walk(69, 3))                                 # last bar missing -> stale
    _, flags = feat.compute(df, FRI, None, expected_session=FRI)
    assert "STALE_PRICE" in flags
    vol = [1e6] * 69 + [1e8]
    df2 = bars(days, walk(70, 4), vol)
    _, flags2 = feat.compute(df2, FRI, None)
    assert "SUSPECT_VOLUME_SPIKE" in flags2
    closes = walk(69, 5) + [0]
    closes[-1] = closes[-2] * 1.10
    df3 = bars(days, closes, spread=0.0)
    df3.iloc[-1, df3.columns.get_loc("Low")] = closes[-2]               # real range, closed at the high
    _, flags3 = feat.compute(df3, FRI, None)
    assert "POSSIBLE_PRICE_BAND" in flags3


def test_sector_relative_needs_three_peers_and_excludes_self():
    fs = {s: {"ret_5": r} for s, r in (("A", 0.10), ("B", 0.02), ("C", 0.04), ("D", 0.00), ("X", 0.05))}
    feat.add_sector_relative(fs, {"A": "Energy", "B": "Energy", "C": "Energy", "D": "Energy", "X": "IT"})
    assert fs["A"]["sector_rs_5"] == pytest.approx(0.10 - 0.02)
    assert fs["X"]["sector_rs_5"] is None and fs["X"]["sector_peer_count"] == 0


# ── rules ────────────────────────────────────────────────────────────────────

GOOD_UP = {"close": 100.0, "ret_5": 0.05, "rs_nifty_5": 0.04, "close_location": 0.9, "abnormal_volume": 2.0,
           "atr_pct": 0.02, "dist_high_20": 0.0, "dist_low_20": 0.08}


def test_rules_up_down_neutral_and_levels():
    up = rules.decide(GOOD_UP, [])
    assert up["direction"] == "UP" and up["setup_type"] == "MOMENTUM_CONTINUATION"
    assert up["stop_loss"] == pytest.approx(97.0) and up["target"] == pytest.approx(104.0)
    down = rules.decide({**GOOD_UP, "ret_5": -0.05, "rs_nifty_5": -0.04, "close_location": 0.1,
                         "dist_high_20": -0.08, "dist_low_20": 0.0}, [])
    assert down["direction"] == "DOWN" and down["stop_loss"] == pytest.approx(103.0)
    assert rules.decide({**GOOD_UP, "abnormal_volume": 1.2}, [])["direction"] == "NEUTRAL"


@pytest.mark.parametrize("flag", rules.BLOCKING_FLAGS)
def test_rules_return_no_call_for_every_blocking_flag(flag):
    d = rules.decide(GOOD_UP, [flag])
    assert d["direction"] == "NO_CALL" and flag in d["quality_flags"]


def test_rules_no_call_when_a_required_feature_is_missing():
    d = rules.decide({**GOOD_UP, "rs_nifty_5": None}, [])
    assert d["direction"] == "NO_CALL" and "MISSING:rs_nifty_5" in d["quality_flags"]


# ── snapshot service ─────────────────────────────────────────────────────────

@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        for i in range(8):
            sym = f"S{i}.NS"
            s.add(Company(symbol=sym, name=sym, sector="Energy" if i < 5 else "IT"))
            s.add(StockUniverseMember(symbol=sym, category="Large Cap"))
        s.add(Company(symbol="NEW.NS", name="New listing", sector="IT"))
        s.add(EngineRun(run_id="V1RUN", kind="RANKING", status="COMPLETED", engine_version="ranking-v1.0",
                        fqvf_version="fqvf-v1.0"))
        s.add(StockAnalysisResult(run_id="V1RUN", symbol="S0.NS", computed_at=dt.datetime.now(UTC),
                                  stockai_score=70.0, rank=1, eligible=True, engine_version="ranking-v1.0",
                                  fqvf_version="fqvf-v1.0"))
        s.commit()
        universe.seed_from_v1(s)
        universe.add_symbols(s, ["NEW.NS"])
    return eng


def market(end: dt.date, symbols, n=120, breakout=None):
    days = sessions(end, n)
    data = {}
    for i, sym in enumerate(symbols):
        closes = walk(n, 10 + i, vol=0.006)
        vol = [1_000_000.0] * n
        if sym == breakout:                          # strong last 5 sessions on heavy volume
            for k in range(5):
                closes[n - 5 + k] = closes[n - 6] * (1 + 0.015 * (k + 1))
            vol[-1] = 3_000_000.0
        data[sym] = bars(days, closes, vol, opens=None)
        if sym == breakout:
            data[sym].iloc[-1, data[sym].columns.get_loc("High")] = closes[-1] * 1.001
            data[sym].iloc[-1, data[sym].columns.get_loc("Low")] = closes[-1] * 0.98
    data["^NSEI"] = bars(days, walk(n, 99, vol=0.004, start=22000))
    data["NEW.NS"] = bars(days[-6:], walk(6, 50, vol=0.05))
    return data


SYMS = [f"S{i}.NS" for i in range(8)]


def eod(day: dt.date, hh=19, mm=30):
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=IST)


def test_plan_timing_and_targets():
    assert service.plan("TOMORROW_EOD", eod(FRI), [])["target"] == MON
    assert service.plan("TOMORROW_EOD", eod(FRI), ["2026-10-12"])["target"] == dt.date(2026, 10, 13)
    p = service.plan("TODAY_PREOPEN", eod(MON, 8, 30), [])
    assert p["target"] == MON and p["cutoff_session"] == FRI
    for rt, when, msg in (("TODAY_PREOPEN", eod(MON, 9, 20), "before 09:15"),
                          ("TOMORROW_EOD", eod(MON, 15, 0), "final daily bars"),
                          ("TODAY_CONFIRMED", eod(MON, 9, 30), "between 09:45"),
                          ("TOMORROW_EOD", eod(dt.date(2026, 10, 10)), "not a trading day")):
        with pytest.raises(service.RunSkipped, match=msg):
            service.plan(rt, when, [])
    with pytest.raises(service.RunSkipped, match="not a trading day"):
        service.plan("TOMORROW_EOD", eod(FRI), ["2026-10-09"])


def test_eod_snapshot_is_created_frozen_and_complete(db):
    data = market(FRI, SYMS, breakout="S1.NS")
    out = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    assert out["status"] == "COMPLETED" and out["target_session_date"] == "2026-10-12"
    with Session(db) as s:
        run = s.exec(select(PredictionRun)).one()
        preds = {p.symbol: p for p in s.exec(select(Prediction)).all()}
        assert run.status == "COMPLETED" and run.prediction_count == 9 and run.universe_count == 9
        assert run.config["rule_version"] == rules.RULE_VERSION and run.config["holiday_calendar_configured"] is False
        assert preds["S1.NS"].direction == "UP" and preds["S1.NS"].setup_type == "MOMENTUM_CONTINUATION"
        assert preds["S1.NS"].stop_loss < preds["S1.NS"].reference_price < preds["S1.NS"].target
        assert preds["NEW.NS"].direction == "NO_CALL"
        assert "INSUFFICIENT_HISTORY_60" in preds["NEW.NS"].quality_flags
        assert all(p.confidence is None for p in preds.values())       # uncalibrated: never a probability
        assert all(p.reference_date == "2026-10-09" for p in preds.values() if p.symbol != "NEW.NS")
        assert preds["S1.NS"].features["cutoff_date"] == "2026-10-09"
        assert sum(run.counts.values()) == 9
        assert len(s.exec(select(FeatureSnapshot)).all()) == 9


def test_retry_and_duplicate_triggers_never_change_a_snapshot(db):
    data = market(FRI, SYMS, breakout="S1.NS")
    first = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    with Session(db) as s:
        before = [(p.prediction_id, p.direction, p.reference_price) for p in s.exec(select(Prediction)).all()]
    changed = {k: v * 1.5 for k, v in data.items()}
    again = service.run_predictions(db, "TOMORROW_EOD", eod(FRI, 21), fetch=lambda syms: changed)
    assert again["status"] == "SKIPPED" and again["run_id"] == first["run_id"]
    with Session(db) as s:
        assert [(p.prediction_id, p.direction, p.reference_price) for p in s.exec(select(Prediction)).all()] == before
        assert len(s.exec(select(PredictionRun)).all()) == 1


def test_only_one_running_run_per_type_across_processes(db):
    with Session(db) as s:
        s.add(PredictionRun(run_id="R1", idempotency_key="TOMORROW_EOD:2026-10-08", engine_version="v",
                            rule_version="r", feature_set_version="f", run_type="TOMORROW_EOD",
                            trading_date="2026-10-08", target_session_date="2026-10-09",
                            data_cutoff_at=eod(dt.date(2026, 10, 8)), config_hash="x", status="RUNNING",
                            started_at=dt.datetime.now(UTC)))
        s.commit()
    out = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: market(FRI, SYMS))
    assert out["status"] == "SKIPPED" and "in progress" in out["reason"]


def test_data_not_ready_skips_without_blocking_a_later_retry(db):
    data = market(dt.date(2026, 10, 8), SYMS)                          # no bar for Friday yet
    out = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    assert out["status"] == "SKIPPED" and "NIFTY 50 bar" in out["reason"]
    with Session(db) as s:
        assert not s.exec(select(PredictionRun)).all()
    ok = service.run_predictions(db, "TOMORROW_EOD", eod(FRI, 20), fetch=lambda syms: market(FRI, SYMS))
    assert ok["status"] == "COMPLETED"


def test_quality_gate_fails_the_run_and_publishes_nothing(db):
    data = market(FRI, SYMS)
    broken = {k: (v if k == "^NSEI" else None) for k, v in data.items()}   # provider outage
    out = service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: broken)
    assert out["status"] == "FAILED" and "usable data" in out["reason"]
    with Session(db) as s:
        assert s.exec(select(PredictionRun)).one().status == "FAILED"
        assert not s.exec(select(Prediction)).all()


def test_confirmed_snapshot_is_skipped_without_an_intraday_feed(db):
    out = service.run_predictions(db, "TODAY_CONFIRMED", eod(MON, 10, 0), fetch=lambda syms: market(FRI, SYMS))
    assert out["status"] == "SKIPPED" and "intraday" in out["reason"]


def test_confirmed_snapshot_keeps_only_calls_the_opening_move_confirms(db):
    data = market(FRI, SYMS, breakout="S1.NS")
    ref = float(data["S1.NS"]["Close"].iloc[-1])
    idx = pd.date_range("2026-10-12 09:15", periods=7, freq="5min", tz="Asia/Kolkata")
    up_open = pd.DataFrame({"Open": [ref * 1.002] * 7, "High": [ref * 1.01] * 7, "Low": [ref] * 7,
                            "Close": [ref * 1.005] * 7, "Volume": [1e5] * 7}, index=idx)
    late = pd.DataFrame({"Open": [ref * 0.5]}, index=pd.date_range("2026-10-12 10:30", periods=1, tz="Asia/Kolkata"))
    intraday = {"S1.NS": pd.concat([up_open, late.assign(High=ref, Low=ref, Close=ref * 0.5, Volume=1)])}
    out = service.run_predictions(db, "TODAY_CONFIRMED", eod(MON, 9, 50), fetch=lambda syms: data,
                                  fetch_intraday=lambda syms, now: intraday, intraday_enabled=True)
    assert out["status"] == "COMPLETED"
    with Session(db) as s:
        p = s.exec(select(Prediction).where(Prediction.symbol == "S1.NS")).one()
        assert p.direction == "UP"
        assert p.features["intraday"]["early_bars"] == 7                # the 10:30 bar is after the cutoff
        others = s.exec(select(Prediction).where(Prediction.direction.in_(("UP", "DOWN")))).all()
        assert all(o.symbol == "S1.NS" for o in others)


def test_v2_runs_do_not_touch_v1_tables(db):
    with Session(db) as s:
        v1_before = [(r.symbol, r.stockai_score, r.rank) for r in s.exec(select(StockAnalysisResult)).all()]
        members_before = sorted((m.symbol, m.category) for m in s.exec(select(StockUniverseMember)).all())
    service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: market(FRI, SYMS))
    with Session(db) as s:
        assert [(r.symbol, r.stockai_score, r.rank) for r in s.exec(select(StockAnalysisResult)).all()] == v1_before
        assert sorted((m.symbol, m.category) for m in s.exec(select(StockUniverseMember)).all()) == members_before
        assert ("NEW.NS", "Large Cap") not in members_before          # added to v2 only


# ── outcomes ─────────────────────────────────────────────────────────────────

def test_outcomes_are_direction_adjusted_cost_adjusted_and_idempotent(db):
    data = market(FRI, SYMS, breakout="S1.NS")
    service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    later = sessions(dt.date(2026, 10, 16), 5)                        # Mon..Fri after the snapshot
    ext = {}
    for sym, df in data.items():
        last = float(df["Close"].iloc[-1])
        ext[sym] = pd.concat([df, bars(later, [last * (1 + 0.01 * (k + 1)) for k in range(5)])])
    now = dt.datetime(2026, 10, 16, 18, 0, tzinfo=IST)
    first = outcomes.evaluate_due(db, now, fetch=lambda syms: ext)
    assert first["evaluated"] == 27                                    # 9 predictions x 3 horizons
    assert outcomes.evaluate_due(db, now, fetch=lambda syms: ext)["evaluated"] == 0
    with Session(db) as s:
        p = s.exec(select(Prediction).where(Prediction.symbol == "S1.NS")).one()
        o1 = s.exec(select(PredictionOutcome).where(PredictionOutcome.prediction_id == p.prediction_id,
                                                    PredictionOutcome.horizon_sessions == 1)).one()
        assert o1.outcome_status == "EVALUATED" and o1.start_price == p.reference_price
        assert o1.stock_return == pytest.approx(0.01) and o1.directional_return == pytest.approx(0.01)
        assert o1.cost_adjusted_return == pytest.approx(0.01 - 0.0020) and o1.hit is True
        assert o1.mfe >= o1.stock_return >= o1.mae - 1e-9
        neutral = s.exec(select(Prediction).where(Prediction.direction == "NEUTRAL")).first()
        on = s.exec(select(PredictionOutcome).where(PredictionOutcome.prediction_id == neutral.prediction_id)).first()
        assert on.outcome_status == "NOT_APPLICABLE" and on.hit is None and on.stock_return is not None
        nc = s.exec(select(Prediction).where(Prediction.symbol == "NEW.NS")).one()
        assert s.exec(select(PredictionOutcome).where(PredictionOutcome.prediction_id == nc.prediction_id)
                      ).first().outcome_status == "NOT_APPLICABLE"


def test_outcomes_wait_for_complete_windows(db):
    data = market(FRI, SYMS)
    service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    monday = {k: pd.concat([v, bars([MON], [float(v["Close"].iloc[-1])])]) for k, v in data.items()}
    out = outcomes.evaluate_due(db, dt.datetime(2026, 10, 12, 15, 0, tzinfo=IST), fetch=lambda syms: monday)
    assert out["evaluated"] == 0                                       # Monday's bar not final at 15:00
    out = outcomes.evaluate_due(db, dt.datetime(2026, 10, 12, 18, 0, tzinfo=IST), fetch=lambda syms: monday)
    assert out["evaluated"] == 9 and out["pending"] == 18              # 3- and 5-session windows still open


# ── exits (shadow) ───────────────────────────────────────────────────────────

def pos(**kw):
    base = dict(direction="UP", reference=100.0, atr=2.0, stop=97.0, target=104.0, ref_low=None)
    return exits.Position(**{**base, **kw})


def B(o, h, l, c, **kw):
    return exits.Bar(o, h, l, c, **kw)


def test_exit_gap_through_stop_exits_at_the_open():
    p, t = exits.step(pos(), B(95, 96, 94, 95.5))
    assert p.state == "EXIT" and t.reason == "STOP_LOSS" and t.price == 95


def test_exit_intraday_stop_and_short_mirror():
    p, t = exits.step(pos(), B(99, 100, 96.5, 98))
    assert (p.state, t.reason, t.price) == ("EXIT", "STOP_LOSS", 97.0)
    p, t = exits.step(pos(direction="DOWN", stop=103.0, target=96.0), B(101, 103.5, 100, 101))
    assert (p.state, t.reason) == ("EXIT", "STOP_LOSS")


def test_target_then_trailing_stop():
    p, t = exits.step(pos(), B(101, 104.5, 100.5, 104))
    assert (p.state, t.reason) == ("PARTIAL_EXIT", "TARGET_REACHED") and p.trailing and p.stop == pytest.approx(102.0)
    p, t = exits.step(p, B(104, 106, 103.5, 105.5))
    assert t is None and p.stop == pytest.approx(103.5)
    p, t = exits.step(p, B(105, 105.2, 103.0, 103.2))
    assert (p.state, t.reason) == ("EXIT", "TRAILING_STOP")


def test_failed_breakout_support_breakdown_and_time_stop():
    p, t = exits.step(pos(), B(100, 100.5, 98.2, 98.5))
    assert t.reason == "FAILED_BREAKOUT"
    p, t = exits.step(pos(ref_low=99.5), B(100, 100.5, 99.0, 99.2))
    assert t.reason == "SUPPORT_BREAKDOWN"
    p = pos()
    for _ in range(4):
        p, t = exits.step(p, B(101, 101.5, 100.5, 101))
        assert t is None
    p, t = exits.step(p, B(101, 101.5, 100.5, 101))
    assert t.reason == "TIME_STOP"


def test_missing_bars_repeated_signals_and_terminal_state():
    p, t = exits.step(pos(), None)
    assert t is None and p.state == "HOLD" and p.sessions == 0
    p, t = exits.step(pos(), B(None, 1, 1, 1))
    assert t is None
    tight = pos(state="TIGHTEN")
    p, t = exits.step(tight, B(101, 101.2, 100.8, 100.9, volume_ratio=3.0))  # distribution again -> no repeat
    assert t is None
    done = pos(state="EXIT")
    assert exits.step(done, B(90, 91, 89, 90)) == (done, None)
    assert "HOLD" not in exits.LEGAL["TIGHTEN"] and not exits.LEGAL["EXIT"]


def test_signals_invalidate_and_rs_deterioration():
    p, t = exits.step(pos(), B(101, 101.5, 100.5, 101), exits.Signals(thesis_invalidated=True))
    assert t.reason == "THESIS_INVALIDATED"
    p, t = exits.step(pos(), B(101, 101.5, 100.5, 101), exits.Signals(macro_reversal=True))
    assert t.reason == "MACRO_REVERSAL"
    p, t = exits.step(pos(), B(101, 101.5, 100.5, 100.6, prev_close=101.0, nifty_return=0.03))
    assert (p.state, t.reason) == ("TIGHTEN", "RELATIVE_STRENGTH_DETERIORATION")


def test_exit_states_are_persisted_in_shadow_mode_only(db):
    data = market(FRI, SYMS, breakout="S1.NS")
    service.run_predictions(db, "TOMORROW_EOD", eod(FRI), fetch=lambda syms: data)
    with Session(db) as s:
        p = s.exec(select(Prediction).where(Prediction.symbol == "S1.NS")).one()
        stop = p.stop_loss
    ext = {k: pd.concat([v, bars([MON], [stop * 0.97], opens=[stop * 0.98])]) for k, v in data.items()}
    now = dt.datetime(2026, 10, 12, 18, 0, tzinfo=IST)
    r = outcomes.update_exit_states(db, now, fetch=lambda syms: ext)
    assert r["transitions"] == 1
    assert outcomes.update_exit_states(db, now, fetch=lambda syms: ext)["transitions"] == 0   # no duplicates
    with Session(db) as s:
        st = s.exec(select(ExitState)).one()
        assert (st.state, st.reason) == ("EXIT", "STOP_LOSS")
        assert len(s.exec(select(ExitTransition)).all()) == 1


# ── events ───────────────────────────────────────────────────────────────────

def test_event_ingestion_is_idempotent_and_point_in_time(db, tmp_path):
    f = tmp_path / "events.json"
    f.write_text(json.dumps([
        {"id": "e1", "symbol": "S1.NS", "event_type": "BUSINESS_UPDATE", "title": "Q2 update",
         "published_at": "2026-10-05T14:00:00+00:00", "source_reference": "https://example.test/e1"},
        {"id": "e2", "symbol": "S1.NS", "event_type": "NOT_A_TYPE", "title": "x",
         "published_at": "2026-10-05T15:00:00+00:00"}]), encoding="utf-8")
    prov = ev.FixtureProvider(f)
    since, until = dt.datetime(2026, 10, 1, tzinfo=UTC), dt.datetime(2026, 10, 31, tzinfo=UTC)
    ingested_at = dt.datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    with Session(db) as s:
        out = ev.ingest(s, prov, since, until, now=ingested_at)
        assert out == {"fetched": 2, "inserted": 1, "duplicates": 0, "rejected": 1}
        assert ev.ingest(s, prov, since, until, now=ingested_at)["duplicates"] == 1
        e = s.exec(select(MarketEvent)).one()
        assert e.effective_available_at.replace(tzinfo=UTC) == ingested_at       # backfill: knowable only now
        assert not ev.available_events(s, "S1.NS", dt.datetime(2026, 10, 6, tzinfo=UTC))
        assert ev.available_events(s, "S1.NS", dt.datetime(2026, 10, 10, tzinfo=UTC))
    with pytest.raises(ev.ProviderNotConfigured):
        list(ev.NotConfiguredProvider().fetch(since, until))


def test_event_classification_review_keeps_the_original(db):
    with Session(db) as s:
        e = MarketEvent(source="fixture", source_event_id="x", symbol="S1.NS", event_type="EARNINGS_SURPRISE",
                        title="Q2", published_at=dt.datetime(2026, 10, 5, tzinfo=UTC),
                        effective_available_at=dt.datetime(2026, 10, 5, tzinfo=UTC))
        s.add(e)
        s.commit()
        s.refresh(e)
        c = ev.classify(s, e, classifier_version="rules-0.1", direction="POSITIVE", expected_value=100,
                        actual_value=110)
        assert c.surprise_score == pytest.approx(0.10)
        nothing = ev.classify(s, e, classifier_version="rules-0.1", direction="UNKNOWN")
        assert nothing.surprise_score is None and nothing.expected_value is None      # never invented
        fixed = ev.review(s, c, reviewer_id=1, status="CORRECTED", corrected={"actual_value": 95})
        assert fixed.supersedes_id == c.id and fixed.surprise_score == pytest.approx(-0.05)
        assert s.get(EventClassification, c.id).actual_value == 110


# ── performance & backtest ───────────────────────────────────────────────────

def test_wilson_interval_and_small_sample_suppression():
    lo, hi = performance.wilson(60, 100)
    assert 0.50 < lo < 0.6 < hi < 0.70
    p = Prediction(prediction_id="p", run_id="r", symbol="S", direction="UP", setup_type="X")
    rows = [(p, PredictionOutcome(prediction_id="p", horizon_sessions=1, evaluator_version="v",
                                  outcome_status="EVALUATED", hit=True, cost_adjusted_return=0.01))] * 5
    out = performance.summarize(rows)
    assert out["hit_rate"] is None and "only 5" in out["sample_note"] and out["promotion_gate_met"] is False


def test_backtest_is_chronological_embargoed_and_leak_free():
    n = 260
    days = sessions(dt.date(2026, 9, 30), n)
    hist = {f"S{i}.NS": bars(days, walk(n, 30 + i, vol=0.012), [1e6] * n) for i in range(6)}
    nifty = bars(days, walk(n, 1, vol=0.008, start=20000))
    sectors = {s: ("A" if i < 3 else "B") for i, s in enumerate(hist)}
    res = backtest.run(hist, nifty, sectors, horizon=1, embargo=5, seed=3)
    sp = res.sessions
    assert sp["development"][-1] < sp["validation"][0] < sp["validation"][-1] < sp["holdout"][0]
    gap = days.index(sp["validation"][0]) - days.index(sp["development"][-1])
    assert gap > 6                                                      # embargo + horizon
    report = res.report(cost_bps=20)
    assert set(report["splits"]) == {"development", "validation", "holdout"}
    assert report["splits"]["holdout"]["strategies"]["always_neutral"]["calls"] == 0
    # future changes cannot alter past decisions
    cut = sp["validation"][-1]
    shocked = {k: v.copy() for k, v in hist.items()}
    for v in shocked.values():
        v.loc[v.index > pd.Timestamp(cut) + pd.Timedelta(days=1), ["Open", "High", "Low", "Close"]] *= 2
    res2 = backtest.run(shocked, nifty, sectors, horizon=1, embargo=5, seed=3)
    key = lambda r: [(t.strategy, t.symbol, t.cutoff, t.direction) for t in r.trades
                     if t.cutoff < cut and t.strategy != "random_matched"]
    assert key(res) == key(res2)
    assert backtest.run(hist, nifty, sectors, seed=3).report() == report  # reproducible


def test_backtest_horizon_follows_the_market_calendar_not_the_stocks_own_bars():
    n = 200
    days = sessions(dt.date(2026, 9, 30), n)
    nifty = bars(days, walk(n, 1, vol=0.008, start=20000))
    full = bars(days, walk(n, 5, vol=0.012), [1e6] * n)
    gap_day = days[150]
    holed = full.drop(pd.Timestamp(gap_day))                    # one missing bar
    for h in (1, 3, 5):
        res = backtest.run({"A.NS": full, "B.NS": holed}, nifty, {}, horizon=h, embargo=5, seed=1)
        for t in res.trades:
            if t.strategy != "always_up":
                continue
            k = days.index(t.cutoff)
            end = days[k + h]
            df = full                                           # B is A minus one bar, same prices otherwise
            want = float(df["Close"].loc[pd.Timestamp(end)] / df["Close"].loc[pd.Timestamp(t.cutoff)] - 1)
            assert t.ret == pytest.approx(want)
            n_ret = float(nifty["Close"].loc[pd.Timestamp(end)] / nifty["Close"].loc[pd.Timestamp(t.cutoff)] - 1)
            assert t.excess == pytest.approx(want - n_ret)
            if t.symbol == "B.NS":                              # no trade whose window starts or ends on the hole
                assert gap_day not in (t.cutoff, end)
        b_cutoffs = {t.cutoff for t in res.trades if t.strategy == "always_up" and t.symbol == "B.NS"}
        a_cutoffs = {t.cutoff for t in res.trades if t.strategy == "always_up" and t.symbol == "A.NS"}
        assert a_cutoffs - b_cutoffs == {d for d in a_cutoffs if gap_day in (d, days[days.index(d) + h])}


def test_backtest_regime_breakdown_and_clustered_interval():
    n = 260
    days = sessions(dt.date(2026, 9, 30), n)
    hist = {f"S{i}.NS": bars(days, walk(n, 30 + i, vol=0.012), [1e6] * n) for i in range(6)}
    nifty = bars(days, walk(n, 1, vol=0.008, start=20000))
    res = backtest.run(hist, nifty, {s: "A" for s in hist}, horizon=3, embargo=5, seed=3)
    rows = res.report(cost_bps=10)["splits"]["holdout"]["strategies"]
    up = rows["always_up"]
    assert up["calls"] > 0 and up["mean_net_return"] == pytest.approx(up["mean_gross_return"] - 0.001)
    lo, hi = up["mean_net_return_ci95_clustered"]
    assert lo <= up["mean_net_return"] <= hi
    assert res.report(cost_bps=10) == res.report(cost_bps=10)                # bootstrap is seeded
    bd = res.breakdown("regime_trend", strategies=("always_up",), cost_bps=10)["always_up"]
    assert set(bd) <= {"ABOVE_SMA50", "BELOW_SMA50", "UNKNOWN"}
    assert sum(g["calls"] for g in bd.values()) == up["calls"]
    # regimes use NIFTY bars <= t only: changing the future leaves past regimes unchanged
    cut = days[200]
    shocked = nifty.copy()
    shocked.loc[shocked.index > pd.Timestamp(cut), "Close"] *= 3
    a, b = backtest._regimes(nifty), backtest._regimes(shocked)
    assert all(a[d] == b[d] for d in days if d <= cut)


def test_universe_seed_is_a_copy_and_admin_adds_do_not_touch_v1(db):
    with Session(db) as s:
        v1 = sorted(m.symbol for m in s.exec(select(StockUniverseMember)).all())
        v2 = sorted(m.symbol for m in s.exec(select(V2UniverseMember)).all())
        assert set(v1) <= set(v2) and "NEW.NS" in v2 and "NEW.NS" not in v1
        assert universe.add_symbols(s, ["NOPE.NS", "s1.ns"]) == {"added": [], "reactivated": [],
                                                                  "already": ["S1.NS"], "unknown": ["NOPE.NS"]}
        universe.deactivate(s, ["S7.NS"])
        assert "S7.NS" not in [c.symbol for c in universe.members(s)]
        assert universe.seed_from_v1(s) == 0                           # re-seeding never re-activates or duplicates
        assert sorted(m.symbol for m in s.exec(select(StockUniverseMember)).all()) == v1
