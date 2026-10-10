"""
tests/test_prediction_jobs.py — scheduled v2 jobs: calendar and timing,
slot locking, retries after failure, missed-snapshot monitor, CLI exit codes.
"""
import datetime as dt

import pytest
from sqlmodel import Session, select

import masters.service as masters
import auth.service as auth_service
from db.models.market import ScheduledJobRun
from db.models.prediction import PredictionRun
from scheduling import jobs
from tests.test_prediction_v2 import FRI, MON, SYMS, db, eod, market  # noqa: F401  (fixture)


def _slots(eng, job):
    with Session(eng) as s:
        return s.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == job)).all()


def test_early_trigger_is_skipped_without_using_the_slot(db):
    out = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI, 15, 0), fetch=lambda s: market(FRI, SYMS))
    assert out["status"] == "SKIPPED" and "final daily bars" in out["reason"]
    assert not _slots(db, "predict_eod")


def test_non_trading_day_is_recorded_once(db):
    sat = dt.date(2026, 10, 10)
    for _ in range(2):
        assert jobs.prediction_job(db, "TOMORROW_EOD", eod(sat))["status"] == "SKIPPED"
    rows = _slots(db, "predict_eod")
    assert [(r.slot, r.status) for r in rows] == [("2026-10-10", "SKIPPED")]


def test_configured_holiday_is_respected(db):
    with Session(db) as s:
        masters.set_config(s, auth_service.get_user_by_email(s, "admin@example.com"), "market.holidays", ["2026-10-09"])
    out = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI), fetch=lambda s: market(FRI, SYMS))
    assert out["status"] == "SKIPPED" and "not a trading day" in out["reason"]


def test_success_then_duplicate_trigger(db):
    first = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI), fetch=lambda s: market(FRI, SYMS))
    assert first["status"] == "COMPLETED" and first["duration_s"] >= 0
    again = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI, 20, 15), fetch=lambda s: market(FRI, SYMS))
    assert again["status"] == "SKIPPED"
    slot = _slots(db, "predict_eod")[0]
    assert slot.status == "COMPLETED" and slot.run_id == first["run_id"]


def test_failure_is_recorded_and_a_later_trigger_retries(db):
    def boom(symbols):
        raise ConnectionError("provider down")
    out = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI), fetch=boom)
    assert out["status"] == "FAILED"
    assert _slots(db, "predict_eod")[0].status == "FAILED"
    # a later trigger retries: a new run; the FAILED run stays as a record
    retry = jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI, 20, 15), fetch=lambda s: market(FRI, SYMS))
    assert retry["status"] == "COMPLETED" and retry["run_id"] != out["run_id"]
    with Session(db) as s:
        runs_ = s.exec(select(PredictionRun).order_by(PredictionRun.started_at)).all()
        assert [r.status for r in runs_] == ["FAILED", "COMPLETED"]
        assert runs_[1].idempotency_key == "TOMORROW_EOD:2026-10-09"
    assert _slots(db, "predict_eod")[0].status == "COMPLETED" and _slots(db, "predict_eod")[0].attempts == 2
    # attempts are capped: after 3 failures the slot is not claimed again
    assert jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI, 21), fetch=boom)["status"] == "SKIPPED"


def test_monitor_reports_missing_and_failed_snapshots(db):
    assert jobs.prediction_monitor_job(db, eod(MON, 9, 0))["status"] == "OK"          # nothing due yet
    out = jobs.prediction_monitor_job(db, eod(MON, 9, 45))
    assert out["status"] == "ALERT" and out["problems"] == [{"run_type": "TODAY_PREOPEN", "problem": "MISSING"}]
    assert jobs.prediction_monitor_job(db, eod(dt.date(2026, 10, 11), 21, 0))["status"] == "OK"  # Sunday
    jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI), fetch=lambda s: (_ for _ in ()).throw(RuntimeError("x")))
    out = jobs.prediction_monitor_job(db, eod(FRI, 21, 0))
    kinds = {p["run_type"]: p["problem"] for p in out["problems"]}
    assert kinds == {"TODAY_PREOPEN": "MISSING", "TOMORROW_EOD": "FAILED"}
    assert "RuntimeError" in next(p["reason"] for p in out["problems"] if p["run_type"] == "TOMORROW_EOD")
    assert any(r.job == "prediction_monitor" and r.status == "FAILED" for r in _slots(db, "prediction_monitor"))


def test_cli_exit_codes_for_alert_and_failure(monkeypatch):
    import scripts.scheduled_jobs as cli
    for status, code in (("OK", 0), ("COMPLETED", 0), ("SKIPPED", 0), ("ALERT", 1), ("FAILED", 1)):
        monkeypatch.setitem(cli.JOBS, "prediction_monitor", lambda s=status: {"status": s})
        assert cli.main(["prediction_monitor"]) == code


def test_outcomes_job_waits_for_final_bars_and_is_idempotent(db):
    jobs.prediction_job(db, "TOMORROW_EOD", eod(FRI), fetch=lambda s: market(FRI, SYMS))
    early = jobs.prediction_outcomes_job(db, eod(MON, 15, 0), fetch=lambda s: market(FRI, SYMS))
    assert early["status"] == "SKIPPED"
    out = jobs.prediction_outcomes_job(db, eod(MON, 18, 30), fetch=lambda s: market(FRI, SYMS))
    assert out["status"] == "DONE"
    assert jobs.prediction_outcomes_job(db, eod(MON, 18, 45))["status"] == "SKIPPED"
