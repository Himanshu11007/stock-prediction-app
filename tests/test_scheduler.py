"""
tests/test_scheduler.py — the external scheduler: GitHub Actions workflow
configuration, the authenticated backend job interface (scheduling/remote.py,
api/routes/scheduler.py) and the workflow client (scripts/scheduler_client.py).

Covers authentication, strict job types, duplicate requests, bounded
retries, holidays and a missing calendar, missing market data, stale-run
recovery and the monitor reporting failures as failures. No network.
"""
import datetime as dt
import hashlib
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import auth.service as auth_service
import config
import masters.service as masters
from api.main import app
from api.routes import scheduler as scheduler_route
from auth.security import create_access_token
from db.models.market import ScheduledJobRun
from db.models.prediction import PredictionRun
from prediction_v2 import service as v2
from scheduling import jobs, remote
from scripts import scheduler_client as client_mod
from tests.test_prediction_v2 import FRI, MON, SYMS, db, eod, market  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
TOKEN = "test-scheduler-token-0123456789abcdef"
HOLIDAYS_2026 = ["2026-10-02", "2026-10-20", "2026-11-10"]      # dates used by these tests only
EXPECTED = {   # workflow file -> (job type, UTC crons, IST equivalents)
    "stocklens-today-preopen.yml": ("TODAY_PREOPEN", ["0 3 * * 1-5", "30 3 * * 1-5"], ["08:30", "09:00"]),
    "stocklens-today-confirmed.yml": ("TODAY_CONFIRMED", ["15 4 * * 1-5", "45 4 * * 1-5"], ["09:45", "10:15"]),
    "stocklens-tomorrow-eod.yml": ("TOMORROW_EOD", ["45 10 * * 1-5", "0 14 * * 1-5"], ["16:15", "19:30"]),
    "stocklens-outcome-evaluation.yml": ("OUTCOME_EVALUATION", ["0 11 * * 1-5", "30 15 * * 1-5"], ["16:30", "21:00"]),
    "stocklens-snapshot-monitor.yml": ("SNAPSHOT_MONITOR", ["30 4 * * 1-5", "15 11 * * 1-5", "15 15 * * 1-5"],
                                       ["10:00", "16:45", "20:45"]),
    "stocklens-news-ingestion.yml": ("NEWS_INGESTION", ["40 0-16 * * *"], ["06:10"]),     # hourly to 22:10 IST
}


# ── workflow configuration ───────────────────────────────────────────────────

def _wf(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _ist(cron: str) -> str:
    minute, hour = int(cron.split()[0]), int(cron.split()[1].split("-")[0])
    t = dt.datetime(2026, 1, 5, hour, minute) + dt.timedelta(hours=5, minutes=30)
    return f"{t:%H:%M}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_workflow_is_inactive_by_default_and_never_runs_on_push(name):
    wf = _wf(name)
    on = wf.get("on", wf.get(True))                      # PyYAML reads the key `on` as True
    assert set(on) == {"schedule", "workflow_dispatch"}  # no push / pull_request triggers
    job_type, crons, ist = EXPECTED[name]
    assert [c["cron"] for c in on["schedule"]] == crons
    assert [_ist(c) for c in crons] == ist                # documented IST equivalents match UTC
    if job_type == "NEWS_INGESTION":
        assert crons == ["40 0-16 * * *"]                # every day: weekend news matters for Monday
    else:
        assert all(c.endswith("* * 1-5") for c in crons) # weekdays; holidays are checked by the backend
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["cancel-in-progress"] is False
    (job,) = wf["jobs"].values()
    assert "vars.STOCKLENS_SCHEDULER_ENABLED == 'true'" in job["if"]
    assert 0 < job["timeout-minutes"] <= 45
    run_step = job["steps"][-1]
    assert run_step["run"].startswith(f"python3 scripts/scheduler_client.py --job {job_type} ")
    assert run_step["env"] == {"STOCKLENS_API_URL": "${{ vars.STOCKLENS_API_URL }}",
                               "STOCKLENS_SCHEDULER_TOKEN": "${{ secrets.STOCKLENS_SCHEDULER_TOKEN }}"}
    assert job["steps"][0]["with"]["persist-credentials"] is False
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert text.count("secrets.") == 1                   # the token is only passed via env
    assert not re.search(r"echo .*(TOKEN|secrets)", text)
    assert "DATABASE_URL" not in text                    # workflows never talk to the database


def test_today_confirmed_needs_its_own_switch():
    (job,) = _wf("stocklens-today-confirmed.yml")["jobs"].values()
    assert "vars.STOCKLENS_TODAY_CONFIRMED_ENABLED == 'true'" in job["if"]


def test_documentation_lists_every_schedule_in_utc_and_ist():
    doc = (ROOT / "docs" / "SCHEDULER.md").read_text(encoding="utf-8")
    for name, (job_type, crons, ist) in EXPECTED.items():
        assert name in doc and job_type in doc
        for c, i in zip(crons, ist):
            assert f"`{c}`" in doc and i in doc


def test_no_scheduler_runs_from_render_or_the_old_template():
    assert not (ROOT / "deploy" / "github-actions" / "prediction-jobs.yml").exists()
    for blueprint in (ROOT / "render.yaml", ROOT / "deploy" / "render-free" / "render.yaml"):
        assert "scheduled_jobs.py predict" not in blueprint.read_text(encoding="utf-8")


# ── backend job interface ────────────────────────────────────────────────────

@pytest.fixture()
def api(db, monkeypatch):
    monkeypatch.setattr(config, "SCHEDULER_TOKEN_SHA256", hashlib.sha256(TOKEN.encode()).hexdigest())
    monkeypatch.setattr(remote, "spawn", lambda fn: fn())               # run jobs inline
    monkeypatch.setattr(v2, "_default_fetch", lambda symbols: market(FRI, SYMS))
    remote._last.clear()
    app.dependency_overrides[scheduler_route.get_engine] = lambda: db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(scheduler_route.get_engine, None)


def _calendar(eng, holidays=HOLIDAYS_2026):
    with Session(eng) as s:
        masters.set_config(s, auth_service.get_user_by_email(s, "admin@example.com"), "market.holidays", holidays)


def _at(monkeypatch, when):
    monkeypatch.setattr(remote, "_now", lambda: when.astimezone(dt.timezone.utc))


def _post(api, job, token=TOKEN):
    return api.post(f"/api/v1/scheduler/jobs/{job}", headers={"Authorization": f"Bearer {token}"} if token else {})


def test_interface_is_disabled_without_a_configured_token(api, monkeypatch):
    monkeypatch.setattr(config, "SCHEDULER_TOKEN_SHA256", "")
    assert _post(api, "TOMORROW_EOD").status_code == 404


def test_authentication_is_the_dedicated_token_only(api, db):
    assert _post(api, "TOMORROW_EOD", token=None).status_code == 401
    assert _post(api, "TOMORROW_EOD", token="wrong").status_code == 401
    admin_jwt = create_access_token(subject="admin@example.com", roles=["ADMIN"])
    assert _post(api, "TOMORROW_EOD", token=admin_jwt).status_code == 401          # no user/admin sessions
    r = api.get("/api/v1/scheduler/jobs/TOMORROW_EOD/runs/2026-10-09")
    assert r.status_code == 401
    with Session(db) as s:
        assert not s.exec(select(ScheduledJobRun)).all()                         # nothing ran


@pytest.mark.parametrize("job", ["tomorrow_eod", "RANKING", "DROP TABLE", "TOMORROW_EOD;x"])
def test_job_type_is_strictly_validated(api, job):
    assert _post(api, job).status_code in (404, 422)


def test_status_slot_must_be_a_date(api):
    r = api.get("/api/v1/scheduler/jobs/TOMORROW_EOD/runs/../../x", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code in (404, 405, 422)                  # never reaches a job (path is normalised away)
    r = api.get("/api/v1/scheduler/jobs/TOMORROW_EOD/runs/latest", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 422


def test_missing_holiday_calendar_fails_instead_of_running(api, db, monkeypatch):
    _at(monkeypatch, eod(FRI))
    out = _post(api, "TOMORROW_EOD").json()["data"]
    assert out["status"] == "FAILED" and out["code"] == "CALENDAR_NOT_CONFIGURED"
    _calendar(db, ["2025-10-02"])                                                # last year only
    assert _post(api, "TODAY_PREOPEN").json()["data"]["code"] == "CALENDAR_NOT_CONFIGURED"
    with Session(db) as s:
        assert not s.exec(select(PredictionRun)).all()


def test_eod_publishes_once_and_duplicates_are_no_ops(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(FRI))
    first = _post(api, "TOMORROW_EOD").json()["data"]
    assert first["status"] == "ACCEPTED"
    state = api.get(f"/api/v1/scheduler/jobs/TOMORROW_EOD/runs/{first['slot']}",
                    headers={"Authorization": f"Bearer {TOKEN}"}).json()["data"]
    assert state["status"] == "COMPLETED" and state["run_id"].startswith("PRED-TOMORROW_EOD-20261009")
    again = _post(api, "TOMORROW_EOD").json()["data"]
    assert again["status"] == "COMPLETED" and again["duplicate"] is True
    with Session(db) as s:
        runs_ = s.exec(select(PredictionRun)).all()
        assert [(r.status, r.target_session_date) for r in runs_] == [("COMPLETED", MON.isoformat())]


def test_eod_waits_for_validated_closing_data(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(FRI, 15, 50))                                           # before 16:00 IST
    assert _post(api, "TOMORROW_EOD").json()["data"]["code"] == "OUTSIDE_WINDOW"
    _at(monkeypatch, eod(FRI))
    monkeypatch.setattr(v2, "_default_fetch", lambda symbols: market(dt.date(2026, 10, 8), SYMS))  # no Friday bar
    first = _post(api, "TOMORROW_EOD").json()["data"]
    state = remote.status(db, remote.JobType.TOMORROW_EOD, first["slot"])
    assert state["status"] == "SKIPPED" and "not available yet" in state["result"]["reason"]
    with Session(db) as s:
        assert not s.exec(select(PredictionRun)).all()                          # nothing published
    # the retry trigger (20:30 IST) finds the data and publishes
    monkeypatch.setattr(v2, "_default_fetch", lambda symbols: market(FRI, SYMS))
    _at(monkeypatch, eod(FRI, 20, 30))
    _post(api, "TOMORROW_EOD")
    assert remote.status(db, remote.JobType.TOMORROW_EOD, first["slot"])["status"] == "COMPLETED"


def test_holiday_and_weekend_are_skipped(api, db, monkeypatch):
    _calendar(db, HOLIDAYS_2026 + ["2026-10-09"])
    _at(monkeypatch, eod(FRI))
    assert _post(api, "TOMORROW_EOD").json()["data"]["code"] == "NOT_A_TRADING_DAY"
    _at(monkeypatch, eod(dt.date(2026, 10, 10), 8, 0))
    assert _post(api, "TODAY_PREOPEN").json()["data"]["code"] == "NOT_A_TRADING_DAY"


def test_today_confirmed_stays_disabled_without_an_intraday_feed(api, db, monkeypatch):
    _calendar(db)
    monkeypatch.setattr(config, "PREDICTION_V2_INTRADAY_ENABLED", False)
    _at(monkeypatch, eod(MON, 9, 50))
    out = _post(api, "TODAY_CONFIRMED").json()["data"]
    assert out["status"] == "SKIPPED" and out["code"] == "INTRADAY_FEED_DISABLED"


def test_failed_job_is_reported_failed_and_retries_are_bounded(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(FRI))

    def down(symbols):
        raise ConnectionError("provider down")
    monkeypatch.setattr(v2, "_default_fetch", down)
    slot = FRI.isoformat()
    for attempt in range(1, jobs.MAX_ATTEMPTS + 1):
        _post(api, "TOMORROW_EOD")
        st = remote.status(db, remote.JobType.TOMORROW_EOD, slot)
        assert st["status"] == "FAILED" and st["attempts"] == attempt
    assert st["code"] == "ATTEMPTS_EXHAUSTED"
    out = _post(api, "TOMORROW_EOD").json()["data"]                               # no 4th attempt
    assert out["status"] == "FAILED" and out["code"] == "ATTEMPTS_EXHAUSTED"
    with Session(db) as s:
        assert len(s.exec(select(PredictionRun)).all()) == jobs.MAX_ATTEMPTS


def test_interrupted_run_is_recovered_after_the_stale_limit(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(FRI))
    started = eod(FRI).astimezone(dt.timezone.utc)
    with Session(db) as s:                                   # a worker died mid-run
        s.add(ScheduledJobRun(job="predict_eod", slot=FRI.isoformat(), status="RUNNING", started_at=started))
        s.commit()
    out = _post(api, "TOMORROW_EOD").json()["data"]
    assert out["status"] == "RUNNING" and out["duplicate"] is True             # not stale yet: no second run
    later = eod(FRI) + jobs.STALE_RUNNING + dt.timedelta(minutes=1)
    monkeypatch.setattr(jobs, "_now", lambda: later.astimezone(dt.timezone.utc))
    _at(monkeypatch, later)
    assert remote.status(db, remote.JobType.TOMORROW_EOD, FRI.isoformat())["stale"] is True
    _post(api, "TOMORROW_EOD")
    st = remote.status(db, remote.JobType.TOMORROW_EOD, FRI.isoformat())
    assert st["status"] == "COMPLETED" and st["attempts"] == 2


def test_monitor_reports_missing_snapshot_as_alert_never_success(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(MON, 9, 45))
    out = _post(api, "SNAPSHOT_MONITOR").json()["data"]
    assert out["status"] == "ALERT" and out["problems"] == [{"run_type": "TODAY_PREOPEN", "problem": "MISSING"}]
    with Session(db) as s:
        row = s.exec(select(ScheduledJobRun).where(ScheduledJobRun.job == "prediction_monitor")).one()
        assert row.status == "FAILED"


def test_monitor_flags_a_failed_outcome_evaluation(api, db, monkeypatch):
    _calendar(db)
    with Session(db) as s:
        s.add(ScheduledJobRun(job="prediction_outcomes", slot=MON.isoformat(), status="FAILED", attempts=3,
                              result={"status": "FAILED", "error": "ConnectionError: x"}))
        s.commit()
    out = jobs.prediction_monitor_job(db, eod(MON, 9, 0))
    assert out["status"] == "ALERT"
    assert out["problems"][0]["run_type"] == "OUTCOME_EVALUATION" and out["problems"][0]["problem"] == "FAILED"


def test_outcome_evaluation_waits_for_final_bars(api, db, monkeypatch):
    _calendar(db)
    _at(monkeypatch, eod(FRI, 15, 0))
    assert _post(api, "OUTCOME_EVALUATION").json()["data"]["code"] == "DATA_NOT_FINAL"


# ── workflow client ──────────────────────────────────────────────────────────

class FakeApi:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def __call__(self, method, url, token, timeout=60):
        self.calls.append((method, url, token))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return client_mod.HttpResult(*item)


def _client(script, **kw):
    fake = FakeApi(script)
    t = [0.0]
    code = client_mod.run("TOMORROW_EOD", "https://api.example", TOKEN, transport=fake,
                          sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], **kw)
    return code, fake


HEALTH = (200, {"status": "ok"})


def _d(status, **extra):
    return {"data": {"status": status, "slot": "2026-10-09", "job_type": "TOMORROW_EOD", **extra}}


def test_client_polls_until_completed():
    code, fake = _client([HEALTH, (200, _d("ACCEPTED")), (200, _d("RUNNING")), (200, _d("COMPLETED"))])
    assert code == 0
    assert fake.calls[1][:2] == ("POST", "https://api.example/api/v1/scheduler/jobs/TOMORROW_EOD")
    assert fake.calls[2][1].endswith("/runs/2026-10-09")


@pytest.mark.parametrize("final,code", [("FAILED", 1), ("ALERT", 1), ("SKIPPED", 0), ("OK", 0)])
def test_client_exit_codes(final, code):
    assert _client([HEALTH, (200, _d(final))])[0] == code


def test_client_retries_transient_errors_a_bounded_number_of_times():
    code, fake = _client([HEALTH, (503, {}), ConnectionError("reset"), (200, _d("COMPLETED"))])
    assert code == 0 and sum(1 for c in fake.calls if c[0] == "POST") == 3
    code, fake = _client([HEALTH, (503, {}), (503, {}), (503, {})])
    assert code == 1 and sum(1 for c in fake.calls if c[0] == "POST") == 3


def test_client_does_not_retry_auth_errors_and_times_out_as_failure():
    code, fake = _client([HEALTH, (401, {"detail": "Invalid scheduler token"})])
    assert code == 1 and sum(1 for c in fake.calls if c[0] == "POST") == 1
    code, _ = _client([HEALTH, (200, _d("ACCEPTED"))] + [(200, _d("RUNNING"))] * 200, timeout_minutes=5)
    assert code == 1


def test_client_never_prints_the_token(capsys, monkeypatch):
    monkeypatch.setenv("STOCKLENS_API_URL", "https://api.example")
    monkeypatch.setenv("STOCKLENS_SCHEDULER_TOKEN", TOKEN)
    monkeypatch.setattr(client_mod, "http", FakeApi([HEALTH, (200, _d("FAILED", result={"error": "boom"}))]))
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    assert client_mod.main(["--job", "TOMORROW_EOD"]) == 1
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err


def test_client_requires_configuration(monkeypatch):
    monkeypatch.delenv("STOCKLENS_SCHEDULER_TOKEN", raising=False)
    monkeypatch.setenv("STOCKLENS_API_URL", "https://api.example")
    assert client_mod.main(["--job", "TOMORROW_EOD"]) == 2
    monkeypatch.setenv("STOCKLENS_SCHEDULER_TOKEN", TOKEN)
    monkeypatch.setenv("STOCKLENS_API_URL", "http://insecure.example")
    assert client_mod.main(["--job", "TOMORROW_EOD"]) == 2


# ── news ingestion job ───────────────────────────────────────────────────────

class _OneArticle:
    name = "testwire"

    def __init__(self, fail=False):
        self.fail = fail

    def fetch(self, since, until):
        from catalysts.providers import ProviderError, RawArticle
        if self.fail:
            raise ProviderError("feed down")
        return [RawArticle("x1", "https://www.reuters.com/x1", "RBI unexpectedly cuts repo rate by 50 basis points",
                           until - dt.timedelta(minutes=20))]


def test_news_ingestion_runs_hourly_without_the_trading_calendar(api, db, monkeypatch):
    monkeypatch.setattr(jobs, "news_providers", lambda: [_OneArticle()])
    sat = eod(dt.date(2026, 10, 10), 9, 10)                  # Saturday, no holiday calendar configured
    _at(monkeypatch, sat)
    monkeypatch.setattr(jobs, "_now", lambda: sat.astimezone(dt.timezone.utc))
    first = _post(api, "NEWS_INGESTION").json()["data"]
    assert first["status"] == "ACCEPTED" and first["slot"] == "2026-10-10T09"
    st = api.get(f"/api/v1/scheduler/jobs/NEWS_INGESTION/runs/{first['slot']}",
                 headers={"Authorization": f"Bearer {TOKEN}"}).json()["data"]
    assert st["status"] == "COMPLETED" and st["result"]["providers"][0]["new_events"] == 1
    again = _post(api, "NEWS_INGESTION").json()["data"]
    assert again["duplicate"] is True                                        # same hour: no second run


def test_news_ingestion_failure_is_reported_failed(api, db, monkeypatch):
    monkeypatch.setattr(jobs, "news_providers", lambda: [_OneArticle(fail=True)])
    _at(monkeypatch, eod(FRI, 11, 10))
    out = _post(api, "NEWS_INGESTION").json()["data"]
    st = remote.status(db, remote.JobType.NEWS_INGESTION, out["slot"])
    assert st["status"] == "FAILED" and st["result"]["providers"][0]["status"] == "FAILED"


def test_monitor_flags_stale_news_once_ingestion_exists(api, db, monkeypatch):
    from db.models.news import NewsIngestionRun
    _calendar(db)
    with Session(db) as s:
        s.add(NewsIngestionRun(provider="rss", status="COMPLETED", started_at=eod(MON, 4, 0),
                               finished_at=eod(MON, 4, 1)))
        s.commit()
    out = jobs.prediction_monitor_job(db, eod(MON, 9, 0))
    assert {"run_type": "NEWS_INGESTION", "problem": "STALE"}.items() <= out["problems"][0].items()
    with Session(db) as s:
        s.add(NewsIngestionRun(provider="rss", status="COMPLETED", started_at=eod(MON, 8, 10),
                               finished_at=eod(MON, 8, 11)))
        s.commit()
    assert jobs.prediction_monitor_job(db, eod(MON, 9, 0))["status"] == "OK"


def test_failed_authentication_is_throttled_per_client(api):
    from api.routes import scheduler as route
    for _ in range(route.AUTH_FAIL_LIMIT):
        assert _post(api, "SNAPSHOT_MONITOR", token="wrong").status_code == 401
    r = _post(api, "SNAPSHOT_MONITOR", token="wrong")
    assert r.status_code == 429 and r.headers["Retry-After"] == str(route.AUTH_FAIL_WINDOW)
    assert _post(api, "SNAPSHOT_MONITOR").status_code == 429            # even the right token, until the window passes


def test_triggers_are_rate_limited_per_job_type(api, db, monkeypatch):
    from api.routes import scheduler as route
    monkeypatch.setattr(route, "TRIGGER_LIMIT", 3)
    _calendar(db)
    _at(monkeypatch, eod(MON, 9, 0))
    codes = [_post(api, "SNAPSHOT_MONITOR").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert _post(api, "OUTCOME_EVALUATION").status_code == 200          # other job types are separate
