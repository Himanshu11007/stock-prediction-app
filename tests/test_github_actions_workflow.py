"""
tests/test_github_actions_workflow.py — the GitHub Actions scheduler for the
daily ranking (.github/workflows/daily-ranking.yml) while the backend runs on
Render Free. The workflow only triggers the existing job
(scripts/scheduled_jobs.py ranking -> scheduling.jobs.ranking_job); the
calendar, locking and publish rules stay in the job and are tested in
tests/test_daily_ranking.py.
"""
import datetime as dt
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "daily-ranking.yml"


def _workflow() -> dict:
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    wf["on"] = wf.pop(True, wf.get("on"))          # YAML 1.1 reads the key `on` as True
    return wf


def _job() -> dict:
    return _workflow()["jobs"]["daily-ranking"]


def _run_step() -> dict:
    return next(s for s in _job()["steps"] if s.get("name") == "Run the daily ranking job")


def test_workflow_exists_and_is_named():
    assert WORKFLOW.is_file()
    assert _workflow()["name"] == "StockLens Daily Ranking"


def test_scheduled_after_the_nse_close_on_weekdays_and_manually_runnable():
    on = _workflow()["on"]
    crons = [s["cron"] for s in on["schedule"]]
    assert crons[0] == "0 11 * * 1-5"                               # 16:30 IST
    for cron in crons:
        minute, hour, dom, month, dow = cron.split()
        assert (dom, month, dow) == ("*", "*", "1-5")
        ist = dt.datetime(2026, 10, 5, int(hour), int(minute), tzinfo=dt.timezone.utc).astimezone(
            dt.timezone(dt.timedelta(hours=5, minutes=30)))
        assert dt.time(16, 0) <= ist.time() <= dt.time(21, 0)      # closing data final (16:00 IST)
    assert len(crons) <= 3                                         # not polling
    assert "workflow_dispatch" in on
    assert on["workflow_dispatch"] in (None, {})                   # no inputs that could bypass the job's rules
    assert "pull_request" not in on and "pull_request_target" not in on   # secrets never reach PR code


def test_runs_the_existing_ranking_command_without_hiding_failures():
    script = _run_step()["run"]
    assert "python scripts/scheduled_jobs.py ranking" in script
    assert "|| true" not in script and "continue-on-error" not in _run_step()
    assert 'exit "${code}"' in script and "PIPESTATUS[0]" in script
    assert "continue-on-error" not in _job()
    # nothing else runs ranking code or sends notifications
    all_runs = " ".join(s.get("run", "") for s in _job()["steps"])
    for other in ("execute_run", "create_run", "notify", "alembic", "scheduled_jobs.py prices",
                  "scheduled_jobs.py notifications"):
        assert other not in all_runs


def test_database_url_comes_from_secrets_and_is_never_printed():
    step = _run_step()
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert "DATABASE_URL" not in _job().get("env", {})              # scoped to the one step
    script = step["run"]
    for line in script.splitlines():
        if "echo" in line:
            assert "${DATABASE_URL}" not in line and "$DATABASE_URL" not in line
    assert '-z "${DATABASE_URL}"' in script                         # missing secret fails, never falls back to SQLite


def test_no_credentials_are_hard_coded():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert not re.search(r"postgres(ql)?://[^*\"\s]+@", text)
    for marker in ("password:", "JWT_SECRET_KEY", "GOCSPX", "BEGIN PRIVATE KEY", "render.com/"):
        assert marker not in text
    secrets = set(re.findall(r"secrets\.([A-Z_]+)", text))
    assert secrets == {"DATABASE_URL"}


def test_concurrency_protection_and_production_engine_settings():
    wf = _workflow()
    assert wf["concurrency"] == {"group": "stocklens-daily-ranking", "cancel-in-progress": False}
    assert wf["permissions"] == {"contents": "read"}
    env = _job()["env"]
    assert env["ENGINE_RUN_ALLOW_ML"] == "false"                    # ML stays off (weight 0)
    assert env["ENGINE_RUN_MAX_WORKERS"] == "2"
    install = next(s for s in _job()["steps"] if s.get("name") == "Install production dependencies")
    assert install["run"] == "pip install -r requirements-cloud.txt"
    python = next(s for s in _job()["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    assert python["with"]["python-version"] == "3.11"
    assert _job()["if"] == ("github.repository == 'Himanshu11007/stock-prediction-app' && "
                            "vars.STOCKLENS_DAILY_RANKING_ENABLED == 'true'")
    checkout = next(s for s in _job()["steps"] if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False


def _gh_if(expr: str, repository: str, variables: dict) -> bool:
    """Evaluate the job condition (GitHub expression subset: ==, &&, ||,
    vars.X, github.repository; an unset variable is the empty string)."""
    assert re.fullmatch(r"[\w.\s'()=!&|/-]+", expr), expr
    py = re.sub(r"vars\.(\w+)", lambda m: repr(variables.get(m.group(1), "")), expr)
    py = py.replace("github.repository", repr(repository)).replace("&&", " and ").replace("||", " or ")
    assert "github." not in py and "vars." not in py, py
    return bool(eval(py, {"__builtins__": {}}, {}))             # noqa: S307 - test-only, validated above


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_off_by_default_and_manual_runs_cannot_bypass_the_switch(event):
    cond = _job()["if"]
    assert "event_name" not in cond                              # no event-based exception
    repo = "Himanshu11007/stock-prediction-app"
    for value in (None, "", "false", "TRUE ", "1", "yes"):
        env = {} if value is None else {"STOCKLENS_DAILY_RANKING_ENABLED": value}
        assert _gh_if(cond, repo, env) is False, (event, value)
    assert _gh_if(cond, repo, {"STOCKLENS_DAILY_RANKING_ENABLED": "true"}) is True
    assert _gh_if(cond, "someone/fork", {"STOCKLENS_DAILY_RANKING_ENABLED": "true"}) is False   # never on forks


def test_render_cron_configuration_is_kept_for_paid_plans():
    bp = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    ranking = next(s for s in bp["services"] if s.get("name") == "stocklens-ranking")
    assert ranking["startCommand"] == "python scripts/scheduled_jobs.py ranking"   # same command


# ── the command the workflow runs is the production daily job ───────────────

def test_cli_runs_the_same_job_as_the_admin_button(monkeypatch, capsys):
    import scheduling.jobs as sj
    import scripts.scheduled_jobs as cli
    from db.session import engine
    calls = []
    monkeypatch.setattr(sj, "ranking_job", lambda eng, now=None, fetch_index=None: calls.append(eng) or
                        {"status": "COMPLETED", "run_id": "R1"})
    assert cli.main(["ranking"]) == 0
    assert calls == [engine]                                       # production engine (DATABASE_URL)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "COMPLETED"
    admin_src = (ROOT / "api" / "routes" / "admin_masters.py").read_text(encoding="utf-8")
    assert "target=jobs.ranking_job" in admin_src                  # the admin button runs the same function


@pytest.mark.parametrize("result,code", [
    ({"status": "SKIPPED", "reason": "2026-10-03 is not a trading day"}, 0),
    ({"status": "SKIPPED", "reason": "today's ranking job (2026-10-05) is already running or done"}, 0),
    ({"status": "FAILED", "run_id": "R2"}, 1),                     # visible failure in GitHub Actions
])
def test_cli_exit_code_reflects_the_job_result(monkeypatch, capsys, result, code):
    import scheduling.jobs as sj
    import scripts.scheduled_jobs as cli
    monkeypatch.setattr(sj, "ranking_job", lambda *a, **k: result)
    assert cli.main(["ranking"]) == code


def test_cli_reports_crash_as_failure(monkeypatch, capsys):
    import scheduling.jobs as sj
    import scripts.scheduled_jobs as cli

    def boom(*a, **k):
        raise ConnectionError("could not connect to server")
    monkeypatch.setattr(sj, "ranking_job", boom)
    assert cli.main(["ranking"]) == 1
    assert json.loads(capsys.readouterr().out.strip())["status"] == "FAILED"


def test_frozen_methodology_is_unchanged():
    from ranking.service import DEFAULT_WEIGHTS
    assert DEFAULT_WEIGHTS == {"quality": 25, "valuation": 20, "financial_health": 15, "technical_trend": 10,
                               "momentum": 10, "risk": 10, "sector_outlook": 5, "market_regime": 5,
                               "ml_signal": 0}
