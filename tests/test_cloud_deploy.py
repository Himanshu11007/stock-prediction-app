"""
tests/test_cloud_deploy.py — the Render deployment configuration
(render.yaml, requirements-cloud.txt, docs/DEPLOY_RENDER.md) and the code
paths that differ in the cloud (postgres:// URLs, no local tracker.db).
"""
import importlib
from pathlib import Path

import yaml
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

ROOT = Path(__file__).resolve().parents[1]


def _blueprint() -> dict:
    return yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))


def test_blueprint_web_service_is_production_ready():
    bp = _blueprint()
    web = next(s for s in bp["services"] if s["type"] == "web")
    assert web["buildCommand"] == "pip install -r requirements-cloud.txt"
    assert web["preDeployCommand"] == "alembic upgrade head"
    assert "api.main:app" in web["startCommand"] and "$PORT" in web["startCommand"]
    assert web["healthCheckPath"] == "/api/v1/health"
    env = {e.get("key"): e for e in web["envVars"] if "key" in e}
    assert env["JWT_SECRET_KEY"] == {"key": "JWT_SECRET_KEY", "generateValue": True}   # never committed
    assert env["DATABASE_URL"]["fromDatabase"]["name"] == bp["databases"][0]["name"]
    assert env["CORS_ALLOWED_ORIGINS"]["value"].startswith("https://") and "*" not in env["CORS_ALLOWED_ORIGINS"]["value"]
    shared = {e["key"]: e["value"] for g in bp["envVarGroups"] for e in g["envVars"]}
    assert shared["APP_ENV"] == "production"


def test_blueprint_cron_jobs_match_scheduled_jobs():
    """Every CLI job has exactly one scheduler: the v1 jobs are Render cron
    services in the paid blueprint; the Prediction v2 jobs are triggered by
    the GitHub Actions workflows through the backend job interface
    (docs/SCHEDULER.md) and are deliberately not Render cron services."""
    from scheduling.remote import SLOT_JOB
    from scripts.scheduled_jobs import JOBS
    crons = [s for s in _blueprint()["services"] if s["type"] == "cron"]
    jobs = {s["startCommand"].split()[-1] for s in crons}
    v2_jobs = set(SLOT_JOB.values())
    workflows = {p.name for p in (ROOT / ".github" / "workflows").glob("stocklens-*.yml")}
    assert len(workflows) == len(SLOT_JOB)                       # one workflow per v2 job type
    assert jobs.isdisjoint(v2_jobs)                              # never scheduled twice
    assert jobs | v2_jobs == set(JOBS)                           # nothing left unscheduled
    for c in crons:
        assert c["startCommand"].startswith("python scripts/scheduled_jobs.py ")
        assert len(c["schedule"].split()) == 5
        assert any(e.get("key") == "DATABASE_URL" for e in c["envVars"])


def test_no_secrets_in_blueprint():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8").lower()
    for marker in ("password:", "secret_key:", "begin private key", "postgres://", "postgresql://"):
        assert marker not in text


def test_cloud_requirements_exclude_heavy_optional_ml():
    reqs = {line.split("#")[0].strip().split("[")[0].split(">")[0].split("=")[0].lower()
            for line in (ROOT / "requirements-cloud.txt").read_text(encoding="utf-8").splitlines()
            if line.split("#")[0].strip()}
    assert not reqs & {"torch", "torchvision", "torchaudio", "transformers", "pytest"}
    assert {"fastapi", "uvicorn", "sqlmodel", "alembic", "psycopg2-binary", "yfinance", "h2"} <= reqs


def test_postgres_scheme_is_normalised(monkeypatch):
    import config
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/db")
    try:
        importlib.reload(config)
        assert config.DATABASE_URL == "postgresql://u:p@host:5432/db"
    finally:
        monkeypatch.delenv("DATABASE_URL")
        importlib.reload(config)


def test_performance_overview_reads_legacy_from_database_without_tracker_file(monkeypatch, tmp_path):
    import analytics.performance_overview as po
    from db.models.tracker import Recommendation, RecommendationValidation
    missing = tmp_path / "tracker.db"
    monkeypatch.setattr(po, "TRACKER_DB", missing)
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        for i, (sig, ver, ok) in enumerate([("BUY", None, 1), ("SELL", "v1.0", 0), ("HOLD", "v1.1", 1)]):
            r = Recommendation(saved_date="2026-07-01", symbol=f"S{i}.NS", stock=f"S{i}", signal=sig, cmp=100.0,
                               engine_version=ver)
            s.add(r)
            s.commit()
            s.add(RecommendationValidation(recommendation_id=r.id, validation_date="2026-07-15", return_pct=1.5,
                                           success=ok))
        s.commit()
        out = po.overview(s)["sections"]
    assert not missing.exists()                                   # never created an empty tracker.db
    assert out["legacy_signals"]["count"] == 2 and out["post_fix_signals"]["count"] == 1
    assert out["legacy_signals"]["horizon"].startswith("Validated after at least 5 trading days; actual holding 14")


def test_free_blueprint_fits_the_free_plan():
    bp = yaml.safe_load((ROOT / "deploy" / "render-free" / "render.yaml").read_text(encoding="utf-8"))
    assert bp["databases"][0]["plan"] == "free"
    services = bp["services"]
    assert [s["type"] for s in services] == ["web"]                  # no cron jobs on the free plan
    web = services[0]
    assert web["plan"] == "free" and "preDeployCommand" not in web
    assert web["startCommand"].startswith("alembic upgrade head && uvicorn api.main:app")
    env = {e["key"]: e for e in web["envVars"]}
    assert env["ENGINE_RUN_ALLOW_ML"]["value"] == "false" and env["ENGINE_RUN_MAX_WORKERS"]["value"] == "2"
    assert env["JWT_SECRET_KEY"] == {"key": "JWT_SECRET_KEY", "generateValue": True}
    assert env["APP_ENV"]["value"] == "production"
    text = (ROOT / "deploy" / "render-free" / "render.yaml").read_text(encoding="utf-8").lower()
    assert "postgres://" not in text and "password:" not in text


def test_ml_signal_can_be_disabled_for_small_servers(monkeypatch):
    import config
    import engine_runs.service as runs
    monkeypatch.setenv("ENGINE_RUN_ALLOW_ML", "false")
    try:
        importlib.reload(config)
        assert config.ENGINE_RUN_ALLOW_ML is False
    finally:
        monkeypatch.delenv("ENGINE_RUN_ALLOW_ML")
        importlib.reload(config)
    assert config.ENGINE_RUN_ALLOW_ML is True
    # the engine gates the per-run include_ml request on the server switch
    src = Path(runs.__file__).read_text(encoding="utf-8")
    assert 'cfg.get("include_ml", True) and ENGINE_RUN_ALLOW_ML' in src


def _cron_hours(field: str) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        if part == "*":
            return set(range(24))
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def test_daily_ranking_and_price_crons_fit_the_nse_session():
    crons = {s["name"]: s for s in _blueprint()["services"] if s["type"] == "cron"}
    ranking, prices = crons["stocklens-ranking"], crons["stocklens-prices"]
    minute, hour, _, _, dow = ranking["schedule"].split()
    # Schedules are UTC. The ranking job needs final closing data (16:00 IST =
    # 10:30 UTC); it fires repeatedly after that so a late provider still
    # gets a run, and the per-day slot lock makes the extra triggers no-ops.
    assert dow == "1-5" and minute.startswith("*/")
    assert max(_cron_hours(hour)) * 60 + 45 >= 10 * 60 + 30 and min(_cron_hours(hour)) <= 10
    assert ranking["startCommand"].endswith("scheduled_jobs.py ranking")
    # Prices: every 15 minutes covering 09:15-16:00 IST (03:45-10:30 UTC).
    minute, hour, _, _, dow = prices["schedule"].split()
    assert minute == "*/15" and dow == "1-5" and {4, 5, 6, 7, 8, 9, 10} <= _cron_hours(hour)
    for c in crons.values():
        assert {"fromGroup": "stocklens-shared"} in c["envVars"]
    shared = {e["key"]: e["value"] for g in _blueprint()["envVarGroups"] for e in g["envVars"]}
    assert shared["ENGINE_RUN_ALLOW_ML"] == "false" and shared["ENGINE_RUN_MAX_WORKERS"] == "2"


def test_scheduling_never_runs_as_an_in_process_loop():
    # Render Free has no cron: the daily ranking is started by an operator
    # (admin console) rather than by a background loop inside the API.
    for folder in ("api", "scheduling", "prices"):
        for f in (ROOT / folder).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            assert "while True" not in text, f
            assert "BackgroundScheduler" not in text and "apscheduler" not in text.lower(), f
    free = (ROOT / "deploy" / "render-free" / "render.yaml").read_text(encoding="utf-8")
    assert "DAILY RANKING" in free and "DOES NOT RUN AUTOMATICALLY" in free
