"""
tests/test_vps_deploy.py — the VPS scheduler files (deploy/vps): the cron
schedule (IST) runs each job in its valid window, never TODAY_CONFIRMED, as
the unprivileged user; scripts never print secrets; the job runner reads the
env file literally (special characters in a password are not shell-expanded)
and reports only "set" / "MISSING" for secrets.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VPS = ROOT / "deploy" / "vps"
JOBS = {"NEWS_INGESTION", "TODAY_PREOPEN", "SNAPSHOT_MONITOR", "TOMORROW_EOD", "ranking", "OUTCOME_EVALUATION", "update"}


def _cron():
    rows = []
    for line in (VPS / "stocklens.cron").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" in line.split()[0]:
            continue
        m, h, dom, mon, dow, user, cmd, job = line.split()
        rows.append({"m": int(m), "h": h, "dow": dow, "user": user, "cmd": cmd, "job": job})
    return rows


def _hours(h):
    if "-" in h:
        a, b = h.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in h.split(",")]


def test_cron_runs_every_job_in_its_window_as_the_service_user():
    rows = _cron()
    assert {r["job"] for r in rows} == JOBS                                       # TODAY_CONFIRMED never scheduled
    assert all(r["user"] == "stocklens" and r["cmd"] == "$J" for r in rows)
    times = {j: sorted((hh, r["m"]) for r in rows if r["job"] == j for hh in _hours(r["h"])) for j in JOBS}
    assert all((h, m) < (9, 15) for h, m in times["TODAY_PREOPEN"]) and len(times["TODAY_PREOPEN"]) == 2
    for job in ("ranking", "TOMORROW_EOD", "OUTCOME_EVALUATION"):
        assert all((h, m) >= (16, 0) for h, m in times[job]), job                 # final daily bars exist
    assert times["NEWS_INGESTION"] == [(h, 10) for h in range(6, 23)]            # hourly 06:10-22:10, every day
    assert all(r["dow"] == "1-5" for r in rows if r["job"] not in ("NEWS_INGESTION", "update"))
    assert len(times["ranking"]) == 3                                            # 16:35 plus two backups


def test_scripts_never_print_secrets():
    run = (VPS / "run-job.sh").read_text(encoding="utf-8")
    token = (VPS / "make-token.sh").read_text(encoding="utf-8")
    setup = (VPS / "setup.sh").read_text(encoding="utf-8")
    for text in (run, token, setup):
        assert "set -x" not in text
        assert not re.search(r"echo[^\n]*\$(\{)?(STOCKLENS_SCHEDULER_TOKEN|DATABASE_URL)\b", text)
    assert not re.search(r"echo[^\n]*\$token", token) and "unset token" in token
    example = (VPS / "stocklens.env.example").read_text(encoding="utf-8")
    assert re.search(r"^STOCKLENS_SCHEDULER_TOKEN=$", example, re.M) and re.search(r"^DATABASE_URL=$", example, re.M)
    assert "0640" in setup and "root:stocklens" in setup and "-o root -g stocklens" in token


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock") and sys.platform.startswith("linux")),
                    reason="needs Linux bash and flock")
def test_runner_reads_the_env_file_literally_and_reports_only_presence(tmp_path):
    app, venv, logs, locks = tmp_path / "app", tmp_path / "venv" / "bin", tmp_path / "log", tmp_path / "lock"
    for d in (app, venv, logs, locks):
        d.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    env_file = tmp_path / "stocklens.env"
    env_file.write_text("# comment\nSTOCKLENS_SCHEDULER_TOKEN=tok-$(touch PWNED)&x\n"
                        "DATABASE_URL=postgresql://u:p&a$b;c@h/db\nlower=ignored\n", encoding="utf-8")
    env = {**os.environ, "STOCKLENS_APP": str(app), "STOCKLENS_VENV": str(tmp_path / "venv"),
           "STOCKLENS_ENV_FILE": str(env_file), "STOCKLENS_LOG_DIR": str(logs), "STOCKLENS_LOCK_DIR": str(locks)}
    env.pop("STOCKLENS_API_URL", None)
    out = subprocess.run(["bash", str(VPS / "run-job.sh"), "check"], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "STOCKLENS_SCHEDULER_TOKEN: set" in out.stdout and "DATABASE_URL: set" in out.stdout
    assert "STOCKLENS_API_URL: MISSING" in out.stdout and "Python 3" in out.stdout
    assert "tok-" not in out.stdout and "p&a" not in out.stdout                       # values never printed
    assert not (tmp_path / "PWNED").exists() and not (app / "PWNED").exists()         # nothing executed
    bad = subprocess.run(["bash", str(VPS / "run-job.sh"), "nope"], env=env, capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown job" in (logs / "nope.log").read_text()
