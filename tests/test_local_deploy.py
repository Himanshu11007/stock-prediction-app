"""
tests/test_local_deploy.py — the Windows laptop scripts (deploy/local): the
Task Scheduler entries run the same jobs at the same IST times as the server
schedule (deploy/vps/stocklens.cron), never TODAY_CONFIRMED; no script prints
a secret; the env file is read literally; the database listens on this
laptop only.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "deploy" / "local"
SCRIPTS = {p.name: p.read_text(encoding="utf-8") for p in LOCAL.glob("*.ps1")}


def _cron_times():
    times = {}
    for line in (ROOT / "deploy" / "vps" / "stocklens.cron").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" in line.split()[0]:
            continue
        m, h, _dom, _mon, dow, _user, _cmd, job = line.split()
        hours = range(int(h.split("-")[0]), int(h.split("-")[1]) + 1) if "-" in h else [int(h)]
        times.setdefault(job, set()).update((f"{hh:02d}:{int(m):02d}", dow) for hh in hours)
    return times


def _task_times():
    text = SCRIPTS["install-tasks.ps1"]
    out = {}
    for job, body in re.findall(r"'(\w+)'\s+=\s+@\((.*)\)", text):
        if "Weekdays" in body:
            out[job] = {(t, "1-5") for t in re.findall(r"'(\d\d:\d\d)'", body)}
        elif job == "update":
            out[job] = {(t, "*") for t in re.findall(r"'(\d\d:\d\d)'", body)}
        elif job == "NEWS_INGESTION":
            start = re.search(r"\$news = New-ScheduledTaskTrigger -Daily -At '(\d\d):(\d\d)'", text)
            hours = int(re.search(r"-RepetitionDuration \(New-TimeSpan -Hours (\d+)", text).group(1))
            h0, m0 = int(start.group(1)), start.group(2)
            out[job] = {(f"{h:02d}:{m0}", "*") for h in range(h0, h0 + hours + 1)}
    return out


def test_tasks_match_the_server_schedule():
    assert _task_times() == _cron_times()
    for name in ("install-tasks.ps1", "run-job.ps1"):                             # code, not comments
        code = [l for l in SCRIPTS[name].splitlines() if not l.lstrip().startswith("#")]
        assert not any("TODAY_CONFIRMED" in l for l in code), name


def test_no_script_prints_a_secret_and_env_is_read_literally():
    secret = r"\$(env:)?(STOCKLENS_SCHEDULER_TOKEN|DATABASE_URL|JWT_SECRET_KEY|LOCAL_DB_PASSWORD|dbPassword|token|src|value)\b"
    for name, text in SCRIPTS.items():
        for line in text.splitlines():
            if re.search(r"Write-(Host|Log|Output)", line):
                assert not re.search(secret, line), (name, line)
        assert "Invoke-Expression" not in text and "iex " not in text, name
    common = SCRIPTS["common.ps1"]
    assert "-split '=', 2" in common and "SetEnvironmentVariable" in common       # literal, no expansion
    setup = SCRIPTS["setup.ps1"]
    assert "/inheritance:r" in setup                                              # only the user can read it
    assert "-p 127.0.0.1:5432:5432" in setup                                      # database: this laptop only
    assert "-e POSTGRES_PASSWORD -p" in setup                                     # password passed by name
    copy = SCRIPTS["copy-data.ps1"]
    assert "-AsSecureString" in copy and "read -r SRC" in copy                    # URL hidden, via stdin
    assert "-AsSecureString" in SCRIPTS["set-secret.ps1"]


def test_ps1_files_are_ascii():
    # Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page
    for name, text in SCRIPTS.items():
        assert text.isascii(), name
