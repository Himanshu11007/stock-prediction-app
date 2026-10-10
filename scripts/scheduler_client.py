"""
scripts/scheduler_client.py — trigger one StockLens backend job and wait for
its result. Used by the GitHub Actions workflows (.github/workflows/stocklens-*.yml);
contains no prediction logic. Standard library only.

  STOCKLENS_API_URL=https://<api-host>  STOCKLENS_SCHEDULER_TOKEN=<secret> \
      python scripts/scheduler_client.py --job TOMORROW_EOD [--timeout-minutes 35]

Behaviour:
  * wakes the API (GET /api/v1/health) - a sleeping free-tier instance can
    take minutes to start; bounded retries;
  * POST /api/v1/scheduler/jobs/{job}; connection errors and 502/503/504 are
    retried a bounded number of times (the backend slot lock makes repeats
    harmless); 401/404/422 are not retried;
  * ACCEPTED / RUNNING: polls the slot until a terminal state or the timeout;
  * exit 0 for COMPLETED, DONE, OK and SKIPPED (an expected non-run: holiday,
    duplicate, data not ready - the monitor catches a snapshot that never
    arrives); exit 1 for FAILED, ALERT, NOT_FOUND after acceptance, timeout or
    an HTTP error; exit 2 for missing configuration.
The token is sent only in the Authorization header and is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

JOB_TYPES = ("TODAY_PREOPEN", "TODAY_CONFIRMED", "TOMORROW_EOD", "OUTCOME_EVALUATION", "SNAPSHOT_MONITOR",
             "NEWS_INGESTION")
SUCCESS = ("COMPLETED", "DONE", "OK", "SKIPPED")
FAILURE = ("FAILED", "ALERT")
PENDING = ("ACCEPTED", "RUNNING")
RETRY_HTTP = (502, 503, 504)


class HttpResult:
    def __init__(self, code: int, body: dict):
        self.code, self.body = code, body


def http(method: str, url: str, token: Optional[str], timeout: float = 60.0) -> HttpResult:
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return HttpResult(r.status, json.loads(r.read() or b"{}"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except ValueError:
            body = {}
        return HttpResult(e.code, body)


def log(msg: str, level: str = "notice") -> None:
    print(f"::{level}::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"[{level}] {msg}", flush=True)


def summary(result: dict) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"### StockLens {result.get('job_type')} - {result.get('status')}\n\n```json\n"
                    f"{json.dumps(result, indent=1, default=str)[:6000]}\n```\n")


def run(job: str, base: str, token: str, timeout_minutes: float = 35, poll_seconds: float = 20,
        attempts: int = 3, backoff_seconds: float = 30, wake_minutes: float = 4,
        transport: Callable[..., HttpResult] = http, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic) -> int:
    base = base.rstrip("/")
    # 1. wake the service (free instances sleep)
    deadline = clock() + wake_minutes * 60
    while True:
        try:
            if transport("GET", f"{base}/api/v1/health", None, 30).code == 200:
                break
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if clock() > deadline:
            log("API did not become healthy in time", "error")
            return 1
        sleep(15)
    # 2. trigger (bounded retries on transient errors)
    res = None
    for i in range(attempts):
        try:
            res = transport("POST", f"{base}/api/v1/scheduler/jobs/{job}", token, 120)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log(f"trigger attempt {i + 1}/{attempts}: {type(e).__name__}", "warning")
            res = None
        if res is not None and res.code not in RETRY_HTTP:
            break
        if i + 1 < attempts:
            sleep(backoff_seconds * (i + 1))
    if res is None or res.code != 200:
        log(f"trigger failed: HTTP {res.code if res else 'no response'} "
            f"{(res.body.get('detail') or res.body.get('message')) if res else ''}", "error")
        return 1
    data: dict[str, Any] = res.body.get("data") or {}
    # 3. poll while the backend works
    deadline = clock() + timeout_minutes * 60
    not_found = 0
    while data.get("status") in PENDING or (data.get("status") == "NOT_FOUND" and not_found < 9):
        if clock() > deadline:
            log(f"{job} still {data.get('status')} after {timeout_minutes} min; reported as a failure", "error")
            summary({**data, "status": "TIMEOUT"})
            return 1
        sleep(poll_seconds)
        try:
            r = transport("GET", f"{base}/api/v1/scheduler/jobs/{job}/runs/{data['slot']}", token, 60)
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
        if r.code != 200:
            if r.code in RETRY_HTTP:
                continue
            log(f"status check failed: HTTP {r.code}", "error")
            return 1
        data = r.body.get("data") or {}
        not_found = not_found + 1 if data.get("status") == "NOT_FOUND" else 0
    summary(data)
    state = data.get("status")
    detail = data.get("reason") or (data.get("result") or {}).get("reason") or data.get("code") or ""
    if state in SUCCESS and not (isinstance(data.get("result"), dict) and data["result"].get("status") in FAILURE):
        log(f"{job}: {state} {detail}".strip())
        return 0
    log(f"{job}: {state} {detail} {json.dumps(data.get('result') or data.get('problems') or '', default=str)[:900]}",
        "error")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, choices=JOB_TYPES)
    ap.add_argument("--timeout-minutes", type=float, default=35)
    ap.add_argument("--poll-seconds", type=float, default=20)
    args = ap.parse_args(argv)
    base, token = os.environ.get("STOCKLENS_API_URL", "").strip(), os.environ.get("STOCKLENS_SCHEDULER_TOKEN", "")
    if not base.startswith("https://") and not base.startswith("http://localhost"):
        log("STOCKLENS_API_URL must be an https:// URL (repository variable)", "error")
        return 2
    if not token:
        log("STOCKLENS_SCHEDULER_TOKEN secret is not set", "error")
        return 2
    return run(args.job, base, token, args.timeout_minutes, args.poll_seconds, transport=http)


if __name__ == "__main__":
    sys.exit(main())
