"""
api/routes/scheduler.py — job interface for the external scheduler
(GitHub Actions). Not for people: it accepts only the dedicated scheduler
bearer token, never a user or administrator session.

  POST /scheduler/jobs/{job_type}              start one job (or report it)
  GET  /scheduler/jobs/{job_type}/runs/{slot}  state of that job's slot

job_type is one of scheduling.remote.JobType (exact, upper case); slot is the
IST trading date (YYYY-MM-DD). Responses use the standard envelope; `status`
in `data` is ACCEPTED, RUNNING, COMPLETED, DONE, OK, SKIPPED, FAILED, ALERT or
NOT_FOUND. HTTP 401 for a missing or wrong token, 404 when the interface is
not configured (SCHEDULER_TOKEN_SHA256 unset), 422 for an unknown job type.
The token is compared as a SHA-256 digest in constant time and never logged.
"""
from __future__ import annotations

import hashlib
import hmac
import re

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

import config
from api.schemas import success_envelope
from scheduling import remote
from scheduling.remote import JobType
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/scheduler")

SLOT_PATTERN = r"^\d{4}-\d{2}-\d{2}(T\d{2})?$"          # date, or date + IST hour for NEWS_INGESTION


def get_engine():
    """Database engine used by jobs (tests override this dependency)."""
    from db.session import engine
    return engine


AUTH_FAIL_LIMIT, AUTH_FAIL_WINDOW = 10, 900        # failed authentications per client IP per 15 minutes
TRIGGER_LIMIT, TRIGGER_WINDOW = 30, 3600          # accepted triggers per job type per hour


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def require_scheduler_token(request: Request, engine=Depends(get_engine)) -> None:
    from sqlmodel import Session

    from auth import throttle
    expected = config.SCHEDULER_TOKEN_SHA256
    if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")
    ip = _ip(request)
    with Session(engine) as session:
        if throttle.count_recent(session, "scheduler_auth_fail", ip, window_seconds=AUTH_FAIL_WINDOW) >= AUTH_FAIL_LIMIT:
            logger.warning("SCHEDULER_AUTH_THROTTLED | %s", ip)
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed scheduler authentications",
                                headers={"Retry-After": str(AUTH_FAIL_WINDOW)})
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        digest = hashlib.sha256(token.strip().encode("utf-8")).hexdigest() if token.strip() else ""
        if scheme.lower() != "bearer" or not digest or not hmac.compare_digest(digest, expected):
            throttle.record(session, "scheduler_auth_fail", ip)
            session.commit()
            logger.warning("SCHEDULER_AUTH_REJECTED | %s | %s", ip, "missing" if not digest else "invalid")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "Scheduler token required" if not digest else "Invalid scheduler token",
                                headers={"WWW-Authenticate": "Bearer"})


@router.post("/jobs/{job_type}", dependencies=[Depends(require_scheduler_token)])
def start_job(job_type: JobType, request: Request, engine=Depends(get_engine)):
    """Preflight, then start the job in the background (SNAPSHOT_MONITOR runs
    inline). Duplicate triggers return the existing slot's state. Accepted
    triggers are rate-limited per job type and audit-logged (never the token)."""
    from sqlmodel import Session

    from auth import throttle
    with Session(engine) as session:
        try:
            throttle.check_and_record(session, "scheduler_trigger", job_type.value, limit=TRIGGER_LIMIT,
                                      window_seconds=TRIGGER_WINDOW, message="scheduler trigger rate exceeded")
        except throttle.ThrottledError as e:
            logger.warning("SCHEDULER_TRIGGER_THROTTLED | %s | %s", job_type.value, _ip(request))
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e),
                                headers={"Retry-After": str(e.retry_after_seconds)}) from None
    out = remote.trigger(engine, job_type)
    logger.info("SCHEDULER_AUDIT | trigger | %s | slot=%s | status=%s | code=%s | ip=%s", job_type.value,
                out.get("slot"), out.get("status"), out.get("code"), _ip(request))
    return success_envelope(out, message=f"{job_type.value}: {out['status']}")


@router.get("/jobs/{job_type}/runs/{slot}", dependencies=[Depends(require_scheduler_token)])
def job_status(job_type: JobType, slot: str = Path(..., pattern=SLOT_PATTERN), engine=Depends(get_engine)):
    out = remote.status(engine, job_type, slot)
    return success_envelope(out, message=f"{job_type.value} {slot}: {out['status']}")
