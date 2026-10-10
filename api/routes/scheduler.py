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

SLOT_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


def get_engine():
    """Database engine used by jobs (tests override this dependency)."""
    from db.session import engine
    return engine


def require_scheduler_token(request: Request) -> None:
    expected = config.SCHEDULER_TOKEN_SHA256
    if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        logger.warning("SCHEDULER_AUTH_MISSING | %s", request.client.host if request.client else "?")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Scheduler token required",
                            headers={"WWW-Authenticate": "Bearer"})
    digest = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, expected):
        logger.warning("SCHEDULER_AUTH_REJECTED | %s", request.client.host if request.client else "?")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid scheduler token",
                            headers={"WWW-Authenticate": "Bearer"})


@router.post("/jobs/{job_type}", dependencies=[Depends(require_scheduler_token)])
def start_job(job_type: JobType, engine=Depends(get_engine)):
    """Preflight, then start the job in the background (SNAPSHOT_MONITOR runs
    inline). Duplicate triggers return the existing slot's state."""
    out = remote.trigger(engine, job_type)
    return success_envelope(out, message=f"{job_type.value}: {out['status']}")


@router.get("/jobs/{job_type}/runs/{slot}", dependencies=[Depends(require_scheduler_token)])
def job_status(job_type: JobType, slot: str = Path(..., pattern=SLOT_PATTERN), engine=Depends(get_engine)):
    out = remote.status(engine, job_type, slot)
    return success_envelope(out, message=f"{job_type.value} {slot}: {out['status']}")
