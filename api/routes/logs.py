"""
api/routes/logs.py — Application log viewing and clearing (ADMIN only).

Logs can contain symbols, user ids and request paths, and clearing them
destroys audit evidence, so both routes require the ADMIN role.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api import services
from api.schemas import success_envelope
from auth.dependencies import require_admin

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("/logs/latest")
def latest_logs(lines: int = 100):
    """Return the last N lines from the application log file."""
    log_lines = services.get_latest_logs(lines=lines)
    return success_envelope(
        {"lines": log_lines, "count": len(log_lines)},
        message=f"Retrieved last {len(log_lines)} log line(s)",
    )


@router.delete("/logs/clear")
def clear_logs():
    """Clear (truncate) the application log file."""
    success = services.clear_logs()
    return success_envelope({"cleared": success}, message="Logs cleared")