"""
api/routes/logs.py — Application log viewing and clearing.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import LogsResponse, ClearLogsResponse
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)
router = APIRouter()


@router.get("/logs/latest", response_model=LogsResponse)
def latest_logs(lines: int = 100):
    """Return the last N lines from the application log file."""
    try:
        log_lines = services.get_latest_logs(lines=lines)
        return {"lines": log_lines, "count": len(log_lines)}
    except Exception as e:
        log_exception(logger, "latest_logs failed", e)
        raise HTTPException(status_code=500, detail="Failed to read logs")


@router.delete("/logs/clear", response_model=ClearLogsResponse)
def clear_logs():
    """Clear (truncate) the application log file."""
    try:
        success = services.clear_logs()
        return {"success": success}
    except Exception as e:
        log_exception(logger, "clear_logs failed", e)
        raise HTTPException(status_code=500, detail="Failed to clear logs")