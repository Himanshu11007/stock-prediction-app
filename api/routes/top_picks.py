"""
api/routes/top_picks.py — Background scan trigger, status, and result.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import (
    StartScanRequest, StartScanResponse,
    ScanStatusResponse, TopPicksResultResponse,
)
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)
router = APIRouter()


@router.post("/top-picks/start", response_model=StartScanResponse)
def start_scan(payload: StartScanRequest):
    """
    Trigger a background scan and receive a scan_id to poll.

    Note: the underlying scanner (scanner/background.py) always scans all
    three categories (Large/Mid/Small Cap) in a single background run —
    that existing behaviour is unchanged. This endpoint hands back a
    scan_id bound to the category you asked about, so status/result calls
    know which cached results to report on.
    """
    try:
        result = services.start_top_picks_scan(payload.category)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_exception(logger, "start_scan failed", e)
        raise HTTPException(status_code=500, detail="Failed to start scan")


@router.get("/top-picks/status/{scan_id}", response_model=ScanStatusResponse)
def scan_status(scan_id: str):
    """Poll the status of a previously started scan."""
    try:
        result = services.get_scan_status(scan_id)
        return result
    except Exception as e:
        log_exception(logger, f"scan_status failed for {scan_id}", e)
        raise HTTPException(status_code=500, detail="Failed to fetch scan status")


@router.get("/top-picks/result/{scan_id}", response_model=TopPicksResultResponse)
def scan_result(scan_id: str):
    """Fetch the cached results for a previously started scan."""
    try:
        result = services.get_scan_result(scan_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_exception(logger, f"scan_result failed for {scan_id}", e)
        raise HTTPException(status_code=500, detail="Failed to fetch scan result")