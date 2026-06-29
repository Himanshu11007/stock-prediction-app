"""
api/routes/top_picks.py — Background scan trigger, status, and result.
"""
from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import StartScanRequest, success_envelope

router = APIRouter()


@router.post("/top-picks/start")
def start_scan(payload: StartScanRequest):
    """
    Trigger a background scan and receive a scan_id to poll.

    Note: the underlying scanner (scanner/background.py) always scans all
    three categories (Large/Mid/Small Cap) in a single background run —
    that existing behaviour is unchanged. This endpoint hands back a
    scan_id bound to the category you asked about, so status/result calls
    know which cached results to report on. Returns immediately; does not
    block until the scan completes.
    """
    result = services.start_top_picks_scan(payload.category)
    return success_envelope(result, message="Scan started")


@router.get("/top-picks/status/{scan_id}")
def scan_status(scan_id: str):
    """Poll the status of a previously started scan."""
    result = services.get_scan_status(scan_id)
    return success_envelope(result, message="Scan status retrieved")


@router.get("/top-picks/result/{scan_id}")
def scan_result(scan_id: str):
    """
    Fetch the cached results for a previously started scan.

    Raises ValueError (→ 400) for an unknown scan_id — handled centrally.
    """
    result = services.get_scan_result(scan_id)
    return success_envelope(result, message="Scan result retrieved")