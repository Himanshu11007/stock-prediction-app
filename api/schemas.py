"""
api/schemas.py — Pydantic request/response models for the StockAI Pro API.

These define the API's contract. No business logic lives here — only
type-validated data shapes that routes and services pass between each other
and that FastAPI uses to auto-generate /docs and /redoc.
"""
from __future__ import annotations

from typing import Optional, Any

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Common / error envelope
# ══════════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    success: bool = False
    error:   str
    details: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# 1. Health
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:  str = "ok"
    app:     str = "StockAI Pro API"
    version: str = "0.1.0"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Analyze single stock
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeStockRequest(BaseModel):
    symbol: str = Field(..., examples=["RELIANCE.NS"])


class AnalyzeStockResponse(BaseModel):
    symbol:        str
    stock:         str
    signal:        str
    score:         float
    confidence:    float
    accuracy:      float
    news_score:    float
    regime:        str
    weekly_trend:  str
    daily_trend:   str
    target:        Optional[float] = None
    stop_loss:     Optional[float] = None
    factors:       list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# 3 & 4. Top Picks scan — start / status
# ══════════════════════════════════════════════════════════════════════════════

class StartScanRequest(BaseModel):
    category: str = Field(..., examples=["Large Cap", "Mid Cap", "Small Cap"])


class StartScanResponse(BaseModel):
    scan_id:  str
    status:   str
    category: str


class ScanStatusResponse(BaseModel):
    scan_id:  str
    status:   str   # running | completed | failed | not_found
    progress: int
    total:    int
    message:  str


# ══════════════════════════════════════════════════════════════════════════════
# 5. Top Picks result
# ══════════════════════════════════════════════════════════════════════════════

class TopPickItem(BaseModel):
    symbol:     str
    stock:      str
    signal:     str
    score:      float
    confidence: float
    accuracy:   float


class TopPicksResultResponse(BaseModel):
    scan_id: str
    status:  str
    results: list[dict[str, Any]] = []


# ══════════════════════════════════════════════════════════════════════════════
# 6. Tracker — recommendations
# ══════════════════════════════════════════════════════════════════════════════

class RecommendationSaveRequest(BaseModel):
    symbol:     str
    stock:      str
    signal:     str
    cmp:        float
    score:      float = Field(..., description="Confluence score (0-1)")
    confidence: float
    news_score: float
    accuracy:   Optional[float] = 0.0
    target:     Optional[float] = None
    stop_loss:  Optional[float] = None


class RecommendationSaveResponse(BaseModel):
    success: bool = True
    row_id:  int


# ══════════════════════════════════════════════════════════════════════════════
# 7. Validation
# ══════════════════════════════════════════════════════════════════════════════

class ValidateOldResponse(BaseModel):
    validated_count: int
    status: str = "success"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Performance
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceSummaryResponse(BaseModel):
    total:        int
    successful:   int
    failed:       int
    success_rate: float
    avg_return:   float
    best_return:  float
    worst_return: float


# ══════════════════════════════════════════════════════════════════════════════
# 9. Logs
# ══════════════════════════════════════════════════════════════════════════════

class LogsResponse(BaseModel):
    lines: list[str]
    count: int


class ClearLogsResponse(BaseModel):
    success: bool


# ══════════════════════════════════════════════════════════════════════════════
# Response envelope helper
# ══════════════════════════════════════════════════════════════════════════════

def success_envelope(data, message: str = "OK") -> dict:
    """
    Wrap a successful result in the standard API response shape:
        {"success": true, "data": {...}, "message": "..."}

    `data` can be a dict, list, or any JSON-serialisable value — FastAPI
    handles serialisation of nested Pydantic models / dataclasses / plain
    dicts the same way it would for a raw return value.
    """
    return {"success": True, "data": data, "message": message}