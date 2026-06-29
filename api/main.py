"""
api/main.py — StockAI Pro FastAPI backend MVP.

Run locally:
    uvicorn api.main:app --reload

Docs (auto-generated):
    http://localhost:8000/docs
    http://localhost:8000/redoc

This file adds a REST API alongside the existing Streamlit app (app.py).
It does not modify, replace, or import app.py — both can run independently
against the same underlying engine and the same SQLite database / JSON
caches in storage/.

Response contract
──────────────────
Every endpoint under /api/v1 returns one of two shapes:

  Success:  {"success": true,  "data": {...}, "message": "..."}
  Error:    {"success": false, "error": "...", "details": "..."}

Routes raise plain Python exceptions (ValueError, KeyError, or anything
else) and the centralized handlers below translate them into the error
shape with the correct HTTP status code — 400 / 404 / 500 respectively.
Routes never need to construct HTTPException themselves for these cases.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import analysis, top_picks, tracker, performance, logs
from api.schemas import HealthResponse

from storage.recommendation_validation import migrate_schema
from utils.logger import get_logger, configure_logging
from config import ENABLE_DEBUG_LOGS

# ── Logging setup — reuses the existing centralized logger, same config ──────
configure_logging(debug=ENABLE_DEBUG_LOGS)
logger = get_logger(__name__)

# Ensure the validation table/columns exist before any request touches it
migrate_schema()

API_VERSION = "v1"
API_PREFIX  = f"/api/{API_VERSION}"

app = FastAPI(
    title="StockAI Pro API",
    description=(
        "REST API around the existing StockAI Pro ML + sentiment stock "
        "recommendation engine. Built so the same engine can be consumed "
        "by the Streamlit frontend, a future .NET MAUI mobile app, and a "
        "future subscription product — without duplicating any ML or "
        "scoring logic."
    ),
    version="0.1.0",
)

# ── CORS — open for MVP; restrict allow_origins before production ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response logging middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every API call on entry and exit:
        API_REQUEST  | POST /api/v1/analyze-stock
        API_RESPONSE | POST /api/v1/analyze-stock | 200 | 923ms
    """
    logger.info("API_REQUEST | %s %s", request.method, request.url.path)
    start = time.time()
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "API_RESPONSE | %s %s | %d | %.1fms",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.error(
            "API_RESPONSE | %s %s | 500 | %.1fms | EXCEPTION: %s",
            request.method, request.url.path, duration_ms, e,
        )
        raise


# ── Centralized exception handlers — consistent error envelope ───────────────
# Routes raise plain ValueError / KeyError / Exception; these handlers turn
# them into the {"success": false, "error": ..., "details": ...} shape with
# the correct status code. Routes should not need their own try/except for
# these three cases — see api/routes/*.py for the pattern.

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("ValueError on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"success": False, "error": "Invalid request", "details": str(exc)},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    logger.warning("KeyError on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Not found", "details": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "details": str(exc)},
    )


# ── Routers — all under /api/v1 ───────────────────────────────────────────────
app.include_router(analysis.router,    prefix=API_PREFIX, tags=["Analysis"])
app.include_router(top_picks.router,   prefix=API_PREFIX, tags=["Top Picks"])
app.include_router(tracker.router,     prefix=API_PREFIX, tags=["Tracker"])
app.include_router(performance.router, prefix=API_PREFIX, tags=["Performance"])
app.include_router(logs.router,        prefix=API_PREFIX, tags=["Logs"])


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check — no auth required."""
    return {"status": "ok", "app": "StockAI Pro API", "version": "0.1.0"}


@app.on_event("startup")
def on_startup():
    logger.info("StockAI Pro API starting up")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("StockAI Pro API shutting down")