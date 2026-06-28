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
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import analysis, top_picks, tracker, performance, logs
from api.schemas import HealthResponse, ErrorResponse

from storage.recommendation_validation import migrate_schema
from utils.logger import get_logger, configure_logging
from config import ENABLE_DEBUG_LOGS

# ── Logging setup — reuses the existing centralized logger, same config ──────
configure_logging(debug=ENABLE_DEBUG_LOGS)
logger = get_logger(__name__)

# Ensure the validation table/columns exist before any request touches it
migrate_schema()

app = FastAPI(
    title="StockAI Pro API",
    description=(
        "REST API around the existing StockAI Pro ML + sentiment stock "
        "recommendation engine. Built so the same engine can be consumed "
        "by the Streamlit frontend, a future mobile app, and a future "
        "subscription product — without duplicating any ML or scoring logic."
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


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log endpoint, method, success/failure, and execution time for every call."""
    start = time.time()
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.error(
            "%s %s -> EXCEPTION after %.1fms: %s",
            request.method, request.url.path, duration_ms, e,
        )
        raise


# ── Global exception handler — consistent error envelope ─────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc),
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(analysis.router,    prefix="/api", tags=["Analysis"])
app.include_router(top_picks.router,   prefix="/api", tags=["Top Picks"])
app.include_router(tracker.router,     prefix="/api", tags=["Tracker"])
app.include_router(performance.router, prefix="/api", tags=["Performance"])
app.include_router(logs.router,        prefix="/api", tags=["Logs"])


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check — no auth required."""
    return {"status": "ok", "app": "StockAI Pro API", "version": "0.1.0"}


@app.on_event("startup")
def on_startup():
    logger.info("StockAI Pro API starting up")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("StockAI Pro API shutting down")