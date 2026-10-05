"""
api/main.py — StockLens FastAPI backend MVP.

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
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import (
    analysis, top_picks, tracker, performance, logs, intelligence, auth, auth_sso, auth_otp,
    auth_devices, auth_password, admin, admin_masters, admin_notifications, notifications, watchlist, stocks,
    product, predictions, scheduler,
)
from api.schemas import HealthResponse

from storage.recommendation_validation import migrate_schema
from utils.logger import get_logger, configure_logging
from pathlib import Path

from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (API_DOCS_ENABLED, ENABLE_DEBUG_LOGS, IS_PRODUCTION, CORS_ALLOWED_ORIGINS, PRODUCT_DESCRIPTION,
                    PRODUCT_NAME, PRODUCT_TAGLINE)

# ── Logging setup — reuses the existing centralized logger, same config ──────
configure_logging(debug=ENABLE_DEBUG_LOGS)
logger = get_logger(__name__)

# Ensure the validation table/columns exist before any request touches it
migrate_schema()

API_VERSION = "v1"
API_PREFIX  = f"/api/{API_VERSION}"

app = FastAPI(
    title=f"{PRODUCT_NAME} API",
    description=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}. {PRODUCT_DESCRIPTION}",
    version="0.1.0",
    docs_url=None,     # served below with the StockLens favicon
    redoc_url=None,
    # /openapi.json, /docs and /redoc exist only when API_DOCS_ENABLED (off by
    # default in production: the schema lists every route, admin ones included).
    openapi_url="/openapi.json" if API_DOCS_ENABLED else None,
)

# Brand assets (favicon, app icons, social image) generated from the approved
# master icon by scripts/generate_brand_assets.py.
BRANDING_DIR = Path(__file__).resolve().parents[1] / "branding"
app.mount("/static/branding", StaticFiles(directory=str(BRANDING_DIR)), name="branding")


if API_DOCS_ENABLED:
    @app.get("/docs", include_in_schema=False)
    def swagger_docs():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{PRODUCT_NAME} API - Docs",
                                   swagger_favicon_url="/static/branding/favicon-32.png")

    @app.get("/redoc", include_in_schema=False)
    def redoc_docs():
        return get_redoc_html(openapi_url=app.openapi_url, title=f"{PRODUCT_NAME} API - ReDoc",
                              redoc_favicon_url="/static/branding/favicon-32.png")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(BRANDING_DIR / "favicon.ico")

def _check_production_settings() -> None:
    """Refuse to start a production deployment with insecure defaults."""
    import os
    problems = []
    if not os.environ.get("JWT_SECRET_KEY"):
        problems.append("JWT_SECRET_KEY is not set (tokens would be signed with an ephemeral key)")
    if not CORS_ALLOWED_ORIGINS or "*" in CORS_ALLOWED_ORIGINS:
        problems.append("CORS_ALLOWED_ORIGINS must list explicit origins (not '*')")
    if not os.environ.get("DATABASE_URL"):
        problems.append("DATABASE_URL is not set (would use the local SQLite dev database)")
    if problems:
        raise RuntimeError("Insecure production configuration: " + "; ".join(problems))


if IS_PRODUCTION:
    _check_production_settings()


def _warn_on_incomplete_auth_settings() -> None:
    """Features that fail closed when unconfigured - worth a loud log line,
    not a refusal to start."""
    from config import GOOGLE_ALLOWED_AUDIENCES, OTP_DEV_LOG_CODES, OTP_SMTP_HOST, PASSWORD_RESET_URL
    if not GOOGLE_ALLOWED_AUDIENCES:
        logger.warning("CONFIG | Google sign-in disabled: GOOGLE_OAUTH_CLIENT_ID is not set")
    if not PASSWORD_RESET_URL:
        logger.warning("CONFIG | password reset emails disabled: set PASSWORD_RESET_URL or FRONTEND_BASE_URL")
    if not OTP_SMTP_HOST:
        logger.warning("CONFIG | no email provider (OTP_SMTP_HOST): OTP and password-reset emails cannot be sent")
    if IS_PRODUCTION and OTP_DEV_LOG_CODES:
        logger.error("CONFIG | OTP_DEV_LOG_CODES is enabled in production - OTP codes are being logged")


_warn_on_incomplete_auth_settings()

# ── CORS — origins from config (CORS_ALLOWED_ORIGINS) ────────────────────────
# allow_credentials must be False: browsers reject allow_origins=["*"] with
# credentials, and this API uses bearer tokens, not cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security headers on every response ───────────────────────────────────────
# JSON API responses: no framing, no MIME sniffing, HTTPS only (HSTS is
# ignored by browsers on plain-HTTP development servers). No CSP here: API
# responses render no HTML; the docs pages (development) load Swagger assets.
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


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
    # The client gets a reference id only; the exception text and traceback
    # stay in the server log (they can contain paths, SQL or data values).
    ref = uuid.uuid4().hex[:12]
    logger.exception("Unhandled exception [ref=%s] on %s %s", ref, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error",
                 "details": f"Unexpected server error (reference {ref})"},
    )


# ── Routers — all under /api/v1 ───────────────────────────────────────────────
app.include_router(auth.router,        prefix=API_PREFIX, tags=["Auth"])
app.include_router(auth_sso.router,    prefix=API_PREFIX, tags=["Auth"])
app.include_router(auth_otp.router,    prefix=API_PREFIX, tags=["Auth"])
app.include_router(auth_devices.router, prefix=API_PREFIX, tags=["Auth"])
app.include_router(auth_password.router, prefix=API_PREFIX, tags=["Auth"])
app.include_router(admin.router,       prefix=API_PREFIX, tags=["Admin"])
app.include_router(admin_masters.router, prefix=API_PREFIX, tags=["Admin"])
app.include_router(product.public_router, prefix=API_PREFIX, tags=["App"])
app.include_router(product.router,     prefix=API_PREFIX, tags=["App"])
app.include_router(watchlist.router,   prefix=API_PREFIX, tags=["Watchlist"])
app.include_router(stocks.router,      prefix=API_PREFIX, tags=["Stocks"])
app.include_router(analysis.router,    prefix=API_PREFIX, tags=["Analysis"])
app.include_router(top_picks.router,   prefix=API_PREFIX, tags=["Top Picks"])
app.include_router(tracker.router,     prefix=API_PREFIX, tags=["Tracker"])
app.include_router(performance.router, prefix=API_PREFIX, tags=["Performance"])
app.include_router(logs.router,         prefix=API_PREFIX, tags=["Logs"])
app.include_router(intelligence.router, prefix=API_PREFIX, tags=["Intelligence"])
app.include_router(notifications.router, prefix=API_PREFIX, tags=["Notifications"])
app.include_router(admin_notifications.router, prefix=API_PREFIX, tags=["Admin"])
# Prediction Engine v2 (shadow): administrators only unless PREDICTION_V2_PUBLIC.
app.include_router(predictions.router, prefix=API_PREFIX, tags=["Predictions (shadow)"])
app.include_router(predictions.admin_router, prefix=API_PREFIX, tags=["Admin"])
# External scheduler job interface (scheduler token only; disabled unless configured).
app.include_router(scheduler.router, prefix=API_PREFIX, tags=["Scheduler"])


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check — no auth required."""
    return {"status": "ok", "app": f"{PRODUCT_NAME} API", "version": "0.1.0"}


@app.on_event("startup")
def on_startup():
    logger.info("%s API starting up", PRODUCT_NAME)


@app.on_event("shutdown")
def on_shutdown():
    logger.info("%s API shutting down", PRODUCT_NAME)