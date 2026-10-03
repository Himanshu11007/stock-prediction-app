import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# On Linux (Streamlit Cloud) use /tmp — always writable and survives the session.
# On Windows (local dev) use the local storage/ directory.
STORAGE_DIR = Path("/tmp/stockai_storage") if os.name == "posix" else Path("storage")

CACHE_FILE = STORAGE_DIR / "recommendations_cache.json"
TRACKER_DB = STORAGE_DIR / "tracker.db"

# ── Stock universes ────────────────────────────────────────────────────────────
UNIVERSE_LARGECAP  = DATA_DIR / "largecap.csv"
UNIVERSE_MIDCAP    = DATA_DIR / "midcap.csv"
UNIVERSE_SMALLCAP  = DATA_DIR / "smallcap.csv"
UNIVERSE_LEGACY    = DATA_DIR / "nifty50.csv"

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]

# ── Scanner ────────────────────────────────────────────────────────────────────
SCAN_TTL_SECONDS  = 3600
SCAN_MAX_STOCKS   = 50
SCAN_MAX_WORKERS  = 8

# ── Quality filters ────────────────────────────────────────────────────────────
MIN_ACCURACY              = 0.44  # fast=True single-split runs ~3-4 pts lower than walk-forward CV
MIN_CONFIDENCE            = 55.0
MIN_AVG_VOLUME            = 500_000
MIN_CONFLUENCE_SCORE      = 0.55    # 0-1 normalised; below this → filtered out
RSI_BUY_MIN               = 35
RSI_BUY_MAX               = 70      # relaxed from 65 — ADX filter handles trend
VOLATILITY_SPIKE_MULTIPLIER = 2.5

# ── Confluence signal thresholds (score 0–100) ─────────────────────────────────
STRONG_BUY_MIN  = 72
BUY_MIN         = 58
HOLD_MIN        = 42
SELL_MIN        = 28
# below SELL_MIN → STRONG SELL

# ── Recommendation engine version ────────────────────────────────────────────
# Stamped on every persisted recommendation (recommendation_validation.
# engine_version). NULL / "v1.0" rows were produced before the Phase 11A
# temporal-integrity fix (ML prediction made on a training row, cmp =
# Close[D-1]) and must never be pooled with "v1.1"+ rows for model evaluation.
# See docs/PRODUCTION_TEMPORAL_INTEGRITY.md.
RECOMMENDATION_ENGINE_VERSION    = "v1.1"
PRE_TEMPORAL_FIX_ENGINE_VERSIONS = (None, "v1.0")

# ── Confluence pillar weights (must sum to 1.0) ────────────────────────────────
W_ML_DIR    = 0.15
W_ML_CONF   = 0.05
W_TECH      = 0.30
W_NEWS      = 0.10
W_VOLUME    = 0.05
W_REGIME    = 0.10
W_TIMEFRAME = 0.15
W_MOMENTUM  = 0.10

# ── Database ───────────────────────────────────────────────────────────────────
# Dev default: SQLite file under STORAGE_DIR. Production: set the DATABASE_URL
# env var to a PostgreSQL DSN, e.g. postgresql+psycopg2://user:pass@host/dbname.
# All application code must go through db/session.py — never hardcode a dialect.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'app.db').as_posix()}"
)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR          = STORAGE_DIR / "logs"
LOG_FILE         = LOG_DIR / "app.log"
# ── Deployment environment ───────────────────────────────────────────────────
# "development" (default) or "production". Production refuses to start with
# insecure defaults (see api/main.py:_check_production_settings).
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"

# Comma-separated list of allowed browser origins. The mobile app is not a
# browser and is unaffected by CORS; this only matters for web clients.
# Development default "*"; production must list explicit origins.
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS", "" if IS_PRODUCTION else "*").split(",") if o.strip()
]

# ── Product analysis engine (FQVF + ranking) ─────────────────────────────────
FQVF_ENGINE_VERSION    = "fqvf-v1.0"
RANKING_ENGINE_VERSION = "ranking-v1.0"
# Ranking Engine v1.0 is FROZEN after the out-of-sample validation in
# docs/RANKING_VALIDATION_V1.md: weights, rules and components must not change
# without a new version and a new validation (tests/test_ranking_validation.py).
RANKING_ENGINE_STATUS  = "FROZEN"
# Fundamentals older than this are re-fetched by an engine run.
FUNDAMENTALS_TTL_HOURS = int(os.environ.get("FUNDAMENTALS_TTL_HOURS", "24"))
# A market-data snapshot whose last bar is older than this many calendar days
# is STALE (covers weekends + one holiday).
MARKET_DATA_STALE_DAYS = int(os.environ.get("MARKET_DATA_STALE_DAYS", "4"))
# Fundamentals older than this are STALE for data-health purposes.
FUNDAMENTALS_STALE_DAYS = int(os.environ.get("FUNDAMENTALS_STALE_DAYS", "7"))
ENGINE_RUN_MAX_WORKERS = int(os.environ.get("ENGINE_RUN_MAX_WORKERS", "8"))

ENABLE_DEBUG_LOGS = False   # set True to write DEBUG-level pillar diagnostics

# ── Phase 8 authentication: Google / Apple SSO ──────────────────────────────────
# OAuth client id Google issued for this app - MUST match the `aud` claim of
# every Google id_token this server accepts, or any Google user's token for
# a completely different app would be accepted here. Required in production;
# left unset, Google sign-in always fails closed (see
# auth/external_identity.py:GoogleIdentityVerifier).
GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")

# The Sign in with Apple Services ID registered for this app - MUST match
# the `aud` claim of every Apple identity token this server accepts. Same
# fail-closed behavior as above when unset.
APPLE_SERVICES_ID = os.environ.get("APPLE_SERVICES_ID")

# ── Phase 8 authentication: OTP ──────────────────────────────────────────────────
OTP_CODE_LENGTH               = int(os.environ.get("OTP_CODE_LENGTH", "6"))
OTP_EXPIRE_SECONDS            = int(os.environ.get("OTP_EXPIRE_SECONDS", "300"))       # 5 minutes
OTP_MAX_ATTEMPTS              = int(os.environ.get("OTP_MAX_ATTEMPTS", "5"))
OTP_RESEND_COOLDOWN_SECONDS   = int(os.environ.get("OTP_RESEND_COOLDOWN_SECONDS", "30"))
OTP_MAX_REQUESTS_PER_WINDOW   = int(os.environ.get("OTP_MAX_REQUESTS_PER_WINDOW", "5"))
OTP_REQUEST_WINDOW_SECONDS    = int(os.environ.get("OTP_REQUEST_WINDOW_SECONDS", "1800"))  # 30 minutes

# Logs each generated OTP code to the server log instead of actually
# delivering it - lets local development exercise the full OTP flow without
# a real SMS/email provider account. NEVER set this in production: it
# exists purely so OTP_DEV_LOG_CODES=true can be a loud, explicit opt-in
# rather than an accidental default (see auth/otp_delivery.py).
OTP_DEV_LOG_CODES = os.environ.get("OTP_DEV_LOG_CODES", "false").lower() == "true"

# Production email OTP delivery via SMTP (see auth/otp_delivery.py:
# SmtpOtpDeliveryService) - works with any provider that exposes an SMTP
# endpoint (SendGrid, Mailgun, Amazon SES, Postmark, a corporate relay,
# ...). Leaving OTP_SMTP_HOST unset means no real provider is configured;
# SMTP_PASSWORD must never be committed - set it via environment variable
# or deployment secret storage only.
OTP_SMTP_HOST          = os.environ.get("OTP_SMTP_HOST")  # e.g. "smtp.sendgrid.net" - unset = not configured
OTP_SMTP_PORT          = int(os.environ.get("OTP_SMTP_PORT", "587"))
OTP_SMTP_USERNAME      = os.environ.get("OTP_SMTP_USERNAME", "")
OTP_SMTP_PASSWORD      = os.environ.get("OTP_SMTP_PASSWORD", "")  # SECRET - env var / deployment secret only, never commit
OTP_SMTP_FROM_ADDRESS  = os.environ.get("OTP_SMTP_FROM_ADDRESS", "no-reply@stockaipro.app")
OTP_SMTP_USE_TLS       = os.environ.get("OTP_SMTP_USE_TLS", "true").lower() == "true"