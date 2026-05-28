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
MIN_ACCURACY              = 0.52
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

# ── Confluence pillar weights (must sum to 1.0) ────────────────────────────────
W_ML_DIR   = 0.35
W_ML_CONF  = 0.20
W_TECH     = 0.25
W_NEWS     = 0.10
W_VOLUME   = 0.05
W_REGIME   = 0.05
