from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
STORAGE_DIR = Path("storage")
CACHE_FILE  = STORAGE_DIR / "recommendations_cache.json"   # legacy single-file cache
TRACKER_DB  = STORAGE_DIR / "tracker.db"

# ── Stock universes ────────────────────────────────────────────────────────────
UNIVERSE_LARGECAP  = DATA_DIR / "largecap.csv"
UNIVERSE_MIDCAP    = DATA_DIR / "midcap.csv"
UNIVERSE_SMALLCAP  = DATA_DIR / "smallcap.csv"
UNIVERSE_LEGACY    = DATA_DIR / "nifty50.csv"   # fallback

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]

# ── Scanner ────────────────────────────────────────────────────────────────────
SCAN_TTL_SECONDS  = 3600   # re-scan at most once per hour
SCAN_MAX_STOCKS   = 50     # per-category cap (safety limit)
SCAN_MAX_WORKERS  = 8      # concurrent threads for parallel scanning

# ── Quality filters ────────────────────────────────────────────────────────────
MIN_ACCURACY    = 0.52
MIN_CONFIDENCE  = 55.0
MIN_AVG_VOLUME  = 500_000

RSI_BUY_MIN = 35
RSI_BUY_MAX = 65
VOLATILITY_SPIKE_MULTIPLIER = 2.5

# ── Decision engine ────────────────────────────────────────────────────────────
BUY_SCORE_THRESHOLD  =  0.25
SELL_SCORE_THRESHOLD = -0.25

# ── Prediction display ─────────────────────────────────────────────────────────
STRONG_BUY_SCORE          =  0.75
STRONG_BUY_CONFIDENCE     = 84.0
HOLD_CONFIDENCE_THRESHOLD = 65.0
HOLD_ACCURACY_THRESHOLD   = 0.70
HOLD_SCORE_THRESHOLD      = 0.15
