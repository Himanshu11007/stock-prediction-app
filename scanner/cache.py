"""
Per-category disk cache.

Files: storage/cache_<slug>.json  e.g. cache_large_cap.json
Each file: {"timestamp": <epoch>, "recommendations": [...]}
"""
import json
import time
from pathlib import Path
from config import STORAGE_DIR, SCAN_TTL_SECONDS


def _cache_path(category: str) -> Path:
    slug = category.lower().replace(" ", "_")
    return STORAGE_DIR / f"cache_{slug}.json"


def load_category_cache(category: str) -> list | None:
    """Return cached list for a category if still fresh, else None."""
    path = _cache_path(category)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - payload.get("timestamp", 0) < SCAN_TTL_SECONDS:
            return payload.get("recommendations", [])
    except Exception:
        pass
    return None


def save_category_cache(category: str, recommendations: list) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(category).write_text(
        json.dumps({"timestamp": time.time(), "recommendations": recommendations}),
        encoding="utf-8",
    )


def cache_age_minutes(category: str) -> float | None:
    path = _cache_path(category)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return round((time.time() - payload.get("timestamp", 0)) / 60, 1)
    except Exception:
        return None


def any_cache_exists() -> bool:
    return any(_cache_path(c).exists() for c in ["Large Cap", "Mid Cap", "Small Cap"])


# ── Legacy single-file helpers (kept for backwards compat) ────────────────────
from config import CACHE_FILE

def load_cache() -> list | None:
    if not CACHE_FILE.exists():
        return None
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() - payload.get("timestamp", 0) < SCAN_TTL_SECONDS:
            return payload.get("recommendations", [])
    except Exception:
        pass
    return None

def save_cache(recommendations: list) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"timestamp": time.time(), "recommendations": recommendations}),
        encoding="utf-8",
    )
