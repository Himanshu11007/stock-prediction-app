"""
scanner/background.py — fire-and-forget background scan manager.

Key design:
  - Priority stocks are scanned first so the UI shows results in ~60 s
  - Results are saved progressively every 5 stocks (not at the end)
  - A JSON progress file lets the UI show X / Y completed (global across categories)
  - A lock file prevents duplicate concurrent scans
"""
import json
import threading
import time
from pathlib import Path

from config import (
    STORAGE_DIR, CATEGORIES,
    UNIVERSE_LARGECAP, UNIVERSE_MIDCAP, UNIVERSE_SMALLCAP,
    SCAN_TTL_SECONDS,
)
from scanner.cache import save_category_cache, cache_age_minutes

_LOCK_FILE     = STORAGE_DIR / ".scan_running"
_PROGRESS_FILE = STORAGE_DIR / ".scan_progress.json"

_PRIORITY = {
    "Large Cap": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "HCLTECH.NS", "WIPRO.NS", "LT.NS",
    ],
    "Mid Cap": [
        "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS", "TATAELXSI.NS",
        "FEDERALBNK.NS", "DIXON.NS", "PAGEIND.NS", "AUBANK.NS", "KPITTECH.NS",
    ],
    "Small Cap": [
        "CDSL.NS", "ANGELONE.NS", "HAPPSTMNDS.NS", "TANLA.NS", "BSOFT.NS",
        "APARINDS.NS", "APLAPOLLO.NS", "SAREGAMA.NS", "AMBER.NS", "AETHER.NS",
    ],
}


# ── Lock helpers ──────────────────────────────────────────────────────────────

def is_scan_running() -> bool:
    if not _LOCK_FILE.exists():
        return False
    try:
        if time.time() - _LOCK_FILE.stat().st_mtime > 1800:
            _LOCK_FILE.unlink(missing_ok=True)
            return False
    except Exception:
        pass
    return True


# ── Progress helpers ──────────────────────────────────────────────────────────

def _write_progress(category: str, done: int, total: int) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _PROGRESS_FILE.write_text(
            json.dumps({"category": category, "done": done, "total": total}),
            encoding="utf-8",
        )
    except Exception:
        pass


def scan_progress() -> dict:
    try:
        if _PROGRESS_FILE.exists():
            return json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ── Universe loader ───────────────────────────────────────────────────────────

def _load_universe(csv_path) -> tuple[list, dict]:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        # Drop rows where Symbol is missing or contains spaces (malformed entries)
        df = df[df["Symbol"].notna()]
        df["Symbol"] = df["Symbol"].str.strip()
        df = df[df["Symbol"].str.len() > 0]
        df = df[~df["Symbol"].str.contains(r"\s", regex=True)]
        symbols = df["Symbol"].tolist()
        cmap    = dict(zip(df["Symbol"], df["Company"].str.strip()))
        return symbols, cmap
    except Exception:
        return [], {}


def _ordered_symbols(all_symbols: list, priority: list) -> list:
    seen = set(priority)
    rest = [s for s in all_symbols if s not in seen]
    return priority + rest


# ── Background worker ─────────────────────────────────────────────────────────

def _run_scan(universes: dict, global_company_map: dict) -> None:
    """
    Runs in a daemon thread. Tracks a global done counter across all categories
    so the UI progress never resets mid-scan.
    """
    from scanner.engine import get_recommendations

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch()

    # ── Compute global total across all categories upfront ───────────────────
    global_total = sum(len(syms) for syms, _ in universes.values())
    global_done  = 0          # incremented by every save_callback call
    _write_progress("Starting", 0, global_total)

    # Thread-safe counter so parallel futures don't race on global_done
    _lock = threading.Lock()

    try:
        for category, (all_symbols, cmap) in universes.items():
            if not all_symbols:
                continue

            merged_map   = {**global_company_map, **cmap}
            priority     = _PRIORITY.get(category, [])
            symbols      = _ordered_symbols(all_symbols, priority)
            cat_total    = len(symbols)
            cat_done_ref = [0]   # mutable container so closure can mutate it

            def _save(results, cat=category, c_ref=cat_done_ref):
                save_category_cache(cat, results)
                # Advance both the per-category and global counters by save_interval
                nonlocal global_done
                with _lock:
                    # Each save_callback represents `save_interval` stocks processed
                    c_ref[0] = min(c_ref[0] + 5, cat_total)
                    global_done = min(global_done + 5, global_total)
                    _write_progress(cat, global_done, global_total)

            get_recommendations(
                symbols,
                merged_map,
                use_raw_loader=True,
                save_callback=_save,
                save_interval=5,
            )

            # Mark category as fully done (handle remainder stocks < save_interval)
            with _lock:
                remainder = cat_total - cat_done_ref[0]
                if remainder > 0:
                    global_done = min(global_done + remainder, global_total)
                    _write_progress(category, global_done, global_total)

        _write_progress("done", global_total, global_total)

    except Exception as e:
        _write_progress(f"error: {e}", global_done, global_total)
    finally:
        _LOCK_FILE.unlink(missing_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def start_background_scan(global_company_map: dict) -> bool:
    if is_scan_running():
        return False

    universes = {
        "Large Cap": _load_universe(UNIVERSE_LARGECAP),
        "Mid Cap":   _load_universe(UNIVERSE_MIDCAP),
        "Small Cap": _load_universe(UNIVERSE_SMALLCAP),
    }

    threading.Thread(
        target=_run_scan,
        args=(universes, global_company_map),
        daemon=True,
        name="StockScanner",
    ).start()
    return True


def needs_scan() -> bool:
    if is_scan_running():
        return False
    for cat in CATEGORIES:
        age = cache_age_minutes(cat)
        if age is None or age * 60 >= SCAN_TTL_SECONDS:
            return True
    return False