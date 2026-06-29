"""
scanner/background.py — fire-and-forget background scan manager.
"""
import json
import threading
import time
import uuid
from pathlib import Path

from config import (
    STORAGE_DIR, CATEGORIES,
    UNIVERSE_LARGECAP, UNIVERSE_MIDCAP, UNIVERSE_SMALLCAP,
    SCAN_TTL_SECONDS,
)
from scanner.cache import save_category_cache, cache_age_minutes
from utils.logger import get_logger, log_scan_start, log_scan_end, log_exception

logger = get_logger(__name__)

_LOCK_FILE     = STORAGE_DIR / ".scan_running"
_PROGRESS_FILE = STORAGE_DIR / ".scan_progress.json"

_PRIORITY = {
    "Large Cap": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "SBIN.NS","BHARTIARTL.NS","HCLTECH.NS","WIPRO.NS","LT.NS",
    ],
    "Mid Cap": [
        "PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","LTTS.NS","TATAELXSI.NS",
        "FEDERALBNK.NS","DIXON.NS","PAGEIND.NS","AUBANK.NS","KPITTECH.NS",
    ],
    "Small Cap": [
        "CDSL.NS","ANGELONE.NS","HAPPSTMNDS.NS","TANLA.NS","BSOFT.NS",
        "APARINDS.NS","APLAPOLLO.NS","SAREGAMA.NS","AMBER.NS","AETHER.NS",
    ],
}

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

def _load_universe(csv_path) -> tuple[list, dict]:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df[df["Symbol"].notna()]
        df["Symbol"] = df["Symbol"].str.strip()
        df = df[df["Symbol"].str.len() > 0]
        df = df[~df["Symbol"].str.contains(r"\s", regex=True)]
        symbols = df["Symbol"].tolist()
        cmap    = dict(zip(df["Symbol"], df["Company"].str.strip()))
        return symbols, cmap
    except Exception as e:
        log_exception(logger, f"Failed to load universe from {csv_path}", e)
        return [], {}

def _ordered_symbols(all_symbols: list, priority: list) -> list:
    seen = set(priority)
    rest = [s for s in all_symbols if s not in seen]
    return priority + rest

def _run_scan(universes: dict, global_company_map: dict) -> None:
    from scanner.engine import get_recommendations
    from storage.tracker import generate_scan_id
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch()

    # One scan_id shared by every category in this run, formatted as
    # SCAN-YYYYMMDD-HHMMSS per the spec — used to tag every persisted
    # recommendation so the whole run is traceable end-to-end.
    scan_run_id  = generate_scan_id("SCAN")
    global_total = sum(len(syms) for syms, _ in universes.values())
    global_done  = 0
    _write_progress("Starting", 0, global_total)

    logger.info(
        "═══ SCAN RUN %s STARTED — %d stocks across %d categories ═══",
        scan_run_id, global_total, len(universes),
    )
    _lock = threading.Lock()

    try:
        for category, (all_symbols, cmap) in universes.items():
            if not all_symbols:
                logger.warning("Category '%s': no symbols — skipping", category)
                continue

            merged_map   = {**global_company_map, **cmap}
            priority     = _PRIORITY.get(category, [])
            symbols      = _ordered_symbols(all_symbols, priority)
            cat_total    = len(symbols)
            cat_done_ref = [0]
            cat_passed   = [0]
            cat_scan_id  = f"{scan_run_id}_{category.lower().replace(' ', '_')}"

            log_scan_start(cat_scan_id, category, cat_total)

            def _save(results, cat=category, c_ref=cat_done_ref, p_ref=cat_passed):
                save_category_cache(cat, results)
                nonlocal global_done
                with _lock:
                    c_ref[0] = min(c_ref[0] + 5, cat_total)
                    p_ref[0] = len(results)
                    global_done = min(global_done + 5, global_total)
                    _write_progress(cat, global_done, global_total)

            get_recommendations(
                symbols, merged_map,
                use_raw_loader=True,
                save_callback=_save,
                save_interval=5,
                scan_id=scan_run_id,
            )

            with _lock:
                remainder = cat_total - cat_done_ref[0]
                if remainder > 0:
                    global_done = min(global_done + remainder, global_total)
                    _write_progress(category, global_done, global_total)

            log_scan_end(cat_scan_id, category, cat_total, cat_passed[0])

        _write_progress("done", global_total, global_total)
        logger.info(
            "═══ SCAN RUN %s COMPLETED — %d/%d processed ═══",
            scan_run_id, global_done, global_total,
        )

    except Exception as e:
        log_exception(logger, f"SCAN RUN {scan_run_id} crashed", e)
        _write_progress(f"error: {e}", global_done, global_total)
    finally:
        _LOCK_FILE.unlink(missing_ok=True)

def start_background_scan(global_company_map: dict) -> bool:
    if is_scan_running():
        logger.info("Scan already running — new request ignored")
        return False
    universes = {
        "Large Cap": _load_universe(UNIVERSE_LARGECAP),
        "Mid Cap":   _load_universe(UNIVERSE_MIDCAP),
        "Small Cap": _load_universe(UNIVERSE_SMALLCAP),
    }
    threading.Thread(
        target=_run_scan, args=(universes, global_company_map),
        daemon=True, name="StockScanner",
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