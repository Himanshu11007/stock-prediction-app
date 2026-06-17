"""
utils/logger.py — Centralized logging system for StockAI Pro.

Features
────────
- Rotating file handler: 5 MB max, 5 backups
- Format: timestamp | level | module | message
- Debug toggle via config.ENABLE_DEBUG_LOGS
- Helper functions for structured scan/stock/filter events
- Never crashes the app — all setup errors are silently swallowed

Usage
─────
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

from __future__ import annotations

import logging
import os
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Log file location (mirrors STORAGE_DIR logic from config) ─────────────────
_STORAGE_DIR = (
    Path("/tmp/stockai_storage") if os.name == "posix" else Path("storage")
)
LOG_DIR  = _STORAGE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# ── Format ─────────────────────────────────────────────────────────────────────
_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ── Internal state ─────────────────────────────────────────────────────────────
_initialised = False   # ensure setup runs only once per process


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

def _setup(debug: bool = False) -> None:
    """
    Configure the root logger with a rotating file handler.
    Safe to call multiple times — idempotent after first call.
    """
    global _initialised
    if _initialised:
        return

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG if debug else logging.INFO)

        # Remove any handlers already attached (e.g. basicConfig from other modules)
        for h in root.handlers[:]:
            root.removeHandler(h)

        # ── Rotating file handler ─────────────────────────────────────────────
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG if debug else logging.INFO)
        fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(fh)

        _initialised = True

    except Exception:
        # Logging must never crash the app
        pass


def configure_logging(debug: bool = False) -> None:
    """
    Public entry point — call once at app startup.

    Args:
        debug: When True, DEBUG-level messages are written to the log file.
               When False (default), only INFO / WARNING / ERROR / CRITICAL.
    """
    global _initialised
    _initialised = False   # allow reconfiguration when debug flag changes
    _setup(debug=debug)


# Run default setup at import time so any module that calls get_logger()
# before configure_logging() still gets a working logger.
_setup()


# ══════════════════════════════════════════════════════════════════════════════
# Public factory
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger bound to the centralized handler.
    """
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
# Structured helper functions
# ══════════════════════════════════════════════════════════════════════════════

_scan_logger   = logging.getLogger("scanner.engine")
_filter_logger = logging.getLogger("scanner.filters")


def log_scan_start(scan_id: str, category: str, total_stocks: int) -> None:
    """
    Log the beginning of a category scan.

    Example output:
        SCAN START | id=large_cap_001 | category=Large Cap | stocks=50
    """
    try:
        _scan_logger.info(
            "SCAN START | id=%s | category=%s | stocks=%d",
            scan_id, category, total_stocks,
        )
    except Exception:
        pass


def log_scan_end(
    scan_id:       str,
    category:      str,
    total_scanned: int,
    total_passed:  int,
) -> None:
    """
    Log the end of a category scan with pass/fail counts.

    Example output:
        SCAN END | id=large_cap_001 | category=Large Cap |
        scanned=50 | passed=12 | filtered=38
    """
    try:
        _scan_logger.info(
            "SCAN END   | id=%s | category=%s | scanned=%d | passed=%d | filtered=%d",
            scan_id, category, total_scanned, total_passed,
            total_scanned - total_passed,
        )
    except Exception:
        pass


def log_stock_diagnostics(
    symbol:          str,
    prediction:      int,
    confidence:      float,
    accuracy:        float,
    signal:          str,
    final_score:     float,
    ml_dir:          float,
    ml_conf:         float,
    tech_score:      float,
    news_score:      float,
    volume_score:    float,
    regime_score:    float,
    timeframe_score: float,
    momentum_score:  float,
    weighted_score:  float,
) -> None:
    """
    Log full pillar breakdown for one stock at DEBUG level.

    Only written when ENABLE_DEBUG_LOGS = True (avoids flooding the log
    with per-stock detail during normal operation).

    Example output:
        DIAG | RELIANCE.NS | signal=BUY | score=0.71 | conf=68.20 |
        acc=0.54 | pred=1 | ml_dir=0.70 | ml_conf=0.36 | tech=0.42 |
        news=0.15 | vol=0.30 | regime=1.00 | tf=0.50 | momentum=0.60 |
        weighted=0.42
    """
    try:
        _scan_logger.debug(
            "DIAG  | %-20s | signal=%-11s | score=%.4f | conf=%6.2f | acc=%.4f | "
            "pred=%d | ml_dir=%+.2f | ml_conf=%+.2f | tech=%+.2f | "
            "news=%+.2f | vol=%+.2f | regime=%+.2f | tf=%+.2f | "
            "momentum=%+.2f | weighted=%+.4f",
            symbol, signal, final_score, confidence, accuracy,
            prediction, ml_dir, ml_conf, tech_score,
            news_score, volume_score, regime_score, timeframe_score,
            momentum_score, weighted_score,
        )
    except Exception:
        pass


def log_filter_rejection(
    symbol:     str,
    signal:     str,
    confidence: float,
    accuracy:   float,
    score:      float,
    reason:     str,
) -> None:
    """
    Log why a stock was rejected by quality filters.

    Example output:
        FILTERED | ADANIENT.NS | signal=BUY | conf=72.6 |
        acc=0.46 | score=0.58 | reason=score below MIN_CONFLUENCE_SCORE 0.60
    """
    try:
        _filter_logger.info(
            "FILTERED | %-20s | signal=%-11s | conf=%6.2f | acc=%.4f | "
            "score=%.4f | reason=%s",
            symbol, signal, confidence, accuracy, score, reason,
        )
    except Exception:
        pass


def log_exception(
    logger:  logging.Logger,
    message: str,
    exc:     Optional[BaseException] = None,
) -> None:
    """
    Log an error message with full traceback.

    Args:
        logger:  The module-level logger to write to.
        message: Human-readable description of what failed.
        exc:     The caught exception (optional; uses sys.exc_info if None).
    """
    try:
        if exc is not None:
            logger.error("%s — %s: %s", message, type(exc).__name__, exc)
            logger.debug("Traceback:\n%s", traceback.format_exc())
        else:
            logger.error(message, exc_info=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Log file utilities (used by Streamlit UI)
# ══════════════════════════════════════════════════════════════════════════════

def read_last_log_lines(n: int = 100) -> list[str]:
    """
    Read the last n lines from the active log file.

    Returns an empty list if the file does not exist or cannot be read.
    """
    try:
        if not LOG_FILE.exists():
            return []
        text  = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n:] if len(lines) > n else lines
    except Exception:
        return []


def clear_log_file() -> bool:
    """
    Truncate the active log file.

    Returns True on success, False on failure.
    """
    try:
        if LOG_FILE.exists():
            LOG_FILE.write_text("", encoding="utf-8")
        return True
    except Exception:
        return False


def log_file_size_kb() -> float:
    """Return the current log file size in KB, or 0.0 if it doesn't exist."""
    try:
        return round(LOG_FILE.stat().st_size / 1024, 1) if LOG_FILE.exists() else 0.0
    except Exception:
        return 0.0