"""
storage/recommendation_validation.py — Recommendation Validation Engine

Responsibilities
────────────────
1. Schema  : extend recommendation_validation table with validation columns
2. Fetching: get latest closing price from yfinance
3. Logic   : determine if 5 trading days have elapsed
4. Scoring : calculate return % and success flag per signal type
5. Engine  : validate_old_recommendations() — the single public entry point
6. Queries : load_pending_recommendations(), load_validated_recommendations()

Success definitions
───────────────────
  BUY / STRONG BUY   → success if current_price > cmp
  SELL / STRONG SELL → success if current_price < cmp
  HOLD               → success if abs(return_pct) <= 3.0 %

Trading-day logic
─────────────────
  Uses numpy busday_count (Mon–Fri, no holiday calendar) so we never
  validate a recommendation that is fewer than 5 market sessions old.
"""

from __future__ import annotations

import sqlite3
import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import TRACKER_DB

# ── Module logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────────────────
_TRADING_DAYS_REQUIRED = 5
_HOLD_BAND_PCT         = 3.0   # ±3 % counts as HOLD success
_BULLISH               = {"BUY", "STRONG BUY"}
_BEARISH               = {"SELL", "STRONG SELL"}


# ══════════════════════════════════════════════════════════════════════════════
# Database helpers
# ══════════════════════════════════════════════════════════════════════════════

def _connect() -> sqlite3.Connection:
    """Open a WAL-mode connection, creating the directory if needed."""
    TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(TRACKER_DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


def migrate_schema() -> None:
    """
    Idempotently add validation columns to recommendation_validation.

    Safe to call on every app start — uses ALTER TABLE only when the column
    is absent, so existing rows and data are never touched.

    Columns added (if missing):
        validation_date  TEXT     — ISO date when validated
        validation_price REAL     — closing price on validation day
        return_pct       REAL     — ((validation_price - cmp) / cmp) * 100
        success          INTEGER  — 1 = successful, 0 = failed
        is_validated     INTEGER  — 0 = pending, 1 = done  (DEFAULT 0)
    """
    con = _connect()
    try:
        # First ensure the base table exists (created by tracker.py on first run)
        con.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_validation (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_date       TEXT    NOT NULL,
                symbol           TEXT    NOT NULL,
                stock            TEXT    NOT NULL,
                signal           TEXT    NOT NULL,
                cmp              REAL    NOT NULL,
                confluence_score REAL,
                ml_confidence    REAL,
                news_score       REAL,
                accuracy         REAL,
                target           REAL,
                stop_loss        REAL,
                is_validated     INTEGER DEFAULT 0,
                validation_date  TEXT,
                validation_price REAL,
                return_pct       REAL,
                success          INTEGER
            )
        """)

        # Fetch existing columns
        existing = {
            row[1]
            for row in con.execute("PRAGMA table_info(recommendation_validation)")
        }

        # Add missing columns one by one (ALTER TABLE cannot add multiple at once)
        additions = {
            "is_validated":     "INTEGER DEFAULT 0",
            "validation_date":  "TEXT",
            "validation_price": "REAL",
            "return_pct":       "REAL",
            "success":          "INTEGER",
        }
        for col, definition in additions.items():
            if col not in existing:
                con.execute(
                    f"ALTER TABLE recommendation_validation ADD COLUMN {col} {definition}"
                )
                logger.info("Schema: added column '%s' to recommendation_validation", col)

        # Indexes for fast pending lookups
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_rv_pending
            ON recommendation_validation (is_validated, saved_date)
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_rv_symbol
            ON recommendation_validation (symbol, saved_date)
        """)
        con.commit()
    finally:
        con.close()


# ══════════════════════════════════════════════════════════════════════════════
# Price fetching
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_close(symbol: str) -> Optional[float]:
    """
    Fetch the most recent available closing price for a symbol via yfinance.

    Returns:
        float  — latest adjusted close price
        None   — if the symbol is invalid, delisted, or the network fails
    """
    try:
        df = yf.download(symbol, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning("get_latest_close(%s): empty dataframe", symbol)
            return None

        # Flatten multi-level columns e.g. ("Close", "RELIANCE.NS") → "Close"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        if "Close" not in df.columns:
            logger.warning("get_latest_close(%s): no Close column", symbol)
            return None

        close = df["Close"].dropna()
        if close.empty:
            return None

        return round(float(close.iloc[-1]), 2)

    except Exception as exc:
        logger.warning("get_latest_close(%s): %s", symbol, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Trading-day logic
# ══════════════════════════════════════════════════════════════════════════════

def trading_days_elapsed(saved_date: str) -> int:
    """
    Count business days (Mon–Fri) between saved_date and today (exclusive).

    Args:
        saved_date: ISO date string "YYYY-MM-DD"

    Returns:
        int — number of trading sessions that have completed since saved_date
    """
    start = datetime.date.fromisoformat(saved_date)
    today = datetime.date.today()
    # numpy busday_count counts Mon–Fri days in [start, today)
    return int(np.busday_count(start.isoformat(), today.isoformat()))


def is_ready_for_validation(saved_date: str) -> bool:
    """Return True when at least 5 trading days have elapsed since saved_date."""
    return trading_days_elapsed(saved_date) >= _TRADING_DAYS_REQUIRED


# ══════════════════════════════════════════════════════════════════════════════
# Success calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_return(cmp: float, current_price: float) -> float:
    """
    Return percentage gain/loss from entry.

    Formula: ((current_price - cmp) / cmp) * 100
    """
    if cmp == 0:
        return 0.0
    return round(((current_price - cmp) / cmp) * 100, 4)


def calculate_success(signal: str, return_pct: float) -> int:
    """
    Determine whether a recommendation was successful.

    Rules:
        BUY / STRONG BUY   → success if return_pct > 0  (price rose)
        SELL / STRONG SELL → success if return_pct < 0  (price fell)
        HOLD               → success if abs(return_pct) <= HOLD_BAND_PCT

    Returns:
        1 — successful
        0 — failed
    """
    signal_upper = signal.upper()
    if signal_upper in _BULLISH:
        return 1 if return_pct > 0 else 0
    if signal_upper in _BEARISH:
        return 1 if return_pct < 0 else 0
    # HOLD
    return 1 if abs(return_pct) <= _HOLD_BAND_PCT else 0


# ══════════════════════════════════════════════════════════════════════════════
# Query helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_pending_recommendations(as_of_date: Optional[str] = None) -> list[dict]:
    """
    Return all unvalidated recommendations (is_validated = 0).

    Args:
        as_of_date : Only return rows saved on or before this ISO date.
                     Useful for dry-run inspection before calling the engine.

    Returns:
        list[dict] — one dict per pending recommendation
    """
    con = _connect()
    try:
        query = """
            SELECT
                id,
                saved_date       AS "Date",
                symbol           AS "Symbol",
                stock            AS "Stock",
                signal           AS "Signal",
                cmp              AS "CMP",
                confluence_score AS "Confluence Score",
                ml_confidence    AS "ML Confidence",
                news_score       AS "News Score",
                accuracy         AS "Accuracy",
                target           AS "Target",
                stop_loss        AS "Stop Loss"
            FROM  recommendation_validation
            WHERE is_validated = 0
        """
        params: list = []
        if as_of_date:
            query  += " AND saved_date <= ?"
            params.append(as_of_date)
        query += " ORDER BY saved_date ASC, id ASC"

        rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def load_validated_recommendations(limit: int = 50) -> list[dict]:
    """
    Return the most recent validated recommendations for display in the UI.

    Returns:
        list[dict] with all columns including validation results
    """
    con = _connect()
    try:
        rows = con.execute(
            """
            SELECT
                saved_date       AS "Date",
                symbol           AS "Symbol",
                stock            AS "Stock",
                signal           AS "Signal",
                cmp              AS "CMP",
                confluence_score AS "Confluence Score",
                ml_confidence    AS "ML Confidence",
                target           AS "Target",
                stop_loss        AS "Stop Loss",
                validation_date  AS "Validation Date",
                validation_price AS "Validation Price",
                return_pct       AS "Return %",
                success          AS "Success"
            FROM  recommendation_validation
            WHERE is_validated = 1
            ORDER BY validation_date DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def _update_validated_row(
    con: sqlite3.Connection,
    row_id:          int,
    validation_date: str,
    validation_price: float,
    return_pct:      float,
    success:         int,
) -> None:
    """Write validation result back to the database row."""
    con.execute(
        """
        UPDATE recommendation_validation
        SET    is_validated     = 1,
               validation_date  = ?,
               validation_price = ?,
               return_pct       = ?,
               success          = ?
        WHERE  id = ?
        """,
        (validation_date, validation_price, return_pct, success, row_id),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Validation engine — main entry point
# ══════════════════════════════════════════════════════════════════════════════

def validate_old_recommendations() -> int:
    """
    Find all pending recommendations that are ≥ 5 trading days old,
    fetch their current price, compute return % and success, and
    persist the result.

    The function is intentionally fault-tolerant: a failure on one
    symbol is logged and skipped; the engine continues to the next row.

    Returns:
        int — number of recommendations successfully validated in this run
    """
    # Ensure schema is current before doing anything
    migrate_schema()

    pending = load_pending_recommendations()
    if not pending:
        logger.info("Validation: no pending recommendations found")
        return 0

    today_str  = datetime.date.today().isoformat()
    validated  = 0
    con        = _connect()

    try:
        for rec in pending:
            row_id     = rec["id"]
            symbol     = rec["Symbol"]
            stock      = rec["Stock"]
            saved_date = rec["Date"]
            signal     = rec["Signal"]
            cmp        = rec["CMP"]

            # ── Gate: 5 trading days must have elapsed ────────────────────────
            if not is_ready_for_validation(saved_date):
                elapsed = trading_days_elapsed(saved_date)
                logger.info(
                    "Skip  %s — only %d/%d trading days elapsed",
                    symbol, elapsed, _TRADING_DAYS_REQUIRED,
                )
                continue

            # ── Fetch current price ───────────────────────────────────────────
            current_price = get_latest_close(symbol)
            if current_price is None:
                logger.warning("Skip  %s — could not fetch current price", symbol)
                continue

            # ── Compute return & success ──────────────────────────────────────
            ret     = calculate_return(float(cmp), current_price)
            success = calculate_success(signal, ret)

            # ── Persist ───────────────────────────────────────────────────────
            try:
                _update_validated_row(
                    con,
                    row_id          = row_id,
                    validation_date = today_str,
                    validation_price= current_price,
                    return_pct      = ret,
                    success         = success,
                )
                con.commit()
                validated += 1
            except sqlite3.Error as db_err:
                logger.error("DB write failed for %s (id=%s): %s", symbol, row_id, db_err)
                con.rollback()
                continue

            # ── Human-readable log line ───────────────────────────────────────
            result_label = "SUCCESS" if success else "FAILED"
            direction    = "▲" if ret > 0 else ("▼" if ret < 0 else "→")
            logger.info(
                "Validated %-20s | Signal: %-11s | CMP: %8.2f | "
                "Current: %8.2f | Return: %+.2f%% | %s %s",
                symbol, signal, float(cmp), current_price, ret,
                direction, result_label,
            )

    finally:
        con.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "Validation complete — %d validated out of %d pending",
        validated, len(pending),
    )
    return validated