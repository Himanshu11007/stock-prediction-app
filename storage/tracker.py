import sqlite3
import datetime
from config import TRACKER_DB


def _connect() -> sqlite3.Connection:
    TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(TRACKER_DB))
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            company     TEXT,
            signal      TEXT    NOT NULL,
            score       REAL,
            confidence  REAL,
            accuracy    REAL,
            close_price REAL,
            next_close  REAL,
            correct     INTEGER
        )
    """)
    con.commit()
    return con


def save_signal(
    symbol: str,
    company: str,
    signal: str,
    score: float,
    confidence: float,
    accuracy: float,
    close_price: float,
) -> int:
    """Insert a new prediction record. Returns the row ID."""
    con = _connect()
    cur = con.execute(
        """INSERT INTO signals
           (date, symbol, company, signal, score, confidence, accuracy, close_price)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.date.today().isoformat(),
            symbol, company, signal,
            round(score, 4),
            round(confidence, 2),
            round(accuracy * 100, 2),
            round(float(close_price), 2),
        ),
    )
    con.commit()
    row_id = cur.lastrowid
    con.close()
    return row_id


def update_outcome(row_id: int, next_close: float) -> None:
    """Fill in the actual next-day close and mark the prediction correct/incorrect."""
    con = _connect()
    row = con.execute(
        "SELECT signal, close_price FROM signals WHERE id = ?", (row_id,)
    ).fetchone()
    if row:
        signal, entry_price = row
        correct = int(
            (signal in ("BUY", "STRONG BUY")   and next_close > entry_price) or
            (signal in ("SELL", "STRONG SELL")  and next_close < entry_price)
        )
        con.execute(
            "UPDATE signals SET next_close = ?, correct = ? WHERE id = ?",
            (round(next_close, 2), correct, row_id),
        )
        con.commit()
    con.close()


def get_recent_signals(limit: int = 20) -> list[dict]:
    """Return the most recent saved signals as a list of dicts."""
    con = _connect()
    rows = con.execute(
        """SELECT date, symbol, company, signal, score, confidence, accuracy,
                  close_price, next_close, correct
           FROM signals ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    con.close()
    keys = [
        "Date", "Symbol", "Company", "Signal", "Score",
        "Confidence %", "Model Acc %", "Entry Price", "Next Close", "Correct",
    ]
    return [dict(zip(keys, r)) for r in rows]


def get_accuracy_stats() -> tuple[int | None, int | None]:
    """Return (correct_count, total_count) for predictions that have outcomes."""
    con = _connect()
    row = con.execute(
        "SELECT COUNT(*), SUM(correct) FROM signals WHERE correct IS NOT NULL"
    ).fetchone()
    con.close()
    total, correct = row
    if not total:
        return None, None
    return int(correct or 0), int(total)


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation Validation
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_validation_table(con: sqlite3.Connection) -> None:
    """Create recommendation_validation table + indexes if they don't exist yet."""
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
            validated        INTEGER DEFAULT 0,
            validation_date  TEXT,
            outcome_price    REAL,
            success          INTEGER
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_rv_pending
        ON recommendation_validation (validated, saved_date)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_rv_symbol
        ON recommendation_validation (symbol, saved_date)
    """)
    con.commit()


def save_recommendation(
    symbol:           str,
    stock:            str,
    signal:           str,
    cmp:              float,
    confluence_score: float,
    ml_confidence:    float,
    news_score:       float,
    accuracy:         float,
    target:           float | None,
    stop_loss:        float | None,
    saved_date:       str | None = None,
) -> int:
    """
    Save one scanner recommendation for future 5-day outcome validation.

    Args:
        symbol           : Yahoo Finance ticker e.g. "RELIANCE.NS"
        stock            : Display company name
        signal           : "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
        cmp              : Current market price at save time
        confluence_score : Weighted confluence score (0–1)
        ml_confidence    : Model confidence (0–100)
        news_score       : FinBERT average sentiment score (-1 to +1)
        accuracy         : Model accuracy (0–1)
        target           : ATR-based target price (None for HOLD)
        stop_loss        : ATR-based stop-loss price (None for HOLD)
        saved_date       : Override date (ISO string); defaults to today

    Returns:
        int: rowid of the inserted row
    """
    row_date = saved_date or datetime.date.today().isoformat()
    con = _connect()
    _ensure_validation_table(con)
    cur = con.execute(
        """
        INSERT INTO recommendation_validation (
            saved_date, symbol, stock, signal, cmp,
            confluence_score, ml_confidence, news_score, accuracy,
            target, stop_loss
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_date,
            symbol,
            stock,
            signal,
            round(float(cmp),              2),
            round(float(confluence_score), 4),
            round(float(ml_confidence),    2),
            round(float(news_score),       4),
            round(float(accuracy),         4),
            round(float(target),    2) if target    is not None else None,
            round(float(stop_loss), 2) if stop_loss is not None else None,
        ),
    )
    con.commit()
    row_id = cur.lastrowid
    con.close()
    return row_id


def load_pending_recommendations(as_of_date: str | None = None) -> list[dict]:
    """
    Return all unvalidated recommendations.

    Args:
        as_of_date : Only return rows saved on or before this ISO date.
                     Pass a date 5 trading days ago to get rows ready for validation.
                     Omit to return all pending rows.

    Returns:
        list[dict] with keys: id, Date, Symbol, Stock, Signal, CMP,
        Confluence Score, ML Confidence, News Score, Accuracy, Target, Stop Loss
    """
    con = _connect()
    _ensure_validation_table(con)

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
        WHERE validated = 0
    """
    params = []
    if as_of_date:
        query  += " AND saved_date <= ?"
        params.append(as_of_date)

    query += " ORDER BY saved_date ASC, id ASC"

    rows = con.execute(query, params).fetchall()
    con.close()

    keys = ["id", "Date", "Symbol", "Stock", "Signal", "CMP",
            "Confluence Score", "ML Confidence", "News Score",
            "Accuracy", "Target", "Stop Loss"]
    return [dict(zip(keys, r)) for r in rows]