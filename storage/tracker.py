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
            (signal == "BUY"  and next_close > entry_price) or
            (signal == "SELL" and next_close < entry_price)
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
