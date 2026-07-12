import sqlite3
import datetime
from config import TRACKER_DB
from utils.logger import get_logger

logger = get_logger(__name__)


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
# Recommendation Validation Storage
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_validation_table(con: sqlite3.Connection) -> None:
    """
    Create recommendation_validation table + indexes if not already present,
    and migration-safely add the scan_id column for duplicate prevention.

    Uses is_validated (not 'validated') to match recommendation_validation.py.

    Migration safety: existing data is NEVER deleted. PRAGMA table_info is
    used to check for the scan_id column before attempting to add it, so
    this function is safe to call on every connection regardless of whether
    the table is brand new or has existing rows from before this change.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_validation (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_date        TEXT    NOT NULL,
            symbol            TEXT    NOT NULL,
            stock             TEXT    NOT NULL,
            signal            TEXT    NOT NULL,
            cmp               REAL    NOT NULL,
            confluence_score  REAL,
            ml_confidence     REAL,
            news_score        REAL,
            accuracy          REAL,
            target            REAL,
            stop_loss         REAL,
            is_validated      INTEGER DEFAULT 0,
            validation_date   TEXT,
            validation_price  REAL,
            return_pct        REAL,
            success           INTEGER,
            scan_id           TEXT,
            pillar_ml_dir     REAL,
            pillar_ml_conf    REAL,
            pillar_tech       REAL,
            pillar_news       REAL,
            pillar_volume     REAL,
            pillar_regime     REAL,
            pillar_timeframe  REAL,
            pillar_momentum   REAL,
            weighted_score    REAL,
            sector            TEXT,
            market_regime     TEXT,
            engine_version    TEXT
        )
    """)

    # ── Migration-safe column add: check PRAGMA table_info before ALTER ───────
    # Every column below is nullable, so existing rows remain valid with NULL
    # in the new fields — no backfill, no data loss, no rewrite of old rows.
    existing_cols = {
        row[1] for row in con.execute("PRAGMA table_info(recommendation_validation)")
    }
    _new_columns = {
        "scan_id":          "TEXT",
        "pillar_ml_dir":    "REAL",
        "pillar_ml_conf":   "REAL",
        "pillar_tech":      "REAL",
        "pillar_news":      "REAL",
        "pillar_volume":    "REAL",
        "pillar_regime":    "REAL",
        "pillar_timeframe": "REAL",
        "pillar_momentum":  "REAL",
        "weighted_score":   "REAL",
        "sector":           "TEXT",
        "market_regime":    "TEXT",
        "engine_version":   "TEXT",
    }
    for col_name, col_type in _new_columns.items():
        if col_name not in existing_cols:
            con.execute(
                f"ALTER TABLE recommendation_validation ADD COLUMN {col_name} {col_type}"
            )
            logger.info("Schema: added column '%s' to recommendation_validation", col_name)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_rv_pending
        ON recommendation_validation (is_validated, saved_date)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_rv_symbol
        ON recommendation_validation (symbol, saved_date)
    """)
    # Unique key for duplicate prevention: one row per symbol per day.
    # Using a UNIQUE INDEX (not a UNIQUE constraint on the column) so this
    # is also migration-safe — CREATE UNIQUE INDEX IF NOT EXISTS will simply
    # fail loudly (not silently corrupt data) if pre-existing duplicate rows
    # violate it; see migrate_duplicates() below for the cleanup step that
    # must run once before this index can be created on a table that
    # already has duplicates.
    try:
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rv_unique_symbol_date
            ON recommendation_validation (symbol, saved_date)
        """)
    except sqlite3.IntegrityError:
        logger.warning(
            "Could not create unique index idx_rv_unique_symbol_date — "
            "existing duplicate (symbol, saved_date) rows present. "
            "Run storage.tracker.dedupe_existing_recommendations() once to clean up."
        )
    con.commit()


def dedupe_existing_recommendations() -> int:
    """
    One-time cleanup helper: collapse existing duplicate (symbol, saved_date)
    rows down to the most recent row (highest id) per group, deleting the
    older duplicates. Safe to run multiple times — a no-op once duplicates
    are gone.

    This does NOT delete any data that isn't a confirmed duplicate. It is
    intended to be run once after upgrading to this schema version, before
    the unique index can be created successfully.

    Returns:
        int — number of duplicate rows deleted
    """
    con = _connect()
    try:
        # Ensure base table/columns exist first
        con.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT, saved_date TEXT, symbol TEXT
            )
        """)
        rows_before = con.execute(
            "SELECT COUNT(*) FROM recommendation_validation"
        ).fetchone()[0]

        con.execute("""
            DELETE FROM recommendation_validation
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM recommendation_validation
                GROUP BY symbol, saved_date
            )
        """)
        con.commit()

        rows_after = con.execute(
            "SELECT COUNT(*) FROM recommendation_validation"
        ).fetchone()[0]
        deleted = rows_before - rows_after
        if deleted:
            logger.info("Dedup: removed %d duplicate recommendation row(s)", deleted)
        return deleted
    finally:
        con.close()


def generate_scan_id(prefix: str = "MANUAL") -> str:
    """
    Generate a unique scan_id stamped with the current timestamp.

    Example: "MANUAL-20260628-143522" or "SCAN-20260628-143522"
    """
    return f"{prefix}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def recommendation_exists(symbol: str, saved_date: str | None = None) -> bool:
    """
    Check whether a recommendation already exists for this symbol on this date.

    The duplicate-prevention key is (symbol, saved_date) — one row per stock
    per day, regardless of which scan_id produced it. This matches the
    intended behaviour: re-analysing the same stock on the same day should
    update the existing row, not create a second one.

    Args:
        symbol:     e.g. "RELIANCE.NS"
        saved_date: ISO date string; defaults to today.

    Returns:
        bool — True if a row for (symbol, saved_date) already exists.
    """
    row_date = saved_date or datetime.date.today().isoformat()
    con = _connect()
    _ensure_validation_table(con)
    try:
        row = con.execute(
            "SELECT 1 FROM recommendation_validation WHERE symbol = ? AND saved_date = ? LIMIT 1",
            (symbol, row_date),
        ).fetchone()
        return row is not None
    finally:
        con.close()


def upsert_recommendation(
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
    scan_id:          str | None = None,
    pillar_scores:    dict | None = None,
    weighted_score:   float | None = None,
    sector:           str | None = None,
    market_regime:    str | None = None,
    engine_version:   str | None = None,
) -> int:
    """
    Insert a new recommendation, or update the existing row for the same
    (symbol, saved_date) pair if one already exists.

    This is the duplicate-safe replacement for the old insert-only
    save_recommendation(). Behaviour:

        - If no row exists for (symbol, saved_date): INSERT a new row.
          Logs: RECOMMENDATION_INSERTED
        - If a row already exists for (symbol, saved_date): UPDATE it with
          the new values (latest analysis wins) and reset is_validated to 0
          so the fresh recommendation gets its own 5-day validation window.
          Logs: RECOMMENDATION_UPDATED

    Args:
        scan_id: Optional identifier for the scan/analysis run that produced
                 this recommendation (e.g. "SCAN-20260628-143522" or
                 "MANUAL-20260628-143522"). Stored for traceability; the
                 actual dedup key remains (symbol, saved_date) — see
                 recommendation_exists() docstring for why.
        pillar_scores: Optional dict with keys matching
                 utils.explainability._PILLAR_WEIGHTS (e.g. "ML Direction",
                 "Technical Analysis", ...). When provided, the eight raw
                 pillar scores are persisted alongside the recommendation
                 so the Recommendation Intelligence Engine can analyze them
                 later without ever recomputing history. All optional and
                 nullable — omitting this has no effect on existing callers.
        weighted_score: The pre-mapped [-1, +1] confluence value (distinct
                 from confluence_score, which is the 0-1 mapped value).
        sector:  Stock sector, if known at save time (e.g. from a CSV
                 mapping). Falls back to NULL if not provided — historical
                 rows without a sector can still be backfilled via mapping
                 at analysis time by the intelligence engine.
        market_regime: The regime *label* (e.g. "Bullish", "Sideways") —
                 distinct from pillar_scores["Market Regime"], which is the
                 numeric regime_score.
        engine_version: Free-text tag for which recommendation engine
                 version produced this row (e.g. "v1.0"). Useful for
                 intelligence analysis if the scoring logic changes later.

    Returns:
        int: rowid of the inserted or updated row
    """
    row_date = saved_date or datetime.date.today().isoformat()
    scan_id  = scan_id or generate_scan_id()

    # ── Unpack pillar scores (all nullable — None if not provided) ───────────
    p = pillar_scores or {}
    pillar_ml_dir    = p.get("ML Direction")
    pillar_ml_conf   = p.get("ML Confidence")
    pillar_tech      = p.get("Technical Analysis")
    pillar_news      = p.get("News Sentiment")
    pillar_volume    = p.get("Volume")
    pillar_regime    = p.get("Market Regime")
    pillar_timeframe = p.get("Multi-Timeframe")
    pillar_momentum  = p.get("Momentum")

    con = _connect()
    _ensure_validation_table(con)
    try:
        existing = con.execute(
            "SELECT id FROM recommendation_validation WHERE symbol = ? AND saved_date = ? LIMIT 1",
            (symbol, row_date),
        ).fetchone()

        if existing is None:
            cur = con.execute(
                """
                INSERT INTO recommendation_validation (
                    saved_date, symbol, stock, signal, cmp,
                    confluence_score, ml_confidence, news_score, accuracy,
                    target, stop_loss, scan_id,
                    pillar_ml_dir, pillar_ml_conf, pillar_tech, pillar_news,
                    pillar_volume, pillar_regime, pillar_timeframe, pillar_momentum,
                    weighted_score, sector, market_regime, engine_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_date, symbol, stock, signal,
                    round(float(cmp), 2),
                    round(float(confluence_score), 4),
                    round(float(ml_confidence), 2),
                    round(float(news_score), 4),
                    round(float(accuracy), 4),
                    round(float(target), 2)    if target    is not None else None,
                    round(float(stop_loss), 2) if stop_loss is not None else None,
                    scan_id,
                    round(float(pillar_ml_dir), 4)    if pillar_ml_dir    is not None else None,
                    round(float(pillar_ml_conf), 4)   if pillar_ml_conf   is not None else None,
                    round(float(pillar_tech), 4)      if pillar_tech      is not None else None,
                    round(float(pillar_news), 4)      if pillar_news      is not None else None,
                    round(float(pillar_volume), 4)    if pillar_volume    is not None else None,
                    round(float(pillar_regime), 4)    if pillar_regime    is not None else None,
                    round(float(pillar_timeframe), 4) if pillar_timeframe is not None else None,
                    round(float(pillar_momentum), 4)  if pillar_momentum  is not None else None,
                    round(float(weighted_score), 4)   if weighted_score   is not None else None,
                    sector,
                    market_regime,
                    engine_version,
                ),
            )
            con.commit()
            row_id = cur.lastrowid
            logger.info(
                "RECOMMENDATION_INSERTED | symbol=%s | date=%s | scan_id=%s | id=%d",
                symbol, row_date, scan_id, row_id,
            )
            return row_id

        row_id = existing[0]
        con.execute(
            """
            UPDATE recommendation_validation
            SET stock = ?, signal = ?, cmp = ?,
                confluence_score = ?, ml_confidence = ?, news_score = ?,
                accuracy = ?, target = ?, stop_loss = ?, scan_id = ?,
                pillar_ml_dir = ?, pillar_ml_conf = ?, pillar_tech = ?, pillar_news = ?,
                pillar_volume = ?, pillar_regime = ?, pillar_timeframe = ?, pillar_momentum = ?,
                weighted_score = ?, sector = ?, market_regime = ?, engine_version = ?,
                is_validated = 0, validation_date = NULL,
                validation_price = NULL, return_pct = NULL, success = NULL
            WHERE id = ?
            """,
            (
                stock, signal,
                round(float(cmp), 2),
                round(float(confluence_score), 4),
                round(float(ml_confidence), 2),
                round(float(news_score), 4),
                round(float(accuracy), 4),
                round(float(target), 2)    if target    is not None else None,
                round(float(stop_loss), 2) if stop_loss is not None else None,
                scan_id,
                round(float(pillar_ml_dir), 4)    if pillar_ml_dir    is not None else None,
                round(float(pillar_ml_conf), 4)   if pillar_ml_conf   is not None else None,
                round(float(pillar_tech), 4)      if pillar_tech      is not None else None,
                round(float(pillar_news), 4)      if pillar_news      is not None else None,
                round(float(pillar_volume), 4)    if pillar_volume    is not None else None,
                round(float(pillar_regime), 4)    if pillar_regime    is not None else None,
                round(float(pillar_timeframe), 4) if pillar_timeframe is not None else None,
                round(float(pillar_momentum), 4)  if pillar_momentum  is not None else None,
                round(float(weighted_score), 4)   if weighted_score   is not None else None,
                sector,
                market_regime,
                engine_version,
                row_id,
            ),
        )
        con.commit()
        logger.info(
            "RECOMMENDATION_UPDATED | symbol=%s | date=%s | scan_id=%s | id=%d",
            symbol, row_date, scan_id, row_id,
        )
        return row_id
    finally:
        con.close()


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
    Backward-compatible wrapper around upsert_recommendation().

    Kept so existing callers (e.g. app.py) continue to work unchanged.
    New code should call upsert_recommendation() directly when a scan_id
    is available.
    """
    return upsert_recommendation(
        symbol=symbol, stock=stock, signal=signal, cmp=cmp,
        confluence_score=confluence_score, ml_confidence=ml_confidence,
        news_score=news_score, accuracy=accuracy,
        target=target, stop_loss=stop_loss, saved_date=saved_date,
        scan_id=generate_scan_id("MANUAL"),
    )


def load_pending_recommendations(as_of_date: str | None = None) -> list[dict]:
    """
    Return all recommendations where is_validated = 0.

    Args:
        as_of_date: Only return rows saved on or before this ISO date.

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
            stop_loss        AS "Stop Loss",
            scan_id          AS "Scan ID"
        FROM  recommendation_validation
        WHERE is_validated = 0
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
            "Accuracy", "Target", "Stop Loss", "Scan ID"]
    return [dict(zip(keys, r)) for r in rows]