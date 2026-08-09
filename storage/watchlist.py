from __future__ import annotations
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf

from config import TRACKER_DB
from utils.logger import get_logger

logger  = get_logger(__name__)


def init_watchlist_table() -> None:
    """Create the watchlist table if it doesn't exist."""
    with sqlite3.connect(TRACKER_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT  NOT NULL UNIQUE,
                stock_name TEXT NOT NULL,
                buy_price REAL NOT NULL,
                buy_date TEXT NOT NULL,
                quantity REAL NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                unique(symbol, buy_date)  -- Ensure unique combination of symbol and buy_date
            )
        """)
        conn.commit()   

def add_to_watchlist(symbol: str, stock_name: str, buy_price: float) -> bool:
    """Add a stock to the watchlist."""
    with sqlite3.connect(TRACKER_DB) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("select 1 from watchlist where symbol = ?", (symbol,))
            if cursor.fetchone():
                return False  # Stock already exists in the watchlist
            cursor.execute("""
                INSERT INTO watchlist (symbol, stock_name, buy_price, buy_date, quantity)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, stock_name, buy_price, datetime.now().strftime("%Y-%m-%d"), 1.0))
            conn.commit()
            logger.info(f"Added {symbol} to watchlist.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"{symbol} already exists in the watchlist.")

def remove_from_watchlist(watchlist_id: int) -> None:
    """Remove a stock from the watchlist."""
    with sqlite3.connect(TRACKER_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE id = ?", (watchlist_id,))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Removed stock with ID {watchlist_id} from watchlist.")
            return True
        else:
            logger.warning(f"Stock with ID {watchlist_id} not found in watchlist.")
            return False

def _get_current_price(symbol: str) -> float | None:
    """Fetch the current price of a stock using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            logger.warning(f"No historical data found for {symbol}.")
            return 0.0
        return round(hist['Close'].iloc[-1], 2)  # Return the last closing price rounded to 2 decimal places
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        return 0.0

def get_watchlist() -> pd.DataFrame:
    """Retrieve the watchlist as a pandas DataFrame."""
    with sqlite3.connect(TRACKER_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
        if df.empty:
            logger.info("Watchlist is empty.")
            return df
        # Fetch current prices for each stock in the watchlist
        current_price = []
        for symbol in df['symbol']:
            price = _get_current_price(symbol)
            current_price.append(price)
            
        df['current_price'] = current_price
        df["pl_pct"]= ((df["current_price"] - df["buy_price"]) / df["buy_price"]) * 100
        df["pl_pct"] = df["pl_pct"].round(2)
        df["investment_value"] = df["buy_price"] * df["quantity"].round(2)
        df["current_value"] = df["current_price"] * df["quantity"].round(2)
        df["pl_amount"] = df["current_value"] - df["investment_value"].round(2)
        return df