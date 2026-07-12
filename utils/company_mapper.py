import pandas as pd
from pathlib import Path
import yfinance as yf

# ── Build mapping from CSV at import time (instant, no network) ────────────────
_csv = Path(__file__).parent.parent / "data" / "nse_stocks.csv"
try:
    _df = pd.read_csv(_csv)
    _df.columns = _df.columns.str.strip()
    COMPANY_MAPPING: dict = dict(zip(_df["Symbol"].str.strip(), _df["Company"].str.strip()))
except Exception:
    COMPANY_MAPPING = {}
    _df = None

# ── Sector mapping — built only if a Sector column exists in the CSV ──────────
# Defensive by design: if nse_stocks.csv does not have a "Sector" column (or
# any equivalent), SECTOR_MAPPING stays empty and get_sector() returns None
# for every symbol — callers (e.g. scanner/engine.py, analytics module) treat
# that as "sector unknown" and degrade gracefully rather than crashing or
# guessing. As soon as a Sector column is added to the CSV, sector data
# starts flowing automatically with zero code changes elsewhere.
SECTOR_MAPPING: dict = {}
_SECTOR_COLUMN_CANDIDATES = ["Sector", "sector", "Industry", "industry", "GICS Sector"]

if _df is not None:
    for _col in _SECTOR_COLUMN_CANDIDATES:
        if _col in _df.columns:
            try:
                SECTOR_MAPPING = dict(
                    zip(_df["Symbol"].str.strip(), _df[_col].astype(str).str.strip())
                )
            except Exception:
                SECTOR_MAPPING = {}
            break

# Hardcoded extras not in the CSV



def get_company_names(stock_symbol: str) -> str:
    """Return a human-readable company name for news queries.

    Lookup order:
      1. In-memory dict built from nifty50.csv  (instant)
      2. yfinance Ticker.info                   (network — fallback only)
      3. Strip '.NS' suffix                     (last resort)
    """

    #print(f"Stock symbol received in get_company_names():",{stock_symbol},flush=True)
    if stock_symbol in COMPANY_MAPPING:
        #print(f"Mapped :{stock_symbol} -> {COMPANY_MAPPING[stock_symbol]}",flush=True)
        return COMPANY_MAPPING[stock_symbol]
    return stock_symbol.replace(".NS","")
    # try:
    #     info = yf.Ticker(stock_symbol).info
    #     return info.get("longName", stock_symbol.replace(".NS", ""))
    # except Exception:
    #     return stock_symbol.replace(".NS", "")


def get_stock_symbol(company_name: str):
    """Reverse lookup: company display name → Yahoo Finance symbol."""
    lower = company_name.lower()
    for symbol, name in COMPANY_MAPPING.items():
        if name.lower() == lower:
            return symbol
    return None


def get_sector(stock_symbol: str) -> str | None:
    """
    Return the sector for a stock symbol, if known.

    Lookup order:
      1. SECTOR_MAPPING built from nse_stocks.csv at import time, if the
         CSV has a Sector/Industry column (see _SECTOR_COLUMN_CANDIDATES
         above).
      2. None — sector is genuinely unknown. Callers must handle this
         gracefully (e.g. group under "Unknown" in sector analytics)
         rather than treating it as an error.

    This function deliberately does NOT fall back to a network call
    (e.g. yfinance) — sector lookups happen during the scanner's hot path
    for up to 150 stocks per run, and an extra network round-trip per
    stock would be a meaningful performance regression.
    """
    return SECTOR_MAPPING.get(stock_symbol)