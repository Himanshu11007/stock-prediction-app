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

# Hardcoded extras not in the CSV



def get_company_names(stock_symbol: str) -> str:
    """Return a human-readable company name for news queries.

    Lookup order:
      1. In-memory dict built from nifty50.csv  (instant)
      2. yfinance Ticker.info                   (network — fallback only)
      3. Strip '.NS' suffix                     (last resort)
    """

    print(f"Stock symbol received in get_company_names():",{stock_symbol},flush=True)
    if stock_symbol in COMPANY_MAPPING:
        print(f"Mapped :{stock_symbol} -> {COMPANY_MAPPING[stock_symbol]}",flush=True)
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
