import os
import pickle
import time
import warnings
from pathlib import Path

import yfinance as yf
import streamlit as st

# Suppress noisy yfinance / urllib3 warnings that are not actionable
warnings.filterwarnings("ignore", category=FutureWarning,    module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning,      module="yfinance")
warnings.filterwarnings("ignore", message=".*No price data found.*")
warnings.filterwarnings("ignore", message=".*No timezone found.*")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

_CACHE_DIR = (
    Path("/tmp/stockai_storage/price_cache") if os.name == "posix"
    else Path("storage/price_cache")
)
_TTL = 3600  # 1 hour


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("/", "_")
    return _CACHE_DIR / f"{safe}.pkl"


def _read_disk_cache(symbol: str):
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if time.time() - obj["ts"] < _TTL:
            return obj["data"]
    except Exception:
        pass
    return None


def _write_disk_cache(symbol: str, data) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(symbol).with_suffix(".tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump({"ts": time.time(), "data": data}, f)
        tmp.replace(_cache_path(symbol))
    except Exception:
        pass


def _fetch(symbol: str):
    data = yf.download(symbol, period="1y", progress=False)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data


@st.cache_data(ttl=_TTL)
def load_data(symbol: str):
    #print(f"Stock symbol received in load_data():",{symbol},flush=True)
    """Streamlit-cached fetch for UI session — checks disk cache first."""
    if " " in symbol:
        raise ValueError(f"Invalid yfinance symbol received:") 
    cached = _read_disk_cache(symbol)
    if cached is not None:
        return cached
    data = _fetch(symbol)
    if not data.empty:
        _write_disk_cache(symbol, data)
    return data


def load_data_raw(symbol: str):
    """Thread-safe fetch for background scanner — checks disk cache first."""
    cached = _read_disk_cache(symbol)
    if cached is not None:
        return cached
    data = _fetch(symbol)
    if not data.empty:
        _write_disk_cache(symbol, data)
    return data

def load_multi_timeframe_data(symbol):

    """
    Load stock data for multiple timeframes.
    
    Returns:
        {
            "weekly": weekly_df,
            "daily": daily_df
        }
    """

    try:

        # Weekly timeframe
        weekly_data = yf.download(
            symbol,
            period="2y",
            interval="1wk",
            progress=False,
            auto_adjust=True
        )

        # Daily timeframe
        daily_data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        # Flatten multi-level columns from yfinance (e.g. ("Close", "AAPL") → "Close")
        weekly_data.columns = [col[0] if isinstance(col, tuple) else col for col in weekly_data.columns]
        daily_data.columns  = [col[0] if isinstance(col, tuple) else col for col in daily_data.columns]

        # Clean empty rows
        weekly_data.dropna(inplace=True)
        daily_data.dropna(inplace=True)

        return {
            "weekly": weekly_data,
            "daily": daily_data
        }

    except Exception as e:

        print(f"Multi-timeframe load error for {symbol}: {e}")

        return {
            "weekly": None,
            "daily": None
        }