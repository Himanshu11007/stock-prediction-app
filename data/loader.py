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


def _cache_path_weekly(symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("/", "_")
    return _CACHE_DIR / f"{safe}_weekly.pkl"


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
    # yf.download() can return None on auth/crumb errors instead of raising
    if data is None or not hasattr(data, "columns"):
        return None
    if data.empty:
        return data
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data


@st.cache_data(ttl=_TTL)
def load_data(symbol: str):
    """Streamlit-cached fetch for UI session — checks disk cache first."""
    if " " in symbol:
        raise ValueError(f"Invalid yfinance symbol received: {symbol!r}")
    cached = _read_disk_cache(symbol)
    if cached is not None:
        return cached
    data = _fetch(symbol)
    if data is not None and not data.empty:
        _write_disk_cache(symbol, data)
    return data


def load_data_raw(symbol: str):
    """Thread-safe fetch for background scanner — checks disk cache first."""
    cached = _read_disk_cache(symbol)
    if cached is not None:
        return cached
    data = _fetch(symbol)
    if data is not None and not data.empty:
        _write_disk_cache(symbol, data)
    return data


def load_multi_timeframe_data(symbol: str) -> dict:
    """
    Load stock data for multiple timeframes with disk caching.

    - Daily  : reuses the existing 1y disk cache (no extra download).
    - Weekly : cached separately under <symbol>_weekly.pkl (TTL = 1 h).

    Returns:
        {"weekly": weekly_df, "daily": daily_df}
    """
    # ── Daily: reuse the same disk cache as load_data_raw (zero extra download) ──
    daily_data = _read_disk_cache(symbol)
    if daily_data is None:
        try:
            daily_data = yf.download(
                symbol, period="1y", interval="1d",
                progress=False, auto_adjust=True,
            )
            daily_data.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in daily_data.columns
            ]
            daily_data.dropna(inplace=True)
            if not daily_data.empty:
                _write_disk_cache(symbol, daily_data)
        except Exception as e:
            print(f"Multi-TF daily load error for {symbol}: {e}")
            daily_data = None

    # ── Weekly: separate cache entry ─────────────────────────────────────────
    weekly_path = _cache_path_weekly(symbol)
    weekly_data = None
    if weekly_path.exists():
        try:
            with weekly_path.open("rb") as f:
                obj = pickle.load(f)
            if time.time() - obj["ts"] < _TTL:
                weekly_data = obj["data"]
        except Exception:
            pass

    if weekly_data is None:
        try:
            weekly_data = yf.download(
                symbol, period="2y", interval="1wk",
                progress=False, auto_adjust=True,
            )
            weekly_data.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in weekly_data.columns
            ]
            weekly_data.dropna(inplace=True)
            if not weekly_data.empty:
                tmp = weekly_path.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    pickle.dump({"ts": time.time(), "data": weekly_data}, f)
                tmp.replace(weekly_path)
        except Exception as e:
            print(f"Multi-TF weekly load error for {symbol}: {e}")
            weekly_data = None

    return {"weekly": weekly_data, "daily": daily_data}