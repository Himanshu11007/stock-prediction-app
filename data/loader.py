import pickle
import time
from pathlib import Path

import yfinance as yf
import streamlit as st

_CACHE_DIR = Path("storage/price_cache")
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
    """Streamlit-cached fetch for UI session — checks disk cache first."""
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
