import json
import time
import requests
import streamlit as st
from pathlib import Path
from utils.company_mapper import get_company_names

_NEWS_CACHE_DIR = Path("storage/news_cache")
_NEWS_TTL = 43200  # 12 hours — news doesn't change that fast


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("/", "_")
    return _NEWS_CACHE_DIR / f"{safe}.json"


def _load_news_cache(symbol: str) -> list | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - payload.get("ts", 0) < _NEWS_TTL:
            return payload["headlines"]
    except Exception:
        pass
    return None


def _save_news_cache(symbol: str, headlines: list) -> None:
    _NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(symbol).with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps({"ts": time.time(), "headlines": headlines}),
            encoding="utf-8",
        )
        tmp.replace(_cache_path(symbol))  # atomic rename
    except Exception:
        pass


def fetch_news(symbol_or_name: str) -> list[str]:
    """
    Fetch up to 5 recent headlines for a stock.
    Results are cached on disk for 12 h to avoid exhausting the NewsAPI quota.
    """
    cached = _load_news_cache(symbol_or_name)
    if cached is not None:
        return cached

    try:
        api_key    = st.secrets["API_KEY"]
        query_name = get_company_names(symbol_or_name)
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query_name}&language=en&sortBy=publishedAt&apiKey={api_key}"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        articles  = response.json().get("articles", [])
        headlines = [a["title"] for a in articles[:5] if a.get("title")]
    except Exception:
        headlines = []

    _save_news_cache(symbol_or_name, headlines)
    return headlines
