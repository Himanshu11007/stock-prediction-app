import json
import time
import requests
import streamlit as st
from pathlib import Path
from utils.company_mapper import get_company_names
import feedparser 
from urllib.parse import quote_plus

_NEWS_CACHE_DIR = Path("storage/news_cache")
_NEWS_TTL = 3600  # 12 hours — news doesn't change that fast


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

    try:
        #api_key    = st.secrets["API_KEY"]
        query_name = get_company_names(symbol_or_name)
        query_name = (
            query_name.lower().replace(" limited","").replace(" ltd","").strip()
        )
        cache_key = query_name.replace(" ","_")
        cached = _load_news_cache(cache_key)
        if cached is not None:
            return cached
        query = quote_plus(f"{query_name} stock")
        url = (
            f"https://news.google.com/rss/search?q={query}&hl=en&gl=IN&cdid=IN:en"
        )
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:10]:
            headlines.append(entry.title)
        _save_news_cache(cache_key, headlines)
        return headlines

    except Exception as e:
        print("Google News RSS errors:",e,flush=True)
        return []