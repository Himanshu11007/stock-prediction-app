import json
import time
import streamlit as st
from pathlib import Path
from utils.company_mapper import get_company_names
from utils.logger import get_logger
import feedparser
from urllib.parse import quote_plus
from config import STORAGE_DIR
logger = get_logger(__name__)

_NEWS_CACHE_DIR = STORAGE_DIR/ "news_cache"
_NEWS_TTL = 3600


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
        tmp.replace(_cache_path(symbol))
    except Exception as e:
        logger.warning("Failed to save news cache for %s: %s", symbol, e)


def fetch_news(symbol_or_name: str) -> list[str]:
    """Fetch up to 10 recent headlines. Results cached on disk for 1 h."""
    try:
        query_name = get_company_names(symbol_or_name)
        query_name = (
            query_name.lower().replace(" limited", "").replace(" ltd", "").strip()
        )
        cache_key = query_name.replace(" ", "_")

        cached = _load_news_cache(cache_key)
        if cached is not None:
            logger.debug("News cache hit for %s (%d headlines)", symbol_or_name, len(cached))
            return cached

        logger.debug("Fetching news for %s (query: %s)", symbol_or_name, query_name)
        query = quote_plus(f"{query_name} stock")
        url   = f"https://news.google.com/rss/search?q={query}&hl=en&gl=IN&cdid=IN:en"
        feed  = feedparser.parse(url)

        headlines = [entry.title for entry in feed.entries[:10]]
        logger.info("News fetched for %s: %d headlines", symbol_or_name, len(headlines))

        _save_news_cache(cache_key, headlines)
        return headlines

    except Exception as e:
        logger.error("News fetch failed for %s: %s", symbol_or_name, e)
        return []