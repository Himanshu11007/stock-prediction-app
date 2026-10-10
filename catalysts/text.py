"""
catalysts/text.py — handling of untrusted article text.

  sanitize(text, limit)   strip markup, control characters and scripts;
                          collapse whitespace; cap the length
  canonical_url(url)      scheme/host lower-case, no tracking parameters,
                          no fragment, no trailing slash
  tokens(text)            lower-case word tokens without stop words
  fingerprint(title)      order-independent signature of the title tokens
  similarity(a, b)        Jaccard similarity of title token sets
  language(text)          "en" or "unknown" (conservative; non-English text is
                          kept but not classified)
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MAX_TITLE = 300
MAX_EXCERPT = 500
TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|cmpid$|ito$|from$)", re.I)
STOP = frozenset("""a an and are as at be by for from has have in into is it its of on or that the this to was
were will with after over under amid says said say new news update updates report reports today live""".split())
_TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)
_WORD = re.compile(r"[a-z0-9][a-z0-9&.'-]*")


def sanitize(text: str | None, limit: int = MAX_EXCERPT) -> str:
    if not text:
        return ""
    t = html.unescape(_TAG.sub(" ", str(text)))
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C" or ch in " \n\t")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def canonical_url(url: str) -> str:
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not TRACKING.match(k)))
    host = p.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunsplit(((p.scheme or "https").lower(), host, path, query, ""))


def domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def tokens(text: str) -> set[str]:
    return {w.strip(".'-") for w in _WORD.findall(text.lower()) if w not in STOP and len(w.strip(".'-")) > 1}


def fingerprint(title: str) -> str:
    return hashlib.sha1(" ".join(sorted(tokens(title))).encode("utf-8")).hexdigest()[:16]


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


FOREIGN = frozenset("""le la les des du et est une pour dans sur el los las del por con una und der die das
mit nicht ist ein eine il che di per sono não uma com são ke ki hai aur""".split())


def language(text: str) -> str:
    """'en' when the text is mostly Latin letters and not dominated by common
    function words of other Latin-script languages; otherwise 'unknown'.
    Headlines often have no English function words, so their absence is
    not evidence of another language."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z") / len(letters)
    words = re.findall(r"[a-z]+", text.lower())
    foreign = sum(1 for w in words if w in FOREIGN)
    return "en" if latin > 0.9 and foreign < max(2, 0.15 * len(words)) else "unknown"
