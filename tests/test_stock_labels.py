"""
tests/test_stock_labels.py — factual labels on ranked stocks (presentation
only): Uptrend / Downtrend / Expensive from the stored component scores with
the existing 70 / 30 thresholds, a one-line explanation per label, and
Positive / Negative news labels only for recent, credible, clear-cut,
non-contradicted reports about the stock (point in time). The StockLens Score
and the ranking are never changed.
"""
import datetime as dt

from sqlmodel import Session

from catalysts import CLASSIFIER_VERSION
from db.models.market import StockAnalysisResult
from db.models.news import EventEntity, NewsArticle
from db.models.prediction import EventClassification, MarketEvent
from ranking import news_labels, presenter
from tests.test_product_api import H, _seed_run, client, db  # noqa: F401  (fixtures)

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _result(**scores):
    return StockAnalysisResult(run_id="R", symbol="AAA.NS", computed_at=NOW, stockai_score=60.0, score_coverage=0.9,
                               eligible=True, rank=3, components={k: {"score": v} for k, v in scores.items()},
                               fqvf={"coverage": 0.9}, engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0")


def _keys(r, limit=20):
    return [l["key"] for l in presenter.labels(r, limit, {"status": "FRESH"})]


def test_trend_and_expensive_labels_use_the_existing_thresholds():
    assert "uptrend" in _keys(_result(technical_trend=70.0)) and "uptrend" not in _keys(_result(technical_trend=69.9))
    assert "downtrend" in _keys(_result(technical_trend=30.0)) and "downtrend" not in _keys(_result(technical_trend=30.1))
    assert "expensive" in _keys(_result(valuation=25.0)) and "expensive" not in _keys(_result(valuation=31.0))
    assert _keys(_result(technical_trend=None)) == ["top_candidate"]            # missing data -> no label, never a guess
    both = _keys(_result(quality=80, valuation=75, financial_health=72, momentum=90, risk=20, technical_trend=85))
    assert both == ["top_candidate", "quality", "valuation", "financial_health", "momentum", "high_risk", "uptrend"]


def test_every_score_label_explains_itself():
    r = _result(quality=82, risk=18, technical_trend=22, valuation=12)
    items = presenter.labels(r, 20, {"status": "FRESH"})
    d = presenter.label_details(r, items)
    assert d["quality"] == "Quality score 82/100 (70 or more)"
    assert d["high_risk"].startswith("Risk score 18/100 (30 or less")
    assert d["downtrend"] == "Trend score 22/100 (30 or less)" and d["expensive"] == "Valuation score 12/100 (30 or less)"
    assert d["top_candidate"] == "Rank 3 by StockLens Score in the latest ranking"
    assert set(d) <= {i["key"] for i in items}


def _news(s, symbol, title, sentiment, available, tier="TIER1", contradicted=False, ext=1):
    ev = MarketEvent(source="news", source_event_id=f"e{ext}", event_type="BUSINESS_UPDATE", title=title,
                     published_at=available - dt.timedelta(minutes=20), effective_available_at=available)
    s.add(ev)
    s.flush()
    s.add(EventClassification(event_id=ev.id, classifier_version=CLASSIFIER_VERSION, sentiment=sentiment,
                              category="EARNINGS", contradicted=contradicted, created_at=available))
    s.add(EventEntity(event_id=ev.id, entity_type="COMPANY", entity_key=symbol, symbol=symbol, relation="SUBJECT",
                      created_at=available))
    s.add(NewsArticle(provider="official_rss", provider_article_id=f"a{ext}", url=f"https://www.reuters.com/{ext}",
                      canonical_url=f"https://reuters.com/{ext}", title=title, source_domain="reuters.com",
                      source_tier=tier, published_at=available - dt.timedelta(minutes=20),
                      ingested_at=available, effective_available_at=available, fingerprint=f"f{ext}", event_id=ev.id))
    s.commit()


def test_news_labels_only_for_recent_credible_clear_reports(db):
    with Session(db) as s:
        _news(s, "AAA.NS", "Alpha Q2 profit jumps 23%", 0.6, NOW - dt.timedelta(hours=3), ext=1)
        _news(s, "BBB.NS", "Beta faces SEBI penalty", -0.5, NOW - dt.timedelta(hours=5), ext=2)
        _news(s, "CCC.NS", "Gamma unclear update", 0.1, NOW - dt.timedelta(hours=1), ext=3)              # not clear-cut
        _news(s, "DDD.NS", "Delta rumour on a blog", 0.7, NOW - dt.timedelta(hours=1), tier="TIER3", ext=4)
        _news(s, "EEE.NS", "Epsilon disputed result", 0.8, NOW - dt.timedelta(hours=1), contradicted=True, ext=5)
        _news(s, "FFF.NS", "Phi old news", 0.8, NOW - dt.timedelta(days=4), ext=6)                     # older than 72 h
        _news(s, "GGG.NS", "Gamma future story", 0.8, NOW + dt.timedelta(hours=1), ext=7)              # not yet available
        out = news_labels.news_for(s, ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS", "FFF.NS", "GGG.NS"], NOW)
    assert set(out) == {"AAA.NS", "BBB.NS"}
    a, b = out["AAA.NS"][0], out["BBB.NS"][0]
    assert (a["tone"], a["title"], a["source"], a["url"]) == ("positive", "Alpha Q2 profit jumps 23%", "reuters.com",
                                                              "https://www.reuters.com/1")
    assert b["tone"] == "negative" and b["published_at"] < b["available_at"]
    assert news_labels.news_labels(out["AAA.NS"]) == [{"key": "news_positive", "label": "Positive news", "tone": "positive"}]
    assert news_labels.news_labels(out["BBB.NS"]) == [{"key": "news_negative", "label": "Negative news", "tone": "negative"}]
    assert news_labels.news_detail(out["BBB.NS"], "negative").startswith("Beta faces SEBI penalty (reuters.com, reported ")
    assert news_labels.news_for(s, [], NOW) == {}


def test_top_candidates_carry_news_details_and_the_not_advice_note(client, db):
    _seed_run(db)
    with Session(db) as s:
        _news(s, "AAA.NS", "Alpha wins large order", 0.6, dt.datetime.now(UTC) - dt.timedelta(hours=2), ext=11)
    data = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert "not predictions or buy/sell advice" in data["labels_note"]
    item = next(i for i in data["items"] if i["symbol"] == "AAA.NS")
    assert {"key": "news_positive", "label": "Positive news", "tone": "positive"} in item["labels"]
    assert item["news"][0]["title"] == "Alpha wins large order" and item["news"][0]["url"].startswith("https://")
    assert item["label_details"]["news_positive"].startswith("Alpha wins large order")
    other = next(i for i in data["items"] if i["symbol"] == "BBB.NS")
    assert other["news"] == [] and not any(l["key"].startswith("news_") for l in other["labels"])
    assert [i["rank"] for i in data["items"]] == sorted(i["rank"] for i in data["items"])     # ranking order untouched
